#!/usr/bin/env python3
"""
train.py — SmolVLA 파인튜닝 (train/val split + early stopping + 시각화)
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


# ── 인자 파싱 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo", default="local/smolvla_finetune_object")
    p.add_argument("--dataset_root", default="data/lerobot_datasets")
    p.add_argument("--pretrained",   default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--output_dir",   default="outputs/smolvla_ft_object")
    p.add_argument("--steps",        type=int,   default=20000)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--save_freq",    type=int,   default=2000)
    p.add_argument("--val_freq",     type=int,   default=500,
                   help="몇 스텝마다 validation 실행할지")
    p.add_argument("--val_ratio",    type=float, default=0.1,
                   help="전체 에피소드 중 val 비율 (기본 10%%)")
    p.add_argument("--patience",     type=int,   default=10,
                   help="early stopping patience (val 체크 횟수 기준)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--wandb",        action="store_true")
    return p.parse_args()


# ── 모델 & 데이터 로드 ────────────────────────────────────────────────────────

def load_policy(args, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"  모델 로드 시작: {args.pretrained}")
    policy = SmolVLAPolicy.from_pretrained(args.pretrained)

    # 전체 동결 후 선택적 해제
    for param in policy.parameters():
        param.requires_grad = False

    # 학습 대상:
    #   - model.vlm_with_expert.lm_expert  : action expert transformer (96.7M)
    #   - model.action_in_proj / action_out_proj / action_time_mlp_* / state_proj : 0.8M
    # 동결 대상:
    #   - model.vlm_with_expert.vlm        : 언어 모델 (507.5M)
    for name, param in policy.named_parameters():
        if "vlm_with_expert.vlm" in name:
            continue  # VLM 동결 유지
        param.requires_grad = True

    total     = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  전체 파라미터: {total/1e6:.1f}M")
    print(f"  학습 파라미터: {trainable/1e6:.1f}M  ({trainable/total*100:.1f}%)")
    return policy.to(device)


def load_datasets(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import random

    action_horizon, fps = 50, 10.0
    delta = {
        "action": [i / fps for i in range(action_horizon)],
        "observation.images.image": [0.0],
        "observation.images.image2": [0.0],
        "observation.state": [0.0],
    }

    # 전체 에피소드 수 파악
    tmp = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root)
    total_eps = tmp.num_episodes
    all_eps = list(range(total_eps))

    random.seed(args.seed)
    random.shuffle(all_eps)
    n_val = max(1, int(total_eps * args.val_ratio))
    val_eps   = sorted(all_eps[:n_val])
    train_eps = sorted(all_eps[n_val:])

    print(f"  총 에피소드: {total_eps}  →  train: {len(train_eps)}, val: {len(val_eps)}")

    train_ds = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root,
                              episodes=train_eps, delta_timestamps=delta)
    val_ds   = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root,
                              episodes=val_eps,   delta_timestamps=delta)
    return train_ds, val_ds


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        """val_loss가 개선되면 False, patience 초과하면 True 반환"""
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── 로깅 & 시각화 ─────────────────────────────────────────────────────────────

class TrainLogger:
    def __init__(self, output_dir: Path, use_wandb: bool, total_steps: int):
        self.output_dir = output_dir
        self.log_path = output_dir / "train_log.jsonl"
        self.loss_window = deque(maxlen=100)
        self.start_time = time.time()
        self.total_steps = total_steps
        # 시각화용 기록
        self.train_steps, self.train_losses = [], []
        self.val_steps,   self.val_losses   = [], []
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project="smolvla_finetune")
                self.wandb = wandb
            except Exception:
                pass

    def log_train(self, step: int, loss: float, lr: float, step_time: float) -> dict:
        self.loss_window.append(loss)
        avg_loss = sum(self.loss_window) / len(self.loss_window)
        remaining = (self.total_steps - step) * step_time
        h, m = divmod(int(remaining), 3600); m, s = divmod(m, 60)
        record = {
            "step": step, "loss": round(loss, 5), "avg_loss": round(avg_loss, 5),
            "lr": f"{lr:.2e}", "eta": f"{h:02d}:{m:02d}:{s:02d}"
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb:
            self.wandb.log({"train/loss": loss, "train/avg_loss": avg_loss, "lr": lr}, step=step)
        self.train_steps.append(step)
        self.train_losses.append(loss)
        return record

    def log_val(self, step: int, val_loss: float, best_loss: float, patience_count: int):
        record = {"step": step, "val_loss": round(val_loss, 5),
                  "best_val_loss": round(best_loss, 5), "patience": patience_count}
        with open(self.output_dir / "val_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb:
            self.wandb.log({"val/loss": val_loss, "val/best": best_loss}, step=step)
        self.val_steps.append(step)
        self.val_losses.append(val_loss)

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Train loss (100-step 이동평균 포함)
            ax = axes[0]
            ax.plot(self.train_steps, self.train_losses, alpha=0.25, color="steelblue", label="train loss (raw)")
            if len(self.train_losses) >= 50:
                window = 100
                smooth = []
                buf = deque(maxlen=window)
                for v in self.train_losses:
                    buf.append(v); smooth.append(sum(buf) / len(buf))
                ax.plot(self.train_steps, smooth, color="steelblue", linewidth=2, label=f"train loss (avg{window})")
            ax.set_xlabel("Step"); ax.set_ylabel("Loss")
            ax.set_title("Train Loss"); ax.legend(); ax.grid(True, alpha=0.3)

            # Val loss
            ax2 = axes[1]
            if self.val_losses:
                ax2.plot(self.val_steps, self.val_losses, marker="o", color="tomato",
                         linewidth=2, markersize=4, label="val loss")
                best_idx = self.val_losses.index(min(self.val_losses))
                ax2.axvline(self.val_steps[best_idx], color="green", linestyle="--",
                            label=f"best val @ step {self.val_steps[best_idx]}")
            ax2.set_xlabel("Step"); ax2.set_ylabel("Loss")
            ax2.set_title("Validation Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.output_dir / "loss_curve.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  그래프 저장: {save_path}")
        except Exception as e:
            print(f"  [경고] 시각화 실패: {e}")

    def finish(self):
        self.plot()
        if self.wandb:
            self.wandb.finish()


# ── Validation 루프 ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(policy, val_loader, tokenizer, accelerator):
    policy.eval()
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
        device = accelerator.device
        text_inputs = tokenizer(
            batch["task"], return_tensors="pt",
            padding="max_length", max_length=22, truncation=True
        )
        batch["observation.language.tokens"]        = text_inputs["input_ids"].to(device)
        batch["observation.language.attention_mask"] = text_inputs["attention_mask"].bool().to(device)

        with accelerator.autocast():
            output = policy.forward(batch)
            loss = output[0] if isinstance(output, (list, tuple)) else output
        total_loss += loss.item()
        n_batches  += 1

    policy.train()
    return total_loss / max(n_batches, 1)


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device, is_main = accelerator.device, accelerator.is_main_process
    torch.manual_seed(args.seed + accelerator.process_index)

    # 1. 컴포넌트 초기화
    policy = load_policy(args, device)
    train_ds, val_ds = load_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, drop_last=False)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Accelerator 준비
    policy, optimizer, train_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, scheduler
    )
    val_loader = accelerator.prepare(val_loader)

    logger       = TrainLogger(output_dir, args.wandb, args.steps) if is_main else None
    early_stop   = EarlyStopping(patience=args.patience) if is_main else None
    best_ckpt_dir = output_dir / "best_checkpoint"

    # 3. 학습 루프
    policy.train()
    data_iter = iter(train_loader)
    pbar = tqdm(total=args.steps, desc="학습", disable=not is_main)

    stopped_early = False

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        text_inputs = tokenizer(
            batch["task"], return_tensors="pt",
            padding="max_length", max_length=22, truncation=True
        )
        batch["observation.language.tokens"]        = text_inputs["input_ids"].to(device)
        batch["observation.language.attention_mask"] = text_inputs["attention_mask"].bool().to(device)

        t0 = time.perf_counter()
        with accelerator.autocast():
            output = policy.forward(batch)
            loss = output[0] if isinstance(output, (list, tuple)) else output

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            rec = logger.log_train(step, loss_val, current_lr, time.perf_counter() - t0)
            pbar.set_postfix(loss=f"{rec['loss']:.4f}", lr=rec['lr'])
            pbar.update(1)

        # ── Validation & Early Stopping ──────────────────────────────────────
        if step % args.val_freq == 0:
            accelerator.wait_for_everyone()
            val_loss = run_validation(policy, val_loader, tokenizer, accelerator)

            if is_main:
                stop = early_stop.update(val_loss)
                logger.log_val(step, val_loss, early_stop.best_loss, early_stop.counter)

                status = f"patience {early_stop.counter}/{args.patience}"
                if early_stop.counter == 0:
                    status = "best! 모델 저장"
                    # best 체크포인트 저장
                    accelerator.unwrap_model(policy).save_pretrained(str(best_ckpt_dir / "policy"))

                pbar.write(
                    f"  [val step {step:6d}]  val_loss={val_loss:.5f}  "
                    f"best={early_stop.best_loss:.5f}  {status}"
                )

                # early stop 신호를 파일로 브로드캐스트
                stop_flag = output_dir / ".stop"
                if stop:
                    stop_flag.touch()

            accelerator.wait_for_everyone()
            # 모든 프로세스가 stop 플래그 확인
            if (output_dir / ".stop").exists():
                if is_main:
                    pbar.write(f"\n  Early stopping! patience {args.patience} 소진 (step {step})")
                stopped_early = True
                break

        # ── 정기 체크포인트 저장 ─────────────────────────────────────────────
        if step % args.save_freq == 0:
            accelerator.wait_for_everyone()
            if is_main:
                ckpt = output_dir / f"checkpoint_{step:07d}"
                accelerator.unwrap_model(policy).save_pretrained(str(ckpt / "policy"))
                print(f"\n  체크포인트 저장: {ckpt}")

    # 4. 마무리
    if is_main:
        pbar.close()
        # stop 플래그 정리
        stop_flag = output_dir / ".stop"
        if stop_flag.exists():
            stop_flag.unlink()

        if not stopped_early:
            # 마지막 스텝도 저장
            final_ckpt = output_dir / f"checkpoint_{args.steps:07d}"
            if not final_ckpt.exists():
                accelerator.unwrap_model(policy).save_pretrained(str(final_ckpt / "policy"))

        print(f"\n  best val loss: {early_stop.best_loss:.5f}  →  {best_ckpt_dir}")
        logger.finish()
        print("\n학습 완료!")


if __name__ == "__main__":
    main()
