#!/usr/bin/env python3
"""
train.py — SmolVLA 파인튜닝 (LoRA + 낮은 LR + 잦은 validation)
----------------------------------------------------------------
주요 변경 사항:
  1. LoRA   : lm_expert attention 레이어에만 적용 → catastrophic forgetting 방지
  2. LR     : 5e-6 (기존 2e-5 대비 4배 낮음)
  3. val_freq: 200 스텝 (기존 500)
  4. steps  : 3000 (기존 20000)

단일 GPU 실행:
    python scripts/train.py

멀티 GPU:
    accelerate launch scripts/train.py


cd /media/idna/Data/study1/smolvla_finetune
/media/idna/Data/envs/smolvla/bin/python scripts/train.py --epochs 5




"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ── 인자 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo", default="local/smolvla_finetune_object")
    p.add_argument("--dataset_root", default="data/lerobot_datasets")
    p.add_argument("--pretrained",   default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--output_dir",   default="outputs/smolvla_ft_lora")

    # ── 학습 하이퍼파라미터 ──
    p.add_argument("--epochs",       type=int,   default=5,
                   help="총 학습 에포크 수 (권장: 3~10). --steps 지정 시 steps 우선")
    p.add_argument("--steps",        type=int,   default=None,
                   help="총 학습 스텝 수 (지정하면 epochs 무시)")
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=5e-6,
                   help="학습률 (LoRA는 5e-6~1e-5 권장)")
    p.add_argument("--save_freq",    type=int,   default=1,
                   help="몇 에포크마다 체크포인트 저장 (기본 1=매 에포크)")
    p.add_argument("--val_freq",     type=int,   default=1,
                   help="몇 에포크마다 validation 실행 (기본 1=매 에포크)")
    p.add_argument("--val_ratio",    type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=5,
                   help="early stopping patience (에포크 기준)")
    p.add_argument("--seed",         type=int,   default=42)

    # ── LoRA ──
    p.add_argument("--lora_rank",    type=int,   default=16,
                   help="LoRA rank (높을수록 표현력↑, forgetting↑)")
    p.add_argument("--lora_alpha",   type=int,   default=32,
                   help="LoRA scaling = alpha/rank")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora",      action="store_true",
                   help="LoRA 없이 전체 파인튜닝 (권장하지 않음)")

    p.add_argument("--wandb",        action="store_true")
    return p.parse_args()


# ── 모델 로드 (LoRA 적용) ─────────────────────────────────────────────────────

def load_policy(args, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"  모델 로드: {args.pretrained}")
    policy = SmolVLAPolicy.from_pretrained(args.pretrained)

    # 전체 동결
    for param in policy.parameters():
        param.requires_grad = False

    if args.no_lora:
        # ── LoRA 없는 기존 방식 ──────────────────────────────────────────────
        for name, param in policy.named_parameters():
            if "vlm_with_expert.vlm" in name:
                continue
            param.requires_grad = True
        print("  [!] LoRA 비활성화: lm_expert + projection 전체 학습")

    else:
        # ── LoRA 적용 ────────────────────────────────────────────────────────
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("pip install peft 를 먼저 실행하세요.")

        # lm_expert 내부의 attention projection 레이어 이름 탐색
        target_modules = set()
        for name, module in policy.named_modules():
            if "lm_expert" not in name:
                continue
            if isinstance(module, torch.nn.Linear):
                short = name.split(".")[-1]
                # attention q/k/v/o proj 와 ffn 레이어만 타겟
                if short in ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"):
                    target_modules.add(short)

        if not target_modules:
            # fallback: 모든 Linear 이름 수집
            for name, module in policy.named_modules():
                if "lm_expert" in name and isinstance(module, torch.nn.Linear):
                    target_modules.add(name.split(".")[-1])

        print(f"  LoRA 타겟 레이어: {sorted(target_modules)}")

        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=sorted(target_modules),
            bias="none",
        )
        policy = get_peft_model(policy, lora_cfg)

        # LoRA 외에 action projection 레이어도 학습
        for name, param in policy.named_parameters():
            if any(k in name for k in (
                "action_in_proj", "action_out_proj",
                "action_time_mlp", "state_proj",
            )):
                param.requires_grad = True

        policy.print_trainable_parameters()

    total     = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  전체: {total/1e6:.1f}M  |  학습: {trainable/1e6:.2f}M  "
          f"({trainable/total*100:.2f}%)")

    return policy.to(device)


# ── 데이터셋 로드 ──────────────────────────────────────────────────────────────

def load_datasets(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import random

    fps, horizon = 10.0, 50
    delta = {
        "action":                           [i / fps for i in range(horizon)],
        "observation.images.image":         [0.0],
        "observation.images.image2":        [0.0],
        "observation.state":                [0.0],
    }

    tmp = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root)
    total_eps = tmp.num_episodes
    all_eps   = list(range(total_eps))

    random.seed(args.seed)
    random.shuffle(all_eps)
    n_val     = max(1, int(total_eps * args.val_ratio))
    val_eps   = sorted(all_eps[:n_val])
    train_eps = sorted(all_eps[n_val:])
    print(f"  에피소드: 총 {total_eps}  →  train {len(train_eps)} / val {len(val_eps)}")

    train_ds = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root,
                              episodes=train_eps, delta_timestamps=delta)
    val_ds   = LeRobotDataset(repo_id=args.dataset_repo, root=args.dataset_root,
                              episodes=val_eps,   delta_timestamps=delta)
    return train_ds, val_ds


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience):
        self.patience    = patience
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def update(self, val_loss):
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── 로거 & 시각화 ──────────────────────────────────────────────────────────────

class TrainLogger:
    def __init__(self, output_dir: Path, use_wandb: bool, total_steps: int):
        self.output_dir  = output_dir
        self.log_path    = output_dir / "train_log.jsonl"
        self.loss_window = deque(maxlen=100)
        self.start_time  = time.time()
        self.total_steps = total_steps
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

    def log_train(self, step, loss, lr, step_time):
        self.loss_window.append(loss)
        avg = sum(self.loss_window) / len(self.loss_window)
        rem = (self.total_steps - step) * step_time
        h, m = divmod(int(rem), 3600); m, s = divmod(m, 60)
        rec = {"step": step, "loss": round(loss, 5), "avg_loss": round(avg, 5),
               "lr": f"{lr:.2e}", "eta": f"{h:02d}:{m:02d}:{s:02d}"}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb:
            self.wandb.log({"train/loss": loss, "lr": lr}, step=step)
        self.train_steps.append(step); self.train_losses.append(loss)
        return rec

    def log_val(self, step, val_loss, best_loss, patience_count):
        rec = {"step": step, "val_loss": round(val_loss, 5),
               "best_val_loss": round(best_loss, 5), "patience": patience_count}
        with open(self.output_dir / "val_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb:
            self.wandb.log({"val/loss": val_loss}, step=step)
        self.val_steps.append(step); self.val_losses.append(val_loss)

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.plot(self.train_steps, self.train_losses,
                    alpha=0.2, color="steelblue", label="raw")
            if len(self.train_losses) >= 20:
                buf, smooth = deque(maxlen=50), []
                for v in self.train_losses:
                    buf.append(v); smooth.append(sum(buf)/len(buf))
                ax.plot(self.train_steps, smooth, color="steelblue",
                        linewidth=2, label="avg50")
            ax.set_title("Train Loss"); ax.legend(); ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            if self.val_losses:
                ax2.plot(self.val_steps, self.val_losses,
                         marker="o", color="tomato", linewidth=2, label="val loss")
                best_i = self.val_losses.index(min(self.val_losses))
                ax2.axvline(self.val_steps[best_i], color="green",
                            linestyle="--", label=f"best @ {self.val_steps[best_i]}")
            ax2.set_title("Validation Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            p = self.output_dir / "loss_curve.png"
            plt.savefig(p, dpi=150); plt.close()
            print(f"  그래프 저장: {p}")
        except Exception as e:
            print(f"  [경고] 시각화 실패: {e}")

    def finish(self):
        self.plot()
        if self.wandb:
            self.wandb.finish()


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(policy, val_loader, tokenizer, accelerator):
    policy.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        device = accelerator.device
        tok = tokenizer(batch["task"], return_tensors="pt",
                        padding="max_length", max_length=22, truncation=True)
        batch["observation.language.tokens"]         = tok["input_ids"].to(device)
        batch["observation.language.attention_mask"] = tok["attention_mask"].bool().to(device)
        with accelerator.autocast():
            out  = policy.forward(batch)
            loss = out[0] if isinstance(out, (list, tuple)) else out
        total += loss.item(); n += 1
    policy.train()
    return total / max(n, 1)


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device, is_main = accelerator.device, accelerator.is_main_process
    torch.manual_seed(args.seed + accelerator.process_index)

    # 1. 초기화
    policy           = load_policy(args, device)
    train_ds, val_ds = load_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, drop_last=False)

    # 에포크 vs 스텝 결정
    steps_per_epoch = len(train_loader)
    if args.steps is not None:
        total_steps = args.steps
        total_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    else:
        total_epochs = args.epochs
        total_steps  = total_epochs * steps_per_epoch

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=total_steps, pct_start=0.05)

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Instruct", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Accelerator 준비
    policy, optimizer, train_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare(val_loader)

    if is_main:
        mode = "LoRA (forgetting 방지)" if not args.no_lora else "Full FT"
        print("=" * 65)
        print(f"  SmolVLA 파인튜닝  [{mode}]")
        print(f"  epochs={total_epochs}  steps/epoch={steps_per_epoch}  "
              f"total_steps={total_steps}")
        print(f"  lr={args.lr}  batch={args.batch_size}")
        if not args.no_lora:
            print(f"  lora_rank={args.lora_rank}  alpha={args.lora_alpha}  "
                  f"dropout={args.lora_dropout}")
        print(f"  val_freq={args.val_freq}에포크  patience={args.patience}에포크")
        print("=" * 65)

    logger        = TrainLogger(output_dir, args.wandb, total_steps) if is_main else None
    early_stop    = EarlyStopping(args.patience)
    best_ckpt_dir = output_dir / "best_checkpoint"
    global_step   = 0

    # 3. 에포크 학습 루프
    policy.train()

    for epoch in range(1, total_epochs + 1):
        epoch_losses = []
        epoch_start  = time.perf_counter()

        for batch in train_loader:
            global_step += 1

            tok = tokenizer(batch["task"], return_tensors="pt",
                            padding="max_length", max_length=22, truncation=True)
            batch["observation.language.tokens"]         = tok["input_ids"].to(device)
            batch["observation.language.attention_mask"] = tok["attention_mask"].bool().to(device)

            t0 = time.perf_counter()
            with accelerator.autocast():
                out  = policy.forward(batch)
                loss = out[0] if isinstance(out, (list, tuple)) else out

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step(); optimizer.zero_grad(); scheduler.step()

            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            epoch_losses.append(loss_val)

            if is_main:
                lr  = optimizer.param_groups[0]["lr"]
                rec = logger.log_train(global_step, loss_val, lr, time.perf_counter() - t0)
                batch_in_epoch = len(epoch_losses)
                print(f"[ep {epoch:3d}/{total_epochs} | "
                      f"step {batch_in_epoch:4d}/{steps_per_epoch}]  "
                      f"loss={loss_val:.5f}  avg={rec['avg_loss']:.5f}  "
                      f"lr={rec['lr']}")

        # ── 에포크 요약 ──────────────────────────────────────────────────────
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.perf_counter() - epoch_start

        if is_main:
            remaining = (total_epochs - epoch) * epoch_time
            h, m = divmod(int(remaining), 3600); m, s = divmod(m, 60)
            print(f"\n{'─'*65}")
            print(f"  [에포크 {epoch:3d}/{total_epochs} 완료]  "
                  f"avg_loss={epoch_loss:.5f}  "
                  f"time={epoch_time:.1f}s  eta={h:02d}:{m:02d}:{s:02d}")

        # ── Validation & Early Stopping (매 val_freq 에포크) ─────────────────
        if epoch % args.val_freq == 0:
            accelerator.wait_for_everyone()
            val_loss = run_validation(policy, val_loader, tokenizer, accelerator)

            if is_main:
                stop   = early_stop.update(val_loss)
                logger.log_val(epoch, val_loss, early_stop.best_loss, early_stop.counter)

                status = f"patience {early_stop.counter}/{args.patience}"
                if early_stop.counter == 0:
                    status = "★ best! 모델 저장"
                    unwrapped = accelerator.unwrap_model(policy)
                    if hasattr(unwrapped, "merge_and_unload"):
                        unwrapped.merge_and_unload().save_pretrained(
                            str(best_ckpt_dir / "policy"))
                    else:
                        unwrapped.save_pretrained(str(best_ckpt_dir / "policy"))

                print(f"  >>> [val ep {epoch:3d}]  "
                      f"val={val_loss:.5f}  best={early_stop.best_loss:.5f}  {status}")
                print(f"{'─'*65}\n")

                if stop:
                    (output_dir / ".stop").touch()

            accelerator.wait_for_everyone()
            if (output_dir / ".stop").exists():
                if is_main:
                    print(f"  Early stopping! (ep {epoch}, "
                          f"patience {args.patience} 소진)")
                break

        # ── 체크포인트 저장 (매 save_freq 에포크) ────────────────────────────
        if epoch % args.save_freq == 0 and is_main:
            ckpt      = output_dir / f"checkpoint_ep{epoch:04d}"
            unwrapped = accelerator.unwrap_model(policy)
            if hasattr(unwrapped, "merge_and_unload"):
                unwrapped.merge_and_unload().save_pretrained(str(ckpt / "policy"))
            else:
                unwrapped.save_pretrained(str(ckpt / "policy"))
            print(f"  체크포인트 저장: {ckpt}\n")

    # 4. 마무리
    accelerator.wait_for_everyone()
    if is_main:
        stop_flag = output_dir / ".stop"
        if stop_flag.exists():
            stop_flag.unlink()

        print(f"\n  best val loss: {early_stop.best_loss:.5f}  →  {best_ckpt_dir}")
        logger.finish()
        print("\n학습 완료!")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
