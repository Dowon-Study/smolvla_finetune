#!/usr/bin/env python3
"""
train.py — SmolVLA 파인튜닝 (OOM 및 Chunk 에러 완벽 해결본)
================================================================

실행 (4-GPU DDP):
    accelerate launch --num_processes=4 scripts/train.py
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


# ── 인자 파싱 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo", default="local/smolvla_finetune_object")
    p.add_argument("--dataset_root", default="data/lerobot_datasets",
                   help="LeRobot 데이터셋 로컬 경로")
    p.add_argument("--pretrained",   default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--output_dir",   default="outputs/smolvla_ft_object")
    p.add_argument("--steps",        type=int,   default=20000,
                   help="총 학습 스텝 수")
    
    # ⭐ [오류 해결 1] 3080 10GB 메모리 초과(OOM)를 막기 위해 기본 배치를 2로 축소
    p.add_argument("--batch_size",   type=int,   default=2,
                   help="GPU당 배치 크기 (3080 10GB 환경에서는 2 권장)")
    
    p.add_argument("--lr",           type=float, default=2e-5,
                   help="학습률")
    p.add_argument("--warmup_steps", type=int,   default=500,
                   help="LR 워밍업 스텝 수")
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="gradient clipping norm")
    p.add_argument("--save_freq",    type=int,   default=2000,
                   help="체크포인트 저장 주기 (스텝)")
    p.add_argument("--log_freq",     type=int,   default=20,
                   help="콘솔 출력 주기 (스텝)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       action="store_true",
                   help="이전 체크포인트에서 재시작")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project", default="smolvla_finetune")
    return p.parse_args()


# ── 모델 & 데이터 로드 ────────────────────────────────────────────────────────

def load_policy(args, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"  사전학습 모델 로드: {args.pretrained}")
    policy = SmolVLAPolicy.from_pretrained(args.pretrained)

    # VLM 동결 및 Action Expert만 학습
    for param in policy.parameters():
        param.requires_grad = True

    vlm_keywords = ["vision", "language", "text_model", "embed_tokens", "layers.", "norm.", "vlm"]
    
    for name, param in policy.named_parameters():
        name_lower = name.lower()
        if any(k in name_lower for k in vlm_keywords) and not any(a in name_lower for a in ["action", "head", "expert", "decoder", "flow"]):
            param.requires_grad = False

    total   = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    if trainable == 0:
        for param in policy.parameters():
            param.requires_grad = True
        trainable = total

    print(f"  전체 파라미터  : {total/1e6:.1f}M")
    print(f"  학습 파라미터  : {trainable/1e6:.2f}M  ({100*trainable/total:.2f}%)")

    return policy.to(device)


def load_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"  데이터셋 로드: {args.dataset_repo}  (root={args.dataset_root})")

    # ⭐ [최종 해결책] 데이터셋을 "생성하는 시점"에 50스텝을 요구하도록 강제합니다.
    # 에피소드가 20스텝이더라도, LeRobot이 자동으로 마지막 프레임으로 나머지를 패딩해줍니다.
    action_horizon = 50
    fps = 10.0

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo,
        root=args.dataset_root,
        delta_timestamps={
            "action": [i / fps for i in range(action_horizon)],
            "observation.images.image": [0.0],
            "observation.images.image2": [0.0],
            "observation.state": [0.0],
        }
    )
    
    print(f"  에피소드: {dataset.num_episodes}개  |  프레임: {len(dataset)}개")
    return dataset


def make_dataloader(dataset, args):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def make_optimizer_scheduler(policy, args, total_steps):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / max(total_steps, 1),
        anneal_strategy="cos",
    )
    return optimizer, scheduler


# ── 로깅 유틸 ────────────────────────────────────────────────────────────────

class TrainLogger:
    def __init__(self, output_dir: Path, use_wandb: bool, wandb_project: str,
                 total_steps: int, batch_size: int, num_gpus: int):
        self.output_dir  = output_dir
        self.total_steps = total_steps
        self.batch_size  = batch_size
        self.num_gpus    = num_gpus
        self.log_path    = output_dir / "train_log.jsonl"

        self.loss_window     = deque(maxlen=100)
        self.grad_window     = deque(maxlen=100)
        self.step_time_window = deque(maxlen=50)

        self.best_loss   = float("inf")
        self.start_time  = time.time()

        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config={
                    "total_steps": total_steps,
                    "batch_size": batch_size * num_gpus,
                    "num_gpus": num_gpus,
                })
                self.wandb = wandb
            except ImportError:
                pass

    def log_step(self, step: int, loss: float, grad_norm: float,
                 lr: float, step_time: float, extra: dict | None = None):
        self.loss_window.append(loss)
        self.grad_window.append(grad_norm)
        self.step_time_window.append(step_time)

        avg_loss = sum(self.loss_window) / len(self.loss_window)
        avg_grad = sum(self.grad_window) / len(self.grad_window)
        avg_step_t = sum(self.step_time_window) / len(self.step_time_window)

        if loss < self.best_loss:
            self.best_loss = loss

        remaining = (self.total_steps - step) * avg_step_t
        h, m = divmod(int(remaining), 3600)
        m, s  = divmod(m, 60)
        eta   = f"{h:02d}:{m:02d}:{s:02d}"

        elapsed = time.time() - self.start_time
        eh, em  = divmod(int(elapsed), 3600)
        em, es  = divmod(em, 60)
        elapsed_str = f"{eh:02d}:{em:02d}:{es:02d}"

        samples_per_sec = self.batch_size * self.num_gpus / max(avg_step_t, 1e-6)

        record = {
            "step": step,
            "loss": round(loss, 5),
            "loss_avg100": round(avg_loss, 5),
            "grad_norm": round(grad_norm, 4),
            "grad_norm_avg100": round(avg_grad, 4),
            "lr": lr,
            "step_time_s": round(step_time, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "best_loss": round(self.best_loss, 5),
            "eta": eta,
            "elapsed": elapsed_str,
        }
        if extra:
            record.update(extra)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if self.wandb:
            self.wandb.log(record, step=step)

        return record

    def print_step(self, step: int, record: dict):
        pct  = 100 * step / self.total_steps
        bar_len = 20
        filled  = int(bar_len * step / self.total_steps)
        bar     = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r[{bar}] {pct:5.1f}%  "
            f"step={step:>6}/{self.total_steps}  "
            f"loss={record['loss']:.4f}(avg={record['loss_avg100']:.4f})  "
            f"grad={record['grad_norm']:.3f}  "
            f"lr={record['lr']:.2e}  "
            f"ETA={record['eta']}  "
            f"elapsed={record['elapsed']}",
            end="", flush=True
        )

    def print_milestone(self, step: int, record: dict):
        print(f"\n{'─'*80}")
        print(
            f"  ★ step {step:>6}/{self.total_steps}"
            f"  loss={record['loss_avg100']:.5f}"
            f"  best={record['best_loss']:.5f}"
            f"  grad={record['grad_norm_avg100']:.3f}"
            f"  lr={record['lr']:.2e}"
            f"  {record['samples_per_sec']:.0f} samples/s"
            f"  ETA={record['eta']}"
        )
        print(f"{'─'*80}")

    def close(self):
        if self.wandb:
            self.wandb.finish()


# ── 체크포인트 저장 / 로드 ────────────────────────────────────────────────────

def save_checkpoint(step: int, policy, optimizer, scheduler,
                    output_dir: Path, accelerator: Accelerator):
    ckpt_dir = output_dir / f"checkpoint_{step:07d}"
    accelerator.save_state(str(ckpt_dir))

    unwrapped = accelerator.unwrap_model(policy)
    unwrapped.save_pretrained(str(ckpt_dir / "policy_model"))

    link = output_dir / "checkpoint_latest"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(ckpt_dir.name)

    print(f"\n  💾 체크포인트 저장: {ckpt_dir}")


def load_latest_checkpoint(output_dir: Path, policy, optimizer, scheduler,
                            accelerator: Accelerator):
    link = output_dir / "checkpoint_latest"
    if not link.exists():
        return 0

    ckpt_dir = output_dir / link.resolve().name
    print(f"  이전 체크포인트 로드: {ckpt_dir}")
    accelerator.load_state(str(ckpt_dir))

    step = int(ckpt_dir.name.split("_")[-1])
    print(f"  재시작 step: {step}")
    return step


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accelerator (DDP)
    accelerator = Accelerator(mixed_precision="bf16")
    device      = accelerator.device
    num_gpus    = accelerator.num_processes
    is_main     = accelerator.is_main_process

    torch.manual_seed(args.seed + accelerator.process_index)

    if is_main:
        print("=" * 80)
        print("  SmolVLA 파인튜닝 (Action Expert만 학습)")
        print(f"  GPU {num_gpus}개  |  device={device}  |  mixed_precision=bf16")
        print(f"  batch_size={args.batch_size}/GPU  →  유효 배치={args.batch_size * num_gpus}")
        print(f"  steps={args.steps}  lr={args.lr}  warmup={args.warmup_steps}")
        print(f"  출력: {output_dir}")
        print("=" * 80)

    if is_main:
        print("\n[1/5] 모델 로드...")
    policy = load_policy(args, device)

    if is_main:
        print("\n[2/5] 데이터셋 로드...")
    dataset    = load_dataset(args)
    dataloader = make_dataloader(dataset, args)

    if is_main:
        print("\n[3/5] 옵티마이저 설정...")
    optimizer, scheduler = make_optimizer_scheduler(policy, args, args.steps)
    
    if is_main:
        print("\n[4/5] 토크나이저 준비...")
    
    # SmolVLM2 공식 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print("\n[5/5] Accelerator 래핑...")
    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    start_step = 0
    if args.resume:
        start_step = load_latest_checkpoint(
            output_dir, policy, optimizer, scheduler, accelerator
        )

    logger = None
    if is_main:
        logger = TrainLogger(
            output_dir, args.wandb, args.wandb_project,
            args.steps, args.batch_size, num_gpus,
        )

    if is_main:
        print(f"\n{'='*80}")
        print(f"  학습 시작  (step {start_step} → {args.steps})")
        print(f"{'='*80}\n")

    policy.train()
    data_iter = iter(dataloader)
    step = start_step

    pbar = tqdm(
        total=args.steps,
        initial=start_step,
        desc="학습",
        unit="step",
        disable=not is_main,
        dynamic_ncols=True,
    )

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        # ====================================================================
        # ⭐ [오류 해결 3] 텍스트 마스크 추가 & 길이 완벽 동기화 (max_length=22)
        # ====================================================================
        if "observation.language.tokens" not in batch and "task" in batch:
            text_inputs = tokenizer(
                batch["task"],
                return_tensors="pt",
                padding="max_length",
                max_length=22,      # SmolVLA 시퀀스 템플릿과 정확히 일치하는 길이
                truncation=True
            )
            batch["observation.language.tokens"] = text_inputs["input_ids"].to(device)
            # ⭐ .bool()을 추가해서 True/False 타입으로 강제 변환해 줍니다!
            batch["observation.language.attention_mask"] = text_inputs["attention_mask"].bool().to(device)
        t0 = time.perf_counter()

        with accelerator.autocast():
            loss, loss_dict = policy.forward(batch)

        accelerator.backward(loss)

        if args.grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(
                policy.parameters(), args.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        step += 1
        step_time = time.perf_counter() - t0

        if is_main:
            lr  = optimizer.param_groups["lr"]
            extra = {k: round(float(v), 5) for k, v in loss_dict.items()
                     if k != "loss"}
            record = logger.log_step(
                step, loss.item(), float(grad_norm),
                lr, step_time, extra
            )

            pbar.set_postfix({
                "loss": f"{record['loss']:.4f}",
                "avg":  f"{record['loss_avg100']:.4f}",
                "grad": f"{record['grad_norm']:.3f}",
                "lr":   f"{record['lr']:.1e}",
                "ETA":  record["eta"],
            }, refresh=False)
            pbar.update(1)

            if step % args.log_freq == 0:
                logger.print_step(step, record)

        if step % args.save_freq == 0 or step == args.steps:
            accelerator.wait_for_everyone()
            if is_main:
                logger.print_milestone(step, record)
                save_checkpoint(step, policy, optimizer, scheduler,
                                output_dir, accelerator)

    pbar.close()

    accelerator.wait_for_everyone()
    if is_main:
        elapsed = time.time() - logger.start_time
        h, m = divmod(int(elapsed), 3600)
        m, s  = divmod(m, 60)
        print(f"\n{'='*80}")
        print(f"  학습 완료!")
        print(f"  총 시간   : {h:02d}:{m:02d}:{s:02d}")
        print(f"  최저 loss : {logger.best_loss:.5f}")
        print(f"  출력 경로 : {output_dir}")
        print(f"{'='*80}")

        final_dir = output_dir / "final_policy"
        accelerator.unwrap_model(policy).save_pretrained(str(final_dir))
        print(f"  최종 모델 : {final_dir}")

        logger.close()


if __name__ == "__main__":
    main()