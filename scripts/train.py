#!/usr/bin/env python3
"""
train.py — SmolVLA 파인튜닝 (오타 수정 및 최종 안정화 버전)
================================================================
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
    p.add_argument("--dataset_root", default="data/lerobot_datasets",
                   help="LeRobot 데이터셋 로컬 경로")
    p.add_argument("--pretrained",   default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--output_dir",   default="outputs/smolvla_ft_object")
    p.add_argument("--steps",        type=int,   default=20000)
    p.add_argument("--batch_size",   type=int,   default=1) # 10GB VRAM 최적화
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--save_freq",    type=int,   default=2000)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project", default="smolvla_finetune")
    return p.parse_args()


# ── 모델 & 데이터 로드 ────────────────────────────────────────────────────────

def load_policy(args, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"  사전학습 모델 로드: {args.pretrained}")
    policy = SmolVLAPolicy.from_pretrained(args.pretrained)

    # VLM 동결 설정
    for param in policy.parameters():
        param.requires_grad = False

    for name, param in policy.named_parameters():
        if any(key in name for key in ["action_head", "diffusion_model", "action_expert"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  학습 파라미터 : {trainable/1e6:.2f}M")

    return policy.to(device)


def load_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    action_horizon, fps = 50, 10.0
    
    return LeRobotDataset(
        repo_id=args.dataset_repo,
        root=args.dataset_root,
        delta_timestamps={
            "action": [i / fps for i in range(action_horizon)],
            "observation.images.image": [0.0],
            "observation.images.image2": [0.0],
            "observation.state": [0.0],
        }
    )


# ── 로깅 유틸 ────────────────────────────────────────────────────────────────

class TrainLogger:
    def __init__(self, output_dir: Path, use_wandb, total_steps):
        self.log_path = output_dir / "train_log.jsonl"
        self.loss_window = deque(maxlen=100)
        self.start_time = time.time()
        self.total_steps = total_steps
        self.wandb = None
        if use_wandb:
            import wandb
            wandb.init(project="smolvla_finetune")
            self.wandb = wandb

    def log(self, step, loss, lr, step_time):
        self.loss_window.append(loss)
        avg_loss = sum(self.loss_window) / len(self.loss_window)
        record = {"step": step, "loss": round(loss, 5), "avg_loss": round(avg_loss, 5), "lr": lr}
        with open(self.log_path, "a") as f: f.write(json.dumps(record) + "\n")
        if self.wandb: self.wandb.log(record, step=step)
        return record


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device, is_main = accelerator.device, accelerator.is_main_process

    # 1. 모델 & 데이터
    policy = load_policy(args, device)
    dataset = load_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # 2. 옵티마이저
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    # 3. 토크나이저
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 4. 준비
    policy, optimizer, dataloader, scheduler = accelerator.prepare(policy, optimizer, dataloader, scheduler)
    logger = TrainLogger(output_dir, args.wandb, args.steps) if is_main else None

    policy.train()
    data_iter = iter(dataloader)
    pbar = tqdm(total=args.steps, desc="학습", disable=not is_main)

    for step in range(1, args.steps + 1):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(dataloader); batch = next(data_iter)

        # 텍스트 처리
        text_inputs = tokenizer(batch["task"], return_tensors="pt", padding="max_length", max_length=22, truncation=True)
        batch["observation.language.tokens"] = text_inputs["input_ids"].to(device)
        batch["observation.language.attention_mask"] = text_inputs["attention_mask"].bool().to(device)

        t0 = time.perf_counter()
        with accelerator.autocast():
            loss, _ = policy.forward(batch)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if is_main:
            # ⭐ [수정 완료] 리스트 인덱싱 추가!
            current_lr = optimizer.param_groups["lr"]
            rec = logger.log(step, loss.item(), current_lr, time.perf_counter() - t0)
            pbar.set_postfix(loss=f"{rec['loss']:.4f}", lr=f"{current_lr:.1e}")
            pbar.update(1)

        if step % args.save_freq == 0:
            accelerator.wait_for_everyone()
            if is_main:
                ckpt = output_dir / f"checkpoint_{step:07d}"
                accelerator.unwrap_model(policy).save_pretrained(str(ckpt / "policy"))
                print(f"\n  💾 저장 완료: {ckpt}")

    if is_main: print("\n🎉 모든 고난을 뚫고 학습 완료!"); logger.wandb.finish() if args.wandb else None

if __name__ == "__main__":
    main()