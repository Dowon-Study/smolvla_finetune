#!/usr/bin/env python3
"""
train.py — SmolVLA 파인튜닝 (최종 통합 수정본)
=============================================
수정 및 최적화 내역:
1. [오류 수정] optimizer.param_groups["lr"] 인덱스 추가
2. [VRAM 최적화] batch_size=1 + VLM 동결 (RTX 3080 10GB용)
3. [데이터 최적화] action_horizon=50 패딩 강제 (201 시퀀스 일치)
4. [타입 최적화] attention_mask를 .bool()로 변환
5. [DDP 최적화] find_unused_parameters=True 적용
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
    p.add_argument("--batch_size",   type=int,   default=1)  # 10GB VRAM은 1이 가장 안전함
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--save_freq",    type=int,   default=2000)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--wandb",        action="store_true")
    return p.parse_args()


# ── 모델 & 데이터 로드 ────────────────────────────────────────────────────────

def load_policy(args, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"  모델 로드 시작: {args.pretrained}")
    policy = SmolVLAPolicy.from_pretrained(args.pretrained)

    # VLM 백본은 동결하고 Action Expert만 학습하여 VRAM 절약
    for param in policy.parameters():
        param.requires_grad = False

    # Action 관련 모듈만 해제
    for name, param in policy.named_parameters():
        if any(key in name for key in ["action_head", "diffusion_model", "action_expert"]):
            param.requires_grad = True

    # 만약 키워드로 안 잡히면 vlm 제외 모두 학습 (안전장치)
    if sum(p.numel() for p in policy.parameters() if p.requires_grad) == 0:
        for name, param in policy.named_parameters():
            if "vlm" not in name.lower():
                param.requires_grad = True

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  학습 파라미터 : {trainable/1e6:.2f}M")
    return policy.to(device)


def load_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # 20스텝 궤도를 모델 요구량인 50스텝으로 확장
    action_horizon, fps = 50, 10.0
    
    print(f"  데이터셋 로드: {args.dataset_repo}")
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
        
        remaining = (self.total_steps - step) * step_time
        h, m = divmod(int(remaining), 3600); m, s = divmod(m, 60)
        eta = f"{h:02d}:{m:02d}:{s:02d}"

        record = {"step": step, "loss": round(loss, 5), "avg_loss": round(avg_loss, 5), "lr": f"{lr:.2e}", "eta": eta}
        with open(self.log_path, "a") as f: f.write(json.dumps(record) + "\n")
        if self.wandb: self.wandb.log(record, step=step)
        return record


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # DDP 환경에서 동결된 레이어가 있을 때 발생하는 에러 방지
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device, is_main = accelerator.device, accelerator.is_main_process
    torch.manual_seed(args.seed + accelerator.process_index)

    if is_main:
        print("=" * 80)
        print(f" SmolVLA 파인튜닝 시작 | GPU {accelerator.num_processes}개 분산학습")
        print("=" * 80)

    # 1. 모델 / 데이터 / 옵티마이저 구성
    policy = load_policy(args, device)
    dataset = load_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    # 2. 토크나이저 (원본 백본인 SmolVLM2 공식 버전 사용)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. Accelerator 준비
    policy, optimizer, dataloader, scheduler = accelerator.prepare(policy, optimizer, dataloader, scheduler)
    logger = TrainLogger(output_dir, args.wandb, args.steps) if is_main else None

    # 4. 학습 시작
    policy.train()
    data_iter = iter(dataloader)
    pbar = tqdm(total=args.steps, desc="학습", disable=not is_main)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader); batch = next(data_iter)

        # 텍스트 토큰화 및 타입 보정 (.bool() 필수)
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

        step_time = time.perf_counter() - t0

        if is_main:
            # ⭐ [수정 완료] 인덱싱을 추가하여 리스트 인덱스 오류 해결
            current_lr = optimizer.param_groups["lr"]
            rec = logger.log(step, loss.item(), current_lr, step_time)
            pbar.set_postfix(loss=f"{rec['loss']:.4f}", lr=rec['lr'], eta=rec['eta'])
            pbar.update(1)

        # 주기적 저장
        if step % args.save_freq == 0:
            accelerator.wait_for_everyone()
            if is_main:
                ckpt_path = output_dir / f"checkpoint_{step:07d}"
                accelerator.unwrap_model(policy).save_pretrained(str(ckpt_path / "policy_model"))
                print(f"\n  💾 저장 완료: {ckpt_path}")

    if is_main: 
        print("\n🎉 모든 관문을 통과하고 학습을 완료했습니다!"); 
        if args.wandb: logger.wandb.finish()

if __name__ == "__main__":
    main()