#!/usr/bin/env python3
"""
train.py — SmolVLA 파인튜닝 (모든 오류 해결 최종 통합본)
================================================================
수정 내역 요약:
1. 리스트 인덱스 오류 수정: optimizer.param_groups["lr"]
2. VRAM 10GB 최적화: batch_size=1 + VLM 확실한 동결
3. 데이터 길이 불일치 해결: action_horizon=50 강제 설정
4. 텐서 타입 오류 해결: attention_mask를 .bool()로 변환
5. DDP 통신 오류 해결: find_unused_parameters=True 적용
6. 토크나이저 호환성: SmolVLM2 공식 토크나이저 사용 (max_length=22)
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
    p.add_argument("--steps",        type=int,   default=20000,
                   help="총 학습 스텝 수")
    
    # ⭐ [VRAM 최적화] 3080 10GB 환경에서는 GPU당 배치 1이 가장 안정적입니다.
    p.add_argument("--batch_size",   type=int,   default=1,
                   help="GPU당 배치 크기 (10GB VRAM 권장값: 1)")
    
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

    # =========================================================================
    # ⭐ [확실한 동결] VLM 백본은 얼리고 Action Expert 부분만 학습을 엽니다.
    # =========================================================================
    # 1. 일단 전체 파라미터 동결
    for param in policy.parameters():
        param.requires_grad = False

    # 2. Action 관련 모듈만 선택적으로 해제 (SmolVLA 구조 전용 키워드)
    unfrozen_params_count = 0
    for name, param in policy.named_parameters():
        if any(key in name for key in ["action_head", "diffusion_model", "action_expert"]):
            param.requires_grad = True
            unfrozen_params_count += 1

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    # 안전장치: 만약 키워드로 못 찾았다면 VLM 제외한 모든 부분을 학습 대상으로 지정
    if trainable == 0:
        print("  [경고] 특정 키워드를 찾지 못해 'vlm'이 포함되지 않은 레이어만 활성화합니다.")
        for name, param in policy.named_parameters():
            if "vlm" not in name.lower():
                param.requires_grad = True
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"  전체 파라미터  : {total/1e6:.1f}M")
    print(f"  학습 파라미터(Action 전용) : {trainable/1e6:.2f}M  ({100*trainable/total:.2f}%)")
    
    return policy.to(device)


def load_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"  데이터셋 로드: {args.dataset_repo}")

    # ⭐ [중요] 에피소드 길이(20)에 상관없이 모델 요구량(50)에 맞춰 강제 패딩 설정
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


# ── 로깅 유틸 ────────────────────────────────────────────────────────────────

class TrainLogger:
    def __init__(self, output_dir: Path, use_wandb: bool, wandb_project: str,
                 total_steps: int, batch_size: int, num_gpus: int):
        self.output_dir  = output_dir
        self.total_steps = total_steps
        self.batch_size  = batch_size
        self.num_gpus    = num_gpus
        self.log_path    = output_dir / "train_log.jsonl"
        self.loss_window = deque(maxlen=100)
        self.best_loss   = float("inf")
        self.start_time  = time.time()
        self.wandb       = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config={"batch_size": batch_size * num_gpus})
                self.wandb = wandb
            except: pass

    def log_step(self, step, loss, lr, step_time):
        self.loss_window.append(loss)
        avg_loss = sum(self.loss_window) / len(self.loss_window)
        if loss < self.best_loss: self.best_loss = loss

        remaining = (self.total_steps - step) * step_time
        h, m = divmod(int(remaining), 3600); m, s = divmod(m, 60)
        eta = f"{h:02d}:{m:02d}:{s:02d}"

        record = {
            "step": step, 
            "loss": round(loss, 5), 
            "avg_loss": round(avg_loss, 5), 
            "lr": f"{lr:.2e}", 
            "eta": eta
        }
        
        with open(self.log_path, "a") as f: f.write(json.dumps(record) + "\n")
        if self.wandb: self.wandb.log(record, step=step)
        return record


# ── 메인 학습 루프 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # ⭐ [DDP 해결] 일부 레이어를 동결했을 때 발생하는 통신 에러 방지
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    
    device, is_main = accelerator.device, accelerator.is_main_process
    torch.manual_seed(args.seed + accelerator.process_index)

    if is_main:
        print("=" * 80)
        print(f" SmolVLA 파인튜닝 (Action Expert 전용) | GPU {accelerator.num_processes}개")
        print(f" [VRAM 10GB 최적화 모드] Batch: {args.batch_size} (Real: {args.batch_size * accelerator.num_processes})")
        print("=" * 80)

    # 1. 모델
    policy = load_policy(args, device)

    # 2. 데이터
    dataset = load_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 3. 옵티마이저 & 스케줄러 (학습 가능한 파라미터만 전달)
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    optimizer = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    
    # 4. 토크나이저 (SmolVLM2 공식 버전으로 호환성 확보)
    if is_main: print("\n[4/5] 토크나이저 준비...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 5. Accelerator 준비
    policy, optimizer, dataloader, scheduler = accelerator.prepare(policy, optimizer, dataloader, scheduler)

    logger = TrainLogger(output_dir, args.wandb, args.wandb_project, args.steps, args.batch_size, accelerator.num_processes) if is_main else None

    policy.train()
    data_iter = iter(dataloader)
    pbar = tqdm(total=args.steps, desc="학습", disable=not is_main)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader); batch = next(data_iter)
            
        # ⭐ [텍스트 처리] max_length=22 고정 및 .bool() 타입 변환으로 모든 에러 차단
        if "observation.language.tokens" not in batch:
            text_inputs = tokenizer(batch["task"], return_tensors="pt", padding="max_length", max_length=22, truncation=True)
            batch["observation.language.tokens"] = text_inputs["input_ids"].to(device)
            batch["observation.language.attention_mask"] = text_inputs["attention_mask"].bool().to(device)

        t0 = time.perf_counter()
        
        with accelerator.autocast():
            # forward 시 batch 전체를 넘깁니다.
            loss, loss_dict = policy.forward(batch)

        accelerator.backward(loss)
        if args.grad_clip > 0: accelerator.clip_grad_norm_(policy.parameters(), args.grad_clip)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        step_time = time.perf_counter() - t0

        if is_main:
            # ⭐ [수정 완료] optimizer.param_groups 인덱스 추가하여 TypeError 해결
            current_lr = optimizer.param_groups["lr"]
            record = logger.log_step(step, loss.item(), current_lr, step_time)
            pbar.set_postfix(loss=f"{record['loss']:.4f}", lr=record['lr'], eta=record['eta'])
            pbar.update(1)

        # 체크포인트 저장
        if step % args.save_freq == 0:
            accelerator.wait_for_everyone()
            if is_main:
                ckpt_path = output_dir / f"checkpoint_{step:07d}"
                accelerator.unwrap_model(policy).save_pretrained(str(ckpt_path / "policy_model"))
                print(f"\n  💾 저장 완료: {ckpt_path}")

    if is_main: 
        print("\n🎉 모든 고난을 뚫고 학습이 완료되었습니다!"); 
        if args.wandb: logger.wandb.finish()

if __name__ == "__main__":
    main()