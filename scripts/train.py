#!/usr/bin/env python3
"""
train.py
========
SmolVLA LoRA 파인튜닝 — lerobot_train 래퍼

실행 (단일 GPU):
    python scripts/train.py

실행 (4-GPU DDP):
    accelerate launch --num_processes=4 scripts/train.py

주요 옵션:
    --steps 20000
    --batch_size 8
    --lr 2e-5
    --output_dir outputs/run1
    --dataset_root data/lerobot_datasets
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo", default="local/smolvla_finetune_object")
    p.add_argument("--dataset_root", default="data/lerobot_datasets")
    p.add_argument("--pretrained",   default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--output_dir",   default="outputs/smolvla_ft_object")
    p.add_argument("--steps",        type=int,   default=20000)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--save_freq",    type=int,   default=2000)
    p.add_argument("--log_freq",     type=int,   default=100)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--lora_r",       type=int,   default=16,
                   help="LoRA rank (줄이면 VRAM 절약, 기본 16)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--wandb",        action="store_true",
                   help="WandB 로깅 활성화")
    p.add_argument("--wandb_project", default="smolvla_finetune")
    return p.parse_args()


def find_lerobot_train():
    """lerobot_train.py 경로 탐색"""
    import lerobot
    lerobot_root = Path(lerobot.__file__).parent
    candidates = [
        lerobot_root / "scripts" / "lerobot_train.py",
        lerobot_root / "scripts" / "train.py",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("lerobot_train.py를 찾을 수 없습니다.")


def main():
    args = parse_args()

    train_script = find_lerobot_train()
    print(f"학습 스크립트: {train_script}")
    print(f"데이터셋    : {args.dataset_repo}  (root={args.dataset_root})")
    print(f"사전학습    : {args.pretrained}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"steps={args.steps}  batch={args.batch_size}  lr={args.lr}  lora_r={args.lora_r}")

    cmd = [
        sys.executable, train_script,
        f"--policy.type=smolvla",
        f"--policy.pretrained={args.pretrained}",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--dataset.root={args.dataset_root}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        f"--output_dir={args.output_dir}",
        f"--seed={args.seed}",
        f"--policy.optimizer_lr={args.lr}",
        # LoRA 설정
        f"--peft.method_type=LORA",
        f"--peft.r={args.lora_r}",
        f"--peft.lora_alpha={args.lora_r * 2}",
        f"--peft.target_modules=all-linear",
    ]

    if args.wandb:
        cmd += [
            f"--wandb.enable=true",
            f"--wandb.project={args.wandb_project}",
        ]

    print("\n" + "="*60)
    print("실행 명령어:")
    print(" ".join(cmd))
    print("="*60 + "\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
