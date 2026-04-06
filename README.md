# SmolVLA LIBERO 파인튜닝

qpos 노이즈 데이터를 이용한 SmolVLA LoRA 파인튜닝 레포입니다.

## 서버 셋업 (처음 한 번)

```bash
git clone https://github.com/YOUR_USERNAME/smolvla_finetune.git
cd smolvla_finetune
bash setup.sh
```

## 데이터 전송

로컬 PC에서 서버로 HDF5 파일 복사:
```bash
scp /path/to/finetune_object_all.hdf5 user@server:~/smolvla_finetune/data/
```

## 실행 순서

```bash
conda activate smolvla

# 1. HDF5 → LeRobot 포맷 변환 (처음 한 번)
python scripts/convert_hdf5_to_lerobot.py

# 2. 학습 (4-GPU)
accelerate launch --num_processes=4 scripts/train.py

# 학습 옵션 조정 예시
accelerate launch --num_processes=4 scripts/train.py \
    --steps 30000 \
    --batch_size 8 \
    --lr 2e-5 \
    --lora_r 16 \
    --wandb
```

## 폴더 구조

```
smolvla_finetune/
├── scripts/
│   ├── convert_hdf5_to_lerobot.py   # HDF5 변환
│   └── train.py                      # 학습 실행
├── data/                             # .gitignore (직접 복사 필요)
│   └── finetune_object_all.hdf5
├── outputs/                          # .gitignore (학습 결과)
├── setup.sh                          # 환경 셋업
└── README.md
```

## VRAM 부족 시

```bash
# batch_size 줄이기
accelerate launch --num_processes=4 scripts/train.py --batch_size 4

# LoRA rank 줄이기
accelerate launch --num_processes=4 scripts/train.py --lora_r 8
```
