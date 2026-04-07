#!/bin/bash
# setup.sh — 서버에서 한 번만 실행
# 사용법: bash setup.sh
set -e

echo "======================================================"
echo " SmolVLA 파인튜닝 환경 셋업"
echo "======================================================"

# 1. conda 환경 확인
if ! command -v conda &>/dev/null; then
    echo "[오류] conda가 없습니다. Miniconda를 먼저 설치하세요."
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "[1/5] conda 환경 생성 (smolvla, python=3.12)..."
conda create -n smolvla python=3.12 -y || echo "이미 존재 — 스킵"

# conda 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smolvla

echo "[2/5] PyTorch (CUDA 12.1) 설치..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "[3/5] lerobot[smolvla] 설치..."
# lerobot이 같은 레포에 있으면 로컬 설치, 없으면 GitHub에서
if [ -d "../lerobot" ]; then
    pip install -e "../lerobot[smolvla]" -q
else
    pip install "lerobot[smolvla]" -q
fi

echo "[4/5] LIBERO 설치..."
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
pip install -e LIBERO -q

echo "[4.5/5] 추가 패키지 설치..."
pip install peft h5py -q

# lerobot exist_ok 버그 패치
METADATA_PY=$(python -c "import lerobot; import os; print(os.path.join(os.path.dirname(lerobot.__file__), 'datasets/dataset_metadata.py'))" 2>/dev/null)
if [ -f "$METADATA_PY" ]; then
    sed -i 's/exist_ok=False/exist_ok=True/' "$METADATA_PY"
    echo "  lerobot exist_ok 패치 완료: $METADATA_PY"
fi

echo "[5/5] accelerate 설정..."
python -c "
from accelerate.utils import write_basic_config
write_basic_config(mixed_precision='bf16')
" || accelerate config

echo ""
echo "======================================================"
echo " GPU 확인"
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPU 수: {torch.cuda.device_count()}')"
echo "======================================================"
echo ""
echo "다음 단계:"
echo "  1. data/ 폴더에 HDF5 파일 복사"
echo "     scp your_pc:/path/to/finetune_object_all.hdf5 data/"
echo ""
echo "  2. HDF5 → LeRobot 포맷 변환"
echo "     conda activate smolvla"
echo "     python scripts/convert_hdf5_to_lerobot.py"
echo ""
echo "  3. 학습 (4-GPU)"
echo "     accelerate launch --num_processes=4 scripts/train.py"
echo "======================================================"
