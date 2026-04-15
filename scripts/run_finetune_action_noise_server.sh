#!/bin/bash
# =============================================================================
# run_finetune_action_noise_server.sh
#
# augment_action_noise.py 로 생성한 증강 데이터로 SmolVLA 파인튜닝 (4-GPU 서버)
#
# 실행 전 meta 파일 보완:
#   python fix_meta_action_noise.py
#
# 실행:
#   bash run_finetune_action_noise_server.sh
#   bash run_finetune_action_noise_server.sh --steps 50000
#   bash run_finetune_action_noise_server.sh --resume
# =============================================================================

set -e

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
PYTHON=/home/bigcom/anaconda3/envs/smolvla/bin/python

# HuggingFace 캐시 경로
export HF_HOME="${HOME}/.cache/huggingface"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${HOME}/.cache/huggingface/hub"

# 사전 학습 모델
PRETRAINED_MODEL="HuggingFaceVLA/smolvla_libero"

# 증강 데이터 경로
AUGMENTED_DIR="./outputs/augmented_action_noise"

# 출력 디렉터리
OUTPUT_DIR="./outputs/train/smolvla_action_noise_$(date +%Y%m%d_%H%M%S)"

# 학습 하이퍼파라미터
STEPS=30000
BATCH_SIZE=8          # GPU 1장당 → 유효 배치 = 8 × 4GPU = 32
NUM_WORKERS=4
SAVE_FREQ=5000
LOG_FREQ=100
SEED=1000

# WandB
WANDB_ENABLE=false
WANDB_PROJECT="smolvla_action_noise"

# HuggingFace Hub 업로드
PUSH_TO_HUB=false
HF_REPO_ID="YOUR_HF_USERNAME/smolvla_action_noise"

# ─────────────────────────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────────────────────────
RESUME=false
for arg in "$@"; do
    case $arg in
        --steps=*)  STEPS="${arg#*=}"  ;;
        --steps)    shift; STEPS="$1"  ;;
        --resume)   RESUME=true        ;;
    esac
done

# ─────────────────────────────────────────────────────────────
# 사전 확인
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SmolVLA 파인튜닝 (Action Noise 증강 데이터, 4-GPU)"
echo "============================================================"
echo "  기반 모델    : $PRETRAINED_MODEL"
echo "  데이터 경로  : $AUGMENTED_DIR"
echo "  출력 경로    : $OUTPUT_DIR"
echo "  학습 스텝    : $STEPS"
echo "  배치 (GPU당) : $BATCH_SIZE  →  유효 배치 $((BATCH_SIZE * 4))"
echo "  재개 여부    : $RESUME"
echo "============================================================"
echo ""

if [ ! -d "$AUGMENTED_DIR/data" ]; then
    echo "[ERROR] 증강 데이터 없음: $AUGMENTED_DIR/data"
    exit 1
fi

if [ ! -f "$AUGMENTED_DIR/meta/info.json" ]; then
    echo "[ERROR] meta/info.json 없음. run_augment_4gpu.sh 병합 완료 여부 확인"
    exit 1
fi

if [ ! -d "$AUGMENTED_DIR/meta/episodes" ]; then
    echo "[INFO] meta/episodes/ 없음 → fix_meta_action_noise.py 자동 실행"
    $PYTHON fix_meta_action_noise.py --dataset_dir "$AUGMENTED_DIR"
fi

# 데이터셋 정보 출력
TOTAL_EP=$($PYTHON -c "import json; d=json.load(open('$AUGMENTED_DIR/meta/info.json')); print(d['total_episodes'])" 2>/dev/null || echo "?")
TOTAL_FR=$($PYTHON -c "import json; d=json.load(open('$AUGMENTED_DIR/meta/info.json')); print(d['total_frames'])" 2>/dev/null || echo "?")
echo "  데이터셋 : $TOTAL_EP 에피소드 / $TOTAL_FR 프레임"
echo ""

# ─────────────────────────────────────────────────────────────
# LeRobot train 스크립트 경로 자동 탐색
# ─────────────────────────────────────────────────────────────
LEROBOT_TRAIN=$($PYTHON -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('lerobot')
if spec and spec.submodule_search_locations:
    p = pathlib.Path(list(spec.submodule_search_locations)[0])
    t = p / 'scripts' / 'lerobot_train.py'
    print(t if t.exists() else '')
" 2>/dev/null)

if [ -z "$LEROBOT_TRAIN" ]; then
    echo "[ERROR] lerobot_train.py 를 찾을 수 없습니다."
    echo "  lerobot 설치를 확인하세요: pip install -e /path/to/lerobot"
    exit 1
fi
echo "  lerobot_train: $LEROBOT_TRAIN"

# ─────────────────────────────────────────────────────────────
# 재개 처리
# ─────────────────────────────────────────────────────────────
RESUME_ARGS=""
if [ "$RESUME" = "true" ]; then
    LAST_CKPT=$(find ./outputs/train -name "train_config.json" \
                  -path "*/pretrained_model/train_config.json" \
                  -printf "%T@ %p\n" 2>/dev/null \
                | sort -rn | head -1 | awk '{print $2}')
    if [ -z "$LAST_CKPT" ]; then
        echo "[ERROR] 재개할 체크포인트를 찾을 수 없습니다."
        exit 1
    fi
    echo "  ▶ 재개 체크포인트: $LAST_CKPT"
    RESUME_ARGS="--resume=true --config_path=$LAST_CKPT"
    OUTPUT_DIR=$(dirname $(dirname $(dirname $LAST_CKPT)))
fi

# ─────────────────────────────────────────────────────────────
# GPU 감지
# ─────────────────────────────────────────────────────────────
NUM_GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "  감지된 GPU: ${NUM_GPUS}장"
ACCELERATE_ARGS="--num_processes=${NUM_GPUS} --mixed_precision=no"

# ─────────────────────────────────────────────────────────────
# 학습 실행
# ─────────────────────────────────────────────────────────────
$PYTHON -m accelerate.commands.launch \
    $ACCELERATE_ARGS \
    "$LEROBOT_TRAIN" \
    \
    --policy.path="$PRETRAINED_MODEL" \
    --policy.push_to_hub=$PUSH_TO_HUB \
    --policy.repo_id="$HF_REPO_ID" \
    \
    --dataset.repo_id="augmented_action_noise" \
    --dataset.root="$AUGMENTED_DIR" \
    --dataset.use_imagenet_stats=false \
    \
    --output_dir="$OUTPUT_DIR" \
    --job_name="smolvla_action_noise" \
    --seed=$SEED \
    \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --num_workers=$NUM_WORKERS \
    --save_freq=$SAVE_FREQ \
    --log_freq=$LOG_FREQ \
    \
    --wandb.enable=$WANDB_ENABLE \
    --wandb.project="$WANDB_PROJECT" \
    \
    $RESUME_ARGS

echo ""
echo "학습 완료. 체크포인트: $OUTPUT_DIR"
