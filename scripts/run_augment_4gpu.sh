#!/bin/bash
# =============================================================================
# run_augment_4gpu.sh
#
# augment_action_noise.py 를 4-GPU 서버에서 병렬 실행합니다.
#
# 분배:
#   GPU 0 → task  0- 9  (libero_10,      max_steps=500)
#   GPU 1 → task 10-19  (libero_goal,    max_steps=350)
#   GPU 2 → task 20-29  (libero_object,  max_steps=320)
#   GPU 3 → task 30-39  (libero_spatial, max_steps=250)
#
# episode / frame 오프셋 계산 (비중첩 보장):
#   GPU 당 최대 에피소드 수: N_SUCCESS(10) × tasks(10) × ep_per_success(2) = 200
#   GPU 당 최대 프레임 수:   200 ep × 1500 frames/ep(상한) = 300,000
#   실제는 이보다 적지만 충분한 여백으로 설정
#
# 실행:
#   bash run_augment_4gpu.sh
#
# 4-GPU 완료 후 메타데이터 자동 병합:
#   info.json (total_episodes / total_frames 업데이트)
#   partial_gpu*.json 파일 삭제
# =============================================================================

set -e
cd "$(dirname "$0")"

PYTHON=/media/idna/Data/envs/smolvla/bin/python
OUT_DIR=./outputs/augmented_action_noise

# ── 오프셋 설정 (GPU 당 여유 있게 잡음)
MAX_EP_PER_GPU=200       # N_SUCCESS(10) × 10tasks × 2(orig+aug) = 200
MAX_FRAMES_PER_GPU=300000  # 200 ep × 1500 frames 상한

EP_OFFSET_0=0
EP_OFFSET_1=$((EP_OFFSET_0 + MAX_EP_PER_GPU))
EP_OFFSET_2=$((EP_OFFSET_1 + MAX_EP_PER_GPU))
EP_OFFSET_3=$((EP_OFFSET_2 + MAX_EP_PER_GPU))

FR_OFFSET_0=0
FR_OFFSET_1=$((FR_OFFSET_0 + MAX_FRAMES_PER_GPU))
FR_OFFSET_2=$((FR_OFFSET_1 + MAX_FRAMES_PER_GPU))
FR_OFFSET_3=$((FR_OFFSET_2 + MAX_FRAMES_PER_GPU))

echo "============================================================"
echo "  Action Noise 증강 (4-GPU 병렬)"
echo "============================================================"
echo "  GPU 0: task  0- 9  ep_offset=${EP_OFFSET_0}  fr_offset=${FR_OFFSET_0}"
echo "  GPU 1: task 10-19  ep_offset=${EP_OFFSET_1}  fr_offset=${FR_OFFSET_1}"
echo "  GPU 2: task 20-29  ep_offset=${EP_OFFSET_2}  fr_offset=${FR_OFFSET_2}"
echo "  GPU 3: task 30-39  ep_offset=${EP_OFFSET_3}  fr_offset=${FR_OFFSET_3}"
echo "  출력: ${OUT_DIR}"
echo "============================================================"
echo ""

mkdir -p "${OUT_DIR}/data"
mkdir -p "${OUT_DIR}/logs"

# ── 4개 프로세스 병렬 실행
$PYTHON augment_action_noise.py \
    --gpu_id 0 --task_start 0  --task_end 10 \
    --ep_offset ${EP_OFFSET_0} --frame_offset ${FR_OFFSET_0} \
    > "${OUT_DIR}/logs/gpu0.log" 2>&1 &
PID0=$!

$PYTHON augment_action_noise.py \
    --gpu_id 1 --task_start 10 --task_end 20 \
    --ep_offset ${EP_OFFSET_1} --frame_offset ${FR_OFFSET_1} \
    > "${OUT_DIR}/logs/gpu1.log" 2>&1 &
PID1=$!

$PYTHON augment_action_noise.py \
    --gpu_id 2 --task_start 20 --task_end 30 \
    --ep_offset ${EP_OFFSET_2} --frame_offset ${FR_OFFSET_2} \
    > "${OUT_DIR}/logs/gpu2.log" 2>&1 &
PID2=$!

$PYTHON augment_action_noise.py \
    --gpu_id 3 --task_start 30 --task_end 40 \
    --ep_offset ${EP_OFFSET_3} --frame_offset ${FR_OFFSET_3} \
    > "${OUT_DIR}/logs/gpu3.log" 2>&1 &
PID3=$!

echo "  PID: GPU0=${PID0}  GPU1=${PID1}  GPU2=${PID2}  GPU3=${PID3}"
echo "  로그: ${OUT_DIR}/logs/gpu{0-3}.log"
echo ""

# ── 진행 상황 모니터링 (30초마다)
monitor_progress() {
    while kill -0 $PID0 $PID1 $PID2 $PID3 2>/dev/null; do
        sleep 30
        echo -n "[$(date '+%H:%M:%S')] 진행: "
        for g in 0 1 2 3; do
            LOG="${OUT_DIR}/logs/gpu${g}.log"
            if [ -f "$LOG" ]; then
                COUNT=$(grep -c "→ ✓" "$LOG" 2>/dev/null || echo 0)
                echo -n "GPU${g}=${COUNT}/10태스크 "
            fi
        done
        echo ""
    done
}
monitor_progress &
MON_PID=$!

# ── 모든 프로세스 완료 대기
wait $PID0 && echo "[완료] GPU 0" || echo "[실패] GPU 0 (exit $?)"
wait $PID1 && echo "[완료] GPU 1" || echo "[실패] GPU 1 (exit $?)"
wait $PID2 && echo "[완료] GPU 2" || echo "[실패] GPU 2 (exit $?)"
wait $PID3 && echo "[완료] GPU 3" || echo "[실패] GPU 3 (exit $?)"

kill $MON_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "  모든 GPU 완료. 메타데이터 병합 중..."
echo "============================================================"

# ── partial_gpu*.json 병합 → info.json 생성
$PYTHON - << 'PYEOF'
import json, pathlib, glob

out = pathlib.Path("./outputs/augmented_action_noise")
partials = sorted(glob.glob(str(out / "partial_gpu*.json")))

if not partials:
    print("[경고] partial 파일 없음 - 병합 스킵")
    exit(0)

total_episodes = 0
total_frames   = 0
all_tasks      = []

for p in partials:
    with open(p) as f:
        d = json.load(f)
    ep_count = d["ep_end"] - d["ep_offset"]
    total_episodes += ep_count
    total_frames   += d["total_frames"]
    all_tasks.extend(d["tasks"])
    print(f"  {pathlib.Path(p).name}: ep={ep_count}  frames={d['total_frames']:,}")

# info.json 읽고 total_episodes / total_frames 업데이트
info_path = out / "info.json"
if info_path.exists():
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = total_episodes
    info["total_frames"]   = total_frames
    info["splits"]         = {"train": f"0:{total_episodes}"}
    info["total_chunks"]   = (total_episodes + info.get("chunks_size", 1000) - 1) // info.get("chunks_size", 1000)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\n  info.json 업데이트: total_episodes={total_episodes}  total_frames={total_frames:,}")

# 통합 summary 저장
summary_path = out / "summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "total_episodes": total_episodes,
        "total_frames":   total_frames,
        "tasks": all_tasks,
    }, f, indent=2, ensure_ascii=False)
print(f"  summary.json 저장 완료")

# partial 파일 정리
for p in partials:
    pathlib.Path(p).unlink()
print(f"  partial 파일 {len(partials)}개 삭제 완료")
PYEOF

echo ""
echo "============================================================"
echo "  증강 완료"
echo "  출력 경로: ${OUT_DIR}"
echo "  로그:      ${OUT_DIR}/logs/"
echo "============================================================"
