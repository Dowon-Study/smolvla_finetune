#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_noisy_recovery.py

HuggingFaceVLA/libero 데이터셋의 각 에피소드에 노이즈 복구 청크를 삽입합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【데이터 생성 원리】
  1. 각 에피소드의 첫 프레임 EEF 위치(clean_eef)를 데이터셋에서 직접 읽음
  2. LIBERO 벤치마크 init_state에 관절 노이즈(std=0.05) 적용 → noisy_init
  3. 시뮬레이션 실행: noisy_init 에서 50스텝 동안 clean_eef 로 선형 복구
     - 이미지는 np.flipud 적용 후 저장 (HF 데이터셋과 동일한 방향)
  4. 복구 청크(50프레임) + 원본 에피소드를 이어붙여 저장

【액션 청크 관점】
  SmolVLA의 chunk_size=50 이므로 복구 청크 50프레임이 정확히
  1개의 추론 청크에 해당합니다.
  모델은 "노이즈 있는 초기 관측 → 50스텝 복구 액션 시퀀스"를 학습합니다.

【실행 방식】
  GPU 1장으로 task 0-39 전체를 순차 처리합니다.
  task_index  0- 9 → libero_10
  task_index 10-19 → libero_goal
  task_index 20-29 → libero_object
  task_index 30-39 → libero_spatial

출력:
  outputs/augmented_data/noise_0_0500/data/chunk-{c:03d}/file-{f:03d}.parquet

실행:
  /media/idna/Data/envs/smolvla/bin/python augment_noisy_recovery.py
"""

import json

import os
import pathlib
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import datasets as hf_datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from scipy.spatial.transform import Rotation

# 로컬 PC: X11 디스플레이 사용 (서버 헤드리스 환경이면 "egl" 또는 "osmesa" 로 변경)
os.environ["MUJOCO_GL"]           = "glx"
os.environ["HF_DATASETS_OFFLINE"] = "1"   # 로컬 캐시 사용 (오프라인)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ============================================================
# ★ 핵심 설정  ← 학습 목적에 맞게 여기만 수정하세요
# ============================================================

DATASET_ID      = "HuggingFaceVLA/libero"
DATASET_CACHE   = "/media/idna/Data/Noise_State_Augmentation"  # 로컬 HF 캐시 경로

OUTPUT_BASE     = pathlib.Path("./outputs/augmented_data")

# 【★ 노이즈 강도】 관절 각도(rad) 표준편차
NOISE_STD       = 0.05

# 【★ 복구 프레임 수】 SmolVLA chunk_size=50 과 일치해야 합니다
RECOVERY_FRAMES = 50

CHUNK_SIZE      = 1000          # 폴더당 에피소드 수 (LeRobot v3.0 규약)
SEED            = 42
FPS             = 10
DT              = 1.0 / FPS     # 0.1 s
ENV_RES         = 256           # 이미지 해상도 (학습 데이터와 동일해야 함)

# 액션 클리핑 범위 (SmolVLA libero 기준)
ACTION_CLIP_ARM     = 0.9375    # EEF delta dims 0-5
ACTION_CLIP_GRIPPER = 1.0       # gripper dim 6

# 【★ task_index 매핑】 HF dataset task_index → (suite, task_id_within_suite)
#
# tasks.parquet 의 task 이름과 LIBERO 벤치마크 task 이름을 대조해 도출한 정확한 매핑입니다.
# (이전의 순차 매핑 0-9=spatial, 10-19=object 등은 잘못된 가정이었습니다.)
#
# HF task_index 0- 9 → libero_10    (나무 테이블, 머그/냄비/바구니 등)
# HF task_index 10-19 → libero_goal  (서랍/선반 등 목표 기반)
# HF task_index 20-29 → libero_object (바구니에 물건 담기)
# HF task_index 30-39 → libero_spatial (검은 그릇 → 접시 위)
TASK_MAP: Dict[int, Tuple[str, int]] = {
    # libero_10  (HF 0-9)
    0:  ("libero_10",     4),
    1:  ("libero_10",     6),
    2:  ("libero_10",     9),
    3:  ("libero_10",     2),
    4:  ("libero_10",     7),
    5:  ("libero_10",     0),
    6:  ("libero_10",     8),
    7:  ("libero_10",     1),
    8:  ("libero_10",     3),
    9:  ("libero_10",     5),
    # libero_goal  (HF 10-19)
    10: ("libero_goal",   8),
    11: ("libero_goal",   9),
    12: ("libero_goal",   3),
    13: ("libero_goal",   6),
    14: ("libero_goal",   2),
    15: ("libero_goal",   5),
    16: ("libero_goal",   7),
    17: ("libero_goal",   1),
    18: ("libero_goal",   4),
    19: ("libero_goal",   0),
    # libero_object  (HF 20-29)
    20: ("libero_object", 9),
    21: ("libero_object", 4),
    22: ("libero_object", 1),
    23: ("libero_object", 3),
    24: ("libero_object", 0),
    25: ("libero_object", 7),
    26: ("libero_object", 2),
    27: ("libero_object", 6),
    28: ("libero_object", 5),
    29: ("libero_object", 8),
    # libero_spatial  (HF 30-39)
    30: ("libero_spatial", 6),
    31: ("libero_spatial", 4),
    32: ("libero_spatial", 5),
    33: ("libero_spatial", 7),
    34: ("libero_spatial", 0),
    35: ("libero_spatial", 3),
    36: ("libero_spatial", 8),
    37: ("libero_spatial", 1),
    38: ("libero_spatial", 2),
    39: ("libero_spatial", 9),
}

_SUITE_MAX_STEPS = {
    "libero_10":      500,
    "libero_goal":    350,
    "libero_object":  320,
    "libero_spatial": 250,
}
MAX_STEPS_MAP: Dict[int, int] = {
    hf_idx: _SUITE_MAX_STEPS[suite]
    for hf_idx, (suite, _) in TASK_MAP.items()
}

# GPU별 담당 태스크 그룹 (suite 단위로 묶어 env 재생성 최소화)
GPU_TASK_GROUPS: List[List[int]] = [
    list(range(0,  10)),   # GPU 0 → libero_10
    list(range(10, 20)),   # GPU 1 → libero_goal
    list(range(20, 30)),   # GPU 2 → libero_object
    list(range(30, 40)),   # GPU 3 → libero_spatial
]

# ============================================================
# PyArrow 스키마
# ============================================================

IMAGE_STRUCT = pa.struct([
    pa.field("bytes", pa.binary()),
    pa.field("path",  pa.large_string()),
])

SCHEMA = pa.schema([
    pa.field("observation.images.image",  IMAGE_STRUCT),
    pa.field("observation.images.image2", IMAGE_STRUCT),
    pa.field("observation.state",         pa.list_(pa.float32())),
    pa.field("action",                    pa.list_(pa.float32())),
    pa.field("timestamp",                 pa.float32()),
    pa.field("frame_index",               pa.int64()),
    pa.field("episode_index",             pa.int64()),
    pa.field("index",                     pa.int64()),
    pa.field("task_index",                pa.int64()),
])

# ============================================================
# 이미지 변환
# ============================================================

def rgb_to_png_bytes(arr: np.ndarray) -> bytes:
    """HxWx3 uint8 RGB ndarray → PNG bytes."""
    import cv2
    _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


# ============================================================
# LIBERO 시뮬레이션 유틸
# ============================================================

def make_libero_env(suite_name: str, task_id: int, seed: int = SEED):
    """OffScreenRenderEnv + init_states 반환."""
    bd    = benchmark.get_benchmark_dict()
    suite = bd[suite_name]()
    task  = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    bddl  = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env   = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=ENV_RES,
        camera_widths=ENV_RES,
    )
    env.seed(seed)
    return env, init_states


def set_delta(env):
    for robot in env.robots:
        robot.controller.use_delta = True


def obs_to_state8(obs: dict) -> np.ndarray:
    """
    LIBERO obs → 8D state: [eef_pos(3), axis_angle(3), gripper(2)]

    ★ LiberoProcessorStep 과 완전히 동일한 변환을 사용해야 합니다.
    쿼터니언 순서: LIBERO는 (w,x,y,z), scipy는 (x,y,z,w) 입니다.
    """
    eef_pos   = obs["robot0_eef_pos"].astype(np.float32)
    q_wxyz    = obs["robot0_eef_quat"].astype(np.float64)
    rotvec    = Rotation.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]  # xyzw 순서로 변환
    ).as_rotvec().astype(np.float32)
    gripper   = obs["robot0_gripper_qpos"].astype(np.float32)  # (2,)
    return np.concatenate([eef_pos, rotvec, gripper])          # (8,)


def get_recovery_sim_frames(
    env,
    noisy_init:   np.ndarray,   # 92D benchmark init state (noisy joints, 원본 물체 배치)
    clean_eef:    np.ndarray,   # (3,) 원본 데이터셋 첫 프레임 EEF 위치 (ground truth)
    first_action: np.ndarray,   # 원본 에피소드 첫 액션 → gripper 값만 참조
) -> list:
    """
    복구 청크 50프레임을 실제 시뮬레이션으로 수집합니다.

    【핵심 설계】
      clean_eef 는 시뮬레이션으로 구하지 않고
      원본 데이터셋의 첫 프레임 observation.state[:3] 을 직접 사용합니다.

      이유:
        - init_states[ep_pos_in_task] 의 물체 배치가 실제 에피소드와
          다를 수 있어 시뮬로 구한 clean_eef 는 부정확합니다.
        - 데이터셋 첫 프레임 EEF 가 실제 에피소드의 정확한 clean 위치입니다.

      noisy_init 에는 원본 init_state 의 물체 배치가 그대로 유지됩니다.
      (noisy_init[:7] 만 변경, 물체 관련 dims 는 원본과 동일)

    【복구 액션 공식】
      arm_delta = (clean_eef - noisy_eef) / RECOVERY_FRAMES
      → 50스텝에 걸쳐 선형적으로 clean 위치로 복귀, 회전 보정 없음
    """
    # Noisy 초기 상태로 시뮬 설정 (물체 배치는 원본 init_state 와 동일)
    env.reset()
    obs = env.set_init_state(noisy_init)
    set_delta(env)
    noisy_eef = obs["robot0_eef_pos"].copy()   # (3,)

    # 복구 액션 계산 (clean_eef = 데이터셋 ground truth)
    arm_delta    = (clean_eef - noisy_eef) / RECOVERY_FRAMES
    arm_delta    = np.clip(arm_delta, -ACTION_CLIP_ARM, ACTION_CLIP_ARM).astype(np.float32)
    gripper_val  = float(np.clip(first_action[6], -ACTION_CLIP_GRIPPER, ACTION_CLIP_GRIPPER))
    recovery_act = np.concatenate([
        arm_delta,
        np.zeros(3, np.float32),
        [gripper_val],
    ])  # (7,)

    # 50스텝 실행 + 프레임 수집
    # np.flipud: LIBERO OffScreenRenderEnv 는 OpenGL 규약(상하 반전)으로 출력하므로
    # HF 원본 데이터셋과 동일한 방향(정방향)이 되도록 저장 전 뒤집습니다.
    frames = []
    for _ in range(RECOVERY_FRAMES):
        frames.append({
            "img_bytes":  rgb_to_png_bytes(np.flipud(obs["agentview_image"])),
            "img2_bytes": rgb_to_png_bytes(np.flipud(obs["robot0_eye_in_hand_image"])),
            "state":      obs_to_state8(obs),
            "action":     recovery_act.copy(),
        })
        obs, _, _, _ = env.step(recovery_act.tolist())

    return frames


# ============================================================
# 데이터셋 로딩 (부모 프로세스에서 1회)
# ============================================================

def load_libero_raw() -> hf_datasets.Dataset:
    """이미지를 디코딩하지 않고 원본 PNG 바이트 그대로 로드합니다."""
    ds = hf_datasets.load_dataset(
        DATASET_ID,
        cache_dir=DATASET_CACHE,
        split="train",
    )
    ds = ds.cast_column("observation.images.image",  hf_datasets.Image(decode=False))
    ds = ds.cast_column("observation.images.image2", hf_datasets.Image(decode=False))
    return ds


def build_task_ep_order(
    ds: hf_datasets.Dataset,
    ep_to_rows: dict,
) -> Dict[int, List[int]]:
    """task_index → [ep_id, ...] (에피소드 순서 유지)."""
    task_ep_order: Dict[int, List[int]] = defaultdict(list)
    for ep_id in sorted(ep_to_rows.keys()):
        row_ids  = ep_to_rows[ep_id]
        task_idx = int(ds[row_ids[0]]["task_index"])
        task_ep_order[task_idx].append(ep_id)
    return dict(task_ep_order)


def group_by_episode(ds: hf_datasets.Dataset) -> dict:
    """episode_index → 행 인덱스 리스트 (frame_index 정렬)."""
    ep_to_rows: dict = {}
    for i, (ep, fi) in enumerate(zip(ds["episode_index"], ds["frame_index"])):
        if ep not in ep_to_rows:
            ep_to_rows[ep] = []
        ep_to_rows[ep].append((fi, i))
    return {ep: [idx for _, idx in sorted(rows)]
            for ep, rows in ep_to_rows.items()}


# ============================================================
# 프레임 → PyArrow Table
# ============================================================

def rows_to_table(
    sim_frames: list,
    orig_rows:  list,
    episode_index: int,
    frame_global_start: int,  # 전체 프레임 순번 시작값
) -> pa.Table:
    """
    복구(시뮬 이미지) 50프레임 + 원본 에피소드 프레임을 합쳐 하나의 Table 생성.

    ★ frame_index 연속성:
      복구 0~49, 원본 50~(50+T-1)
      → SmolVLA가 timestamp/frame_index 로 순서를 파악합니다.
    """
    img_col, img2_col = [], []
    states, actions   = [], []
    timestamps, frame_indices, ep_indices, indices, task_indices = [], [], [], [], []

    task_idx_val = int(orig_rows[0]["task_index"])

    # ---- 복구 청크 ----
    for t, f in enumerate(sim_frames):
        img_col.append({"bytes": f["img_bytes"],  "path": None})
        img2_col.append({"bytes": f["img2_bytes"], "path": None})
        states.append(f["state"].tolist())
        actions.append(f["action"].tolist())
        timestamps.append(float(t * DT))
        frame_indices.append(t)
        ep_indices.append(episode_index)
        indices.append(frame_global_start + t)
        task_indices.append(task_idx_val)

    # ---- 원본 프레임 (타임스탬프·frame_index에 offset 추가) ----
    ts_offset = RECOVERY_FRAMES * DT   # 2.5 s
    fi_offset = RECOVERY_FRAMES        # 50

    for i, r in enumerate(orig_rows):
        img_col.append({"bytes":  r["observation.images.image"]["bytes"],  "path": None})
        img2_col.append({"bytes": r["observation.images.image2"]["bytes"], "path": None})
        states.append([float(x) for x in r["observation.state"]])
        actions.append([float(x) for x in r["action"]])
        timestamps.append(float(r["timestamp"]) + ts_offset)
        frame_indices.append(int(r["frame_index"]) + fi_offset)
        ep_indices.append(episode_index)
        indices.append(frame_global_start + RECOVERY_FRAMES + i)
        task_indices.append(int(r["task_index"]))

    arrays = [
        pa.array(img_col,        type=IMAGE_STRUCT),
        pa.array(img2_col,       type=IMAGE_STRUCT),
        pa.array(states,         type=pa.list_(pa.float32())),
        pa.array(actions,        type=pa.list_(pa.float32())),
        pa.array(timestamps,     type=pa.float32()),
        pa.array(frame_indices,  type=pa.int64()),
        pa.array(ep_indices,     type=pa.int64()),
        pa.array(indices,        type=pa.int64()),
        pa.array(task_indices,   type=pa.int64()),
    ]
    return pa.table(dict(zip(SCHEMA.names, arrays)), schema=SCHEMA)


# ============================================================
# meta 폴더 생성 (info.json / stats.json / tasks.parquet)
# ============================================================

# 원본 meta 폴더 경로 (stats.json / tasks.parquet 복사 원본)
ORIG_META_DIR = pathlib.Path("/media/idna/Data/Noise_State_Augmentation/meta")


def write_meta_dir(out_root: pathlib.Path, num_episodes: int, total_frames: int):
    """
    원본 meta 폴더 양식에 맞추어 3개 파일을 생성합니다.
      out_root/meta/info.json
      out_root/meta/stats.json      ← 원본 복사 (정규화 범위 재사용)
      out_root/meta/tasks.parquet   ← 원본 복사 (task 설명 재사용)
    """
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) info.json : 원본과 동일한 키/값 구조
    info = {
        "codebase_version": "v3.0",
        "robot_type":       "panda",
        "total_episodes":   num_episodes,
        "total_frames":     total_frames,
        "total_tasks":      40,
        "chunks_size":      CHUNK_SIZE,
        "fps":              float(FPS),
        "splits":           {"train": f"0:{num_episodes}"},
        "data_path":        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path":       "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        # 증강 관련 부가 정보 (LeRobot 로더가 무시하는 extra 필드)
        "noise_std":        NOISE_STD,
        "recovery_frames":  RECOVERY_FRAMES,
        "features": {
            "observation.images.image": {
                "dtype": "image",
                "shape": [ENV_RES, ENV_RES, 3],
                "names": ["height", "width", "channel"],
                "fps":   float(FPS),
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": [ENV_RES, ENV_RES, 3],
                "names": ["height", "width", "channel"],
                "fps":   float(FPS),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["state"],
                "fps":   float(FPS),
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["actions"],
                "fps":   float(FPS),
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
                "fps":   float(FPS),
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
                "fps":   float(FPS),
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
                "fps":   float(FPS),
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
                "fps":   float(FPS),
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
                "fps":   float(FPS),
            },
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    # ── 2) stats.json : 원본 통계 복사
    #    (정규화 min/max 범위는 동일 환경이므로 그대로 재사용 가능)
    import shutil
    orig_stats = ORIG_META_DIR / "stats.json"
    if orig_stats.exists():
        shutil.copy2(orig_stats, meta_dir / "stats.json")
    else:
        print(f"  [경고] stats.json 원본 없음: {orig_stats}")

    # ── 3) tasks.parquet : 원본 복사
    orig_tasks = ORIG_META_DIR / "tasks.parquet"
    if orig_tasks.exists():
        shutil.copy2(orig_tasks, meta_dir / "tasks.parquet")
    else:
        print(f"  [경고] tasks.parquet 원본 없음: {orig_tasks}")

    # ── 4) meta/episodes/chunk-000/file-000.parquet
    _write_episodes_parquet(out_root)


def _write_episodes_parquet(out_root: pathlib.Path):
    """
    data/ 파일들을 순회하며 에피소드별 통계를 계산하고
    meta/episodes/chunk-000/file-000.parquet 를 생성합니다.
    LeRobotDatasetMetadata._load_metadata() 가 이 파일을 필요로 합니다.
    """
    import glob as _glob

    ep_dir = out_root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(_glob.glob(str(out_root / "data" / "chunk-*" / "file-*.parquet")))
    if not data_files:
        print("  [경고] data/ 파일 없음 - meta/episodes 스킵")
        return

    print("  meta/episodes 생성 중 (데이터 스캔)...", flush=True)

    # 에피소드별 집계
    ep_records: Dict[int, dict] = {}

    for fpath in tqdm.tqdm(data_files, desc="episodes meta scan", leave=False):
        tbl = pq.read_table(fpath)
        df  = tbl.to_pydict()

        n = len(df["episode_index"])
        for i in range(n):
            ep = int(df["episode_index"][i])
            fidx  = int(df["frame_index"][i])
            gidx  = int(df["index"][i])
            tidx  = int(df["task_index"][i])
            ts    = float(df["timestamp"][i])
            state = list(df["observation.state"][i])
            act   = list(df["action"][i])

            if ep not in ep_records:
                ep_records[ep] = {
                    "task_index": tidx,
                    "frame_min": fidx, "frame_max": fidx,
                    "index_min": gidx, "index_max": gidx,
                    "ts_min": ts,      "ts_max": ts,
                    "state_rows": [], "action_rows": [],
                    "frame_rows": [], "index_rows": [], "ts_rows": [],
                    "count": 0,
                    "chunk_index": 0, "file_index": 0,
                }
                # 청크/파일 인덱스 파싱
                import re as _re
                m = _re.search(r"chunk-(\d+)/file-(\d+)\.parquet", fpath)
                if m:
                    ep_records[ep]["chunk_index"] = int(m.group(1))
                    ep_records[ep]["file_index"]  = int(m.group(2))

            r = ep_records[ep]
            r["frame_min"] = min(r["frame_min"], fidx)
            r["frame_max"] = max(r["frame_max"], fidx)
            r["index_min"] = min(r["index_min"], gidx)
            r["index_max"] = max(r["index_max"], gidx)
            r["ts_min"]    = min(r["ts_min"],    ts)
            r["ts_max"]    = max(r["ts_max"],    ts)
            r["state_rows"].append(state)
            r["action_rows"].append(act)
            r["frame_rows"].append(fidx)
            r["index_rows"].append(gidx)
            r["ts_rows"].append(ts)
            r["count"] += 1

    # tasks.parquet 에서 task_index → task_description 매핑
    tasks_pq_path = out_root / "meta" / "tasks.parquet"
    task_desc_map: Dict[int, str] = {}
    if tasks_pq_path.exists():
        t = pq.read_table(tasks_pq_path).to_pydict()
        for tidx, tdesc in zip(t["task_index"], t["task"]):
            task_desc_map[int(tidx)] = str(tdesc)

    # 정렬된 에피소드 목록
    sorted_eps = sorted(ep_records.keys())

    rows: Dict[str, list] = {col: [] for col in [
        "episode_index", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index",
        "tasks", "length",
        "stats/observation.images.image/min",
        "stats/observation.images.image/max",
        "stats/observation.images.image/mean",
        "stats/observation.images.image/std",
        "stats/observation.images.image/count",
        "stats/observation.images.image2/min",
        "stats/observation.images.image2/max",
        "stats/observation.images.image2/mean",
        "stats/observation.images.image2/std",
        "stats/observation.images.image2/count",
        "stats/observation.state/min",
        "stats/observation.state/max",
        "stats/observation.state/mean",
        "stats/observation.state/std",
        "stats/observation.state/count",
        "stats/action/min",
        "stats/action/max",
        "stats/action/mean",
        "stats/action/std",
        "stats/action/count",
        "stats/timestamp/min",
        "stats/timestamp/max",
        "stats/timestamp/mean",
        "stats/timestamp/std",
        "stats/timestamp/count",
        "stats/frame_index/min",
        "stats/frame_index/max",
        "stats/frame_index/mean",
        "stats/frame_index/std",
        "stats/frame_index/count",
        "stats/episode_index/min",
        "stats/episode_index/max",
        "stats/episode_index/mean",
        "stats/episode_index/std",
        "stats/episode_index/count",
        "stats/index/min",
        "stats/index/max",
        "stats/index/mean",
        "stats/index/std",
        "stats/index/count",
        "stats/task_index/min",
        "stats/task_index/max",
        "stats/task_index/mean",
        "stats/task_index/std",
        "stats/task_index/count",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ]}

    for ep in sorted_eps:
        r   = ep_records[ep]
        cnt = r["count"]
        sa  = np.array(r["state_rows"],  dtype=np.float64)  # (N, 8)
        aa  = np.array(r["action_rows"], dtype=np.float64)  # (N, 7)
        fra = np.array(r["frame_rows"],  dtype=np.float64)
        ida = np.array(r["index_rows"],  dtype=np.float64)
        tsa = np.array(r["ts_rows"],     dtype=np.float64)

        tidx = r["task_index"]
        task_str = task_desc_map.get(tidx, f"task_{tidx}")

        rows["episode_index"].append(ep)
        rows["data/chunk_index"].append(r["chunk_index"])
        rows["data/file_index"].append(r["file_index"])
        rows["dataset_from_index"].append(int(r["index_min"]))
        rows["dataset_to_index"].append(int(r["index_max"]) + 1)
        rows["tasks"].append([task_str])
        rows["length"].append(cnt)

        # image stats: placeholder (normalized [0,1] 범위)
        for cam in ["observation.images.image", "observation.images.image2"]:
            rows[f"stats/{cam}/min"].append([[[ 0.0]] for _ in range(3)])
            rows[f"stats/{cam}/max"].append([[[ 1.0]] for _ in range(3)])
            rows[f"stats/{cam}/mean"].append([[[0.3]] for _ in range(3)])
            rows[f"stats/{cam}/std"].append([[[0.15]] for _ in range(3)])
            rows[f"stats/{cam}/count"].append([cnt])

        # state stats
        rows["stats/observation.state/min"].append(sa.min(axis=0).tolist())
        rows["stats/observation.state/max"].append(sa.max(axis=0).tolist())
        rows["stats/observation.state/mean"].append(sa.mean(axis=0).tolist())
        rows["stats/observation.state/std"].append(sa.std(axis=0).tolist())
        rows["stats/observation.state/count"].append([cnt])

        # action stats
        rows["stats/action/min"].append(aa.min(axis=0).tolist())
        rows["stats/action/max"].append(aa.max(axis=0).tolist())
        rows["stats/action/mean"].append(aa.mean(axis=0).tolist())
        rows["stats/action/std"].append(aa.std(axis=0).tolist())
        rows["stats/action/count"].append([cnt])

        # scalar stats
        for key, arr in [
            ("timestamp",    tsa),
            ("frame_index",  fra),
            ("index",        ida),
        ]:
            rows[f"stats/{key}/min"].append([float(arr.min())])
            rows[f"stats/{key}/max"].append([float(arr.max())])
            rows[f"stats/{key}/mean"].append([float(arr.mean())])
            rows[f"stats/{key}/std"].append([float(arr.std())])
            rows[f"stats/{key}/count"].append([cnt])

        for key in ("episode_index", "task_index"):
            val = float(ep if key == "episode_index" else tidx)
            rows[f"stats/{key}/min"].append([val])
            rows[f"stats/{key}/max"].append([val])
            rows[f"stats/{key}/mean"].append([val])
            rows[f"stats/{key}/std"].append([0.0])
            rows[f"stats/{key}/count"].append([cnt])

        rows["meta/episodes/chunk_index"].append(0)
        rows["meta/episodes/file_index"].append(0)

    tbl = pa.Table.from_pydict(rows)
    out_path = ep_dir / "file-000.parquet"
    pq.write_table(tbl, out_path)
    print(f"  meta/episodes/chunk-000/file-000.parquet 생성 완료  ({len(sorted_eps)} episodes)")


# ============================================================
# 워커 함수 (GPU 1개 담당)
# ============================================================

def _augment_worker(
    gpu_id:       int,
    task_indices: List[int],
    ds,                         # HuggingFace Dataset (fork로 공유)
    ep_to_rows:   dict,
    task_ep_order: Dict[int, List[int]],
    ep_offset:    int,          # 전체 에피소드 순번 시작값
    frame_offset: int,          # 전체 프레임 순번 시작값
    out_dir:      pathlib.Path,
    result_path:  str,
):
    """태스크 목록을 순차 처리합니다 (싱글 GPU / 멀티 GPU 모두 동작)."""

    noise_rng     = np.random.default_rng(SEED + gpu_id * 1000)
    ep_global_idx = ep_offset
    frame_global  = frame_offset
    total_frames  = 0
    task_results  = []

    for task_idx in task_indices:
        suite_name, task_id = TASK_MAP[task_idx]
        ep_list = task_ep_order.get(task_idx, [])
        if not ep_list:
            continue

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][GPU{gpu_id}] Task {task_idx:2d} {suite_name}/{task_id}"
              f"  ({len(ep_list)} episodes)", flush=True)

        # 태스크당 LIBERO env 1개 생성
        env, init_states = make_libero_env(suite_name, task_id)
        task_frames = 0

        for ep_pos_in_task, ep_id in enumerate(
            tqdm.tqdm(ep_list, desc=f"GPU{gpu_id} task{task_idx:2d}", leave=False)
        ):
            row_ids      = ep_to_rows[ep_id]
            ep_data      = ds.select(row_ids)
            first_action = np.array(ep_data[0]["action"], dtype=np.float32)

            # ★ clean_eef: 데이터셋 첫 프레임의 EEF 위치 (ground truth)
            #   시뮬레이션으로 추정하지 않고 직접 읽습니다.
            #   observation.state = [eef_pos(3), axis_angle(3), gripper(2)]
            clean_eef = np.array(ep_data[0]["observation.state"][:3], dtype=np.float32)

            # ★ noisy_init: 벤치마크 init_state의 물체 배치는 유지하고
            #   로봇 관절 각도([:7])에만 노이즈를 적용합니다.
            base_init        = init_states[ep_pos_in_task % len(init_states)].copy()
            noisy_init       = base_init.copy()
            noisy_init[:7]  += noise_rng.standard_normal(7).astype(np.float32) * NOISE_STD

            # 시뮬 실행 → 복구 50프레임 수집
            sim_frames = get_recovery_sim_frames(env, noisy_init, clean_eef, first_action)

            orig_rows  = [ep_data[i] for i in range(len(ep_data))]
            n_orig     = len(orig_rows)

            table = rows_to_table(
                sim_frames=sim_frames,
                orig_rows=orig_rows,
                episode_index=ep_id,
                frame_global_start=frame_global,
            )

            chunk_idx = ep_global_idx // CHUNK_SIZE
            file_idx  = ep_global_idx % CHUNK_SIZE
            chunk_dir = out_dir / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                chunk_dir / f"file-{file_idx:03d}.parquet",
                compression="snappy",
            )

            ep_global_idx += 1
            frame_global  += RECOVERY_FRAMES + n_orig
            task_frames   += RECOVERY_FRAMES + n_orig
            total_frames  += RECOVERY_FRAMES + n_orig

        env.close()
        task_results.append({
            "task_idx":  task_idx,
            "suite":     suite_name,
            "n_episodes": len(ep_list),
            "n_frames":  task_frames,
        })
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][GPU{gpu_id}] Task {task_idx:2d} done  "
              f"frames={task_frames:,}", flush=True)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu_id":       gpu_id,
            "total_frames": total_frames,
            "tasks":        task_results,
        }, f, indent=2)


# ============================================================
# 메인
# ============================================================

def main():
    # ── 출력 디렉터리
    noise_tag = f"noise_{NOISE_STD:.4f}".replace(".", "_")
    out_root  = OUTPUT_BASE / noise_tag
    out_dir   = out_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터셋 로드
    print(f"\n[1/3] 데이터셋 로드: {DATASET_ID}")
    ds = load_libero_raw()
    print(f"  총 프레임: {len(ds):,}")

    print(f"[2/3] 에피소드 그룹화...")
    ep_to_rows    = group_by_episode(ds)
    task_ep_order = build_task_ep_order(ds, ep_to_rows)
    num_episodes  = sum(len(v) for v in task_ep_order.values())
    print(f"  에피소드: {num_episodes}개  태스크: {len(task_ep_order)}개")

    # ── 단일 GPU: task 0-39 전체를 순차 처리
    all_tasks = list(range(40))
    print(f"\n[3/3] 증강 시작  noise_std={NOISE_STD}  (단일 GPU 순차 실행)")
    print(f"  처리 태스크: {all_tasks[0]}~{all_tasks[-1]}  총 {num_episodes}개 에피소드")

    result_path = str(out_root / "result.json")
    _augment_worker(
        gpu_id=0,
        task_indices=all_tasks,
        ds=ds,
        ep_to_rows=ep_to_rows,
        task_ep_order=task_ep_order,
        ep_offset=0,
        frame_offset=0,
        out_dir=out_dir,
        result_path=result_path,
    )

    # ── 결과 수집 + meta 생성
    total_frames_all = 0
    if pathlib.Path(result_path).exists():
        with open(result_path) as f:
            r = json.load(f)
        total_frames_all = r["total_frames"]
        pathlib.Path(result_path).unlink()

    write_meta_dir(out_root, num_episodes, total_frames_all)

    print(f"\n{'='*64}")
    print(f"  증강 완료")
    print(f"  저장 경로   : {out_root}")
    print(f"  에피소드 수 : {num_episodes:,}")
    print(f"  총 프레임   : {total_frames_all:,}")
    print(f"  (원본 + {RECOVERY_FRAMES}프레임 복구 청크 × {num_episodes} 에피소드)")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
