#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_action_noise.py

SmolVLA 추론 성공 궤적 기반 Action Noise "샌드위치" 데이터 증강

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【증강 원리】

  원본: s_GT(t) ─a_GT(t)→ s_GT(t+1) ─a_GT(t+1)→ s_GT(t+2)
                                 ↑
  증강: s_GT(t) ─a(t)→  s(t+1)  ──── a_rec ─────→ s_GT(t+2)
        │        └─ a_GT(t) + ε      (s(t+1) → GT(t+2) 방향)
        ε ~ N(0, σ),  σ = NOISE_RATIO × normalized_action_range

  증강 데이터 쌍 (per step t):
    Aug-1: (s_GT(t),   a(t)  )  ← 노이즈 액션으로 s(t+1) 유도
    Aug-2: (s(t+1),    a_rec )  ← 노이즈 상태에서 GT(t+2)로 복구

  INCLUDE_ORIGINAL=True (기본값):
    - 성공한 원본 궤적 에피소드 저장
    - 증강 쌍 에피소드 저장 (별도 episode_index)
  INCLUDE_ORIGINAL=False:
    - 증강 쌍 에피소드만 저장

【실행 흐름】
  1. SmolVLA 추론 → 성공한 궤적 수집 (각 스텝 sim_state 저장)
  2. 성공 에피소드마다 증강 쌍 생성
     - sim을 step t로 복원 → 노이즈 액션 적용 → s_noisy(t+1) 관측
     - s_noisy(t+1) EEF와 s_GT(t+2) EEF의 delta로 복구 액션 계산
  3. LeRobot v3.0 parquet 저장

【복구 액션 스케일 추정】
  각 성공 에피소드에서 EEF 이동량 / action 값으로 실험적 추정
  (fallback: DEFAULT_POS_SCALE=0.05 m, DEFAULT_ROT_SCALE=0.15 rad)

【서버 환경 (멀티 GPU)】
  MUJOCO_GL=egl  (헤드리스, 디스플레이 불필요)
  --gpu_id 인자로 담당 GPU를 지정합니다.
  run_augment_4gpu.sh 가 4개 프로세스를 병렬 실행하며 task를 분배합니다.

  GPU 0 → tasks  0- 9  (libero_10)
  GPU 1 → tasks 10-19  (libero_goal)
  GPU 2 → tasks 20-29  (libero_object)
  GPU 3 → tasks 30-39  (libero_spatial)

  각 GPU는 동일한 출력 디렉터리에 비중첩 episode_index / frame index로 씁니다.
  ep_offset / frame_offset 은 run_augment_4gpu.sh 가 미리 계산하여 전달합니다.

단독 실행 (단일 GPU):
  /media/idna/Data/envs/smolvla/bin/python augment_action_noise.py

4-GPU 병렬 실행:
  bash run_augment_4gpu.sh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import copy
import json
import os
import pathlib
import sys
import types as _types

# ── 서버 headless 렌더링 (EGL)
os.environ.setdefault("MUJOCO_GL", "egl")

# ── TF 비호환 방지: lerobot import 전에 tensorflow를 None으로 설정
# (ModuleType 객체를 먼저 넣으면 torch._dynamo가 __spec__=None 에서 ValueError 발생)
sys.modules["tensorflow"] = None  # type: ignore[assignment]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm
from scipy.spatial.transform import Rotation

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline

# ── lerobot import 완료 후 fake TF 모듈 세팅 (Tensor/Variable 접근 대비)
_fake_tf = _types.ModuleType("tensorflow")
class _FakeTFTensor: pass
_fake_tf.Tensor   = _FakeTFTensor
_fake_tf.Variable = _FakeTFTensor
sys.modules["tensorflow"] = _fake_tf

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import cv2

# ============================================================
# 설정
# ============================================================

MODEL_ID         = "HuggingFaceVLA/smolvla_libero"
OUTPUT_BASE      = pathlib.Path("./outputs/augmented_action_noise")

# ── 수집 설정
N_SUCCESS        = 10    # 태스크당 수집할 성공 에피소드 수
MAX_ATTEMPTS     = 50    # 태스크당 최대 추론 시도 횟수
NUM_WAIT         = 10    # 초기 안정화 스텝 (기록 X)
ENV_RES          = 256   # 이미지 해상도 (px)
CHUNK_SIZE       = 1000  # parquet 파일당 에피소드 수
FPS              = 10
SEED             = 42

# ── 노이즈 설정
# 정규화 액션 범위 [-1, 1] → range=2 → 10% = 0.2
# σ=0.10 이면 95% 샘플이 ±0.20 이내 (range의 10%)
NOISE_RATIO      = 0.10                  # 범위 대비 노이즈 비율
ACTION_RANGE     = 2.0                   # 정규화 범위 크기 (-1 ~ 1)
NOISE_STD        = NOISE_RATIO * ACTION_RANGE / 2.0   # = 0.10
NOISE_DIMS       = 6                     # pos(3)+rot(3) 만 노이즈 (gripper 제외)
ACTION_CLIP      = 1.0                   # 클립 범위

# ── 복구 액션 스케일 (OSC_POSE 컨트롤러 기본값, 실험적으로 덮어씀)
DEFAULT_POS_SCALE = 0.05   # action=1.0 → ~0.05 m
DEFAULT_ROT_SCALE = 0.15   # action=1.0 → ~0.15 rad

# ── 원본 궤적 포함 여부
INCLUDE_ORIGINAL = True    # True: 원본+증강 모두 저장 / False: 증강만 저장

# ── TASK_MAP: HF task_index(0-39) → (suite_name, libero_task_id)
# (HF 데이터셋 task_index와 일치하도록 이름 기반 매핑)
TASK_MAP = {
    # libero_10  (HF 0-9)
    0:  ("libero_10",     4),  1:  ("libero_10",     6),  2:  ("libero_10",     9),
    3:  ("libero_10",     2),  4:  ("libero_10",     7),  5:  ("libero_10",     0),
    6:  ("libero_10",     8),  7:  ("libero_10",     1),  8:  ("libero_10",     3),
    9:  ("libero_10",     5),
    # libero_goal  (HF 10-19)
    10: ("libero_goal",   8),  11: ("libero_goal",   9),  12: ("libero_goal",   3),
    13: ("libero_goal",   6),  14: ("libero_goal",   2),  15: ("libero_goal",   5),
    16: ("libero_goal",   7),  17: ("libero_goal",   1),  18: ("libero_goal",   4),
    19: ("libero_goal",   0),
    # libero_object  (HF 20-29)
    20: ("libero_object", 9),  21: ("libero_object", 4),  22: ("libero_object", 1),
    23: ("libero_object", 3),  24: ("libero_object", 0),  25: ("libero_object", 7),
    26: ("libero_object", 2),  27: ("libero_object", 6),  28: ("libero_object", 5),
    29: ("libero_object", 8),
    # libero_spatial  (HF 30-39)
    30: ("libero_spatial", 6), 31: ("libero_spatial", 4), 32: ("libero_spatial", 5),
    33: ("libero_spatial", 7), 34: ("libero_spatial", 0), 35: ("libero_spatial", 3),
    36: ("libero_spatial", 8), 37: ("libero_spatial", 1), 38: ("libero_spatial", 2),
    39: ("libero_spatial", 9),
}

_SUITE_MAX_STEPS = {
    "libero_10":      500,
    "libero_goal":    350,
    "libero_object":  320,
    "libero_spatial": 250,
}
MAX_STEPS_MAP = {i: _SUITE_MAX_STEPS[s] for i, (s, _) in TASK_MAP.items()}

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
# 유틸
# ============================================================

def rgb_to_png_bytes(arr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def obs_to_state8(obs: dict) -> list:
    """LIBERO obs dict → 8D float32 list [eef_pos(3), axis_angle(3), gripper(2)]"""
    eef_pos   = obs["robot0_eef_pos"].astype(np.float32)
    quat_wxyz = obs["robot0_eef_quat"].astype(np.float64)
    rotvec    = Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_rotvec().astype(np.float32)
    gripper   = obs["robot0_gripper_qpos"].astype(np.float32)
    return np.concatenate([eef_pos, rotvec, gripper]).tolist()


def make_policy_batch(obs: dict) -> dict:
    return {
        "pixels": {
            "image":  obs["agentview_image"],
            "image2": obs["robot0_eye_in_hand_image"],
        },
        "robot_state": {
            "eef": {
                "pos":  obs["robot0_eef_pos"][np.newaxis],
                "quat": obs["robot0_eef_quat"][np.newaxis],
                "mat":  None,
            },
            "gripper": {
                "qpos": obs["robot0_gripper_qpos"][np.newaxis],
                "qvel": None,
            },
            "joints": {"pos": None, "vel": None},
        },
    }


def set_delta_mode(env):
    """OSC 컨트롤러를 delta 모드로 설정."""
    for robot in env.robots:
        robot.controller.use_delta = True


def make_env(suite_name: str, task_id: int, seed: int):
    bd    = benchmark.get_benchmark_dict()
    suite = bd[suite_name]()
    task  = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    bddl_path   = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=ENV_RES,
        camera_widths=ENV_RES,
    )
    env.seed(seed)
    return env, task.language, init_states


# ============================================================
# Phase 1: 에피소드 추론 + 궤적 수집
# ============================================================

def run_inference_episode(
    env,
    task_desc: str,
    policy,
    env_pre,
    pre,
    post,
    init_state: np.ndarray,
    max_steps:  int,
) -> tuple:
    """
    SmolVLA 추론을 실행하고 성공 시 궤적을 반환합니다.

    Returns:
        success:     bool
        obs_list:    list[dict]           - 각 step의 관측값
        action_list: list[np.ndarray]     - 각 step에서 선택한 7D 액션
        sim_states:  list[MjSimState]     - 각 step 진입 직전 시뮬레이터 상태
    """
    env.reset()
    obs = env.set_init_state(init_state)
    set_delta_mode(env)
    policy.reset()

    # 초기 안정화 (기록 X)
    for _ in range(NUM_WAIT):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

    obs_list    = []
    action_list = []
    sim_states  = []
    success     = False

    for _ in range(max_steps):
        # ★ step 진입 직전 sim 상태 저장 (mid-episode 복원용)
        sim_states.append(copy.deepcopy(env.sim.get_state()))

        # 액션 예측
        batch = preprocess_observation(make_policy_batch(obs))
        batch["task"] = task_desc
        batch = env_pre(batch)
        batch = pre(batch)

        with torch.inference_mode():
            action_tensor = policy.select_action(batch)

        action_np = post(action_tensor).squeeze(0).cpu().numpy()  # (7,)

        obs_list.append({k: v.copy() for k, v in obs.items()})
        action_list.append(action_np.copy())

        obs, _, done, _ = env.step(action_np.tolist())

        if done:
            success = bool(env.check_success())
            break

    return success, obs_list, action_list, sim_states


# ============================================================
# Phase 2: 복구 액션 스케일 추정
# ============================================================

def estimate_action_scales(obs_list: list, action_list: list) -> tuple:
    """
    성공 궤적의 EEF 이동량으로 position / rotation action scale을 추정합니다.
    scale: action=1.0 일 때 실제 이동량 (m 또는 rad)
    """
    pos_scales, rot_scales = [], []

    for t in range(len(obs_list) - 1):
        a_pos = action_list[t][:3]
        a_rot = action_list[t][3:6]

        pos_t  = obs_list[t]["robot0_eef_pos"].astype(np.float64)
        pos_t1 = obs_list[t+1]["robot0_eef_pos"].astype(np.float64)
        dpos   = pos_t1 - pos_t

        quat_t  = obs_list[t]["robot0_eef_quat"].astype(np.float64)
        quat_t1 = obs_list[t+1]["robot0_eef_quat"].astype(np.float64)
        r_t     = Rotation.from_quat([quat_t[1],  quat_t[2],  quat_t[3],  quat_t[0]])
        r_t1    = Rotation.from_quat([quat_t1[1], quat_t1[2], quat_t1[3], quat_t1[0]])
        drot    = (r_t.inv() * r_t1).as_rotvec()

        # 액션이 충분히 큰 차원만 사용 (노이즈 방지)
        mask_p = np.abs(a_pos) > 0.05
        if mask_p.any():
            scales = np.abs(dpos[mask_p]) / (np.abs(a_pos[mask_p]) + 1e-9)
            pos_scales.extend(scales[scales > 1e-6].tolist())

        mask_r = np.abs(a_rot) > 0.05
        if mask_r.any():
            scales = np.abs(drot[mask_r]) / (np.abs(a_rot[mask_r]) + 1e-9)
            rot_scales.extend(scales[scales > 1e-6].tolist())

    pos_scale = float(np.clip(np.median(pos_scales), 0.005, 0.20)) if len(pos_scales) > 5 else DEFAULT_POS_SCALE
    rot_scale = float(np.clip(np.median(rot_scales), 0.005, 0.50)) if len(rot_scales) > 5 else DEFAULT_ROT_SCALE
    return pos_scale, rot_scale


# ============================================================
# Phase 2: 복구 액션 계산
# ============================================================

def compute_recovery_action(
    obs_noisy:   dict,
    obs_gt_t2:   dict,
    gripper_ref: float,
    pos_scale:   float,
    rot_scale:   float,
) -> np.ndarray:
    """
    s(t+1) [noisy] → s_GT(t+2) 방향의 복구 액션 (7D) 계산.

    pos:     EEF 위치 차이 / pos_scale
    rot:     EEF 쿼터니언 차이(body-frame axis-angle) / rot_scale
    gripper: a_GT(t+1) gripper 유지
    """
    # 위치 복구
    eef_n  = obs_noisy["robot0_eef_pos"].astype(np.float64)
    eef_t2 = obs_gt_t2["robot0_eef_pos"].astype(np.float64)
    delta_pos = (eef_t2 - eef_n) / pos_scale

    # 회전 복구 (body frame 상대 axis-angle)
    qn  = obs_noisy["robot0_eef_quat"].astype(np.float64)
    qt2 = obs_gt_t2["robot0_eef_quat"].astype(np.float64)
    r_n   = Rotation.from_quat([qn[1],  qn[2],  qn[3],  qn[0]])
    r_t2  = Rotation.from_quat([qt2[1], qt2[2], qt2[3], qt2[0]])
    delta_rot = (r_n.inv() * r_t2).as_rotvec() / rot_scale

    a_rec = np.array([*delta_pos, *delta_rot, gripper_ref], dtype=np.float32)
    return np.clip(a_rec, -ACTION_CLIP, ACTION_CLIP)


# ============================================================
# Phase 2: 증강 에피소드 프레임 생성
# ============================================================

def build_aug_frames(
    env,
    obs_list:    list,
    action_list: list,
    sim_states:  list,
    ep_idx:      int,
    frame_start: int,
    task_idx:    int,
    pos_scale:   float,
    rot_scale:   float,
    rng:         np.random.Generator,
) -> list:
    """
    한 에피소드의 (Aug-1, Aug-2) 쌍 전체를 생성합니다.

    프레임 구조 (step t마다):
      frame 2t  : Aug-1  (s_GT(t),    a_GT(t)+ε)
      frame 2t+1: Aug-2  (s_noisy(t+1), a_recovery)

    Returns:
        frames: list of frame dicts
    """
    frames    = []
    aug_fidx  = 0   # episode 내 frame_index (0-based)
    T         = len(obs_list)
    DT        = 1.0 / FPS

    for t in tqdm.tqdm(range(T - 2), desc="  aug gen", leave=False):
        # ── sim을 step t 시점으로 복원
        env.sim.set_state(sim_states[t])
        env.sim.forward()
        set_delta_mode(env)

        # ── 노이즈 액션: pos(3)+rot(3) 에만 적용, gripper(1) 유지
        a_noisy = action_list[t].copy()
        noise   = rng.standard_normal(NOISE_DIMS).astype(np.float32) * NOISE_STD
        a_noisy[:NOISE_DIMS] = np.clip(
            a_noisy[:NOISE_DIMS] + noise, -ACTION_CLIP, ACTION_CLIP
        )

        # ── a_noisy 적용 → s_noisy(t+1) 관측
        obs_noisy, _, _, _ = env.step(a_noisy.tolist())

        # ── 복구 액션: s_noisy(t+1) → s_GT(t+2)
        gripper_ref = float(action_list[t + 1][6])
        a_rec = compute_recovery_action(
            obs_noisy, obs_list[t + 2],
            gripper_ref, pos_scale, rot_scale,
        )

        # ── Aug-1: (s_GT(t), a_noisy)
        frames.append({
            "img_bytes":     rgb_to_png_bytes(np.flipud(obs_list[t]["agentview_image"])),
            "img2_bytes":    rgb_to_png_bytes(np.flipud(obs_list[t]["robot0_eye_in_hand_image"])),
            "state":         obs_to_state8(obs_list[t]),
            "action":        a_noisy.tolist(),
            "timestamp":     float(aug_fidx * DT),
            "frame_index":   aug_fidx,
            "episode_index": ep_idx,
            "index":         frame_start + aug_fidx,
            "task_index":    task_idx,
        })
        aug_fidx += 1

        # ── Aug-2: (s_noisy(t+1), a_recovery)
        frames.append({
            "img_bytes":     rgb_to_png_bytes(np.flipud(obs_noisy["agentview_image"])),
            "img2_bytes":    rgb_to_png_bytes(np.flipud(obs_noisy["robot0_eye_in_hand_image"])),
            "state":         obs_to_state8(obs_noisy),
            "action":        a_rec.tolist(),
            "timestamp":     float(aug_fidx * DT),
            "frame_index":   aug_fidx,
            "episode_index": ep_idx,
            "index":         frame_start + aug_fidx,
            "task_index":    task_idx,
        })
        aug_fidx += 1

    return frames


def build_orig_frames(
    obs_list:    list,
    action_list: list,
    ep_idx:      int,
    frame_start: int,
    task_idx:    int,
) -> list:
    """원본 궤적을 LeRobot 포맷 프레임 리스트로 변환합니다."""
    frames = []
    DT     = 1.0 / FPS
    for t, (obs, action) in enumerate(zip(obs_list, action_list)):
        frames.append({
            "img_bytes":     rgb_to_png_bytes(np.flipud(obs["agentview_image"])),
            "img2_bytes":    rgb_to_png_bytes(np.flipud(obs["robot0_eye_in_hand_image"])),
            "state":         obs_to_state8(obs),
            "action":        action.tolist(),
            "timestamp":     float(t * DT),
            "frame_index":   t,
            "episode_index": ep_idx,
            "index":         frame_start + t,
            "task_index":    task_idx,
        })
    return frames


# ============================================================
# LeRobot v3.0 저장
# ============================================================

def frames_to_table(frames: list) -> pa.Table:
    arrays = [
        pa.array([{"bytes": f["img_bytes"],  "path": None} for f in frames], type=IMAGE_STRUCT),
        pa.array([{"bytes": f["img2_bytes"], "path": None} for f in frames], type=IMAGE_STRUCT),
        pa.array([f["state"]         for f in frames], type=pa.list_(pa.float32())),
        pa.array([f["action"]        for f in frames], type=pa.list_(pa.float32())),
        pa.array([f["timestamp"]     for f in frames], type=pa.float32()),
        pa.array([f["frame_index"]   for f in frames], type=pa.int64()),
        pa.array([f["episode_index"] for f in frames], type=pa.int64()),
        pa.array([f["index"]         for f in frames], type=pa.int64()),
        pa.array([f["task_index"]    for f in frames], type=pa.int64()),
    ]
    return pa.table(dict(zip(SCHEMA.names, arrays)), schema=SCHEMA)


def save_episode(frames: list, ep_idx: int, data_dir: pathlib.Path):
    if not frames:
        return
    chunk_idx = ep_idx // CHUNK_SIZE
    file_idx  = ep_idx % CHUNK_SIZE
    chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        frames_to_table(frames),
        chunk_dir / f"file-{file_idx:03d}.parquet",
        compression="snappy",
    )


def write_info_json(out_root: pathlib.Path, total_episodes: int, total_frames: int):
    info = {
        "codebase_version":  "v3.0",
        "data_path":         "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "fps":               float(FPS),
        "splits":            {"train": f"0:{total_episodes}"},
        "total_episodes":    total_episodes,
        "total_frames":      total_frames,
        "total_tasks":       40,
        "total_chunks":      (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size":       CHUNK_SIZE,
        "include_original":  INCLUDE_ORIGINAL,
        "noise_std":         float(NOISE_STD),
        "noise_dims":        NOISE_DIMS,
        "noise_ratio":       NOISE_RATIO,
        "model_id":          MODEL_ID,
        "features": {
            "observation.images.image":  {
                "dtype": "image", "shape": [ENV_RES, ENV_RES, 3],
                "names": ["height", "width", "channel"], "fps": float(FPS),
            },
            "observation.images.image2": {
                "dtype": "image", "shape": [ENV_RES, ENV_RES, 3],
                "names": ["height", "width", "channel"], "fps": float(FPS),
            },
            "observation.state": {
                "dtype": "float32", "shape": [8],
                "names": ["eef_pos_x","eef_pos_y","eef_pos_z",
                          "eef_axis_x","eef_axis_y","eef_axis_z",
                          "gripper_pos","gripper_vel"],
                "fps": float(FPS),
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": ["dx","dy","dz","dax","day","daz","gripper"],
                "fps": float(FPS),
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None, "fps": float(FPS)},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None, "fps": float(FPS)},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None, "fps": float(FPS)},
            "index":         {"dtype": "int64",   "shape": [1], "names": None, "fps": float(FPS)},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None, "fps": float(FPS)},
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ============================================================
# 메인
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Action Noise 샌드위치 데이터 증강")
    p.add_argument("--gpu_id",      type=int, default=0,
                   help="사용할 GPU 번호 (CUDA_VISIBLE_DEVICES 설정)")
    p.add_argument("--task_start",  type=int, default=0,
                   help="담당 task_index 시작 (inclusive)")
    p.add_argument("--task_end",    type=int, default=40,
                   help="담당 task_index 끝 (exclusive)")
    p.add_argument("--ep_offset",   type=int, default=0,
                   help="전체 episode_index 시작 오프셋 (GPU 간 비중첩 보장)")
    p.add_argument("--frame_offset",type=int, default=0,
                   help="전체 frame index 시작 오프셋 (GPU 간 비중첩 보장)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── GPU 지정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    rng = np.random.default_rng(SEED + args.gpu_id * 1000)
    torch.manual_seed(SEED + args.gpu_id)

    task_range = list(range(args.task_start, args.task_end))

    data_dir = OUTPUT_BASE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[GPU {args.gpu_id}] tasks {args.task_start}-{args.task_end - 1}"
          f"  ep_offset={args.ep_offset}  frame_offset={args.frame_offset}")

    # ── 모델 로드
    print(f"[GPU {args.gpu_id}][1/2] 모델 로딩: {MODEL_ID}")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()
    print(f"[GPU {args.gpu_id}]      device={device}  chunk_size={policy.config.chunk_size}")

    pre, post = make_pre_post_processors(policy.config, MODEL_ID)
    env_pre   = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    print(f"[GPU {args.gpu_id}]  σ={NOISE_STD:.3f}  INCLUDE_ORIGINAL={INCLUDE_ORIGINAL}")

    # ── 수집 + 증강
    print(f"[GPU {args.gpu_id}][2/2] 수집 및 증강 시작")
    global_ep_idx    = args.ep_offset
    global_frame_idx = args.frame_offset
    total_frames     = 0
    summary          = []

    for task_idx in task_range:
        suite_name, task_id = TASK_MAP[task_idx]
        max_steps           = MAX_STEPS_MAP[task_idx]

        print(f"\n[GPU {args.gpu_id}] {'─'*56}")
        print(f"[GPU {args.gpu_id}]  task {task_idx:2d}  {suite_name}/{task_id}")

        env, task_desc, init_states = make_env(suite_name, task_id, SEED + args.gpu_id)
        print(f"[GPU {args.gpu_id}]  {task_desc[:68]}")

        collected   = 0
        attempt     = 0
        task_frames = 0

        pbar = tqdm.tqdm(total=N_SUCCESS, desc=f"[GPU{args.gpu_id}] task{task_idx:2d}", unit="ep")

        while collected < N_SUCCESS and attempt < MAX_ATTEMPTS:
            init_state = init_states[attempt % len(init_states)].copy()

            success, obs_list, action_list, sim_states = run_inference_episode(
                env=env, task_desc=task_desc,
                policy=policy, env_pre=env_pre, pre=pre, post=post,
                init_state=init_state, max_steps=max_steps,
            )
            attempt += 1

            if not success or len(obs_list) < 3:
                continue

            # ── 스케일 추정
            pos_scale, rot_scale = estimate_action_scales(obs_list, action_list)

            ep_frames_count = 0

            # ── 원본 궤적 저장 (INCLUDE_ORIGINAL=True)
            if INCLUDE_ORIGINAL:
                orig_frames = build_orig_frames(
                    obs_list=obs_list,
                    action_list=action_list,
                    ep_idx=global_ep_idx,
                    frame_start=global_frame_idx,
                    task_idx=task_idx,
                )
                save_episode(orig_frames, global_ep_idx, data_dir)
                n_orig = len(orig_frames)
                global_ep_idx    += 1
                global_frame_idx += n_orig
                ep_frames_count  += n_orig
                task_frames      += n_orig
                total_frames     += n_orig

            # ── 증강 에피소드 저장
            aug_frames = build_aug_frames(
                env=env,
                obs_list=obs_list,
                action_list=action_list,
                sim_states=sim_states,
                ep_idx=global_ep_idx,
                frame_start=global_frame_idx,
                task_idx=task_idx,
                pos_scale=pos_scale,
                rot_scale=rot_scale,
                rng=rng,
            )
            save_episode(aug_frames, global_ep_idx, data_dir)
            n_aug = len(aug_frames)
            global_ep_idx    += 1
            global_frame_idx += n_aug
            ep_frames_count  += n_aug
            task_frames      += n_aug
            total_frames     += n_aug

            collected += 1
            pbar.update(1)
            pbar.set_postfix(att=attempt, frames=ep_frames_count,
                             ps=f"{pos_scale:.3f}", rs=f"{rot_scale:.3f}")

        pbar.close()
        env.close()

        status = "✓" if collected >= N_SUCCESS else f"✗({collected}/{N_SUCCESS})"
        print(f"[GPU {args.gpu_id}]  → {status}  시도:{attempt}  수집:{collected}  프레임:{task_frames:,}")

        summary.append({
            "task_index":  task_idx,
            "suite":       suite_name,
            "task_id":     task_id,
            "description": task_desc,
            "collected":   collected,
            "attempts":    attempt,
            "frames":      task_frames,
        })

    # ── GPU별 부분 결과 저장 (나중에 run_augment_4gpu.sh 가 병합)
    partial_path = OUTPUT_BASE / f"partial_gpu{args.gpu_id}.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu_id":        args.gpu_id,
            "task_start":    args.task_start,
            "task_end":      args.task_end,
            "ep_offset":     args.ep_offset,
            "ep_end":        global_ep_idx,
            "frame_offset":  args.frame_offset,
            "frame_end":     global_frame_idx,
            "total_frames":  total_frames,
            "tasks":         summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[GPU {args.gpu_id}] {'='*56}")
    print(f"[GPU {args.gpu_id}]  완료  frames={total_frames:,}  eps={global_ep_idx - args.ep_offset}")
    print(f"[GPU {args.gpu_id}]  부분 결과: {partial_path}")
    print(f"[GPU {args.gpu_id}] {'='*56}")


if __name__ == "__main__":
    main()
