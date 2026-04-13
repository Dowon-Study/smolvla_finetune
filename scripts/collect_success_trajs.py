#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_success_trajs.py

SmolVLA 모델 추론을 통해 성공하는 궤적을 수집합니다.

- 40개 태스크(libero_spatial×10 / libero_object×10 / libero_goal×10 / libero_10×10)
- 태스크당 10개의 성공 에피소드 수집 (최대 MAX_ATTEMPTS 시도)
- 각 성공 에피소드를 LeRobot v3.0 parquet 형식으로 저장

출력:
  outputs/collected_trajs/data/chunk-{c:03d}/file-{f:03d}.parquet
  outputs/collected_trajs/info.json

실행:
  /media/idna/Data/envs/smolvla/bin/python collect_success_trajs.py
"""

import json
import os
import pathlib
import sys
import types as _types

os.environ["MUJOCO_GL"] = "glx"

# ------------------------------------------------------------------
# TF / NumPy 비호환 문제 회피 (datasets import 전에 처리)
# ------------------------------------------------------------------
sys.modules["tensorflow"] = None  # type: ignore[assignment]

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline

_fake_tf = _types.ModuleType("tensorflow")
class _FakeTFTensor: pass
_fake_tf.Tensor   = _FakeTFTensor
_fake_tf.Variable = _FakeTFTensor
sys.modules["tensorflow"] = _fake_tf

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm
from scipy.spatial.transform import Rotation

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import cv2

# ============================================================
# 설정
# ============================================================

MODEL_ID       = "HuggingFaceVLA/smolvla_libero"
OUTPUT_BASE    = pathlib.Path("./outputs/collected_trajs")
N_SUCCESS      = 10          # 태스크당 수집할 성공 에피소드 수
MAX_ATTEMPTS   = 100         # 태스크당 최대 시도 횟수
NUM_WAIT       = 10          # 초기 안정화 스텝
ENV_RES        = 256
CHUNK_SIZE     = 1000
SEED           = 7

# task_index (0-39) → (suite_name, task_id, max_steps)
TASK_MAP = {}
MAX_STEPS_MAP = {}
for _i in range(10):
    TASK_MAP[_i]      = ("libero_spatial", _i);  MAX_STEPS_MAP[_i]      = 250
    TASK_MAP[_i + 10] = ("libero_object",  _i);  MAX_STEPS_MAP[_i + 10] = 320
    TASK_MAP[_i + 20] = ("libero_goal",    _i);  MAX_STEPS_MAP[_i + 20] = 350
    TASK_MAP[_i + 30] = ("libero_10",      _i);  MAX_STEPS_MAP[_i + 30] = 500

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


def obs_to_state8(obs: dict) -> np.ndarray:
    """LIBERO obs → 8D [eef_pos(3), axis_angle(3), gripper(2)]."""
    eef_pos   = obs["robot0_eef_pos"].astype(np.float32)
    quat_wxyz = obs["robot0_eef_quat"].astype(np.float64)
    rotvec    = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2],
                                    quat_wxyz[3], quat_wxyz[0]]).as_rotvec().astype(np.float32)
    gripper   = obs["robot0_gripper_qpos"].astype(np.float32)
    return np.concatenate([eef_pos, rotvec, gripper])


def make_env(suite_name: str, task_id: int, seed: int):
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
    return env, task.language, init_states


def set_delta(env):
    for robot in env.robots:
        robot.controller.use_delta = True


def make_obs_batch(obs: dict) -> dict:
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


# ============================================================
# 에피소드 실행 + 프레임 수집
# ============================================================

def run_episode_collect(
    env,
    task_desc: str,
    policy,
    env_pre,
    pre,
    post,
    init_state: np.ndarray,
    max_steps: int,
    episode_index: int,
    global_frame_start: int,
    task_index: int,
) -> tuple:
    """
    에피소드를 실행하고 (성공 여부, 프레임 리스트)를 반환합니다.

    프레임 리스트 = [
      {img, img2, state, action, timestamp, frame_index, episode_index, index, task_index},
      ...
    ]
    Wait 스텝은 기록하지 않습니다.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    set_delta(env)
    policy.reset()

    # 안정화 스텝 (기록 X)
    for _ in range(NUM_WAIT):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

    frames  = []
    success = False
    FPS     = 20
    DT      = 1.0 / FPS

    try:
        for t in range(max_steps):
            # 관측 → 액션 예측
            batch = preprocess_observation(make_obs_batch(obs))
            batch["task"] = task_desc
            batch = env_pre(batch)
            batch = pre(batch)

            with torch.inference_mode():
                action_tensor = policy.select_action(batch)

            action_np = post(action_tensor).squeeze(0).cpu().numpy()  # (7,)

            # 현재 프레임 기록
            frames.append({
                "img_bytes":  rgb_to_png_bytes(obs["agentview_image"]),
                "img2_bytes": rgb_to_png_bytes(obs["robot0_eye_in_hand_image"]),
                "state":      obs_to_state8(obs).tolist(),
                "action":     action_np.tolist(),
                "timestamp":  float(t * DT),
                "frame_index":  t,
                "episode_index": episode_index,
                "index":      global_frame_start + t,
                "task_index": task_index,
            })

            obs, _, done, _ = env.step(action_np.tolist())

            if done:
                success = bool(env.check_success())
                break

    except Exception as e:
        print(f"    [오류] {e}")

    return success, frames


# ============================================================
# 프레임 리스트 → PyArrow Table
# ============================================================

def frames_to_table(frames: list) -> pa.Table:
    arrays = [
        pa.array([{"bytes": f["img_bytes"],  "path": None} for f in frames], type=IMAGE_STRUCT),
        pa.array([{"bytes": f["img2_bytes"], "path": None} for f in frames], type=IMAGE_STRUCT),
        pa.array([f["state"]        for f in frames], type=pa.list_(pa.float32())),
        pa.array([f["action"]       for f in frames], type=pa.list_(pa.float32())),
        pa.array([f["timestamp"]    for f in frames], type=pa.float32()),
        pa.array([f["frame_index"]  for f in frames], type=pa.int64()),
        pa.array([f["episode_index"] for f in frames], type=pa.int64()),
        pa.array([f["index"]        for f in frames], type=pa.int64()),
        pa.array([f["task_index"]   for f in frames], type=pa.int64()),
    ]
    return pa.table(dict(zip(SCHEMA.names, arrays)), schema=SCHEMA)


# ============================================================
# info.json
# ============================================================

def write_info_json(out_root: pathlib.Path, total_episodes: int,
                    total_frames: int):
    num_chunks = (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE
    info = {
        "codebase_version": "v3.0",
        "data_path":  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "fps":        20,
        "splits":     {"train": f"0:{total_episodes}"},
        "total_episodes":  total_episodes,
        "total_frames":    total_frames,
        "total_tasks":     40,
        "total_videos":    0,
        "total_chunks":    num_chunks,
        "chunks_size":     CHUNK_SIZE,
        "repo_id":         "smolvla_libero_success_trajs",
        "model_id":        MODEL_ID,
        "n_success_per_task": N_SUCCESS,
        "features": {
            "observation.images.image":  {"dtype": "image", "shape": [ENV_RES, ENV_RES, 3], "names": None},
            "observation.images.image2": {"dtype": "image", "shape": [ENV_RES, ENV_RES, 3], "names": None},
            "observation.state": {
                "dtype": "float32", "shape": [8],
                "names": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                          "eef_axis_x", "eef_axis_y", "eef_axis_z",
                          "gripper_pos", "gripper_vel"],
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": ["delta_eef_x", "delta_eef_y", "delta_eef_z",
                          "delta_rot_x", "delta_rot_y", "delta_rot_z", "gripper"],
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    with open(out_root / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ============================================================
# 메인
# ============================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    data_dir = OUTPUT_BASE / "data"
    data_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    print(f"[1/2] 모델 로딩: {MODEL_ID}")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()
    print(f"      device={device}  chunk_size={policy.config.chunk_size}")

    pre, post = make_pre_post_processors(policy.config, MODEL_ID)
    env_pre   = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    # ------------------------------------------------------------------
    # 수집
    # ------------------------------------------------------------------
    print(f"\n[2/2] 성공 궤적 수집 시작")
    print(f"      태스크: 40개  /  태스크당 목표: {N_SUCCESS}개  /  최대 시도: {MAX_ATTEMPTS}회\n")

    global_ep_idx    = 0   # 전체 에피소드 인덱스
    global_frame_idx = 0   # 전체 프레임 인덱스
    total_episodes   = 0
    total_frames     = 0
    task_summary     = []

    for task_index in range(40):
        suite_name, task_id = TASK_MAP[task_index]
        max_steps           = MAX_STEPS_MAP[task_index]

        print(f"{'─'*64}")
        print(f"  task {task_index:2d}  {suite_name}/{task_id}")

        env, task_desc, init_states = make_env(suite_name, task_id, SEED)
        print(f"  {task_desc[:70]}")

        collected   = 0   # 이 태스크에서 수집한 성공 수
        attempt     = 0   # 시도 횟수
        task_frames = 0

        pbar = tqdm.tqdm(total=N_SUCCESS, desc=f"  task {task_index:2d}", unit="success")

        while collected < N_SUCCESS and attempt < MAX_ATTEMPTS:
            init_state = init_states[attempt % len(init_states)].copy()

            success, frames = run_episode_collect(
                env=env,
                task_desc=task_desc,
                policy=policy,
                env_pre=env_pre,
                pre=pre,
                post=post,
                init_state=init_state,
                max_steps=max_steps,
                episode_index=global_ep_idx,
                global_frame_start=global_frame_idx,
                task_index=task_index,
            )

            attempt += 1

            if success and len(frames) > 0:
                # parquet 저장
                table     = frames_to_table(frames)
                chunk_idx = global_ep_idx // CHUNK_SIZE
                file_idx  = global_ep_idx % CHUNK_SIZE
                chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
                chunk_dir.mkdir(exist_ok=True)
                pq.write_table(
                    table,
                    chunk_dir / f"file-{file_idx:03d}.parquet",
                    compression="snappy",
                )

                global_frame_idx += len(frames)
                global_ep_idx    += 1
                collected        += 1
                task_frames      += len(frames)
                pbar.update(1)
                pbar.set_postfix(attempt=attempt, frames=len(frames))

        pbar.close()
        env.close()

        status = "✓" if collected >= N_SUCCESS else f"✗ ({collected}/{N_SUCCESS})"
        print(f"  → {status}  {attempt}회 시도  프레임 {task_frames}개")

        task_summary.append({
            "task_index":  task_index,
            "suite":       suite_name,
            "task_id":     task_id,
            "description": task_desc,
            "collected":   collected,
            "attempts":    attempt,
            "frames":      task_frames,
        })
        total_episodes += collected
        total_frames   += task_frames

    # ------------------------------------------------------------------
    # info.json + 요약
    # ------------------------------------------------------------------
    write_info_json(OUTPUT_BASE, total_episodes, total_frames)

    print(f"\n{'='*64}")
    print(f"  수집 완료")
    print(f"  총 에피소드: {total_episodes}  (목표 {40 * N_SUCCESS})")
    print(f"  총 프레임:   {total_frames:,}")
    print(f"  저장 경로:   {OUTPUT_BASE}")
    print(f"{'='*64}")

    # 태스크별 요약 저장
    summary_path = OUTPUT_BASE / "collection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_episodes": total_episodes,
            "total_frames":   total_frames,
            "n_success_target": N_SUCCESS,
            "tasks": task_summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"  요약 저장:   {summary_path}")


if __name__ == "__main__":
    main()
