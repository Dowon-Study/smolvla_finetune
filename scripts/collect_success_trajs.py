#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_success_trajs_multigpu.py

GPU 4장을 사용해 성공 궤적을 병렬 수집합니다.

GPU 분배:
  GPU 0 → tasks  0- 9  (libero_spatial)    episodes   0- 99
  GPU 1 → tasks 10-19  (libero_object)     episodes 100-199
  GPU 2 → tasks 20-29  (libero_goal)       episodes 200-299
  GPU 3 → tasks 30-39  (libero_10)         episodes 300-399

실행:
  /media/idna/Data/envs/smolvla/bin/python collect_success_trajs_multigpu.py
"""

# ──────────────────────────────────────────────────────────────
#  NOTE: 멀티프로세싱 spawn 방식을 사용하므로 CUDA 초기화 전에
#  모든 설정이 이루어집니다.  모듈 레벨에서는 torch 를 import
#  하지 않습니다.
# ──────────────────────────────────────────────────────────────

import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from typing import List, Tuple

# ============================================================
# 공통 설정 (워커에서도 참조)
# ============================================================

MODEL_ID     = "HuggingFaceVLA/smolvla_libero"
OUTPUT_BASE  = pathlib.Path("./outputs/collected_trajs")
N_SUCCESS    = 10
MAX_ATTEMPTS = 100
NUM_WAIT     = 10
ENV_RES      = 256
CHUNK_SIZE   = 1000
SEED         = 7

# task_index → (suite_name, task_id, max_steps)
TASK_MAP      = {}
MAX_STEPS_MAP = {}
for _i in range(10):
    TASK_MAP[_i]      = ("libero_spatial", _i);  MAX_STEPS_MAP[_i]      = 250
    TASK_MAP[_i + 10] = ("libero_object",  _i);  MAX_STEPS_MAP[_i + 10] = 320
    TASK_MAP[_i + 20] = ("libero_goal",    _i);  MAX_STEPS_MAP[_i + 20] = 350
    TASK_MAP[_i + 30] = ("libero_10",      _i);  MAX_STEPS_MAP[_i + 30] = 500

# GPU별 태스크 범위 (task_indices, ep_offset)
#   ep_offset: 에피소드 index 충돌 방지를 위해 GPU별 시작점 분리
GPU_TASK_GROUPS: List[Tuple[List[int], int]] = [
    (list(range(0,  10)), 0),    # GPU 0
    (list(range(10, 20)), 100),  # GPU 1
    (list(range(20, 30)), 200),  # GPU 2
    (list(range(30, 40)), 300),  # GPU 3
]


# ============================================================
# 워커 함수
# ============================================================

def _worker(gpu_id: int, task_indices: List[int], ep_offset: int,
            data_dir: str, log_dir: str, result_path: str):
    """각 GPU에서 독립적으로 실행되는 수집 워커."""

    # ── CUDA 장치 고정 (spawn 이후, torch 초기화 전에 설정)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"]            = "egl"

    # ── TF stub
    import types as _types
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

    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import tqdm
    from scipy.spatial.transform import Rotation

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # ── 로그 파일
    log_path = pathlib.Path(log_dir) / f"worker_{gpu_id}.log"
    log_f    = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}][GPU{gpu_id}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    # ── PyArrow 스키마
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

    # ── 유틸 함수들
    def rgb_to_png_bytes(arr):
        _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return buf.tobytes()

    def obs_to_state8(obs):
        eef_pos   = obs["robot0_eef_pos"].astype(np.float32)
        quat_wxyz = obs["robot0_eef_quat"].astype(np.float64)
        rotvec    = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2],
                                        quat_wxyz[3], quat_wxyz[0]]).as_rotvec().astype(np.float32)
        gripper   = obs["robot0_gripper_qpos"].astype(np.float32)
        return np.concatenate([eef_pos, rotvec, gripper])

    def make_env(suite_name, task_id):
        bd    = benchmark.get_benchmark_dict()
        suite = bd[suite_name]()
        task  = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        bddl  = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env   = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=ENV_RES,
            camera_widths=ENV_RES,
        )
        env.seed(SEED)
        return env, task.language, inits

    def set_delta(env):
        for robot in env.robots:
            robot.controller.use_delta = True

    def make_obs_batch(obs):
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

    def run_episode(env, task_desc, policy, env_pre, pre, post,
                    init_state, max_steps, ep_idx, frame_start, task_idx):
        env.reset()
        obs = env.set_init_state(init_state)
        set_delta(env)
        policy.reset()

        for _ in range(NUM_WAIT):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

        frames  = []
        success = False
        DT      = 1.0 / 20

        try:
            for t in range(max_steps):
                batch = preprocess_observation(make_obs_batch(obs))
                batch["task"] = task_desc
                batch = env_pre(batch)
                batch = pre(batch)

                with torch.inference_mode():
                    action_tensor = policy.select_action(batch)

                action_np = post(action_tensor).squeeze(0).cpu().numpy()

                frames.append({
                    "img_bytes":     rgb_to_png_bytes(obs["agentview_image"]),
                    "img2_bytes":    rgb_to_png_bytes(obs["robot0_eye_in_hand_image"]),
                    "state":         obs_to_state8(obs).tolist(),
                    "action":        action_np.tolist(),
                    "timestamp":     float(t * DT),
                    "frame_index":   t,
                    "episode_index": ep_idx,
                    "index":         frame_start + t,
                    "task_index":    task_idx,
                })

                obs, _, done, _ = env.step(action_np.tolist())

                if done:
                    success = bool(env.check_success())
                    break
        except Exception as e:
            log(f"  에피소드 오류: {e}")

        return success, frames

    def frames_to_table(frames):
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

    # ── 모델 로드
    log(f"모델 로딩: {MODEL_ID}")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES=gpu_id 이므로 항상 :0
    policy = policy.to(device).eval()
    log(f"  device={device}  chunk_size={policy.config.chunk_size}")

    pre, post = make_pre_post_processors(policy.config, MODEL_ID)
    env_pre   = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    # ── 수집 루프
    data_path        = pathlib.Path(data_dir)
    global_ep_idx    = ep_offset
    global_frame_idx = ep_offset * 500   # 충분한 offset (최대 max_steps=500)
    task_summaries   = []
    total_episodes   = 0
    total_frames     = 0

    for task_index in task_indices:
        suite_name, task_id = TASK_MAP[task_index]
        max_steps           = MAX_STEPS_MAP[task_index]

        log(f"── task {task_index:2d}  {suite_name}/{task_id}")
        env, task_desc, init_states = make_env(suite_name, task_id)
        log(f"  {task_desc[:70]}")

        collected   = 0
        attempt     = 0
        task_frames = 0

        pbar = tqdm.tqdm(
            total=N_SUCCESS,
            desc=f"GPU{gpu_id} task{task_index:2d}",
            unit="ep",
            position=gpu_id,
            leave=True,
        )

        while collected < N_SUCCESS and attempt < MAX_ATTEMPTS:
            init_state = init_states[attempt % len(init_states)].copy()

            success, frames = run_episode(
                env, task_desc, policy, env_pre, pre, post,
                init_state, max_steps,
                ep_idx=global_ep_idx,
                frame_start=global_frame_idx,
                task_idx=task_index,
            )
            attempt += 1

            if success and frames:
                chunk_idx = global_ep_idx // CHUNK_SIZE
                file_idx  = global_ep_idx % CHUNK_SIZE
                chunk_dir = data_path / f"chunk-{chunk_idx:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)

                table = frames_to_table(frames)
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
        log(f"  → {status}  {attempt}회 시도  ep {global_ep_idx - collected}~{global_ep_idx - 1}  프레임 {task_frames}개")

        task_summaries.append({
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

    log_f.close()

    # ── 결과를 파일에 저장 (큐 대신 파일로 공유)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu_id":         gpu_id,
            "total_episodes": total_episodes,
            "total_frames":   total_frames,
            "tasks":          task_summaries,
        }, f, indent=2, ensure_ascii=False)


# ============================================================
# 메인
# ============================================================

def main():
    mp.set_start_method("spawn", force=True)

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU: {n_gpus}장")
    if n_gpus < 4:
        print(f"  경고: GPU가 {n_gpus}장만 감지되었습니다. "
              f"가용 GPU 범위로 자동 조정합니다.")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    data_dir = OUTPUT_BASE / "data"
    data_dir.mkdir(exist_ok=True)
    log_dir  = OUTPUT_BASE / "worker_logs"
    log_dir.mkdir(exist_ok=True)

    n_workers = min(n_gpus, len(GPU_TASK_GROUPS))
    print(f"워커 수: {n_workers}  (GPU 1장당 태스크 10개 병렬 처리)\n")

    result_paths = []
    processes    = []

    for gpu_id in range(n_workers):
        task_indices, ep_offset = GPU_TASK_GROUPS[gpu_id]
        result_path = str(OUTPUT_BASE / f"result_gpu{gpu_id}.json")
        result_paths.append(result_path)

        p = mp.Process(
            target=_worker,
            args=(gpu_id, task_indices, ep_offset,
                  str(data_dir), str(log_dir), result_path),
            name=f"worker-gpu{gpu_id}",
        )
        p.start()
        processes.append(p)
        print(f"  워커 시작: GPU {gpu_id}  tasks {task_indices[0]}~{task_indices[-1]}")

    print(f"\n{'─'*64}")
    print(f"  모든 워커 실행 중... (완료까지 기다립니다)")
    print(f"{'─'*64}\n")

    for p in processes:
        p.join()

    # ──────────────────────────────────────────────────────────
    # 결과 병합 및 info.json 생성
    # ──────────────────────────────────────────────────────────
    all_tasks      = []
    total_episodes = 0
    total_frames   = 0

    for rp in result_paths:
        if not pathlib.Path(rp).exists():
            print(f"  경고: 결과 파일 없음 → {rp}")
            continue
        with open(rp, encoding="utf-8") as f:
            data = json.load(f)
        total_episodes += data["total_episodes"]
        total_frames   += data["total_frames"]
        all_tasks.extend(data["tasks"])

    # info.json
    num_chunks = (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE
    info = {
        "codebase_version":    "v3.0",
        "data_path":           "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "fps":                 20,
        "splits":              {"train": f"0:{total_episodes}"},
        "total_episodes":      total_episodes,
        "total_frames":        total_frames,
        "total_tasks":         40,
        "total_videos":        0,
        "total_chunks":        num_chunks,
        "chunks_size":         CHUNK_SIZE,
        "repo_id":             "smolvla_libero_success_trajs",
        "model_id":            MODEL_ID,
        "n_success_per_task":  N_SUCCESS,
        "features": {
            "observation.images.image":  {"dtype": "image",   "shape": [ENV_RES, ENV_RES, 3]},
            "observation.images.image2": {"dtype": "image",   "shape": [ENV_RES, ENV_RES, 3]},
            "observation.state":         {"dtype": "float32", "shape": [8]},
            "action":                    {"dtype": "float32", "shape": [7]},
            "timestamp":                 {"dtype": "float32", "shape": [1]},
            "frame_index":               {"dtype": "int64",   "shape": [1]},
            "episode_index":             {"dtype": "int64",   "shape": [1]},
            "index":                     {"dtype": "int64",   "shape": [1]},
            "task_index":                {"dtype": "int64",   "shape": [1]},
        },
    }
    with open(OUTPUT_BASE / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # collection_summary.json
    with open(OUTPUT_BASE / "collection_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_episodes":    total_episodes,
            "total_frames":      total_frames,
            "n_success_target":  N_SUCCESS,
            "tasks":             all_tasks,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*64}")
    print(f"  수집 완료")
    print(f"  총 에피소드: {total_episodes}  (목표 {40 * N_SUCCESS})")
    print(f"  총 프레임:   {total_frames:,}")
    print(f"  저장 경로:   {OUTPUT_BASE}")
    print(f"{'='*64}")

    # 워커별 임시 결과 파일 정리
    for rp in result_paths:
        try: pathlib.Path(rp).unlink()
        except Exception: pass


if __name__ == "__main__":
    main()
