#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_finetuned_action_noise.py

Action Noise 증강 데이터로 파인튜닝된 SmolVLA 모델을
4가지 LIBERO 수트에 대해 태스크당 10회씩 평가합니다.

실행 (서버):
    python eval_finetuned_action_noise.py \
      --model_path ./outputs/train/smolvla_action_noise_YYYYMMDD_HHMMSS/checkpoints/030000/pretrained_model

    # 중단 후 재개:
    python eval_finetuned_action_noise.py \
      --model_path ./outputs/train/smolvla_action_noise_YYYYMMDD_HHMMSS/checkpoints/030000/pretrained_model \
      --start_suite spatial --start_task 0 \
      --resume_log ./outputs/eval_action_noise/eval_YYYY-MM-DD_HH-MM-SS.txt
"""

import os
import sys
import argparse

os.environ["MUJOCO_GL"] = "egl"

import json
import logging
import pathlib
import re
import types as _types
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import tqdm

sys.modules["tensorflow"] = None  # type: ignore[assignment]

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline

_fake_tf = _types.ModuleType("tensorflow")
class _FakeTFTensor:
    pass
_fake_tf.Tensor = _FakeTFTensor
_fake_tf.Variable = _FakeTFTensor
sys.modules["tensorflow"] = _fake_tf

import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ===========================================================================
# 상수
# ===========================================================================

ENV_IMG_RES = 256
NUM_WAIT    = 10
SEED        = 7
OUTPUT_DIR  = pathlib.Path("./outputs/eval_action_noise")

ALL_SUITES = [
    ("libero_10",      "LIBERO-10",      "다양한 장기 태스크",          500),
    ("libero_spatial", "LIBERO-Spatial", "물체 간 공간적 관계 이해",    250),
    ("libero_object",  "LIBERO-Object",  "다양한 물체 조작",            320),
    ("libero_goal",    "LIBERO-Goal",    "변화하는 목표 지점 대응",     350),
]

SUITE_KEY_MAP = {
    "10":      "libero_10",
    "spatial": "libero_spatial",
    "object":  "libero_object",
    "goal":    "libero_goal",
}

# ===========================================================================
# 로깅
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _log(msg: str, f=None):
    logger.info(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


# ===========================================================================
# LIBERO 환경 유틸
# ===========================================================================

def _make_env(task, resolution: int, seed: int):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _set_delta(env):
    for robot in env.robots:
        robot.controller.use_delta = True


def _make_obs(obs: dict) -> dict:
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


# ===========================================================================
# 에피소드 실행
# ===========================================================================

def run_episode(
    env,
    task_desc: str,
    policy: SmolVLAPolicy,
    env_pre,
    pre,
    post,
    initial_state,
    max_steps: int,
    log_file=None,
) -> bool:
    env.reset()
    obs = env.set_init_state(initial_state)
    _set_delta(env)
    policy.reset()

    success = False
    t = 0

    try:
        while t < max_steps + NUM_WAIT:
            if t < NUM_WAIT:
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                t += 1
                continue

            batch = preprocess_observation(_make_obs(obs))
            batch["task"] = task_desc
            batch = env_pre(batch)
            batch = pre(batch)

            with torch.inference_mode():
                action = policy.select_action(batch)

            action_np = post(action).squeeze(0).numpy()
            obs, _, done, _ = env.step(action_np.tolist())

            if done:
                success = bool(env.check_success())
                break

            t += 1

    except Exception as e:
        _log(f"    [에피소드 오류] {e}", log_file)

    return success


# ===========================================================================
# 태스크 실행
# ===========================================================================

def run_task(
    task_suite,
    task_id: int,
    trials: int,
    policy,
    env_pre,
    pre,
    post,
    max_steps: int,
    log_file=None,
) -> Dict:
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    env, desc = _make_env(task, ENV_IMG_RES, SEED)

    _log(f"  [{task_id:2d}] {desc}", log_file)

    successes = 0
    for ep in tqdm.tqdm(range(trials), desc=f"    task {task_id:2d}", leave=False):
        init = init_states[ep % len(init_states)]
        ok   = run_episode(env, desc, policy, env_pre, pre, post, init, max_steps, log_file)
        successes += int(ok)

    try:
        env.close()
    except Exception:
        pass

    rate = successes / trials
    _log(f"       → {successes}/{trials}  ({rate:.0%})", log_file)

    return {
        "task_id":      task_id,
        "description":  desc,
        "trials":       trials,
        "successes":    successes,
        "success_rate": rate,
    }


# ===========================================================================
# 수트 실행
# ===========================================================================

def run_suite(
    suite_key: str,
    suite_label: str,
    trials: int,
    max_steps: int,
    policy,
    env_pre,
    pre,
    post,
    log_file=None,
    start_task_id: int = 0,
    prefilled_tasks: Optional[List[Dict]] = None,
) -> Dict:
    bd    = benchmark.get_benchmark_dict()
    suite = bd[suite_key]()

    _log(f"\n{'─' * 60}", log_file)
    _log(f"  {suite_label}  ({suite.n_tasks} tasks × {trials} trials)", log_file)
    if start_task_id > 0:
        _log(f"  [재개] task {start_task_id}부터 시작", log_file)
    _log(f"{'─' * 60}", log_file)

    task_results = list(prefilled_tasks) if prefilled_tasks else []

    for tid in range(start_task_id, suite.n_tasks):
        res = run_task(suite, tid, trials, policy, env_pre, pre, post, max_steps, log_file)
        task_results.append(res)

    task_results.sort(key=lambda r: r["task_id"])

    total_suc  = sum(r["successes"] for r in task_results)
    total_tri  = sum(r["trials"]    for r in task_results)
    suite_rate = total_suc / total_tri if total_tri > 0 else 0.0

    return {
        "suite_key":       suite_key,
        "suite_label":     suite_label,
        "n_tasks":         suite.n_tasks,
        "trials_each":     trials,
        "total_trials":    total_tri,
        "total_successes": total_suc,
        "success_rate":    suite_rate,
        "task_results":    task_results,
    }


# ===========================================================================
# 이전 로그 파싱 (재개용)
# ===========================================================================

def parse_completed_tasks_from_log(log_path: str) -> Dict[str, List[Dict]]:
    suite_map = {
        "LIBERO-10":      "libero_10",
        "LIBERO-Spatial": "libero_spatial",
        "LIBERO-Object":  "libero_object",
        "LIBERO-Goal":    "libero_goal",
    }

    completed: Dict[str, List[Dict]] = {}
    current_suite = None

    task_re   = re.compile(r"\[\s*(\d+)\]\s+(.+)")
    result_re = re.compile(r"→\s+(\d+)/(\d+)")

    try:
        lines = pathlib.Path(log_path).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"[경고] 로그 파싱 실패: {e}")
        return {}

    pending_task = None
    for line in lines:
        for label, key in suite_map.items():
            if label in line and "tasks ×" in line:
                current_suite = key
                if key not in completed:
                    completed[key] = []
                break

        if current_suite is None:
            continue

        m = task_re.search(line)
        if m:
            pending_task = {
                "task_id": int(m.group(1)), "description": m.group(2).strip(),
                "trials": 0, "successes": 0, "success_rate": 0.0,
            }
            continue

        if pending_task is not None:
            m2 = result_re.search(line)
            if m2:
                suc = int(m2.group(1))
                tri = int(m2.group(2))
                pending_task["successes"]    = suc
                pending_task["trials"]       = tri
                pending_task["success_rate"] = suc / tri if tri > 0 else 0.0
                completed[current_suite].append(pending_task)
                pending_task = None

    total = sum(len(v) for v in completed.values())
    print(f"[재개] 이전 로그에서 {total}개 태스크 결과 로드:")
    for k, v in completed.items():
        if v:
            print(f"  {k}: task {[r['task_id'] for r in v]} 완료")
    return completed


# ===========================================================================
# 결과 출력 (표 형식)
# ===========================================================================

def _bar(rate: float, width: int = 20) -> str:
    filled = int(rate * width)
    return "█" * filled + "░" * (width - filled)


def print_results(model_path: str, suite_results: List[Dict], log_file=None):
    SEP  = "═" * 74
    SEP2 = "─" * 74
    n_tasks_total = sum(sr["n_tasks"] for sr in suite_results)

    lines = [
        "",
        SEP,
        f"  {'Action Noise SmolVLA  LIBERO 평가 결과':^68}",
        f"  {'모델: ' + str(model_path):^68}",
        SEP,
    ]

    for sr in suite_results:
        rate = sr["success_rate"]
        bar  = _bar(rate)
        lines += [
            f"\n  ▶ {sr['suite_label']}",
            f"    전체: {sr['total_successes']:3d}/{sr['total_trials']}  "
            f"성공률: {rate:6.1%}  [{bar}]",
            SEP2,
            f"    {'#':>2}  {'태스크 설명':<50}  {'결과':>10}",
            SEP2,
        ]
        for r in sr["task_results"]:
            desc   = r["description"]
            if len(desc) > 50:
                desc = desc[:47] + "..."
            result = f"{r['successes']:2d}/{r['trials']}  {r['success_rate']:5.0%}"
            marker = "✓" if r["success_rate"] >= 0.5 else "✗"
            lines.append(f"  {marker} {r['task_id']:>2}  {desc:<50}  {result:>10}")

    all_suc = sum(sr["total_successes"] for sr in suite_results)
    all_tri = sum(sr["total_trials"]    for sr in suite_results)
    overall = all_suc / all_tri if all_tri > 0 else 0.0

    lines += [
        "",
        SEP,
        f"  {'종합 성능 요약':^68}",
        SEP,
        f"  {'수트':<18}  {'태스크':>5}  {'성공/시도':>10}  {'성공률':>7}  {'바 차트':<22}",
        SEP2,
    ]
    for sr in suite_results:
        rate = sr["success_rate"]
        lines.append(
            f"  {sr['suite_label']:<18}  {sr['n_tasks']:>5}개  "
            f"{sr['total_successes']:>4}/{sr['total_trials']:<5}  "
            f"{rate:>7.1%}  [{_bar(rate, 16)}]"
        )
    lines += [
        SEP2,
        f"  {'전체 (' + str(len(suite_results)) + ' 수트)':<18}  {n_tasks_total:>5}개  "
        f"{all_suc:>4}/{all_tri:<5}  "
        f"{overall:>7.1%}  [{_bar(overall, 16)}]",
        SEP,
        "",
    ]

    for line in lines:
        _log(line, log_file)


# ===========================================================================
# CLI 파싱
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Action Noise 파인튜닝 SmolVLA 모델 LIBERO 평가 스크립트"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="체크포인트 경로 (예: ./outputs/train/smolvla_action_noise_YYYYMMDD/checkpoints/030000/pretrained_model)\n"
             "생략 시 최신 smolvla_action_noise_* 체크포인트를 자동 탐색",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=10,
        help="태스크당 추론 횟수 (기본값: 10)",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        choices=["10", "spatial", "object", "goal"],
        default=["10", "spatial", "object", "goal"],
        help="평가할 수트 목록 (기본값: 10 spatial object goal)",
    )
    p.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="사용할 GPU ID (기본값: 0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"랜덤 시드 (기본값: {SEED})",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"결과 저장 경로 (기본값: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--start_suite",
        type=str,
        default=None,
        choices=["10", "spatial", "object", "goal"],
        help="이 수트부터 재개 (이전 수트는 --resume_log 에서 로드)",
    )
    p.add_argument(
        "--start_task",
        type=int,
        default=0,
        help="start_suite 안에서 이 task_id부터 재개 (기본값: 0)",
    )
    p.add_argument(
        "--resume_log",
        type=str,
        default=None,
        help="이전 실행 로그 파일 경로 — 완료된 태스크 결과를 파싱해 자동으로 채움",
    )
    return p.parse_args()


# ===========================================================================
# 메인
# ===========================================================================

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 모델 경로 결정
    if args.model_path:
        model_path = pathlib.Path(args.model_path)
    else:
        model_path = None

    if model_path is None or not model_path.exists():
        # 최신 smolvla_action_noise_* 체크포인트 자동 탐색
        candidates = sorted(pathlib.Path("./outputs/train").glob(
            "smolvla_action_noise_*/checkpoints/*/pretrained_model"
        ))
        if not candidates:
            # pretrained_model 바로 아래에 있는 경우도 탐색
            candidates = sorted(pathlib.Path("./outputs/train").glob(
                "smolvla_action_noise_*/pretrained_model"
            ))
        if candidates:
            model_path = candidates[-1]
            print(f"[자동 탐색] 체크포인트: {model_path}")
        else:
            print("[오류] smolvla_action_noise_* 체크포인트를 찾을 수 없습니다.")
            print("  --model_path 로 직접 경로를 지정해 주세요.")
            sys.exit(1)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts           = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path     = out_dir / f"eval_{ts}.txt"
    results_path = out_dir / f"eval_{ts}.json"
    log_file     = open(log_path, "w", encoding="utf-8")

    _log("=" * 60, log_file)
    _log("  Action Noise SmolVLA  LIBERO 평가", log_file)
    _log("=" * 60, log_file)
    _log(f"  모델    : {model_path}", log_file)
    _log(f"  수트    : {' / '.join(args.suites)}", log_file)
    _log(f"  시도/태스크: {args.trials}회", log_file)
    _log(f"  시드    : {args.seed}", log_file)

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    _log(f"\n[1/2] 모델 로딩: {model_path}", log_file)
    policy = SmolVLAPolicy.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()
    _log(f"      device={device}  n_action_steps={policy.config.n_action_steps}  "
         f"chunk_size={policy.config.chunk_size}", log_file)

    pre, post = make_pre_post_processors(policy.config, str(model_path))
    env_pre   = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    # ------------------------------------------------------------------
    # 재개 설정
    # ------------------------------------------------------------------
    start_suite_key = SUITE_KEY_MAP.get(args.start_suite) if args.start_suite else None
    prefilled_all: Dict[str, List[Dict]] = {}
    if args.resume_log:
        prefilled_all = parse_completed_tasks_from_log(args.resume_log)

    # ------------------------------------------------------------------
    # 선택된 수트 순차 평가
    # ------------------------------------------------------------------
    _log("\n[2/2] 평가 시작", log_file)
    suite_results = []

    selected = [s for s in ALL_SUITES if s[0].replace("libero_", "") in args.suites
                or ("10" in args.suites and s[0] == "libero_10")]

    suite_order = [s[0] for s in ALL_SUITES]

    for suite_key, suite_label, suite_desc, max_steps in selected:
        # start_suite 이전 수트는 이전 로그 결과로 채우고 스킵
        if start_suite_key and suite_order.index(suite_key) < suite_order.index(start_suite_key):
            prefilled = prefilled_all.get(suite_key, [])
            if prefilled:
                total_suc = sum(r["successes"] for r in prefilled)
                total_tri = sum(r["trials"]    for r in prefilled)
                _log(f"\n  [스킵] {suite_label} — 이전 결과 사용 ({total_suc}/{total_tri})", log_file)
                suite_results.append({
                    "suite_key":       suite_key,
                    "suite_label":     suite_label,
                    "n_tasks":         len(prefilled),
                    "trials_each":     args.trials,
                    "total_trials":    total_tri,
                    "total_successes": total_suc,
                    "success_rate":    total_suc / total_tri if total_tri > 0 else 0.0,
                    "task_results":    prefilled,
                })
            continue

        st        = args.start_task if (suite_key == start_suite_key) else 0
        prefilled = prefilled_all.get(suite_key, []) if st > 0 else []

        _log(f"\n  수트: {suite_label}  —  {suite_desc}", log_file)
        result = run_suite(
            suite_key, suite_label, args.trials, max_steps,
            policy, env_pre, pre, post, log_file,
            start_task_id=st,
            prefilled_tasks=prefilled,
        )
        suite_results.append(result)

    # ------------------------------------------------------------------
    # 결과 출력
    # ------------------------------------------------------------------
    print_results(str(model_path), suite_results, log_file)

    # JSON 저장
    all_suc = sum(sr["total_successes"] for sr in suite_results)
    all_tri = sum(sr["total_trials"]    for sr in suite_results)
    save_data = {
        "meta": {
            "model":       "smolvla_action_noise",
            "model_path":  str(model_path),
            "trials_each": args.trials,
            "seed":        args.seed,
            "timestamp":   datetime.now().isoformat(),
            "suites":      args.suites,
        },
        "overall": {
            "total_trials":    all_tri,
            "total_successes": all_suc,
            "success_rate":    all_suc / all_tri if all_tri > 0 else 0.0,
        },
        "suites": suite_results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    _log(f"\n로그  저장 → {log_path}", log_file)
    _log(f"JSON 저장 → {results_path}", log_file)
    log_file.close()


if __name__ == "__main__":
    main()
