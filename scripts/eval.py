#!/usr/bin/env python3
"""
eval.py — 파인튜닝 vs Pretrained SmolVLA 비교 평가 (GUI 지원)
==============================================================
단일 모델 평가:
    MUJOCO_GL=egl python scripts/eval.py --render
    MUJOCO_GL=egl python scripts/eval.py --task_id 0 --n_episodes 5 --render

파인튜닝 vs Pretrained 분할화면 비교:
    MUJOCO_GL=egl python scripts/eval.py --compare --render
    MUJOCO_GL=egl python scripts/eval.py --compare --render --task_id 2 --n_episodes 5
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ACTION_LABELS = ["x", "y", "z", "rx", "ry", "rz", "gripper"]

# ── 공통 색상 ──
C_GREEN  = (60,  220,  60)
C_RED    = (60,   60, 200)
C_CYAN   = (0,   210, 210)
C_WHITE  = (220, 220, 220)
C_DARK   = (22,   22,  22)
C_FT     = (0,   200, 100)   # 파인튜닝: 초록-민트
C_PT     = (30,  120, 255)   # Pretrained: 파랑


# ══════════════════════════════════════════════════════════════════════════════
# 인자
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="outputs/smolvla_ft_object/best_checkpoint/policy_migrated",
                   help="파인튜닝 모델 경로")
    p.add_argument("--pretrained", default="HuggingFaceVLA/smolvla_libero",
                   help="Pretrained 모델 HF hub ID 또는 로컬 경로")
    p.add_argument("--suite",      default="libero_object",
                   choices=["libero_spatial","libero_object","libero_goal","libero_10","libero_90"])
    p.add_argument("--task_id",    type=int,   default=0)
    p.add_argument("--n_episodes", type=int,   default=10)
    p.add_argument("--max_steps",  type=int,   default=280)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--obs_width",  type=int,   default=256)
    p.add_argument("--obs_height", type=int,   default=256)
    p.add_argument("--render",      action="store_true",
                   help="GUI 실시간 시각화")
    p.add_argument("--compare",     action="store_true",
                   help="파인튜닝 vs Pretrained 분할화면 비교 모드")
    p.add_argument("--qpos_noise",  type=float, default=0.0,
                   help="초기 joint 각도 노이즈 ±rad (기본 0=비활성, 권장 0.03)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 모델 / 환경
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_id_or_path: str, device: str, label: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    path = Path(model_id_or_path)
    mid  = str(path) if path.exists() else model_id_or_path
    src  = f"로컬: {mid}" if path.exists() else f"HF Hub: {mid}"
    print(f"  [{label}] {src}")

    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(mid).to(device)
    policy.eval()
    print(f"  [{label}] 로드 완료 ({time.time()-t0:.1f}s)")

    pre, post = make_pre_post_processors(
        policy.config, mid,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def make_env(suite_name: str, task_id: int, obs_w: int, obs_h: int):
    from lerobot.envs.libero import LiberoEnv, _get_suite
    suite = _get_suite(suite_name)
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        observation_width=obs_w, observation_height=obs_h,
        init_states=True,
    )
    return env


# ══════════════════════════════════════════════════════════════════════════════
# obs → 텐서 (원본 smolvla_libero_eval.py 와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    x, y, z, w = quat
    w = np.clip(w, -1.0, 1.0)
    theta = 2.0 * np.arccos(np.abs(w))
    sh = np.sqrt(max(0.0, 1.0 - w * w))
    if sh < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (np.array([x, y, z], dtype=np.float32) / sh * theta).astype(np.float32)


def obs_to_tensors(obs: dict, device: str) -> dict:
    tensors = {}
    for cam_key, img in obs.get("pixels", {}).items():
        t = torch.from_numpy(img.copy()).float() / 255.0
        t = torch.flip(t.permute(2, 0, 1).unsqueeze(0), dims=[2, 3])
        tensors[f"observation.images.{cam_key}"] = t.to(device)
    rs = obs.get("robot_state", {})
    if rs:
        eef_pos   = rs["eef"]["pos"].astype(np.float32)
        eef_quat  = rs["eef"]["quat"].astype(np.float32)
        grip_qpos = rs["gripper"]["qpos"].astype(np.float32)
        state = np.concatenate([eef_pos, quat_to_axisangle(eef_quat), grip_qpos])
        tensors["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    return tensors


# ══════════════════════════════════════════════════════════════════════════════
# 프레임 렌더링
# ══════════════════════════════════════════════════════════════════════════════

def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    r = h / max(img.shape[0], 1)
    return cv2.resize(img, (max(1, int(img.shape[1] * r)), h))


def draw_action_bars(action: np.ndarray, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    n, bar_h = len(action), 34
    gap = max(2, (height - n * bar_h - 36) // (n + 1))

    cv2.putText(panel, "Action", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_CYAN, 1, cv2.LINE_AA)

    for i, (lbl, val) in enumerate(zip(ACTION_LABELS, action)):
        y0  = 36 + gap + i * (bar_h + gap)
        y1  = y0 + bar_h
        mid = 55 + (width - 70) // 2

        cv2.rectangle(panel, (55, y0), (width - 10, y1), (50, 50, 50), -1)
        cv2.line(panel, (mid, y0 - 2), (mid, y1 + 2), (100, 100, 100), 1)

        bar   = int(min(abs(val), 1.0) * (width - 70) / 2)
        color = C_GREEN if val >= 0 else C_RED
        if val >= 0:
            cv2.rectangle(panel, (mid,       y0+4), (mid+bar,  y1-4), color, -1)
        else:
            cv2.rectangle(panel, (mid-bar,   y0+4), (mid,      y1-4), color, -1)

        cv2.putText(panel, lbl,              (4,         y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{val:+.2f}",   (width - 52, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_CYAN,  1, cv2.LINE_AA)
    return panel


def make_panel(obs: dict, action: np.ndarray,
               step: int, max_steps: int,
               ep: int, n_ep: int,
               label: str, label_color,
               results: list,
               success: bool,
               active: bool,
               obs_h: int, obs_w: int,
               panel_w: int = 780, panel_h: int = 512) -> np.ndarray:
    """
    한 모델의 패널:
      [AgentView cam] [Wrist cam (상단 절반)] [Action bars]
    active=False일 때는 어둡게 처리 + "WAITING..." 표시
    """
    img_a = obs["pixels"].get("image",  np.zeros((obs_h, obs_w, 3), dtype=np.uint8))
    img_w = obs["pixels"].get("image2", np.zeros((obs_h, obs_w, 3), dtype=np.uint8))

    # 카메라 영역 너비 계산 (action bar 고정 140px)
    act_w    = 140
    wrist_w  = panel_h // 2             # 정사각형 손목 카메라
    cam_w    = panel_w - wrist_w - act_w

    cam_bgr   = _resize_h(cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR), panel_h)
    cam_bgr   = cv2.resize(cam_bgr, (cam_w, panel_h))

    wrist_bgr = _resize_h(cv2.cvtColor(img_w, cv2.COLOR_RGB2BGR), panel_h // 2)
    wrist_bgr = cv2.resize(wrist_bgr, (wrist_w, panel_h // 2))
    wrist_pad = np.full((panel_h - wrist_bgr.shape[0], wrist_w, 3), 20, dtype=np.uint8)
    wrist_col = np.vstack([wrist_bgr, wrist_pad])

    act_panel = draw_action_bars(action, act_w, panel_h)

    # ── 카메라에 오버레이 ──────────────────────────────────────────
    bw = 4
    border = label_color if active else (80, 80, 80)
    cam_bgr[:bw,:]=border; cam_bgr[-bw:,:]=border
    cam_bgr[:,:bw]=border; cam_bgr[:,-bw:]=border

    # 상단 배너
    overlay = cam_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 44), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, cam_bgr, 0.4, 0, cam_bgr)

    # 모델 레이블
    cv2.putText(cam_bgr, label, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, label_color, 2, cv2.LINE_AA)

    # 스텝 / 에피소드
    if active:
        pct      = step / max(max_steps, 1)
        bar_len  = cam_w - 16
        cv2.rectangle(cam_bgr, (8, 34), (8 + bar_len, 40), (60, 60, 60), -1)
        cv2.rectangle(cam_bgr, (8, 34), (8 + int(bar_len * pct), 40), label_color, -1)
    cv2.putText(cam_bgr, f"Ep {ep+1}/{n_ep}", (cam_w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)

    # 성공
    if success:
        cv2.putText(cam_bgr, "SUCCESS!", (8, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 128), 2, cv2.LINE_AA)

    if not active:
        dark = np.full_like(cam_bgr, 0)
        cv2.addWeighted(dark, 0.45, cam_bgr, 0.55, 0, cam_bgr)
        cv2.putText(cam_bgr, "WAITING...", (cam_w//2 - 70, panel_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2, cv2.LINE_AA)

    # 성공률 히스토리 (하단)
    n_done = len(results)
    if n_done > 0:
        sr = sum(results) / n_done * 100
        hist = "".join("▮" if s else "▯" for s in results[-12:])
        cv2.putText(cam_bgr, f"SR {sr:5.1f}%  {hist}", (6, panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)

    # 손목 카메라 레이블
    cv2.putText(wrist_col, "Wrist", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1, cv2.LINE_AA)

    return np.hstack([cam_bgr, wrist_col, act_panel])


def make_single_frame(obs: dict, action: np.ndarray,
                      step: int, max_steps: int,
                      ep: int, n_ep: int,
                      results: list, success: bool,
                      obs_h: int, obs_w: int) -> np.ndarray:
    """단일 모델 전체 창 프레임."""
    return make_panel(obs, action, step, max_steps, ep, n_ep,
                      "SmolVLA (Finetuned)", C_FT, results, success,
                      active=True, obs_h=obs_h, obs_w=obs_w,
                      panel_w=1280, panel_h=512)


def make_compare_frame(
    obs_ft, action_ft, step_ft, success_ft, results_ft, active_ft,
    obs_pt, action_pt, step_pt, success_pt, results_pt, active_pt,
    ep: int, n_ep: int, max_steps: int,
    obs_h: int, obs_w: int,
) -> np.ndarray:
    """분할화면: 왼쪽=파인튜닝, 오른쪽=Pretrained."""
    PANEL_W, PANEL_H = 760, 512

    left  = make_panel(obs_ft, action_ft, step_ft, max_steps, ep, n_ep,
                       "Finetuned", C_FT, results_ft, success_ft,
                       active_ft, obs_h, obs_w, PANEL_W, PANEL_H)
    right = make_panel(obs_pt, action_pt, step_pt, max_steps, ep, n_ep,
                       "Pretrained", C_PT, results_pt, success_pt,
                       active_pt, obs_h, obs_w, PANEL_W, PANEL_H)

    divider = np.full((PANEL_H, 4, 3), 80, dtype=np.uint8)
    return np.hstack([left, divider, right])


# ══════════════════════════════════════════════════════════════════════════════
# 에피소드 실행
# ══════════════════════════════════════════════════════════════════════════════

def apply_qpos_noise(env, noise_std: float):
    """env.reset() 직후 로봇 arm joint 7개에 가우시안 노이즈 주입."""
    if noise_std <= 0:
        return
    sim = env._env.sim
    noise = np.random.normal(0, noise_std, size=7)
    sim.data.qpos[:7] += noise
    sim.forward()


def _step_model(obs, policy, preprocess, postprocess, task_desc, device):
    obs_t = obs_to_tensors(obs, device)
    obs_t["task"] = task_desc
    proc  = preprocess(obs_t)
    with torch.no_grad():
        act_t = policy.select_action(proc)
    act_t = postprocess(act_t)
    return act_t.squeeze(0).cpu().numpy().astype(np.float32)


def run_episode_single(env, policy, pre, post, task_desc, args,
                       ep_idx: int, win: str | None,
                       results: list) -> dict:
    """단일 모델 에피소드."""
    obs, _ = env.reset()
    apply_qpos_noise(env, args.qpos_noise)
    if hasattr(policy, "reset"):
        policy.reset()

    total_reward, success, steps = 0.0, False, 0
    action = np.zeros(7, dtype=np.float32)

    for step in range(args.max_steps):
        action = _step_model(obs, policy, pre, post, task_desc, args.device)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("is_success"):
            success = True

        if win:
            frame = make_single_frame(obs, action, step+1, args.max_steps,
                                      ep_idx, args.n_episodes,
                                      results, success,
                                      args.obs_height, args.obs_width)
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                return {"success": success, "reward": total_reward,
                        "steps": steps, "quit": True}

        if success:
            if win:
                cv2.waitKey(600)
            break
        if term or trunc:
            break

    return {"success": success, "reward": total_reward, "steps": steps, "quit": False}


def run_episode_compare(
    env_ft, policy_ft, pre_ft, post_ft,
    env_pt, policy_pt, pre_pt, post_pt,
    task_desc, args, ep_idx: int, win: str | None,
    results_ft: list, results_pt: list,
) -> tuple[dict, dict]:
    """
    파인튜닝 에피소드 → Pretrained 에피소드 순서로 실행.
    GUI 있을 때 한쪽이 실행 중이면 반대쪽은 마지막 obs/action 프리즈.
    """
    BLANK_OBS = {
        "pixels": {
            "image":  np.zeros((args.obs_height, args.obs_width, 3), np.uint8),
            "image2": np.zeros((args.obs_height, args.obs_width, 3), np.uint8),
        },
        "robot_state": {},
    }
    BLANK_ACT = np.zeros(7, dtype=np.float32)

    def run_one(env, policy, pre, post, active_side,
                frozen_obs, frozen_act, frozen_step, frozen_success,
                res_active, res_frozen):
        obs, _ = env.reset()
        apply_qpos_noise(env, args.qpos_noise)
        if hasattr(policy, "reset"):
            policy.reset()
        total_reward, success, steps = 0.0, False, 0
        action = BLANK_ACT.copy()

        for step in range(args.max_steps):
            action = _step_model(obs, policy, pre, post, task_desc, args.device)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            if info.get("is_success"):
                success = True

            if win:
                if active_side == "ft":
                    frame = make_compare_frame(
                        obs,         action,      step+1, success,      res_active,  True,
                        frozen_obs,  frozen_act,  frozen_step, frozen_success, res_frozen, False,
                        ep_idx, args.n_episodes, args.max_steps,
                        args.obs_height, args.obs_width,
                    )
                else:
                    frame = make_compare_frame(
                        frozen_obs,  frozen_act,  frozen_step, frozen_success, res_frozen, False,
                        obs,         action,      step+1, success,      res_active,  True,
                        ep_idx, args.n_episodes, args.max_steps,
                        args.obs_height, args.obs_width,
                    )
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    return {"success": success, "reward": total_reward,
                            "steps": steps, "quit": True}, obs, action, step+1

            if success:
                if win:
                    cv2.waitKey(600)
                break
            if term or trunc:
                break

        return ({"success": success, "reward": total_reward, "steps": steps, "quit": False},
                obs, action, steps)

    # ── 1. 파인튜닝 실행 ────────────────────────────────────────────────────────
    print(f"  [ep {ep_idx+1}] Finetuned 실행 중...", end=" ", flush=True)
    r_ft, last_obs_ft, last_act_ft, last_step_ft = run_one(
        env_ft, policy_ft, pre_ft, post_ft,
        "ft", BLANK_OBS, BLANK_ACT, 0, False,
        results_ft, results_pt,
    )
    ft_success = r_ft["success"]
    if r_ft.get("quit"):
        return r_ft, {"success": False, "reward": 0, "steps": 0, "quit": True}

    ft_status = "SUCCESS" if ft_success else "FAIL"
    print(f"{ft_status}")

    # ── 2. Pretrained 실행 (파인튜닝 마지막 프레임 왼쪽 고정) ──────────────────
    print(f"  [ep {ep_idx+1}] Pretrained 실행 중...", end=" ", flush=True)
    frozen_results_ft = results_ft + [int(ft_success)]   # 이번 ep 결과 포함
    r_pt, _, _, _ = run_one(
        env_pt, policy_pt, pre_pt, post_pt,
        "pt", last_obs_ft, last_act_ft, last_step_ft, ft_success,
        results_pt, frozen_results_ft,
    )
    pt_status = "SUCCESS" if r_pt["success"] else "FAIL"
    print(f"{pt_status}")

    return r_ft, r_pt


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(label, results):
    n = len(results)
    if n == 0:
        return
    n_ok = sum(r["success"] for r in results)
    avg_r = np.mean([r["reward"] for r in results])
    avg_s = np.mean([r["steps"]  for r in results])
    print(f"  {label:<20}  {n_ok:>2}/{n} ({100*n_ok/n:5.1f}%)  "
          f"avg_reward={avg_r:.3f}  avg_steps={avg_s:.1f}")


def main():
    args = parse_args()

    print("=" * 70)
    print("  SmolVLA LIBERO 평가")
    print(f"  finetuned : {args.model}")
    print(f"  pretrained: {args.pretrained}")
    print(f"  suite={args.suite}  task_id={args.task_id}  "
          f"episodes={args.n_episodes}  max_steps={args.max_steps}")
    print(f"  mode={'compare' if args.compare else 'single'}  "
          f"GUI={'ON (q=종료)' if args.render else 'OFF'}")
    print("=" * 70)

    WIN = None
    if args.render:
        WIN = "SmolVLA LIBERO"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        if args.compare:
            cv2.resizeWindow(WIN, 1524, 512)
        else:
            cv2.resizeWindow(WIN, 1280, 512)

    # ── 비교 모드 ────────────────────────────────────────────────────────────
    if args.compare:
        print("\n[모델 로드]")
        ft_path = Path(args.model)
        ft_id   = str(ft_path) if ft_path.exists() else args.pretrained   # fallback
        if not ft_path.exists():
            print(f"  [경고] 파인튜닝 모델({args.model}) 없음 → pretrained로 대체")

        policy_ft, pre_ft, post_ft = load_model(ft_id,         args.device, "Finetuned")
        policy_pt, pre_pt, post_pt = load_model(args.pretrained, args.device, "Pretrained")

        print("\n[환경 생성]")
        env_ft = make_env(args.suite, args.task_id, args.obs_width, args.obs_height)
        env_pt = make_env(args.suite, args.task_id, args.obs_width, args.obs_height)
        task_desc = env_ft.task_description
        print(f"  task: {task_desc}")

        results_ft, results_pt = [], []
        sr_ft_list, sr_pt_list = [], []

        print(f"\n{'─'*70}")
        for ep in range(args.n_episodes):
            r_ft, r_pt = run_episode_compare(
                env_ft, policy_ft, pre_ft, post_ft,
                env_pt, policy_pt, pre_pt, post_pt,
                task_desc, args, ep, WIN,
                [int(r["success"]) for r in results_ft],
                [int(r["success"]) for r in results_pt],
            )
            results_ft.append(r_ft)
            results_pt.append(r_pt)

            n_done = ep + 1
            sr_ft  = sum(r["success"] for r in results_ft) / n_done * 100
            sr_pt  = sum(r["success"] for r in results_pt) / n_done * 100
            sr_ft_list.append(sr_ft); sr_pt_list.append(sr_pt)

            diff = sr_ft - sr_pt
            arrow = ("▲" if diff > 0 else "▼") if abs(diff) > 0.01 else "="
            print(f"  ep {ep+1:3d}  FT={'O' if r_ft['success'] else 'X'}({sr_ft:5.1f}%)  "
                  f"PT={'O' if r_pt['success'] else 'X'}({sr_pt:5.1f}%)  "
                  f"diff={arrow}{abs(diff):.1f}%p")

            if r_ft.get("quit") or r_pt.get("quit"):
                break

        env_ft.close(); env_pt.close()

        # 최종 비교 화면 (5초 표시)
        if WIN and results_ft and results_pt:
            _show_final_compare(WIN, results_ft, results_pt, task_desc)

        print(f"\n{'='*70}")
        print("  최종 결과 비교")
        print(f"  task: {task_desc}")
        print(f"{'─'*70}")
        print_summary("Finetuned", results_ft)
        print_summary("Pretrained", results_pt)
        n_ft = sum(r["success"] for r in results_ft)
        n_pt = sum(r["success"] for r in results_pt)
        n    = len(results_ft)
        diff = (n_ft - n_pt) / n * 100
        sign = "+" if diff >= 0 else ""
        print(f"{'─'*70}")
        print(f"  파인튜닝 개선: {sign}{diff:.1f}%p  ({n_ft} vs {n_pt} 성공/{n} 에피소드)")
        print(f"{'='*70}")

    # ── 단일 모델 모드 ────────────────────────────────────────────────────────
    else:
        print("\n[모델 로드]")
        ft_path = Path(args.model)
        ft_id   = str(ft_path) if ft_path.exists() else args.pretrained
        if not ft_path.exists():
            print(f"  [경고] 파인튜닝 모델({args.model}) 없음 → pretrained로 대체")

        policy, pre, post = load_model(ft_id, args.device, "SmolVLA")

        print("\n[환경 생성]")
        env = make_env(args.suite, args.task_id, args.obs_width, args.obs_height)
        task_desc = env.task_description
        print(f"  task: {task_desc}")

        results = []
        print(f"\n{'─'*70}")
        for ep in range(args.n_episodes):
            print(f"  ep {ep+1:3d}/{args.n_episodes} 시작...", end=" ", flush=True)
            t0 = time.time()
            r  = run_episode_single(env, policy, pre, post, task_desc, args,
                                    ep, WIN,
                                    [int(x["success"]) for x in results])
            r["wall_time"] = time.time() - t0
            results.append(r)

            sr = sum(x["success"] for x in results) / len(results) * 100
            print(f"{'SUCCESS' if r['success'] else 'FAIL'}  "
                  f"reward={r['reward']:.2f}  steps={r['steps']}  "
                  f"time={r['wall_time']:.1f}s  sr={sr:.1f}%")

            if r.get("quit"):
                break

        env.close()

        n = len(results); n_ok = sum(r["success"] for r in results)
        print(f"\n{'='*70}")
        print(f"  결과: {n_ok}/{n} 성공 ({100*n_ok/n:.1f}%)")
        print(f"  task: {task_desc}")
        print(f"{'='*70}")

    if WIN:
        cv2.destroyAllWindows()


def _show_final_compare(win, results_ft, results_pt, task_desc):
    """비교 최종 결과 화면을 5초간 표시."""
    n    = len(results_ft)
    n_ft = sum(r["success"] for r in results_ft)
    n_pt = sum(r["success"] for r in results_pt)
    sr_ft = n_ft / n * 100
    sr_pt = n_pt / n * 100

    H, W = 512, 1524
    img = np.full((H, W, 3), 20, dtype=np.uint8)

    # 배경 분할선
    cv2.line(img, (W//2, 0), (W//2, H), (80, 80, 80), 2)

    # 파인튜닝 결과 (왼쪽)
    cv2.putText(img, "Finetuned", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_FT, 3, cv2.LINE_AA)
    cv2.putText(img, f"{sr_ft:.1f}%", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, C_FT, 6, cv2.LINE_AA)
    cv2.putText(img, f"{n_ft} / {n} episodes", (40, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_WHITE, 2, cv2.LINE_AA)
    hist_ft = "".join("▮" if r["success"] else "▯" for r in results_ft)
    cv2.putText(img, hist_ft, (40, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_FT, 2, cv2.LINE_AA)

    # Pretrained 결과 (오른쪽)
    cv2.putText(img, "Pretrained", (W//2 + 40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_PT, 3, cv2.LINE_AA)
    cv2.putText(img, f"{sr_pt:.1f}%", (W//2 + 40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, C_PT, 6, cv2.LINE_AA)
    cv2.putText(img, f"{n_pt} / {n} episodes", (W//2 + 40, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_WHITE, 2, cv2.LINE_AA)
    hist_pt = "".join("▮" if r["success"] else "▯" for r in results_pt)
    cv2.putText(img, hist_pt, (W//2 + 40, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_PT, 2, cv2.LINE_AA)

    # 중앙 하단: diff
    diff   = sr_ft - sr_pt
    sign   = "+" if diff >= 0 else ""
    color  = C_FT if diff >= 0 else C_PT
    label  = "Finetuned wins" if diff > 0 else ("Pretrained wins" if diff < 0 else "Tie")
    cv2.putText(img, f"{label}  {sign}{diff:.1f}%p", (W//2 - 220, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # task 설명 (상단 중앙)
    cv2.putText(img, task_desc[:60], (W//2 - len(task_desc)*5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow(win, img)
    cv2.waitKey(5000)


if __name__ == "__main__":
    main()
