#!/usr/bin/env python3
"""
convert_hdf5_to_lerobot.py
==========================
qpos_finetunning.py 출력 HDF5 → LeRobot v3 데이터셋 포맷 변환

실행:
    python scripts/convert_hdf5_to_lerobot.py \
        --hdf5 data/finetune_object_all.hdf5 \
        --output_repo local/smolvla_finetune_object \
        --root data/lerobot_datasets
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",        default="data/finetune_object_all.hdf5")
    p.add_argument("--output_repo", default="local/smolvla_finetune_object")
    p.add_argument("--root",        default="data/lerobot_datasets",
                   help="로컬 저장 경로 (HF_LEROBOT_HOME 대신 사용)")
    p.add_argument("--fps",         type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    hdf5_path  = Path(args.hdf5)
    output_root = Path(args.root)
    output_root.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "grip_l", "grip_r"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    }

    print(f"입력 HDF5  : {hdf5_path}")
    print(f"출력 레포  : {args.output_repo}  (root={output_root})")

    dataset = LeRobotDataset.create(
        repo_id=args.output_repo,
        fps=args.fps,
        features=features,
        root=output_root,
        robot_type="panda",
        use_videos=True,
        image_writer_threads=4,
    )

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys())
        print(f"에피소드 수: {len(demos)}")

        for ep_idx, dname in enumerate(demos):
            grp = f["data"][dname]
            T   = len(grp["actions"])
            task_desc = grp.attrs.get(
                "task_desc",
                "pick up the object and place it in the basket"
            )

            for t in range(T):
                state = np.concatenate([
                    grp["obs/ee_pos"][t].astype(np.float32),
                    grp["obs/ee_ori"][t].astype(np.float32),
                    grp["obs/gripper_states"][t].astype(np.float32),
                ])

                dataset.add_frame({
                    "observation.images.image":  grp["obs/agentview_rgb"][t],
                    "observation.images.image2": grp["obs/eye_in_hand_rgb"][t],
                    "observation.state": state,
                    "action": grp["actions"][t].astype(np.float32),
                    "task": task_desc,
                })

            dataset.save_episode()

            if (ep_idx + 1) % 50 == 0 or ep_idx == len(demos) - 1:
                print(f"  [{ep_idx+1}/{len(demos)}] {dname} ({T} steps)  "
                      f"task: {task_desc[:50]}")

    dataset.consolidate()
    print(f"\n완료: {output_root / args.output_repo}")


if __name__ == "__main__":
    main()
