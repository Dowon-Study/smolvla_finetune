#!/usr/bin/env python3
"""
fix_meta_action_noise.py

augment_action_noise.py 가 생성하지 않은 meta 파일들을 보완합니다.
  - meta/tasks.parquet
  - meta/stats.json
  - meta/episodes/chunk-000/file-000.parquet

실행:
    python fix_meta_action_noise.py
    python fix_meta_action_noise.py --dataset_dir ./outputs/augmented_action_noise
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── task_index(0-39) → 태스크 설명
# augment_action_noise.py 의 TASK_MAP + LIBERO 태스크 이름 기반
TASK_DESCRIPTIONS = {
    0:  "put the white mug on the left plate and put the yellow and white mug on the right plate",
    1:  "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    2:  "put the yellow and white mug in the microwave and close it",
    3:  "turn on the stove and put the moka pot on it",
    4:  "put both the alphabet soup and the cream cheese box in the basket",
    5:  "put both the alphabet soup and the tomato sauce in the basket",
    6:  "put both moka pots on the stove",
    7:  "put both the cream cheese box and the butter in the basket",
    8:  "put the black bowl in the bottom drawer of the cabinet and close it",
    9:  "pick up the book and place it in the back compartment of the caddy",
    10: "put the bowl on the plate",
    11: "put the wine bottle on the rack",
    12: "open the top drawer and put the bowl inside",
    13: "put the cream cheese in the bowl",
    14: "put the wine bottle on top of the cabinet",
    15: "push the plate to the front of the stove",
    16: "turn on the stove",
    17: "put the bowl on the stove",
    18: "put the bowl on top of the cabinet",
    19: "open the middle drawer of the cabinet",
    20: "pick up the orange juice and place it in the basket",
    21: "pick up the ketchup and place it in the basket",
    22: "pick up the cream cheese and place it in the basket",
    23: "pick up the bbq sauce and place it in the basket",
    24: "pick up the alphabet soup and place it in the basket",
    25: "pick up the milk and place it in the basket",
    26: "pick up the salad dressing and place it in the basket",
    27: "pick up the butter and place it in the basket",
    28: "pick up the tomato sauce and place it in the basket",
    29: "pick up the chocolate pudding and place it in the basket",
    30: "pick up the black bowl next to the cookie box and place it on the plate",
    31: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    32: "pick up the black bowl on the ramekin and place it on the plate",
    33: "pick up the black bowl on the stove and place it on the plate",
    34: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    35: "pick up the black bowl on the cookie box and place it on the plate",
    36: "pick up the black bowl next to the plate and place it on the plate",
    37: "pick up the black bowl next to the ramekin and place it on the plate",
    38: "pick up the black bowl from table center and place it on the plate",
    39: "pick up the black bowl on the wooden cabinet and place it on the plate",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./outputs/augmented_action_noise",
    )
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    meta_dir    = dataset_dir / "meta"
    data_dir    = dataset_dir / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # 1. meta/tasks.parquet 생성
    # ──────────────────────────────────────────────────────────
    tasks_path = meta_dir / "tasks.parquet"
    if not tasks_path.exists():
        print("[1/3] meta/tasks.parquet 생성...")
        df_tasks = pd.DataFrame([
            {"__index_level_0__": desc, "task_index": idx}
            for idx, desc in TASK_DESCRIPTIONS.items()
        ])
        df_tasks = df_tasks.set_index("__index_level_0__")
        df_tasks.to_parquet(tasks_path)
        print(f"      저장: {tasks_path}")
    else:
        print("[1/3] meta/tasks.parquet 이미 존재 — 스킵")

    # ──────────────────────────────────────────────────────────
    # 2. 데이터 로딩 (이미지 제외)
    # ──────────────────────────────────────────────────────────
    print("[2/3] 데이터 로딩 중...")
    all_files = sorted(data_dir.rglob("*.parquet"))
    if not all_files:
        print(f"[오류] data parquet 파일이 없습니다: {data_dir}")
        return

    cols = ["episode_index", "index", "task_index",
            "observation.state", "action", "timestamp", "frame_index"]
    dfs = [pq.read_table(f, columns=cols).to_pandas() for f in all_files]
    df  = pd.concat(dfs, ignore_index=True).sort_values("index").reset_index(drop=True)
    print(f"      전체 프레임: {len(df)}")

    # ──────────────────────────────────────────────────────────
    # 3. meta/stats.json 생성
    # ──────────────────────────────────────────────────────────
    stats_path = meta_dir / "stats.json"
    if not stats_path.exists():
        print("      stats.json 계산 중...")

        def compute_stats(values: np.ndarray) -> dict:
            return {
                "min":   values.min(axis=0).tolist(),
                "max":   values.max(axis=0).tolist(),
                "mean":  values.mean(axis=0).tolist(),
                "std":   values.std(axis=0).tolist(),
                "count": [len(values)],
                "q01":   np.quantile(values, 0.01, axis=0).tolist(),
                "q10":   np.quantile(values, 0.10, axis=0).tolist(),
                "q50":   np.quantile(values, 0.50, axis=0).tolist(),
                "q90":   np.quantile(values, 0.90, axis=0).tolist(),
                "q99":   np.quantile(values, 0.99, axis=0).tolist(),
            }

        states  = np.array(df["observation.state"].tolist(), dtype=np.float32)
        actions = np.array(df["action"].tolist(), dtype=np.float32)
        timestamps   = df["timestamp"].values.reshape(-1, 1).astype(np.float32)
        frame_indices= df["frame_index"].values.reshape(-1, 1).astype(np.float32)
        ep_indices   = df["episode_index"].values.reshape(-1, 1).astype(np.float32)
        indices      = df["index"].values.reshape(-1, 1).astype(np.float32)
        task_indices = df["task_index"].values.reshape(-1, 1).astype(np.float32)

        stats = {
            "observation.state": compute_stats(states),
            "action":            compute_stats(actions),
            "timestamp":         compute_stats(timestamps),
            "frame_index":       compute_stats(frame_indices),
            "episode_index":     compute_stats(ep_indices),
            "index":             compute_stats(indices),
            "task_index":        compute_stats(task_indices),
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"      저장: {stats_path}")
    else:
        print("      stats.json 이미 존재 — 스킵")

    # ──────────────────────────────────────────────────────────
    # 4. meta/episodes/ 생성
    # ──────────────────────────────────────────────────────────
    ep_dir = meta_dir / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_out = ep_dir / "file-000.parquet"

    if ep_out.exists():
        print("[3/3] meta/episodes/chunk-000/file-000.parquet 이미 존재 — 스킵")
    else:
        print("[3/3] meta/episodes/ 생성 중...")

        # 파일별 시작 index 맵
        file_map = {}
        for fpath in all_files:
            chunk_idx = int(fpath.parent.name.replace("chunk-", ""))
            file_idx  = int(fpath.stem.replace("file-", ""))
            tbl = pq.read_table(fpath, columns=["index"])
            for idx in tbl["index"].to_pylist():
                file_map[idx] = (chunk_idx, file_idx)

        def safe_stats(arr_list):
            arr = np.array(arr_list, dtype=np.float64)
            return {
                "min":   arr.min(axis=0).tolist(),
                "max":   arr.max(axis=0).tolist(),
                "mean":  arr.mean(axis=0).tolist(),
                "std":   arr.std(axis=0).tolist(),
                "count": [len(arr)],
            }

        # 에피소드별 그룹핑
        grouped = df.groupby("episode_index")
        rows = []
        for ep_idx, grp in grouped:
            grp = grp.sort_values("frame_index")
            from_idx = int(grp["index"].min())
            to_idx   = int(grp["index"].max()) + 1
            length   = len(grp)
            task_idx = int(grp["task_index"].iloc[0])
            task_str = TASK_DESCRIPTIONS.get(task_idx, f"task_{task_idx}")

            data_chunk, data_file = file_map.get(from_idx, (0, 0))

            s_stats  = safe_stats(grp["observation.state"].tolist())
            a_stats  = safe_stats(grp["action"].tolist())
            t_stats  = safe_stats([[t] for t in grp["timestamp"].tolist()])
            fi_stats = safe_stats([[f] for f in grp["frame_index"].tolist()])
            gi_stats = safe_stats([[i] for i in grp["index"].tolist()])
            ei_stats = {"min":[float(ep_idx)],"max":[float(ep_idx)],
                        "mean":[float(ep_idx)],"std":[0.0],"count":[length]}
            ti_stats = {"min":[float(task_idx)],"max":[float(task_idx)],
                        "mean":[float(task_idx)],"std":[0.0],"count":[length]}

            img_min  = [[[0.0]],[[0.0]],[[0.0]]]
            img_max  = [[[1.0]],[[1.0]],[[1.0]]]
            img_mean = [[[0.3]],[[0.3]],[[0.3]]]
            img_std  = [[[0.15]],[[0.15]],[[0.15]]]
            img_cnt  = [length]

            rows.append({
                "episode_index":                         int(ep_idx),
                "data/chunk_index":                      data_chunk,
                "data/file_index":                       data_file,
                "dataset_from_index":                    from_idx,
                "dataset_to_index":                      to_idx,
                "tasks":                                 [task_str],
                "length":                                length,
                "stats/observation.images.image/min":    img_min,
                "stats/observation.images.image/max":    img_max,
                "stats/observation.images.image/mean":   img_mean,
                "stats/observation.images.image/std":    img_std,
                "stats/observation.images.image/count":  img_cnt,
                "stats/observation.images.image2/min":   img_min,
                "stats/observation.images.image2/max":   img_max,
                "stats/observation.images.image2/mean":  img_mean,
                "stats/observation.images.image2/std":   img_std,
                "stats/observation.images.image2/count": img_cnt,
                "stats/observation.state/min":           s_stats["min"],
                "stats/observation.state/max":           s_stats["max"],
                "stats/observation.state/mean":          s_stats["mean"],
                "stats/observation.state/std":           s_stats["std"],
                "stats/observation.state/count":         s_stats["count"],
                "stats/action/min":                      a_stats["min"],
                "stats/action/max":                      a_stats["max"],
                "stats/action/mean":                     a_stats["mean"],
                "stats/action/std":                      a_stats["std"],
                "stats/action/count":                    a_stats["count"],
                "stats/timestamp/min":                   t_stats["min"],
                "stats/timestamp/max":                   t_stats["max"],
                "stats/timestamp/mean":                  t_stats["mean"],
                "stats/timestamp/std":                   t_stats["std"],
                "stats/timestamp/count":                 t_stats["count"],
                "stats/frame_index/min":                 fi_stats["min"],
                "stats/frame_index/max":                 fi_stats["max"],
                "stats/frame_index/mean":                fi_stats["mean"],
                "stats/frame_index/std":                 fi_stats["std"],
                "stats/frame_index/count":               fi_stats["count"],
                "stats/episode_index/min":               ei_stats["min"],
                "stats/episode_index/max":               ei_stats["max"],
                "stats/episode_index/mean":              ei_stats["mean"],
                "stats/episode_index/std":               ei_stats["std"],
                "stats/episode_index/count":             ei_stats["count"],
                "stats/index/min":                       gi_stats["min"],
                "stats/index/max":                       gi_stats["max"],
                "stats/index/mean":                      gi_stats["mean"],
                "stats/index/std":                       gi_stats["std"],
                "stats/index/count":                     gi_stats["count"],
                "stats/task_index/min":                  ti_stats["min"],
                "stats/task_index/max":                  ti_stats["max"],
                "stats/task_index/mean":                 ti_stats["mean"],
                "stats/task_index/std":                  ti_stats["std"],
                "stats/task_index/count":                ti_stats["count"],
                "meta/episodes/chunk_index":             0,
                "meta/episodes/file_index":              0,
            })

        # 참조 스키마 (fix_meta_episodes.py 가 생성한 파일)
        ref_candidates = [
            "./outputs/augmented_latent_recovery/meta/episodes/chunk-000/file-000.parquet",
            "./outputs/augmented_data/noise_0_0500/meta/episodes/chunk-000/file-000.parquet",
        ]
        ref_schema = None
        for rc in ref_candidates:
            rp = pathlib.Path(rc)
            if rp.exists():
                ref_schema = pq.read_table(rp).schema
                print(f"      참조 스키마: {rp}")
                break

        tbl = pa.Table.from_pylist(rows, schema=ref_schema)
        pq.write_table(tbl, ep_out)
        print(f"      저장: {ep_out}  ({len(rows)} 에피소드)")

    print("\n완료!")
    print(f"  이제 파인튜닝 스크립트를 실행하세요:")
    print(f"  bash run_finetune_action_noise_server.sh")


if __name__ == "__main__":
    main()
