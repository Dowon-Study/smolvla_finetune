#!/usr/bin/env python3
"""
fix_meta_episodes.py

augment_latent_recovery.py 가 생성하지 않은 meta/episodes/ 디렉토리를
기존 data/ parquet 파일에서 재구성합니다.

실행:
    python fix_meta_episodes.py
    python fix_meta_episodes.py --dataset_dir ./outputs/augmented_latent_recovery
"""

import argparse
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./outputs/augmented_latent_recovery",
    )
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    data_dir    = dataset_dir / "data"
    meta_dir    = dataset_dir / "meta"
    ep_dir      = meta_dir / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # ── tasks.parquet 로드 (task_index → description 매핑)
    tasks_tbl  = pq.read_table(meta_dir / "tasks.parquet")
    task_dict  = dict(zip(
        tasks_tbl["task_index"].to_pylist(),
        tasks_tbl["__index_level_0__"].to_pylist(),
    ))

    # ── 모든 data parquet 읽기
    print("데이터 파일 로딩 중...")
    all_files = sorted(data_dir.rglob("*.parquet"))
    tables = [pq.read_table(f) for f in all_files]
    full_table = pa.concat_tables(tables)

    # index 기준으로 정렬 (이미지 바이너리 제외하고 정렬 후 재결합)
    import pandas as pd
    df = full_table.select(["episode_index", "index", "task_index",
                             "observation.state", "action", "timestamp",
                             "frame_index"]).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)

    print(f"  전체 프레임: {len(df)}")

    # ── 에피소드별 그룹핑
    ep_indices   = df["episode_index"].tolist()
    indices      = df["index"].tolist()
    task_indices = df["task_index"].tolist()
    states       = df["observation.state"].tolist()
    actions      = df["action"].tolist()
    timestamps   = df["timestamp"].tolist()
    frame_indices= df["frame_index"].tolist()

    # 에피소드 경계 계산
    episodes = {}
    for i, ep_idx in enumerate(ep_indices):
        if ep_idx not in episodes:
            episodes[ep_idx] = {
                "from": i, "to": i,
                "task_index": task_indices[i],
                "states": [], "actions": [], "timestamps": [], "frame_idxs": [],
            }
        episodes[ep_idx]["to"] = i
        episodes[ep_idx]["states"].append(states[i])
        episodes[ep_idx]["actions"].append(actions[i])
        episodes[ep_idx]["timestamps"].append(timestamps[i])
        episodes[ep_idx]["frame_idxs"].append(frame_indices[i])

    print(f"  에피소드 수: {len(episodes)}")

    # ── data chunk/file 인덱스 매핑 (frame index → 파일 위치)
    # 각 파일의 시작 frame index 기록
    file_map = {}  # global_index → (chunk_idx, file_idx)
    for fpath in all_files:
        parts = fpath.parts
        chunk_idx = int(fpath.parent.name.replace("chunk-", ""))
        file_idx  = int(fpath.stem.replace("file-", ""))
        tbl = pq.read_table(fpath, columns=["index"])
        for idx in tbl["index"].to_pylist():
            file_map[idx] = (chunk_idx, file_idx)

    # ── 에피소드별 stats 계산
    def safe_stats(arr_list):
        arr = np.array(arr_list, dtype=np.float64)
        return {
            "min":   arr.min(axis=0).tolist(),
            "max":   arr.max(axis=0).tolist(),
            "mean":  arr.mean(axis=0).tolist(),
            "std":   arr.std(axis=0).tolist(),
            "count": [len(arr)],
        }

    # ── 에피소드 rows 구성
    rows = []
    for ep_idx in sorted(episodes.keys()):
        ep = episodes[ep_idx]
        from_idx = indices[ep["from"]]
        to_idx   = indices[ep["to"]] + 1
        length   = ep["to"] - ep["from"] + 1
        task_idx = ep["task_index"]
        task_str = task_dict.get(task_idx, f"task_{task_idx}")

        data_chunk, data_file = file_map.get(from_idx, (0, 0))

        s_stats = safe_stats(ep["states"])
        a_stats = safe_stats(ep["actions"])
        t_stats = safe_stats([[t] for t in ep["timestamps"]])
        fi_stats= safe_stats([[f] for f in ep["frame_idxs"]])
        ei_stats= {"min":[float(ep_idx)],"max":[float(ep_idx)],
                   "mean":[float(ep_idx)],"std":[0.0],"count":[length]}
        gi_stats= safe_stats([[float(indices[ep["from"]+k])] for k in range(length)])

        # image stats (더미값 — use_imagenet_stats=false 이므로 학습에 영향 없음)
        img_min  = [[[0.0]],[[0.0]],[[0.0]]]
        img_max  = [[[1.0]],[[1.0]],[[1.0]]]
        img_mean = [[[0.3]],[[0.3]],[[0.3]]]
        img_std  = [[[0.15]],[[0.15]],[[0.15]]]
        img_cnt  = [length]

        rows.append({
            "episode_index":                         ep_idx,
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
            "stats/task_index/min":                  [float(task_idx)],
            "stats/task_index/max":                  [float(task_idx)],
            "stats/task_index/mean":                 [float(task_idx)],
            "stats/task_index/std":                  [0.0],
            "stats/task_index/count":                [length],
            "meta/episodes/chunk_index":             0,
            "meta/episodes/file_index":              0,
        })

    # ── 참조 스키마 로드 (기존 동작 데이터셋에서)
    ref_path = pathlib.Path(
        "./outputs/augmented_data/noise_0_0500/meta/episodes/chunk-000/file-000.parquet"
    )
    if ref_path.exists():
        ref_schema = pq.read_table(ref_path).schema
        print(f"  참조 스키마 사용: {ref_path}")
    else:
        ref_schema = None
        print("  참조 스키마 없음 — 자동 추론")

    # ── PyArrow 테이블 생성 & 저장
    tbl = pa.Table.from_pylist(rows, schema=ref_schema)
    out_path = ep_dir / "file-000.parquet"
    pq.write_table(tbl, out_path)
    print(f"\n  저장 완료: {out_path}")
    print(f"  에피소드 수: {len(rows)}")


if __name__ == "__main__":
    main()
