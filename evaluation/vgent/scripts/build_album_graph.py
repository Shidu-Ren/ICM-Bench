#!/usr/bin/env python3
"""Construct Vgent graphs for the public ICM-Bench clip manifest."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))

from icm_runtime import activate_upstream, load_vgent_class  # noqa: E402

activate_upstream()
from utils.data import get_subtitles  # noqa: E402

Vgent = load_vgent_class()


def distributed_context() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    return rank, world_size


def read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--n_retrieval", type=int, default=20)
    parser.add_argument("--n_refine", type=int, default=5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--total_pixels", type=int, default=16384)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.task = "album"

    rank, world_size = distributed_context()
    rows = read_manifest(Path(args.manifest_path).expanduser().resolve())
    rows = rows[rank::world_size]
    graph_dir = (
        Path(args.graph_path).expanduser().resolve()
        / f"album_{args.fps}fps_{args.chunk_size}"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    vgent = Vgent(args)

    for row in tqdm(rows, desc=f"rank {rank} graph construction"):
        video_path = Path(row["video_path"]).expanduser().resolve()
        output = graph_dir / f"{Path(row['video_name']).stem}.pkl"
        if output.exists():
            continue
        try:
            raw_video, _, _, _, fps, video_inputs, _ = vgent.load_video(
                str(video_path), args
            )
            if not isinstance(video_inputs, list):
                video_inputs = [video_inputs]
            subtitles = get_subtitles(
                row.get("subtitle"),
                len(video_inputs[0]),
                fps=args.fps,
                data=row,
            )
            video_graph, entity_graph = vgent.construct_graph(video_inputs, subtitles)
            with output.open("wb") as handle:
                pickle.dump(
                    {"video_graph": video_graph, "entity_graph": entity_graph},
                    handle,
                )
        except Exception as exc:
            if args.fail_fast:
                raise
            print(
                f"[build_album_graph] failed {video_path.name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
