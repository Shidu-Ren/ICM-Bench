#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icm_runtime import activate_upstream  # noqa: E402

activate_upstream()
from utils.data import get_subtitles  # noqa: E402


def clip_num_from_name(name: str) -> int:
    match = re.search(r"(\d+)", Path(name).stem)
    if not match:
        raise ValueError(f"Cannot parse clip number from {name}")
    return int(match.group(1))


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            clip_num = clip_num_from_name(item["video_name"])
            rows[clip_num] = item
    return rows


def subtitles_for(item: dict[str, Any], fps: float) -> list[tuple[int, str]] | None:
    subtitle_path = item.get("subtitle")
    if not subtitle_path:
        return None
    subtitle_file = Path(subtitle_path)
    if not subtitle_file.exists():
        return None
    return get_subtitles(str(subtitle_file), num_frames=0, fps=fps, data=item)


def inject_one(
    pkl_path: Path,
    output_path: Path,
    manifest: dict[int, dict[str, Any]],
    chunk_size: int,
    fps: float,
) -> dict[str, int]:
    clip_num = clip_num_from_name(pkl_path.name)
    saved = pickle.load(pkl_path.open("rb"))
    graph = saved.get("video_graph")
    if graph is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkl_path, output_path)
        return {"files": 1, "nodes": 0, "nodes_with_subtitles": 0, "subtitle_files": 0}

    item = manifest.get(clip_num)
    subtitles = subtitles_for(item, fps) if item else None
    subtitle_files = 1 if subtitles is not None else 0
    nodes = 0
    nodes_with_subtitles = 0

    for node_id, node_data in graph.nodes(data=True):
        nodes += 1
        if subtitles is None:
            node_data["subtitles"] = None
            continue
        start_time = int(node_id) * chunk_size // fps
        end_time = (int(node_id) + 1) * chunk_size // fps
        current = [text for time, text in subtitles if time >= start_time and time < end_time]
        node_data["subtitles"] = current
        if current:
            nodes_with_subtitles += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(saved, f)
    return {
        "files": 1,
        "nodes": nodes,
        "nodes_with_subtitles": nodes_with_subtitles,
        "subtitle_files": subtitle_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-graph-dir", required=True)
    parser.add_argument("--output-graph-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    input_dir = Path(args.input_graph_dir)
    output_dir = Path(args.output_graph_dir)

    totals = {
        "files": 0,
        "nodes": 0,
        "nodes_with_subtitles": 0,
        "subtitle_files": 0,
    }
    for pkl_path in sorted(input_dir.glob("*.pkl"), key=lambda p: clip_num_from_name(p.name)):
        stats = inject_one(
            pkl_path,
            output_dir / pkl_path.name,
            manifest,
            chunk_size=args.chunk_size,
            fps=args.fps,
        )
        for key, value in stats.items():
            totals[key] += value

    print(
        json.dumps(
            {
                "input_graph_dir": str(input_dir),
                "output_graph_dir": str(output_dir),
                "manifest": str(Path(args.manifest)),
                **totals,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
