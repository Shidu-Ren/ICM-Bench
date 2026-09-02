#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def sort_key(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return (0, int(match.group(1))) if match else (1, path.name)


def public_clip_id(path: Path, fallback_index: int, prefix: str) -> str:
    match = re.search(r"(\d+)$", path.stem)
    numeric_id = int(match.group(1)) if match else fallback_index
    return f"{prefix}_{numeric_id:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="clip")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "album_manifest.jsonl"

    clips = sorted(source_dir.glob("*.mp4"), key=sort_key)
    if args.limit > 0:
        clips = clips[:args.limit]
    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, clip in enumerate(clips):
            clip_id = public_clip_id(clip, idx, args.prefix)
            item = {
                "video_id": clip_id,
                "video_name": f"{clip_id}.mp4",
                "video_path": str(clip),
                "subtitle": None,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({
        "manifest": str(manifest_path),
        "clips": len(clips),
        "source_dir": str(source_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
