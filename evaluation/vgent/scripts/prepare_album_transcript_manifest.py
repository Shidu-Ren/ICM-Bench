#!/usr/bin/env python3
"""Attach released ICM-Bench SRT files to a Vgent clip manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def subtitle_for(video_name: str, subtitle_dir: Path) -> Path | None:
    stem = Path(video_name).stem
    candidates = (
        subtitle_dir / f"{stem}.srt",
        subtitle_dir / f"{stem.removeprefix('clip_')}.srt",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach released SRT transcripts to an ICM-Bench Vgent manifest."
    )
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--subtitle-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_manifest = Path(args.base_manifest).expanduser().resolve()
    subtitle_dir = Path(args.subtitle_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    subtitle_count = 0
    with base_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            subtitle_path = subtitle_for(str(item["video_name"]), subtitle_dir)
            item["subtitle"] = str(subtitle_path) if subtitle_path else None
            subtitle_count += int(subtitle_path is not None)
            rows.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "subtitle_dir": str(subtitle_dir),
                "rows": len(rows),
                "subtitle_files": subtitle_count,
                "missing_subtitles": len(rows) - subtitle_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
