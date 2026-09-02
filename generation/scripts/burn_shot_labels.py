#!/usr/bin/env python3
"""Burn per-shot labels into review copies of voiced/subtitled clips.

Source videos are left untouched. Labels are derived from 06_shot_plan.json and
shown only during the matching shot time range.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_clip_range(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    if "-" in value:
        start, end = value.split("-", 1)
        start_num = int(start.replace("clip_", ""))
        end_num = int(end.replace("clip_", ""))
        return [f"clip_{idx:03d}" for idx in range(start_num, end_num + 1)]
    return [value]


def ffmpeg_escape_text(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace(",", "\\,")
    )


def ffmpeg_escape_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_filter(clip: dict, font_path: Path | None) -> str:
    filters: list[str] = []
    cursor = 0.0
    shots = clip.get("shots") or []
    total = len(shots)
    font_option = f"fontfile='{ffmpeg_escape_path(font_path)}':" if font_path else ""
    for index, shot in enumerate(shots, start=1):
        duration = float(shot.get("duration_seconds") or 0)
        start = cursor
        end = cursor + duration
        cursor = end
        label = f"{shot['id']}  shot {index}/{total}"
        enable = f"between(t\\,{start:.3f}\\,{max(start, end - 0.001):.3f})"
        filters.append(
            "drawbox="
            f"x=18:y=18:w=470:h=48:color=black@0.62:t=fill:enable='{enable}'"
        )
        filters.append(
            "drawtext="
            f"{font_option}"
            f"text='{ffmpeg_escape_text(label)}':"
            "x=34:y=30:fontsize=26:fontcolor=white:"
            "borderw=2:bordercolor=black@0.85:"
            f"enable='{enable}'"
        )
    return ",".join(filters) if filters else "null"


def burn_one(
    clip: dict,
    source_dir: Path,
    output_dir: Path,
    font_path: Path | None,
    force: bool,
) -> dict:
    clip_id = clip["id"]
    source_path = source_dir / f"{clip_id}.mp4"
    output_path = output_dir / f"{clip_id}.mp4"
    if not source_path.exists():
        return {"clip_id": clip_id, "status": "missing_source", "source_path": str(source_path)}
    if output_path.exists() and not force:
        return {"clip_id": clip_id, "status": "skipped_existing", "output_path": str(output_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")
    tmp_path.unlink(missing_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-vf",
        build_filter(clip, font_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    tmp_path.replace(output_path)
    return {
        "clip_id": clip_id,
        "status": "done",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "shot_count": len(clip.get("shots") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Burn shot ids into review video copies.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clips", default="clip_001-clip_450")
    parser.add_argument("--source-dir-name", default="clips_gemini_tts_subtitled")
    parser.add_argument("--output-dir-name", default="clips_gemini_tts_subtitled_shot_labeled")
    parser.add_argument("--font-path", help="Optional font file; FFmpeg's default font is used when omitted.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    font_path = Path(args.font_path).expanduser().resolve() if args.font_path else None
    source_dir = output_root / "renders" / args.source_dir_name
    output_dir = output_root / "renders" / args.output_dir_name
    clip_ids = parse_clip_range(args.clips)
    series = json.loads((output_root / "metadata" / "06_shot_plan.json").read_text(encoding="utf-8"))
    clips_by_id = {clip["id"]: clip for clip in series.get("clips", [])}
    selected = [clips_by_id[clip_id] for clip_id in clip_ids if clip_id in clips_by_id]
    missing_metadata = [clip_id for clip_id in clip_ids if clip_id not in clips_by_id]
    if missing_metadata:
        print(f"⚠️ missing metadata: {missing_metadata[:10]}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(burn_one, clip, source_dir, output_dir, font_path, args.force)
            for clip in selected
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']}: {result['clip_id']}")

    summary = {
        "clips_requested": len(clip_ids),
        "clips_completed": sum(1 for item in results if item["status"] in {"done", "skipped_existing"}),
        "clips_rendered": sum(1 for item in results if item["status"] == "done"),
        "missing_metadata": missing_metadata,
        "missing_source": [item["clip_id"] for item in results if item["status"] == "missing_source"],
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_videos_untouched": True,
    }
    manifest_path = output_root / "review" / "shot_label_burn_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["missing_source"] or missing_metadata:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
