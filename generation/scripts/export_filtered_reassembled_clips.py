#!/usr/bin/env python3
"""Export filtered full clips by removing excluded shots and renumbering.

The source clips are left untouched. For each source clip, shots marked "pass"
in the review manifest are cut from the TTS full clip and concatenated into a
new full clip. Source clips with no passing shots are omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KeptShot:
    original_clip_id: str
    original_shot_id: str
    original_shot_index: int | None
    new_clip_id: str
    new_shot_id: str
    new_shot_index: int
    start: float
    end: float
    duration: float
    scene_id: str | None
    visible_characters: list[str]
    focus_characters: list[str]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=650)
    parser.add_argument("--source-dir-name", default="clips_gemini_tts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_kept_shots(clip: dict[str, Any], review: dict[str, Any], new_clip_id: str) -> list[KeptShot]:
    cursor = 0.0
    kept: list[KeptShot] = []
    new_index = 0
    for shot in clip.get("shots", []):
        duration = float(shot.get("duration_seconds") or 0)
        start = cursor
        end = cursor + duration
        cursor = end
        status = (review.get("shots", {}).get(shot["id"]) or {}).get("status")
        if status != "pass":
            continue
        new_index += 1
        kept.append(
            KeptShot(
                original_clip_id=clip["id"],
                original_shot_id=shot["id"],
                original_shot_index=shot.get("shot_index"),
                new_clip_id=new_clip_id,
                new_shot_id=f"{new_clip_id}_shot_{new_index:02d}",
                new_shot_index=new_index,
                start=start,
                end=end,
                duration=duration,
                scene_id=shot.get("scene_id"),
                visible_characters=list(shot.get("visible_characters") or []),
                focus_characters=list(shot.get("focus_characters") or []),
            )
        )
    return kept


def render_filtered_clip(
    source_video: Path,
    output_video: Path,
    kept: list[KeptShot],
    force: bool,
) -> dict[str, Any]:
    if output_video.exists() and output_video.stat().st_size > 0 and not force:
        return {"new_clip_id": kept[0].new_clip_id, "status": "skipped_existing"}
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_video.with_suffix(".tmp.mp4")
    tmp.unlink(missing_ok=True)

    filters: list[str] = []
    concat_inputs: list[str] = []
    for index, shot in enumerate(kept):
        filters.append(
            f"[0:v]trim=start={shot.start:.3f}:end={shot.end:.3f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[0:a]atrim=start={shot.start:.3f}:end={shot.end:.3f},"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filters.append("".join(concat_inputs) + f"concat=n={len(kept)}:v=1:a=1[outv][outa]")

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_video),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(output_video)
    return {"new_clip_id": kept[0].new_clip_id, "status": "done", "output": str(output_video)}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    export_root = Path(args.export_root).expanduser().resolve()
    source_dir = output_root / "renders" / args.source_dir_name
    export_video_dir = export_root / "renders" / "clips"
    metadata_dir = export_root / "metadata"
    review_dir = export_root / "review"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    plan = load_json(output_root / "metadata" / "06_shot_plan.json")
    source_series_path = output_root / "metadata" / "07_series_bible.json"
    if source_series_path.exists():
        source_series = load_json(source_series_path)
        if not plan.get("cast") and source_series.get("cast"):
            plan["cast"] = source_series["cast"]
        if not plan.get("scenes") and source_series.get("scenes"):
            plan["scenes"] = source_series["scenes"]
    review = load_json(output_root / "review" / "shot_review_manifest.json")
    clips_by_id = {clip["id"]: clip for clip in plan.get("clips", [])}

    new_clips: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    jobs: list[tuple[Path, Path, list[KeptShot]]] = []
    excluded_source_clips: list[str] = []

    new_clip_counter = 0
    for clip_num in range(args.start, args.end + 1):
        original_clip_id = f"clip_{clip_num:03d}"
        clip = clips_by_id.get(original_clip_id)
        if not clip:
            continue
        preview_new_clip_id = f"clip_{new_clip_counter + 1:03d}"
        kept = build_kept_shots(clip, review, preview_new_clip_id)
        if not kept:
            excluded_source_clips.append(original_clip_id)
            continue
        new_clip_counter += 1
        new_clip_id = f"clip_{new_clip_counter:03d}"
        if kept[0].new_clip_id != new_clip_id:
            kept = build_kept_shots(clip, review, new_clip_id)

        source_video = source_dir / f"{original_clip_id}.mp4"
        if not source_video.exists():
            raise FileNotFoundError(source_video)
        output_video = export_video_dir / f"{new_clip_id}.mp4"
        jobs.append((source_video, output_video, kept))

        new_clip = json.loads(json.dumps(clip))
        new_clip["id"] = new_clip_id
        new_clip["original_clip_id"] = original_clip_id
        new_clip["source_clip_id"] = original_clip_id
        new_clip["target_runtime_seconds"] = int(round(sum(shot.duration for shot in kept)))
        new_shots = []
        original_shots_by_id = {shot["id"]: shot for shot in clip.get("shots", [])}
        for shot in kept:
            new_shot = json.loads(json.dumps(original_shots_by_id[shot.original_shot_id]))
            new_shot["id"] = shot.new_shot_id
            new_shot["shot_index"] = shot.new_shot_index
            new_shot["original_shot_id"] = shot.original_shot_id
            new_shot["source_shot_id"] = shot.original_shot_id
            new_shots.append(new_shot)
            mappings.append(
                {
                    "new_clip_id": new_clip_id,
                    "new_shot_id": shot.new_shot_id,
                    "new_shot_index": shot.new_shot_index,
                    "original_clip_id": original_clip_id,
                    "original_shot_id": shot.original_shot_id,
                    "original_shot_index": shot.original_shot_index,
                    "start_seconds_in_source": round(shot.start, 3),
                    "end_seconds_in_source": round(shot.end, 3),
                    "duration_seconds": round(shot.duration, 3),
                    "source_full_tts_video": str(source_video),
                    "filtered_full_video": str(output_video),
                    "scene_id": shot.scene_id,
                    "visible_characters": shot.visible_characters,
                    "focus_characters": shot.focus_characters,
                }
            )
        new_clip["shots"] = new_shots
        if len(new_shots) != len(clip.get("shots", [])):
            new_clip["logline"] = (
                f"{clip.get('title', original_clip_id)}: filtered and reassembled "
                "from passing shots only."
            )
            new_clip["memory_facts"] = []
            new_clip["relationship_facts"] = []
            new_clip["continuity_hooks"] = []
        new_clips.append(new_clip)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(render_filtered_clip, source, output, kept, args.force)
            for source, output, kept in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            print(f"{result['status']}: {result['new_clip_id']}")

    filtered_plan = json.loads(json.dumps(plan))
    filtered_plan["clips"] = new_clips
    write_json(metadata_dir / "06_shot_plan.json", filtered_plan)
    write_json(metadata_dir / "01_series_bible.json", filtered_plan)
    write_json(metadata_dir / "07_series_bible.json", filtered_plan)

    for filename in ["00_video_config_snapshot.json", "02_asset_manifest.json", "03_anchor_manifest.json"]:
        source = output_root / "metadata" / filename
        if source.exists():
            target = metadata_dir / filename
            target.write_bytes(source.read_bytes())

    manifest = {
        "source_output_root": str(output_root),
        "export_root": str(export_root),
        "source_clip_range": f"clip_{args.start:03d}-clip_{args.end:03d}",
        "source_video_dir": str(source_dir),
        "filtered_video_dir": str(export_video_dir),
        "renumbering_policy": (
            "Source clips with at least one pass shot are kept in chronological order and "
            "renumbered as clip_001...; pass shots inside each new clip are renumbered as "
            "clip_XXX_shot_01..."
        ),
        "included_clip_count": len(new_clips),
        "included_shot_count": len(mappings),
        "excluded_source_clip_count": len(excluded_source_clips),
        "excluded_source_clips": excluded_source_clips,
        "included_runtime_seconds": round(sum(item["duration_seconds"] for item in mappings), 3),
        "included_runtime_minutes": round(sum(item["duration_seconds"] for item in mappings) / 60, 2),
        "mappings": mappings,
    }
    write_json(review_dir / "filtered_reassembled_manifest.json", manifest)
    (review_dir / "source_shot_review_manifest.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (review_dir / "clip_shot_mapping.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "new_clip_id",
            "new_shot_id",
            "new_shot_index",
            "original_clip_id",
            "original_shot_id",
            "original_shot_index",
            "start_seconds_in_source",
            "end_seconds_in_source",
            "duration_seconds",
            "source_full_tts_video",
            "filtered_full_video",
            "scene_id",
            "visible_characters",
            "focus_characters",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in mappings:
            row = dict(item)
            row["visible_characters"] = ";".join(row.get("visible_characters") or [])
            row["focus_characters"] = ";".join(row.get("focus_characters") or [])
            writer.writerow(row)

    print(json.dumps({k: v for k, v in manifest.items() if k != "mappings"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
