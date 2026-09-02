#!/usr/bin/env python3
"""Generate review subtitles from Gemini TTS manifests.

The script leaves source videos untouched. It writes sidecar SRT files and,
optionally, separate hard-subtitled MP4 review copies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean_srt_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def write_srt(manifest_path: Path, srt_path: Path, include_speaker: bool) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        text = clean_srt_text(segment.get("text") or "")
        if include_speaker:
            speaker = clean_srt_text(segment.get("character_name") or segment.get("char_id") or "")
            if speaker:
                text = f"{speaker}: {text}"
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{srt_timestamp(start)} --> {srt_timestamp(end)}",
                    text,
                ]
            )
        )
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return len(blocks)


def escape_filter_path(path: Path) -> str:
    # FFmpeg filtergraph escaping: backslash first, then colon and apostrophe.
    text = str(path)
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    return text


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_filter = (
        f"subtitles='{escape_filter_path(srt_path)}':"
        "force_style='FontName=Arial,FontSize=24,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
        "BorderStyle=1,Outline=2,Shadow=0,MarginV=42,Alignment=2'"
    )
    tmp_path = output_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        subtitle_filter,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SRT and review MP4 subtitles for TTS clips.")
    parser.add_argument("--output-root", required=True, help="Video run output root.")
    parser.add_argument(
        "--clips",
        default="clip_001-clip_450",
        help="Clip range or comma-separated ids, e.g. clip_001-clip_450.",
    )
    parser.add_argument(
        "--video-dir-name",
        default="clips_gemini_tts",
        help="Source voiced video directory under renders/.",
    )
    parser.add_argument(
        "--subtitled-dir-name",
        default="clips_gemini_tts_subtitled",
        help="Hard-subtitled output directory under renders/.",
    )
    parser.add_argument(
        "--srt-dir-name",
        default="gemini_tts_srt",
        help="SRT output directory under subtitles/.",
    )
    parser.add_argument(
        "--voice-work-dir-name",
        default="voice_work",
        help="Voice work directory under output root containing per-clip tts_manifest.json files.",
    )
    parser.add_argument("--no-burn", action="store_true", help="Only write SRT files.")
    parser.add_argument("--no-speaker", action="store_true", help="Do not prefix subtitles with speaker names.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing subtitled MP4 files.")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    video_dir = output_root / "renders" / args.video_dir_name
    subtitled_dir = output_root / "renders" / args.subtitled_dir_name
    srt_dir = output_root / "subtitles" / args.srt_dir_name
    clip_ids = parse_clip_range(args.clips)

    done = 0
    missing: list[str] = []
    for clip_id in clip_ids:
        video_path = video_dir / f"{clip_id}.mp4"
        manifest_path = output_root / args.voice_work_dir_name / clip_id / "tts_manifest.json"
        srt_path = srt_dir / f"{clip_id}.srt"
        output_path = subtitled_dir / f"{clip_id}.mp4"
        if not video_path.exists() or not manifest_path.exists():
            missing.append(clip_id)
            continue
        line_count = write_srt(manifest_path, srt_path, include_speaker=not args.no_speaker)
        if not args.no_burn:
            burn_subtitles(video_path, srt_path, output_path, force=args.force)
        done += 1
        print(
            f"✅ {clip_id}: {line_count} subtitle line(s)"
            + ("" if args.no_burn else f" -> {output_path}")
        )

    summary = {
        "clips_requested": len(clip_ids),
        "clips_completed": done,
        "missing": missing,
        "srt_dir": str(srt_dir),
        "subtitled_dir": None if args.no_burn else str(subtitled_dir),
        "source_video_dir": str(video_dir),
        "source_videos_untouched": True,
    }
    manifest_path = output_root / "subtitles" / "subtitle_generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
