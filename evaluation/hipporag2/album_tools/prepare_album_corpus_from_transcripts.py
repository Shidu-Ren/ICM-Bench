#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "icm_bench_asr_corpus"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clip_number(video_id: str) -> int:
    if str(video_id).strip() == "clip_000":
        return 0
    match = re.search(r"(\d+)$", str(video_id))
    if not match:
        raise ValueError(f"Cannot parse numeric clip id from {video_id!r}")
    return int(match.group(1))


def strip_srt(text: str, strip_speaker_labels: bool) -> str:
    lines: list[str] = []
    speaker_names = ("Li Ming", "Wang Lin", "Zhang Hua", "Li Jian", "Chen Tao", "Sarah Wu")
    speaker_prefix = re.compile(r"(?:^|\s)(?:" + "|".join(re.escape(name) for name in speaker_names) + r"):\s*")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        if strip_speaker_labels:
            line = speaker_prefix.sub(" ", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def load_transcript(row: dict[str, Any], strip_speaker_labels: bool) -> str:
    subtitle = str(row.get("subtitle") or "").strip()
    if not subtitle:
        return ""
    path = Path(subtitle)
    if not path.is_file():
        return ""
    return strip_srt(path.read_text(encoding="utf-8", errors="ignore"), strip_speaker_labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a HippoRAG2 corpus from transcript/ASR only.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--source-label", default="speakerless ASR transcript")
    parser.add_argument("--title-suffix", default="ASR transcript")
    parser.add_argument("--keep-speaker-labels", action="store_true")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    rows.sort(key=lambda row: clip_number(str(row.get("video_id") or "")))
    if args.limit:
        rows = rows[: args.limit]

    docs: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    empty_count = 0
    for row in rows:
        video_id = str(row.get("video_id") or "")
        number = clip_number(video_id)
        transcript = load_transcript(row, strip_speaker_labels=not args.keep_speaker_labels)
        if not transcript:
            empty_count += 1
            if not args.include_empty:
                continue

        title = f"{video_id}: {args.title_suffix}"
        text_parts = [
            f"Clip id: {video_id}.",
            f"Numeric memory order: {number}.",
            f"Source: {args.source_label}.",
            "Speaker labels: retained." if args.keep_speaker_labels else "Speaker labels: stripped.",
            f"Transcript: {transcript or '[empty]'}",
        ]
        # Public ICM-Bench clip ids are zero-based (clip_000--clip_838), so the
        # corpus index can use the public numeric id directly without remapping.
        idx = number
        docs.append({"title": title, "text": "\n".join(text_parts), "idx": idx})
        manifest_rows.append(
            {
                "idx": idx,
                "clip_id": video_id,
                "m3_clip_number": number,
                "video_path": row.get("video_path"),
                "subtitle": row.get("subtitle"),
                "title": title,
                "has_transcript": bool(transcript),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / f"{args.dataset_name}_corpus.json"
    manifest_path = output_dir / f"{args.dataset_name}_manifest.json"
    manifest = {
        "dataset_name": args.dataset_name,
        "source_manifest": str(Path(args.manifest)),
        "source_label": args.source_label,
        "source_clip_count": len(rows),
        "doc_count": len(docs),
        "empty_transcript_count": empty_count,
        "include_empty": args.include_empty,
        "speaker_labels": "retained" if args.keep_speaker_labels else "stripped",
        "docs": manifest_rows,
    }
    corpus_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(docs)} docs to {corpus_path}")
    print(f"empty transcripts: {empty_count} of {len(rows)}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
