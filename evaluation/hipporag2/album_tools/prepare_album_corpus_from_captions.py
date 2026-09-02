#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "icm_bench_caption_corpus"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clip_number(video_id: str) -> int:
    if str(video_id).strip() == "clip_000":
        return 0
    match = re.search(r"(\d+)$", str(video_id))
    if not match:
        raise ValueError(f"Cannot parse numeric clip id from {video_id!r}")
    return int(match.group(1))


def canonical_clip_number(row: dict[str, Any]) -> int:
    video_id = str(row.get("video_id") or row.get("clip_id") or row.get("caption", {}).get("clip_id"))
    return clip_number(video_id)


def render_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def render_caption_doc(
    row: dict[str, Any],
    source_label: str = "Gemini Pro video caption",
    title_suffix: str = "caption",
    field_keys: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    video_id = str(row.get("video_id") or row.get("clip_id") or row.get("caption", {}).get("clip_id"))
    number = canonical_clip_number(row)
    caption = row.get("caption") or {}
    if not isinstance(caption, dict):
        caption = {"caption": str(caption)}

    title = f"{video_id}: {title_suffix}"
    fields = [
        ("caption", "Caption"),
        ("visual_details", "Visual details"),
        ("dialogue_or_speech", "Dialogue or speech"),
        ("audio_events", "Audio events"),
        ("people", "People"),
        ("objects_places", "Objects and places"),
        ("time_or_event", "Time or event"),
    ]
    if field_keys:
        keep = set(field_keys)
        fields = [(key, label) for key, label in fields if key in keep]
    text_parts = [
        f"Clip id: {video_id}.",
        f"Numeric memory order: {number}.",
        f"Source: {source_label}.",
    ]
    for key, label in fields:
        value = render_value(caption.get(key))
        if value:
            text_parts.append(f"{label}: {value}.")

    extras = []
    for key, value in caption.items():
        if field_keys or key in {k for k, _ in fields} or key == "clip_id":
            continue
        rendered = render_value(value)
        if rendered:
            extras.append(f"{key}: {rendered}")
    if extras:
        text_parts.append("Additional caption fields: " + " | ".join(extras) + ".")

    # Public ICM-Bench clip ids are zero-based (clip_000--clip_838), so the
    # corpus index can use the public numeric id directly without remapping.
    idx = number
    doc = {
        "title": title,
        "text": "\n".join(text_parts),
        "idx": idx,
    }
    manifest_row = {
        "idx": idx,
        "clip_id": video_id,
        "m3_clip_number": number,
        "video_path": row.get("video_path"),
        "caption_model": row.get("caption_model"),
        "title": title,
    }
    return doc, manifest_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--source-label", default="video caption")
    parser.add_argument("--title-suffix", default="caption")
    parser.add_argument(
        "--field-keys",
        default="",
        help=(
            "Comma-separated caption fields to include in corpus docs. "
            "Default includes all supported fields. Example: caption"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.captions))
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    rows.sort(key=canonical_clip_number)
    if args.limit:
        rows = rows[: args.limit]

    docs = []
    manifest_rows = []
    field_keys = [part.strip() for part in args.field_keys.split(",") if part.strip()]
    for row in rows:
        doc, manifest_row = render_caption_doc(row, args.source_label, args.title_suffix, field_keys or None)
        docs.append(doc)
        manifest_rows.append(manifest_row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / f"{args.dataset_name}_corpus.json"
    manifest_path = output_dir / f"{args.dataset_name}_manifest.json"
    manifest = {
        "dataset_name": args.dataset_name,
        "source_captions": str(Path(args.captions)),
        "source_label": args.source_label,
        "source_clip_count": len(rows),
        "field_keys": field_keys or "all",
        "docs": manifest_rows,
    }
    corpus_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(docs)} docs to {corpus_path}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
