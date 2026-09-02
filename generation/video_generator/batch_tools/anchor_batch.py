#!/usr/bin/env python3
"""Submit and collect Gemini Batch API jobs for anchor image generation."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import get_google_api_key, get_image_model
from video_generator.anchor_generator import VideoPreproductionBuilder
from video_generator.planner import load_series_bible


JOB_MANIFEST_NAME = "10_anchor_batch_jobs.json"
UPLOAD_MANIFEST_NAME = "10_anchor_batch_uploads.json"


def _read_json(path: Path, default: dict) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return default


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"💾 已保存: {path}")


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _upload_file_once(
    client: genai.Client,
    path: Path,
    upload_manifest: dict,
) -> types.File:
    key = str(path.resolve())
    fingerprint = _file_fingerprint(path)
    cached = upload_manifest.get("files", {}).get(key)
    if (
        cached
        and cached.get("size") == fingerprint["size"]
        and cached.get("mtime_ns") == fingerprint["mtime_ns"]
        and cached.get("name")
        and cached.get("uri")
    ):
        return types.File(
            name=cached["name"],
            uri=cached["uri"],
            mime_type=cached.get("mime_type") or "image/png",
        )

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    print(f"☁️  上传 Batch 输入图片: {path.name}")
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type=mime_type, display_name=path.name),
    )
    upload_manifest.setdefault("files", {})[key] = {
        **fingerprint,
        "name": uploaded.name,
        "uri": uploaded.uri,
        "mime_type": uploaded.mime_type or mime_type,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    return uploaded


def _target_shots(builder: VideoPreproductionBuilder, clip_ids: set[str] | None) -> list[tuple[Any, Any]]:
    targets: list[tuple[Any, Any]] = []
    for clip in builder.series_bible.clips:
        if clip_ids and clip.id not in clip_ids:
            continue
        for shot in clip.shots:
            selected_path = builder.anchors_dir / clip.id / shot.id / "selected.png"
            if selected_path.exists():
                continue
            targets.append((clip, shot))
    return targets


def _prepare_batch_request(
    builder: VideoPreproductionBuilder,
    client: genai.Client,
    upload_manifest: dict,
    clip: Any,
    shot: Any,
) -> tuple[types.InlinedRequest, dict]:
    # Ensure outfit references and cast board exist. This may generate only the
    # small fixed outfit-ref set interactively; the expensive per-shot anchors go batch.
    cast_board_path = builder._build_cast_board(clip, shot)
    if cast_board_path is None or not cast_board_path.exists():
        raise RuntimeError(f"无法为 {clip.id}/{shot.id} 生成 cast board。")

    scene_refs = builder.asset_manifest.get("scenes", {}).get(shot.scene_id, {}).get("references", [])
    if not scene_refs:
        raise RuntimeError(f"{shot.id} 缺少场景参考图: {shot.scene_id}")
    scene_path = Path(scene_refs[0])
    if not scene_path.exists():
        raise RuntimeError(f"{shot.id} 场景参考图不存在: {scene_path}")

    scene_file = _upload_file_once(client, scene_path, upload_manifest)
    cast_file = _upload_file_once(client, cast_board_path, upload_manifest)

    prompt = builder._build_anchor_prompt(clip, shot)
    contents = [
        prompt,
        "\nThis is the scene reference image.",
        scene_file,
        "End of scene reference.\n",
        "\nThis is the cast board showing the exact people who must appear in frame.",
        cast_file,
        "End of cast board.\n",
    ]
    request = types.InlinedRequest(
        contents=contents,
        metadata={
            "key": shot.id,
            "clip_id": clip.id,
            "shot_id": shot.id,
            "scene_id": shot.scene_id,
        },
        config=types.GenerateContentConfig(
            response_modalities=["Image"],
            image_config=types.ImageConfig(aspect_ratio=builder.aspect_ratio),
        ),
    )
    record = {
        "key": shot.id,
        "clip_id": clip.id,
        "shot_id": shot.id,
        "scene_id": shot.scene_id,
        "cast_board_path": str(cast_board_path),
        "candidate_path": str(builder.anchors_dir / clip.id / shot.id / "candidate_01.png"),
        "selected_path": str(builder.anchors_dir / clip.id / shot.id / "selected.png"),
        "visible_characters": list(shot.visible_characters),
        "clip_character_outfits": {
            char_id: builder._selected_clip_outfit(clip, char_id)
            for char_id in shot.visible_characters
            if builder._selected_clip_outfit(clip, char_id)
        },
    }
    return request, record


def submit(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_dir = output_root / "metadata"
    series_bible = load_series_bible(output_root)
    builder = VideoPreproductionBuilder(
        series_bible=series_bible,
        output_root=output_root,
        config_path=args.config,
    )

    image_model = args.model or get_image_model() or "gemini-3.1-flash-image-preview"
    clip_ids = {part.strip() for part in (args.include_clips or "").split(",") if part.strip()} or None
    targets = _target_shots(builder, clip_ids)
    if args.max_requests:
        targets = targets[: args.max_requests]

    print(f"🎯 待提交 Batch anchor requests: {len(targets)}")
    if args.dry_run:
        planned_jobs = (len(targets) + args.batch_size - 1) // args.batch_size if targets else 0
        print(f"🧪 dry-run: would submit {planned_jobs} Batch jobs with batch_size={args.batch_size}")
        return
    if not targets:
        return

    # These steps reuse existing assets and only fill missing global references.
    builder.generate_character_references()
    builder.generate_scene_references()

    client = genai.Client(api_key=get_google_api_key())

    upload_manifest_path = metadata_dir / UPLOAD_MANIFEST_NAME
    job_manifest_path = metadata_dir / JOB_MANIFEST_NAME
    upload_manifest = _read_json(upload_manifest_path, {"files": {}})
    job_manifest = _read_json(
        job_manifest_path,
        {
            "output_root": str(output_root),
            "model": image_model,
            "jobs": [],
            "requests": {},
        },
    )
    job_manifest["model"] = image_model
    job_manifest["output_root"] = str(output_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for chunk_index in range(0, len(targets), args.batch_size):
        target_chunk = targets[chunk_index : chunk_index + args.batch_size]
        chunk_number = chunk_index // args.batch_size + 1
        total_chunks = (len(targets) + args.batch_size - 1) // args.batch_size
        chunk: list[tuple[types.InlinedRequest, dict]] = []
        print(f"🧩 准备 Batch chunk {chunk_number}/{total_chunks}: {len(target_chunk)} requests")
        for offset, (clip, shot) in enumerate(target_chunk, start=1):
            global_index = chunk_index + offset
            print(f"   准备请求 {global_index}/{len(targets)}: {clip.id}/{shot.id}")
            request, record = _prepare_batch_request(builder, client, upload_manifest, clip, shot)
            chunk.append((request, record))
            if global_index % 25 == 0:
                _write_json(upload_manifest_path, upload_manifest)
                _write_json(builder.asset_manifest_path, builder.asset_manifest)

        _write_json(upload_manifest_path, upload_manifest)
        _write_json(builder.asset_manifest_path, builder.asset_manifest)

        batch_number = len(job_manifest.get("jobs", [])) + 1
        display_name = f"{args.display_name_prefix}_{timestamp}_{batch_number:03d}"
        requests = [item[0] for item in chunk]
        records = [item[1] for item in chunk]

        print(f"🚀 提交 Batch job: {display_name} ({len(requests)} requests)")
        job = client.batches.create(
            model=image_model,
            src=requests,
            config=types.CreateBatchJobConfig(display_name=display_name),
        )
        job_record = {
            "name": job.name,
            "display_name": display_name,
            "state": str(job.state),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "request_keys": [record["key"] for record in records],
        }
        job_manifest.setdefault("jobs", []).append(job_record)
        for record in records:
            record["batch_job_name"] = job.name
            record["batch_display_name"] = display_name
            job_manifest.setdefault("requests", {})[record["key"]] = record
        _write_json(job_manifest_path, job_manifest)

    if not args.dry_run:
        print(f"✅ 已提交 Batch jobs: {len(job_manifest.get('jobs', []))}")
        print(f"📄 Job manifest: {job_manifest_path}")


def _parts_from_response(response: types.GenerateContentResponse) -> list[Any]:
    parts = getattr(response, "parts", None)
    if parts:
        return list(parts)
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) if content else None
        if candidate_parts:
            return list(candidate_parts)
    return []


def _save_response_image(response: types.GenerateContentResponse, path: Path) -> bool:
    for part in _parts_from_response(response):
        inline_data = getattr(part, "inline_data", None)
        if not inline_data:
            continue
        data = getattr(inline_data, "data", None)
        if data is None:
            continue
        if isinstance(data, str):
            raw = base64.b64decode(data)
        else:
            raw = data
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return True
    return False


def _update_anchor_manifest(output_root: Path, request_record: dict) -> None:
    metadata_dir = output_root / "metadata"
    manifest_path = metadata_dir / "03_anchor_manifest.json"
    manifest = _read_json(manifest_path, {"shots": {}})
    selected_path = Path(request_record["selected_path"])
    candidate_path = Path(request_record["candidate_path"])
    if not selected_path.exists() and candidate_path.exists():
        shutil.copyfile(candidate_path, selected_path)
    manifest.setdefault("shots", {})[request_record["shot_id"]] = {
        "clip_id": request_record["clip_id"],
        "strategy": "shot_based",
        "scene_id": request_record["scene_id"],
        "visible_characters": request_record.get("visible_characters", []),
        "clip_character_outfits": request_record.get("clip_character_outfits", {}),
        "cast_board_path": request_record.get("cast_board_path"),
        "candidates": [str(candidate_path)],
        "selected_anchor": str(selected_path),
        "generated_by": "gemini_batch_api",
    }
    _write_json(manifest_path, manifest)


def status(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "metadata" / JOB_MANIFEST_NAME
    manifest = _read_json(manifest_path, {"jobs": []})
    client = genai.Client(api_key=get_google_api_key())
    for job_record in manifest.get("jobs", []):
        job = client.batches.get(name=job_record["name"])
        job_record["state"] = str(job.state)
        job_record["update_time"] = str(job.update_time) if job.update_time else None
        if job.error:
            job_record["error"] = str(job.error)
        stats = getattr(job, "completion_stats", None)
        if stats:
            job_record["completion_stats"] = stats.model_dump(mode="json", exclude_none=True)
        print(job_record["name"], job_record["state"], job_record.get("completion_stats", ""))
    _write_json(manifest_path, manifest)


def collect(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "metadata" / JOB_MANIFEST_NAME
    manifest = _read_json(manifest_path, {"jobs": [], "requests": {}})
    client = genai.Client(api_key=get_google_api_key())
    saved = 0
    failed = 0

    for job_record in manifest.get("jobs", []):
        job = client.batches.get(name=job_record["name"])
        job_record["state"] = str(job.state)
        if str(job.state) not in {
            "JobState.JOB_STATE_SUCCEEDED",
            "JobState.JOB_STATE_PARTIALLY_SUCCEEDED",
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_PARTIALLY_SUCCEEDED",
        }:
            print(f"⏳ job 未完成，跳过 collect: {job.name} {job.state}")
            continue
        responses = getattr(getattr(job, "dest", None), "inlined_responses", None) or []
        print(f"📥 collect {job.name}: {len(responses)} responses")
        for item in responses:
            metadata = item.metadata or {}
            key = metadata.get("key") or metadata.get("shot_id")
            if not key:
                failed += 1
                continue
            record = manifest.get("requests", {}).get(key)
            if not record:
                failed += 1
                continue
            if item.error:
                record["error"] = str(item.error)
                failed += 1
                continue
            candidate_path = Path(record["candidate_path"])
            selected_path = Path(record["selected_path"])
            if candidate_path.exists() and selected_path.exists() and not args.overwrite:
                record["collected"] = True
                continue
            if item.response and _save_response_image(item.response, candidate_path):
                selected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate_path, selected_path)
                _update_anchor_manifest(output_root, record)
                record["collected"] = True
                record["collected_at"] = datetime.now().isoformat(timespec="seconds")
                saved += 1
            else:
                record["error"] = "No image inline_data found in batch response."
                failed += 1
        job_record["collected_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(manifest_path, manifest)
    print(f"✅ collect 完成: saved={saved}, failed={failed}")


def wait(args: argparse.Namespace) -> None:
    while True:
        status(args)
        output_root = Path(args.output_root).expanduser().resolve()
        manifest = _read_json(output_root / "metadata" / JOB_MANIFEST_NAME, {"jobs": []})
        states = [str(job.get("state")) for job in manifest.get("jobs", [])]
        unfinished = [
            state for state in states
            if "SUCCEEDED" not in state and "FAILED" not in state and "CANCELLED" not in state and "EXPIRED" not in state
        ]
        if not unfinished:
            break
        print(f"⏳ {len(unfinished)} jobs still running; sleep {args.poll_seconds}s")
        time.sleep(args.poll_seconds)
    collect(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch API anchor generation helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--output-root", required=True)
        subparser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "video_config.yaml"))

    submit_parser = subparsers.add_parser("submit")
    add_common(submit_parser)
    submit_parser.add_argument("--model", default=None)
    submit_parser.add_argument("--include-clips", default="")
    submit_parser.add_argument("--batch-size", type=int, default=200)
    submit_parser.add_argument("--max-requests", type=int, default=0)
    submit_parser.add_argument("--display-name-prefix", default="anchor_batch")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.set_defaults(func=submit)

    status_parser = subparsers.add_parser("status")
    add_common(status_parser)
    status_parser.set_defaults(func=status)

    collect_parser = subparsers.add_parser("collect")
    add_common(collect_parser)
    collect_parser.add_argument("--overwrite", action="store_true")
    collect_parser.set_defaults(func=collect)

    wait_parser = subparsers.add_parser("wait")
    add_common(wait_parser)
    wait_parser.add_argument("--overwrite", action="store_true")
    wait_parser.add_argument("--poll-seconds", type=int, default=300)
    wait_parser.set_defaults(func=wait)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
