#!/usr/bin/env python3
"""Manual shot review and filtered-video export.

The source videos and source metadata are never modified. Review decisions are
kept in a small JSON manifest, then exported into a filtered release directory.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_REVIEW_PATH = "review/shot_review_manifest.json"
STATUS_VALUES = {"unreviewed", "pass", "exclude"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_clip_range(value: str) -> list[str]:
    value = (value or "").strip()
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


def clip_sort_key(clip_id: str) -> tuple[int, str]:
    try:
        return (int(clip_id.replace("clip_", "")), clip_id)
    except ValueError:
        return (10**9, clip_id)


def shot_sort_key(shot_id: str) -> tuple[int, str]:
    try:
        return (int(shot_id.rsplit("_", 1)[-1]), shot_id)
    except ValueError:
        return (10**9, shot_id)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ffprobe_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return float(result.stdout.strip())
    except Exception:
        return None


class ReviewStore:
    def __init__(self, output_root: Path, review_path: Path) -> None:
        self.output_root = output_root.resolve()
        self.review_path = review_path
        self.metadata_dir = self.output_root / "metadata"
        self.series_path = self.metadata_dir / "06_shot_plan.json"
        self.voice_dir = self.output_root / "voice_work"
        self.subtitles_dir = self.output_root / "subtitles" / "gemini_tts_srt"
        self.segment_dir = self.output_root / "renders" / "segments"
        self.clip_dir = self.output_root / "renders" / "clips_gemini_tts"
        self.subtitled_clip_dir = self.output_root / "renders" / "clips_gemini_tts_subtitled"
        self.series = load_json(self.series_path)
        self.cast_by_id = {member["id"]: member for member in self.series.get("cast", [])}
        self.default_clip_ids = self._discover_rendered_clip_ids()

    def load_review(self) -> dict[str, Any]:
        if self.review_path.exists():
            payload = load_json(self.review_path)
        else:
            payload = {
                "version": 1,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "output_root": str(self.output_root),
                "policy": {
                    "final_dataset_status": "Only shots explicitly marked pass are included.",
                    "unreviewed_status": "Unreviewed shots are excluded from final exports by default.",
                    "source_files_untouched": True,
                },
                "shots": {},
            }
        payload.setdefault("shots", {})
        return payload

    def save_review(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = utc_now()
        write_json(self.review_path, payload)

    def _discover_rendered_clip_ids(self) -> list[str]:
        ids: set[str] = set()
        render_manifest = self.metadata_dir / "04_render_manifest.json"
        if render_manifest.exists():
            payload = load_json(render_manifest)
            clips = payload.get("clips", {})
            if isinstance(clips, dict):
                for clip_id, item in clips.items():
                    if isinstance(item, dict) and item.get("status") == "success":
                        ids.add(clip_id)
        for directory in [self.subtitled_clip_dir, self.clip_dir, self.segment_dir]:
            if not directory.exists():
                continue
            for path in directory.glob("clip_*"):
                stem = path.stem if path.is_file() else path.name
                if stem.startswith("clip_"):
                    ids.add(stem)
        return sorted(ids, key=clip_sort_key)

    def clips(self, clip_ids: list[str] | None = None) -> list[dict[str, Any]]:
        clips = list(self.series.get("clips", []))
        include_ids = clip_ids or self.default_clip_ids
        if include_ids:
            include = set(include_ids)
            clips = [clip for clip in clips if clip.get("id") in include]
        return sorted(clips, key=lambda clip: clip_sort_key(clip.get("id", "")))

    def init_manifest(self, clip_ids: list[str] | None = None) -> dict[str, Any]:
        review = self.load_review()
        shots = review.setdefault("shots", {})
        requested_clip_ids = clip_ids or self.default_clip_ids
        keep_shot_ids = {
            shot["id"]
            for clip in self.clips(requested_clip_ids)
            for shot in clip.get("shots", [])
        }
        if clip_ids is None:
            for shot_id in list(shots):
                if shot_id not in keep_shot_ids:
                    del shots[shot_id]
        for clip in self.clips(requested_clip_ids):
            clip_id = clip["id"]
            for shot in sorted(clip.get("shots", []), key=lambda item: shot_sort_key(item.get("id", ""))):
                shot_id = shot["id"]
                shots.setdefault(
                    shot_id,
                    {
                        "clip_id": clip_id,
                        "shot_index": shot.get("shot_index"),
                        "status": "unreviewed",
                        "reason_tags": [],
                        "note": "",
                        "reviewed_at": None,
                    },
                )
        self.save_review(review)
        return review

    def mark(
        self,
        *,
        status: str,
        shot_ids: list[str] | None = None,
        clip_ids: list[str] | None = None,
        reason_tags: list[str] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        if status not in STATUS_VALUES:
            raise ValueError(f"status must be one of {sorted(STATUS_VALUES)}")
        review = self.init_manifest(clip_ids=None)
        targets: set[str] = set(shot_ids or [])
        if clip_ids:
            for clip in self.clips(clip_ids):
                targets.update(shot["id"] for shot in clip.get("shots", []))
        if not targets:
            raise ValueError("No target shots. Use --shot or --clip.")
        missing = sorted(shot_id for shot_id in targets if shot_id not in review["shots"])
        if missing:
            raise ValueError(f"Unknown shot id(s): {missing[:10]}")
        for shot_id in sorted(targets):
            item = review["shots"][shot_id]
            item["status"] = status
            item["reason_tags"] = reason_tags or []
            item["note"] = note
            item["reviewed_at"] = utc_now() if status != "unreviewed" else None
        self.save_review(review)
        return review

    def shot_runtime(self, shot: dict[str, Any]) -> float:
        return float(shot.get("duration_seconds") or 0)

    def status_for(self, review: dict[str, Any], shot_id: str) -> str:
        return str(review.get("shots", {}).get(shot_id, {}).get("status") or "unreviewed")

    def summary(self, clip_ids: list[str] | None = None) -> dict[str, Any]:
        review = self.init_manifest(clip_ids=None)
        counts = {status: 0 for status in sorted(STATUS_VALUES)}
        seconds = {status: 0.0 for status in sorted(STATUS_VALUES)}
        clips_total = 0
        clips_all_pass = 0
        clips_any_excluded = 0
        clips_fully_reviewed = 0
        for clip in self.clips(clip_ids):
            statuses: list[str] = []
            clips_total += 1
            for shot in clip.get("shots", []):
                status = self.status_for(review, shot["id"])
                statuses.append(status)
                counts[status] = counts.get(status, 0) + 1
                seconds[status] = seconds.get(status, 0.0) + self.shot_runtime(shot)
            if statuses and all(status == "pass" for status in statuses):
                clips_all_pass += 1
            if "exclude" in statuses:
                clips_any_excluded += 1
            if statuses and all(status != "unreviewed" for status in statuses):
                clips_fully_reviewed += 1
        return {
            "clips_total": clips_total,
            "clips_fully_reviewed": clips_fully_reviewed,
            "clips_all_pass": clips_all_pass,
            "clips_any_excluded": clips_any_excluded,
            "shot_counts": counts,
            "runtime_seconds": seconds,
            "pass_minutes": round(seconds.get("pass", 0.0) / 60.0, 2),
            "target_minutes": 100.0,
            "target_met": seconds.get("pass", 0.0) >= 100 * 60,
            "review_manifest": str(self.review_path),
        }

    def clip_payload(self, clip_id: str) -> dict[str, Any]:
        review = self.init_manifest(clip_ids=None)
        matching = [clip for clip in self.clips([clip_id]) if clip.get("id") == clip_id]
        if not matching:
            raise KeyError(clip_id)
        clip = matching[0]
        tts_manifest = self.voice_dir / clip_id / "tts_manifest.json"
        tts_segments = load_json(tts_manifest).get("segments", []) if tts_manifest.exists() else []
        dialogue_by_shot: dict[str, list[dict[str, Any]]] = {}
        for segment in tts_segments:
            dialogue_by_shot.setdefault(segment.get("shot_id", ""), []).append(segment)
        shots = []
        for shot in sorted(clip.get("shots", []), key=lambda item: shot_sort_key(item["id"])):
            shot_id = shot["id"]
            shot_path = self.segment_dir / clip_id / f"{shot_id}.mp4"
            shots.append(
                {
                    "id": shot_id,
                    "shot_index": shot.get("shot_index"),
                    "duration_seconds": shot.get("duration_seconds"),
                    "beat_title": shot.get("beat_title"),
                    "purpose": shot.get("purpose"),
                    "visible_characters": [
                        self.cast_by_id.get(char_id, {}).get("name_en", char_id)
                        for char_id in shot.get("visible_characters", [])
                    ],
                    "evidence_facts": shot.get("evidence_facts", []),
                    "dialogue_lines": shot.get("dialogue_lines", []),
                    "tts_dialogue": [
                        {
                            "speaker": seg.get("character_name") or seg.get("char_id"),
                            "text": seg.get("text"),
                            "original_text": seg.get("original_text"),
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                        }
                        for seg in dialogue_by_shot.get(shot_id, [])
                    ],
                    "status": self.status_for(review, shot_id),
                    "review": review.get("shots", {}).get(shot_id, {}),
                    "video_url": f"/file/{self._rel_url(shot_path)}" if shot_path.exists() else None,
                }
            )
        full_video = self.subtitled_clip_dir / f"{clip_id}.mp4"
        if not full_video.exists():
            full_video = self.clip_dir / f"{clip_id}.mp4"
        return {
            "clip_id": clip_id,
            "title": clip.get("title"),
            "logline": clip.get("logline"),
            "date": clip.get("clip_date"),
            "target_runtime_seconds": clip.get("target_runtime_seconds"),
            "memory_facts": clip.get("memory_facts", []),
            "relationship_facts": clip.get("relationship_facts", []),
            "continuity_hooks": clip.get("continuity_hooks", []),
            "full_video_url": f"/file/{self._rel_url(full_video)}" if full_video.exists() else None,
            "srt_url": f"/file/{self._rel_url(self.subtitles_dir / f'{clip_id}.srt')}",
            "shots": shots,
            "prev_clip_id": self.neighbor_clip_id(clip_id, -1),
            "next_clip_id": self.neighbor_clip_id(clip_id, 1),
            "summary": self.summary(),
        }

    def neighbor_clip_id(self, clip_id: str, offset: int) -> str | None:
        ids = [clip["id"] for clip in self.clips()]
        try:
            index = ids.index(clip_id)
        except ValueError:
            return None
        next_index = index + offset
        if 0 <= next_index < len(ids):
            return ids[next_index]
        return None

    def first_unreviewed_clip_id(self) -> str | None:
        review = self.init_manifest(clip_ids=None)
        for clip in self.clips():
            for shot in clip.get("shots", []):
                if self.status_for(review, shot["id"]) == "unreviewed":
                    return clip["id"]
        return self.clips()[0]["id"] if self.clips() else None

    def todo(self, limit: int = 20) -> list[dict[str, Any]]:
        review = self.init_manifest(clip_ids=None)
        items: list[dict[str, Any]] = []
        for clip in self.clips():
            unreviewed = [
                shot["id"]
                for shot in clip.get("shots", [])
                if self.status_for(review, shot["id"]) == "unreviewed"
            ]
            if unreviewed:
                items.append(
                    {
                        "clip_id": clip["id"],
                        "title": clip.get("title"),
                        "unreviewed_shots": unreviewed,
                    }
                )
            if len(items) >= limit:
                break
        return items

    def export_filtered(
        self,
        export_root: Path,
        include_unreviewed: bool = False,
        keep_partial_clip_facts: bool = False,
    ) -> dict[str, Any]:
        review = self.init_manifest(clip_ids=None)
        export_root = export_root.resolve()
        export_metadata_dir = export_root / "metadata"
        export_review_dir = export_root / "review"
        export_metadata_dir.mkdir(parents=True, exist_ok=True)
        export_review_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in export_metadata_dir.glob("08*.json"):
            stale_path.unlink()
        for stale_path in export_metadata_dir.glob("00_08*_prompt.txt"):
            stale_path.unlink()

        include_statuses = {"pass"}
        if include_unreviewed:
            include_statuses.add("unreviewed")

        filtered_series = json.loads(json.dumps(self.series))
        filtered_clips: list[dict[str, Any]] = []
        dataset_items: list[dict[str, Any]] = []
        excluded_items: list[dict[str, Any]] = []
        total_seconds = 0.0

        reviewable_clip_ids = {clip["id"] for clip in self.clips()}
        for clip in filtered_series.get("clips", []):
            if clip.get("id") not in reviewable_clip_ids:
                continue
            original_shots = clip.get("shots", [])
            kept_shots = []
            excluded_shots = []
            for shot in original_shots:
                status = self.status_for(review, shot["id"])
                item = review.get("shots", {}).get(shot["id"], {})
                if status in include_statuses:
                    kept_shots.append(shot)
                    duration = self.shot_runtime(shot)
                    total_seconds += duration
                    dataset_items.append(
                        {
                            "clip_id": clip["id"],
                            "shot_id": shot["id"],
                            "shot_index": shot.get("shot_index"),
                            "duration_seconds": duration,
                            "status": status,
                            "segment_path": str(self.segment_dir / clip["id"] / f"{shot['id']}.mp4"),
                            "source_full_video_path": str(self.clip_dir / f"{clip['id']}.mp4"),
                            "source_subtitled_video_path": str(self.subtitled_clip_dir / f"{clip['id']}.mp4"),
                            "srt_path": str(self.subtitles_dir / f"{clip['id']}.srt"),
                        }
                    )
                else:
                    excluded_shots.append(shot)
                    excluded_items.append(
                        {
                            "clip_id": clip["id"],
                            "shot_id": shot["id"],
                            "status": status,
                            "reason_tags": item.get("reason_tags", []),
                            "note": item.get("note", ""),
                        }
                    )
            if not kept_shots:
                continue
            clip["shots"] = kept_shots
            clip["target_runtime_seconds"] = int(sum(self.shot_runtime(shot) for shot in kept_shots))
            if excluded_shots and not keep_partial_clip_facts:
                clip["memory_facts"] = []
                clip["relationship_facts"] = []
                clip["continuity_hooks"] = []
                clip["logline"] = f"{clip.get('title', clip['id'])}: filtered partial clip with reviewed passing shots only."
            filtered_clips.append(clip)

        filtered_series["clips"] = filtered_clips

        for filename in ["00_video_config_snapshot.json", "02_asset_manifest.json", "03_anchor_manifest.json"]:
            source = self.metadata_dir / filename
            if source.exists():
                shutil.copy2(source, export_metadata_dir / filename)

        for filename in ["01_series_bible.json", "06_shot_plan.json", "07_series_bible.json"]:
            write_json(export_metadata_dir / filename, filtered_series)

        manifest = {
            "created_at": utc_now(),
            "source_output_root": str(self.output_root),
            "filtered_output_root": str(export_root),
            "review_manifest": str(self.review_path),
            "include_unreviewed": include_unreviewed,
            "keep_partial_clip_facts": keep_partial_clip_facts,
            "included_clip_count": len(filtered_clips),
            "included_shot_count": len(dataset_items),
            "excluded_shot_count": len(excluded_items),
            "included_runtime_seconds": round(total_seconds, 3),
            "included_runtime_minutes": round(total_seconds / 60.0, 2),
            "target_minutes": 100.0,
            "target_met": total_seconds >= 100 * 60,
            "items": dataset_items,
            "excluded": excluded_items,
        }
        write_json(export_review_dir / "final_dataset_manifest.json", manifest)
        shutil.copy2(self.review_path, export_review_dir / self.review_path.name)
        return manifest

    def _rel_url(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.output_root)
        return "/".join(part.replace("%", "%25").replace("#", "%23").replace("?", "%3F") for part in rel.parts)


def render_review_page(initial_clip_id: str) -> str:
    safe_initial = html.escape(initial_clip_id)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shot Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f5; color: #171717; }}
    header {{ position: sticky; top: 0; z-index: 10; background: #ffffff; border-bottom: 1px solid #ddd; padding: 12px 20px; display: flex; align-items: center; gap: 14px; }}
    button, input {{ font: inherit; }}
    button {{ border: 1px solid #999; background: #fff; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
    button.pass {{ border-color: #1f7a3b; color: #1f7a3b; }}
    button.exclude {{ border-color: #b42318; color: #b42318; }}
    button.active.pass {{ background: #dff3e5; }}
    button.active.exclude {{ background: #fde2df; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 18px 20px 40px; }}
    .top {{ display: grid; grid-template-columns: minmax(320px, 1.45fr) minmax(280px, .9fr); gap: 18px; align-items: start; }}
    video {{ width: 100%; background: #111; border-radius: 8px; }}
    h1 {{ font-size: 22px; margin: 0 0 6px; }}
    h2 {{ font-size: 18px; margin: 24px 0 10px; }}
    .meta {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }}
    .muted {{ color: #666; font-size: 14px; }}
    .shot {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin: 14px 0; }}
    .shot-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .shot-grid {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(260px, .9fr); gap: 14px; margin-top: 10px; }}
    .status {{ font-weight: 700; }}
    .status.pass {{ color: #1f7a3b; }}
    .status.exclude {{ color: #b42318; }}
    .status.unreviewed {{ color: #7a5b00; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .tag {{ border: 1px solid #bbb; border-radius: 999px; padding: 4px 8px; font-size: 13px; background: #fafafa; }}
    textarea {{ width: 100%; min-height: 56px; resize: vertical; box-sizing: border-box; }}
    ul {{ padding-left: 18px; }}
    @media (max-width: 860px) {{ .top, .shot-grid {{ grid-template-columns: 1fr; }} header {{ flex-wrap: wrap; }} }}
  </style>
</head>
<body>
  <header>
    <button id="prev">Prev</button>
    <input id="clipInput" value="{safe_initial}" size="12">
    <button id="go">Go</button>
    <button id="next">Next</button>
    <button id="todo">First Unreviewed</button>
    <span id="progress" class="muted"></span>
  </header>
  <main id="app"></main>
  <script>
    let currentClip = "{safe_initial}";
    const reasons = ["wrong_action", "face_drift", "identity_error", "bad_lipsync", "bad_audio", "bad_subtitle", "scene_error", "camera_error", "artifact"];

    async function api(path, options) {{
      const res = await fetch(path, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }}

    function esc(text) {{
      return String(text ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    }}

    async function loadClip(clipId) {{
      currentClip = clipId;
      document.getElementById("clipInput").value = clipId;
      const data = await api(`/api/clip?clip=${{encodeURIComponent(clipId)}}`);
      render(data);
    }}

    function render(data) {{
      document.getElementById("prev").disabled = !data.prev_clip_id;
      document.getElementById("next").disabled = !data.next_clip_id;
      document.getElementById("progress").textContent =
        `pass ${{data.summary.pass_minutes}}min / target 100min, reviewed clips ${{data.summary.clips_fully_reviewed}}/${{data.summary.clips_total}}`;
      const fullVideo = data.full_video_url ? `<video src="${{data.full_video_url}}" controls></video>` : `<p>No full video found.</p>`;
      const facts = [...(data.memory_facts || []), ...(data.relationship_facts || [])].slice(0, 8);
      document.getElementById("app").innerHTML = `
        <section class="top">
          <div>${{fullVideo}}</div>
          <div class="meta">
            <h1>${{esc(data.clip_id)}} · ${{esc(data.title)}}</h1>
            <p>${{esc(data.logline)}}</p>
            <p class="muted">${{esc(data.date)}} · target ${{esc(data.target_runtime_seconds)}}s</p>
            <h2>Facts</h2>
            <ul>${{facts.map(f => `<li>${{esc(f)}}</li>`).join("")}}</ul>
          </div>
        </section>
        <h2>Shots</h2>
        ${{data.shots.map(renderShot).join("")}}
      `;
      document.getElementById("prev").onclick = () => data.prev_clip_id && loadClip(data.prev_clip_id);
      document.getElementById("next").onclick = () => data.next_clip_id && loadClip(data.next_clip_id);
    }}

    function renderShot(shot) {{
      const video = shot.video_url ? `<video src="${{shot.video_url}}" controls></video>` : `<p>No shot segment found.</p>`;
      const tts = (shot.tts_dialogue || []).map(d =>
        `<li><b>${{esc(d.speaker)}}:</b> ${{esc(d.text)}} <span class="muted">(${{esc(d.start)}}-${{esc(d.end)}})</span></li>`
      ).join("");
      const tags = reasons.map(tag => `<label class="tag"><input type="checkbox" data-shot="${{esc(shot.id)}}" value="${{tag}}"> ${{tag}}</label>`).join("");
      return `
        <article class="shot" id="${{esc(shot.id)}}">
          <div class="shot-head">
            <div><b>${{esc(shot.id)}}</b> · ${{esc(shot.beat_title)}} · ${{esc(shot.duration_seconds)}}s</div>
            <div class="status ${{esc(shot.status)}}">${{esc(shot.status)}}</div>
          </div>
          <div class="shot-grid">
            <div>${{video}}</div>
            <div>
              <p>${{esc(shot.purpose)}}</p>
              <p class="muted">Characters: ${{esc((shot.visible_characters || []).join(", "))}}</p>
              <ul>${{(shot.evidence_facts || []).map(f => `<li>${{esc(f)}}</li>`).join("")}}</ul>
              <ul>${{tts}}</ul>
              <div class="tags">${{tags}}</div>
              <textarea id="note-${{esc(shot.id)}}" placeholder="optional note">${{esc(shot.review?.note || "")}}</textarea>
              <p>
                <button class="pass ${{shot.status === "pass" ? "active" : ""}}" onclick="markShot('${{esc(shot.id)}}','pass')">Pass</button>
                <button class="exclude ${{shot.status === "exclude" ? "active" : ""}}" onclick="markShot('${{esc(shot.id)}}','exclude')">Exclude</button>
                <button onclick="markShot('${{esc(shot.id)}}','unreviewed')">Clear</button>
              </p>
            </div>
          </div>
        </article>
      `;
    }}

    async function markShot(shotId, status) {{
      const checked = [...document.querySelectorAll(`input[data-shot="${{shotId}}"]:checked`)].map(el => el.value);
      const note = document.getElementById(`note-${{shotId}}`)?.value || "";
      await api("/api/mark", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{shot_ids: [shotId], status, reason_tags: checked, note}})
      }});
      await loadClip(currentClip);
    }}

    document.getElementById("go").onclick = () => loadClip(document.getElementById("clipInput").value.trim());
    document.getElementById("todo").onclick = async () => {{
      const data = await api("/api/first-unreviewed");
      if (data.clip_id) loadClip(data.clip_id);
    }};
    document.addEventListener("keydown", event => {{
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") return;
      if (event.key === "ArrowLeft") document.getElementById("prev").click();
      if (event.key === "ArrowRight") document.getElementById("next").click();
    }});
    loadClip(currentClip);
  </script>
</body>
</html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            clip_id = parse_qs(parsed.query).get("clip", [self.store.first_unreviewed_clip_id() or "clip_001"])[0]
            self.send_text(render_review_page(clip_id), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/clip":
            clip_id = parse_qs(parsed.query).get("clip", ["clip_001"])[0]
            self.send_json(self.store.clip_payload(clip_id))
            return
        if parsed.path == "/api/first-unreviewed":
            self.send_json({"clip_id": self.store.first_unreviewed_clip_id()})
            return
        if parsed.path.startswith("/file/"):
            self.send_file(parsed.path[len("/file/") :])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/mark":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        self.store.mark(
            status=payload["status"],
            shot_ids=payload.get("shot_ids") or [],
            clip_ids=payload.get("clip_ids") or [],
            reason_tags=payload.get("reason_tags") or [],
            note=payload.get("note") or "",
        )
        self.send_json({"ok": True})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

    def send_json(self, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, rel_url: str) -> None:
        try:
            rel_path = Path(unquote(rel_url))
            abs_path = (self.store.output_root / rel_path).resolve()
            abs_path.relative_to(self.store.output_root)
        except Exception:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not abs_path.exists() or not abs_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(abs_path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(abs_path.stat().st_size))
        self.end_headers()
        with abs_path.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)


def resolve_review_path(output_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = output_root / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review generated video shots and export filtered metadata.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--review-path", default=DEFAULT_REVIEW_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize or extend the review manifest.")
    init_parser.add_argument("--clips", default="clip_001-clip_450")

    serve_parser = subparsers.add_parser("serve", help="Start a local browser review UI.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8787)
    serve_parser.add_argument("--clip", default="")

    mark_parser = subparsers.add_parser("mark", help="Mark a shot or all shots in a clip.")
    mark_parser.add_argument("--shot", action="append", default=[])
    mark_parser.add_argument("--clip", action="append", default=[])
    mark_parser.add_argument("--status", required=True, choices=sorted(STATUS_VALUES))
    mark_parser.add_argument("--reason", default="", help="Comma-separated reason tags.")
    mark_parser.add_argument("--note", default="")

    subparsers.add_parser("summary", help="Print review progress and usable runtime.")

    todo_parser = subparsers.add_parser("todo", help="Show clips with unreviewed shots.")
    todo_parser.add_argument("--limit", type=int, default=20)

    export_parser = subparsers.add_parser("export", help="Export filtered metadata and final dataset manifest.")
    export_parser.add_argument("--export-root", default="")
    export_parser.add_argument("--include-unreviewed", action="store_true")
    export_parser.add_argument("--keep-partial-clip-facts", action="store_true")

    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    review_path = resolve_review_path(output_root, args.review_path)
    store = ReviewStore(output_root, review_path)

    if args.command == "init":
        payload = store.init_manifest(parse_clip_range(args.clips))
        print(f"review_manifest: {review_path}")
        print(json.dumps(store.summary(), indent=2, ensure_ascii=False))
    elif args.command == "serve":
        store.init_manifest(clip_ids=None)
        ReviewHandler.store = store
        server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
        initial_clip = args.clip or store.first_unreviewed_clip_id() or "clip_001"
        print(f"Review UI: http://{args.host}:{args.port}/?clip={initial_clip}")
        print(f"Review manifest: {review_path}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    elif args.command == "mark":
        reasons = [part.strip() for part in args.reason.split(",") if part.strip()]
        store.mark(
            status=args.status,
            shot_ids=args.shot,
            clip_ids=args.clip,
            reason_tags=reasons,
            note=args.note,
        )
        print(json.dumps(store.summary(), indent=2, ensure_ascii=False))
    elif args.command == "summary":
        print(json.dumps(store.summary(), indent=2, ensure_ascii=False))
    elif args.command == "todo":
        print(json.dumps(store.todo(args.limit), indent=2, ensure_ascii=False))
    elif args.command == "export":
        export_root = Path(args.export_root).expanduser().resolve() if args.export_root else output_root.with_name(output_root.name + "_review_filtered")
        manifest = store.export_filtered(
            export_root=export_root,
            include_unreviewed=args.include_unreviewed,
            keep_partial_clip_facts=args.keep_partial_clip_facts,
        )
        print(json.dumps({k: v for k, v in manifest.items() if k not in {"items", "excluded"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
