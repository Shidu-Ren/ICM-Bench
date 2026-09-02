from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini.vertex_nanobanana import VertexNanobanana
from project_config import get_google_api_key, get_image_model, get_text_model, load_video_config
from video_generator.api_usage import ApiUsageLogger
from video_generator.schemas import ClipPlan, SeriesBible, ShotPlan


if hasattr(Image, "Resampling"):
    RESAMPLE = Image.Resampling.LANCZOS
else:
    RESAMPLE = Image.LANCZOS

ANCHOR_GENERATION_METHOD = "album_style_individual_references_v1"


class VideoPreproductionBuilder:
    """Generate reusable character references, scene plates, and shot anchors."""

    def __init__(
        self,
        series_bible: SeriesBible,
        output_root: str | Path,
        config_path: str | None = None,
    ) -> None:
        self.series_bible = series_bible
        self.output_root = Path(output_root)
        self.config = load_video_config(config_path)

        production_cfg = self.config.get("production", {}) if isinstance(
            self.config.get("production"), dict
        ) else {}
        series_cfg = self.config.get("series", {}) if isinstance(self.config.get("series"), dict) else {}
        prompting_cfg = (
            self.config.get("prompting", {}) if isinstance(self.config.get("prompting"), dict) else {}
        )

        self.aspect_ratio = str(production_cfg.get("aspect_ratio", "16:9"))
        self.character_reference_count = int(
            production_cfg.get("character_reference_images_per_character", 2)
        )
        self.scene_reference_count = int(
            production_cfg.get("scene_reference_images_per_scene", 1)
        )
        self.anchor_candidates_per_shot = int(
            production_cfg.get("anchor_candidates_per_shot", 3)
        )
        self.anchor_generation_retries = int(
            production_cfg.get("anchor_generation_retries", 4)
        )
        self.anchor_retry_wait_seconds = int(
            production_cfg.get("anchor_retry_wait_seconds", 5)
        )
        self.anchor_fallback_mode = str(
            production_cfg.get("anchor_fallback_mode", "retry_only")
        ).strip().lower()
        self.anchor_quality_check_enabled = bool(
            production_cfg.get("anchor_quality_check_enabled", True)
        )
        self.anchor_quality_check_existing = bool(
            production_cfg.get("anchor_quality_check_existing", True)
        )
        self.anchor_quality_max_attempts = max(
            1,
            int(production_cfg.get("anchor_quality_max_attempts", 2)),
        )
        self.anchor_quality_min_score = max(
            0.0,
            min(1.0, float(production_cfg.get("anchor_quality_min_score", 0.78))),
        )
        self.anchor_quality_model = str(
            production_cfg.get("anchor_quality_model")
            or get_text_model()
            or "gemini-3.1-pro-preview"
        )
        self.reuse_existing_assets = bool(production_cfg.get("reuse_existing_assets", True))
        self.asset_generation_workers = max(1, int(production_cfg.get("asset_generation_workers", 1)))
        self.anchor_generation_workers = max(1, int(production_cfg.get("anchor_generation_workers", 1)))
        self.max_visible_characters = int(series_cfg.get("max_visible_characters", 6))
        self.default_negative_prompt = (
            prompting_cfg.get("negative_prompt") if isinstance(prompting_cfg.get("negative_prompt"), str) else None
        )

        self.assets_dir = self.output_root / "assets"
        self.characters_dir = self.assets_dir / "characters"
        self.scenes_dir = self.assets_dir / "scenes"
        self.clip_outfits_dir = self.assets_dir / "clip_outfits"
        self.outfit_refs_dir = self.assets_dir / "outfit_refs"
        self.cast_boards_dir = self.assets_dir / "cast_boards"
        self.anchors_dir = self.assets_dir / "anchors"
        self.metadata_dir = self.output_root / "metadata"
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)

        for directory in (
            self.characters_dir,
            self.scenes_dir,
            self.clip_outfits_dir,
            self.outfit_refs_dir,
            self.cast_boards_dir,
            self.anchors_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.nanobanana = VertexNanobanana(
            output_dir=str(self.assets_dir),
            api_key=get_google_api_key(),
            model=get_image_model(),
            usage_logger=self.api_usage_logger,
        )
        self._google_api_key = get_google_api_key()
        self.quality_client = genai.Client(api_key=self._google_api_key)
        self._quality_client_local = threading.local()
        self._quality_client_local.client = self.quality_client
        self._clip_outfit_reference_cache: dict[tuple[str, str], Path | None] = {}
        self._asset_manifest_lock = threading.RLock()
        self._anchor_manifest_lock = threading.RLock()
        self._outfit_reference_locks: dict[tuple[str, str], threading.Lock] = {}
        self._outfit_reference_locks_guard = threading.Lock()
        self.cast_by_id = {member.id: member for member in self.series_bible.cast}
        self.scene_by_id = {scene.id: scene for scene in self.series_bible.scenes}

        self.asset_manifest_path = self.metadata_dir / "02_asset_manifest.json"
        self.anchor_manifest_path = self.metadata_dir / "03_anchor_manifest.json"
        self.asset_manifest = self._load_manifest(
            self.asset_manifest_path,
            default={"characters": {}, "scenes": {}, "clip_outfits": {}, "outfit_refs": {}},
        )
        self.anchor_manifest = self._load_manifest(
            self.anchor_manifest_path,
            default={"shots": {}},
        )
        self.asset_manifest.setdefault("characters", {})
        self.asset_manifest.setdefault("scenes", {})
        self.asset_manifest.setdefault("clip_outfits", {})
        self.asset_manifest.setdefault("outfit_refs", {})

    def _thread_quality_client(self):
        client = getattr(self._quality_client_local, "client", None)
        if client is None:
            client = genai.Client(api_key=self._google_api_key)
            self._quality_client_local.client = client
        return client

    def _load_manifest(self, path: Path, default: dict) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        return default

    def _save_manifest(self, data: dict, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"💾 已保存: {path}")

    def _save_preproduction_manifests(self) -> None:
        with self._asset_manifest_lock, self._anchor_manifest_lock:
            self._save_manifest(self.asset_manifest, self.asset_manifest_path)
            self._save_manifest(self.anchor_manifest, self.anchor_manifest_path)

    def _get_primary_character_reference(self, char_id: str) -> Path | None:
        entry = self.asset_manifest.get("characters", {}).get(char_id, {})
        references = entry.get("references", [])
        if not references:
            return None
        return Path(references[0])

    def _selected_clip_outfit(self, clip: ClipPlan, char_id: str) -> str:
        outfit_text = str((clip.clip_character_outfits or {}).get(char_id, "")).strip()
        if outfit_text:
            return outfit_text
        member = self.cast_by_id.get(char_id)
        if member is None:
            return ""
        wardrobe_options = [option.strip() for option in member.wardrobe_options if option.strip()]
        if wardrobe_options:
            return wardrobe_options[0]
        return member.signature_outfit.strip()

    @staticmethod
    def _outfit_reference_key(char_id: str, outfit_text: str) -> str:
        digest = hashlib.sha256(f"{char_id}\n{outfit_text}".encode("utf-8")).hexdigest()[:16]
        return digest

    def _clip_outfit_reference_entry(self, clip: ClipPlan, char_id: str) -> dict:
        outfit_text = self._selected_clip_outfit(clip, char_id)
        outfit_key = self._outfit_reference_key(char_id, outfit_text)
        char_entry = self.asset_manifest.setdefault("outfit_refs", {}).setdefault(char_id, {})
        return char_entry.setdefault(outfit_key, {})

    def _clip_outfit_reference_path(self, clip: ClipPlan, char_id: str) -> Path | None:
        outfit_text = self._selected_clip_outfit(clip, char_id)
        outfit_key = self._outfit_reference_key(char_id, outfit_text)
        entry = self.asset_manifest.get("outfit_refs", {}).get(char_id, {}).get(outfit_key, {})
        path_text = entry.get("reference")
        if not path_text:
            return None
        return Path(path_text)

    def _clip_outfit_reference_prompt(self, clip: ClipPlan, char_id: str) -> str:
        member = self.cast_by_id[char_id]
        outfit_text = self._selected_clip_outfit(clip, char_id)
        return " ".join(
            [
                "Create a clean full-body character reference photo of the exact same adult person shown in the input reference image.",
                "Preserve face identity, age, hairstyle, body type, skin tone, and overall appearance precisely.",
                f"Dress this person exactly in this locked clip outfit: {outfit_text}.",
                "Keep the pose natural and front-facing with the full body visible.",
                "Use a plain neutral studio-style background and clear realistic lighting.",
                "Do not add props, extra people, text, collage layout, or dramatic action.",
                "This is a reusable single-character outfit reference for this exact character-and-outfit combination, so consistency matters more than style variation.",
            ]
        )

    def _record_clip_outfit_reference(
        self,
        clip: ClipPlan,
        char_id: str,
        selected_outfit: str,
        reference_path: Path | None,
        *,
        fallback_path: Path | None = None,
    ) -> None:
        clip_entry = self.asset_manifest.setdefault("clip_outfits", {}).setdefault(clip.id, {}).setdefault(char_id, {})
        clip_entry["selected_outfit"] = selected_outfit
        if reference_path is not None:
            clip_entry["reference"] = str(reference_path)
            clip_entry.pop("fallback_reference", None)
        elif fallback_path is not None:
            clip_entry.pop("reference", None)
            clip_entry["fallback_reference"] = str(fallback_path)

    @staticmethod
    def _normalize_outfit_map(raw_map: dict | None) -> dict[str, str]:
        return {
            str(char_id).strip(): str(outfit_text).strip()
            for char_id, outfit_text in (raw_map or {}).items()
            if str(char_id).strip() and str(outfit_text).strip()
        }

    def _expected_shot_outfit_map(self, clip: ClipPlan, shot: ShotPlan) -> dict[str, str]:
        return {
            char_id: outfit_text
            for char_id in shot.visible_characters
            if (outfit_text := self._selected_clip_outfit(clip, char_id))
        }

    def _clip_outfit_reference_is_current(
        self,
        clip: ClipPlan,
        char_id: str,
        expected_outfit: str,
    ) -> bool:
        outfit_key = self._outfit_reference_key(char_id, expected_outfit)
        entry = self.asset_manifest.get("outfit_refs", {}).get(char_id, {}).get(outfit_key, {})
        recorded_outfit = str(entry.get("selected_outfit", "")).strip()
        if recorded_outfit != expected_outfit:
            return False
        reference_path = entry.get("reference") or entry.get("fallback_reference")
        if not reference_path:
            return False
        return Path(reference_path).exists()

    def _invalidate_stale_shot_assets(self, clip: ClipPlan, shot: ShotPlan) -> None:
        expected_outfits = self._expected_shot_outfit_map(clip, shot)
        recorded_entry = self.anchor_manifest.get("shots", {}).get(shot.id, {})
        recorded_outfits = self._normalize_outfit_map(recorded_entry.get("clip_character_outfits", {}))

        missing_clip_outfit_refs = [
            char_id
            for char_id, outfit_text in expected_outfits.items()
            if not self._clip_outfit_reference_is_current(clip, char_id, outfit_text)
        ]

        reasons: list[str] = []
        if recorded_outfits != expected_outfits:
            reasons.append("clip outfits changed")
        if missing_clip_outfit_refs:
            reasons.append("clip outfit references missing or stale")
        recorded_method = str(recorded_entry.get("anchor_generation_method") or "").strip()
        if recorded_method != ANCHOR_GENERATION_METHOD:
            reasons.append("anchor generation method changed")

        if not reasons:
            return

        shot_dir = self.anchors_dir / clip.id / shot.id
        if shot_dir.exists():
            shutil.rmtree(shot_dir)

        cast_board_path = self.cast_boards_dir / f"{clip.id}_{shot.id}_cast_board.png"
        if cast_board_path.exists():
            cast_board_path.unlink()

        self.anchor_manifest.get("shots", {}).pop(shot.id, None)
        print(
            f"   ♻️  检测到 {shot.id} 的旧 anchor 资产不再可信，已清理重生："
            + ", ".join(reasons)
        )

    def _outfit_reference_lock(self, cache_key: tuple[str, str]) -> threading.Lock:
        with self._outfit_reference_locks_guard:
            return self._outfit_reference_locks.setdefault(cache_key, threading.Lock())

    def _ensure_clip_outfit_reference(self, clip: ClipPlan, char_id: str) -> Path | None:
        selected_outfit = self._selected_clip_outfit(clip, char_id)
        cache_key = (char_id, selected_outfit)
        with self._outfit_reference_lock(cache_key), self._asset_manifest_lock:
            return self._ensure_clip_outfit_reference_locked(
                clip=clip,
                char_id=char_id,
                selected_outfit=selected_outfit,
                cache_key=cache_key,
            )

    def _ensure_clip_outfit_reference_locked(
        self,
        clip: ClipPlan,
        char_id: str,
        selected_outfit: str,
        cache_key: tuple[str, str],
    ) -> Path | None:
        if cache_key in self._clip_outfit_reference_cache:
            reference = self._clip_outfit_reference_cache[cache_key]
            self._record_clip_outfit_reference(clip, char_id, selected_outfit, reference)
            return reference

        if not selected_outfit:
            reference = self._get_primary_character_reference(char_id)
            self._clip_outfit_reference_cache[cache_key] = reference
            self._record_clip_outfit_reference(clip, char_id, selected_outfit, reference)
            return reference

        primary_reference = self._get_primary_character_reference(char_id)
        if primary_reference is None or not primary_reference.exists():
            self._clip_outfit_reference_cache[cache_key] = None
            return None

        entry = self._clip_outfit_reference_entry(clip, char_id)
        existing_path = self._clip_outfit_reference_path(clip, char_id)
        if (
            existing_path is not None
            and existing_path.exists()
            and self.reuse_existing_assets
            and str(entry.get("selected_outfit", "")).strip() == selected_outfit
        ):
            self._clip_outfit_reference_cache[cache_key] = existing_path
            self._record_clip_outfit_reference(clip, char_id, selected_outfit, existing_path)
            return existing_path

        outfit_key = self._outfit_reference_key(char_id, selected_outfit)
        char_dir = self.outfit_refs_dir / char_id
        char_dir.mkdir(parents=True, exist_ok=True)
        save_path = char_dir / f"{outfit_key}.png"

        opened_reference: Image.Image | None = None
        try:
            opened_reference = Image.open(primary_reference)
            self.nanobanana.generate_with_mixed_content(
                contents=[
                    self._clip_outfit_reference_prompt(clip, char_id),
                    opened_reference,
                ],
                aspect_ratio="3:4",
                save_path=str(save_path),
                max_attempts=self.anchor_generation_retries,
                retry_wait_seconds=self.anchor_retry_wait_seconds,
            )
            entry["selected_outfit"] = selected_outfit
            entry["reference"] = str(save_path)
            entry.pop("fallback_reference", None)
            self._clip_outfit_reference_cache[cache_key] = save_path
            self._record_clip_outfit_reference(clip, char_id, selected_outfit, save_path)
            return save_path
        except Exception as exc:
            print(
                f"   ⚠️  clip outfit reference 生成失败，回退到通用角色参考图: {clip.id}/{char_id} - {exc}"
            )
            entry["selected_outfit"] = selected_outfit
            entry.pop("reference", None)
            entry["fallback_reference"] = str(primary_reference)
            self._clip_outfit_reference_cache[cache_key] = primary_reference
            self._record_clip_outfit_reference(
                clip,
                char_id,
                selected_outfit,
                None,
                fallback_path=primary_reference,
            )
            return primary_reference
        finally:
            if opened_reference is not None:
                opened_reference.close()

    def _reference_path_for_clip_character(self, clip: ClipPlan, char_id: str) -> Path | None:
        return self._ensure_clip_outfit_reference(clip, char_id)

    def _generate_character_reference_entry(self, member) -> tuple[str, dict]:
        char_dir = self.characters_dir / member.id
        char_dir.mkdir(parents=True, exist_ok=True)

        references: list[str] = []
        prompts = member.reference_photo_prompts[: self.character_reference_count]
        if not prompts:
            raise ValueError(f"角色 {member.id} 没有 reference_photo_prompts。")

        print(f"\n👤 角色: {member.name_en} ({member.id})")
        for index, prompt in enumerate(prompts, start=1):
            save_path = char_dir / f"{member.id}_ref_{index:02d}.png"
            if save_path.exists() and self.reuse_existing_assets:
                print(f"  ♻️  复用人物参考图: {save_path}")
            else:
                print(f"  🎨 生成人物参考图 {index}/{len(prompts)}")
                self.nanobanana.generate_image(
                    prompt=prompt.prompt,
                    aspect_ratio=prompt.aspect_ratio or "1:1",
                    response_modalities=["Image"],
                    save_path=str(save_path),
                )
            references.append(str(save_path))

        return member.id, {
            "name_en": member.name_en,
            "references": references,
        }

    def generate_character_references(self) -> None:
        print("\n" + "=" * 60)
        print("步骤 2/4: 生成人物参考图")
        print("=" * 60)

        workers = min(self.asset_generation_workers, max(1, len(self.series_bible.cast)))
        if workers > 1:
            print(f"🚀 并发生成/复用人物参考图: workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._generate_character_reference_entry, member)
                for member in self.series_bible.cast
            ]
            for future in as_completed(futures):
                char_id, entry = future.result()
                self.asset_manifest["characters"][char_id] = entry

        self._save_manifest(self.asset_manifest, self.asset_manifest_path)
        self.api_usage_logger.write_summary()

    def _generate_scene_reference_entry(self, scene) -> tuple[str, dict]:
        scene_dir = self.scenes_dir / scene.id
        scene_dir.mkdir(parents=True, exist_ok=True)

        references: list[str] = []
        print(f"\n🏠 场景: {scene.name_en} ({scene.id})")
        for index in range(1, self.scene_reference_count + 1):
            save_path = scene_dir / f"{scene.id}_bg_{index:02d}.png"
            if save_path.exists() and self.reuse_existing_assets:
                print(f"  ♻️  复用场景参考图: {save_path}")
            else:
                print(f"  🎨 生成场景参考图 {index}/{self.scene_reference_count}")
                self.nanobanana.generate_image(
                    prompt=scene.background_prompt,
                    aspect_ratio=scene.aspect_ratio or self.aspect_ratio,
                    response_modalities=["Image"],
                    save_path=str(save_path),
                )
            references.append(str(save_path))

        return scene.id, {
            "name_en": scene.name_en,
            "references": references,
        }

    def generate_scene_references(self) -> None:
        print("\n" + "=" * 60)
        print("步骤 3/4: 生成场景参考图")
        print("=" * 60)

        workers = min(self.asset_generation_workers, max(1, len(self.series_bible.scenes)))
        if workers > 1:
            print(f"🚀 并发生成/复用场景参考图: workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._generate_scene_reference_entry, scene)
                for scene in self.series_bible.scenes
            ]
            for future in as_completed(futures):
                scene_id, entry = future.result()
                self.asset_manifest["scenes"][scene_id] = entry

        self._save_manifest(self.asset_manifest, self.asset_manifest_path)
        self.api_usage_logger.write_summary()

    def _build_cast_board(self, clip: ClipPlan, shot: ShotPlan) -> Path | None:
        board_path = self.cast_boards_dir / f"{clip.id}_{shot.id}_cast_board.png"
        if board_path.exists() and self.reuse_existing_assets:
            return board_path

        visible_references: list[tuple[str, Path]] = []
        for char_id in shot.visible_characters[: self.max_visible_characters]:
            reference_path = self._reference_path_for_clip_character(clip, char_id)
            if reference_path is not None and reference_path.exists():
                visible_references.append((char_id, reference_path))

        if not visible_references:
            return None

        columns = 3 if len(visible_references) > 4 else 2 if len(visible_references) > 1 else 1
        rows = math.ceil(len(visible_references) / columns)
        cell_width = 420
        cell_height = 560
        label_height = 44
        gap = 18
        board_width = columns * cell_width + (columns + 1) * gap
        board_height = rows * (cell_height + label_height) + (rows + 1) * gap

        board = Image.new("RGB", (board_width, board_height), color=(247, 244, 238))
        draw = ImageDraw.Draw(board)
        font = ImageFont.load_default()

        for index, (char_id, reference_path) in enumerate(visible_references):
            row = index // columns
            col = index % columns
            x = gap + col * cell_width
            y = gap + row * (cell_height + label_height)

            with Image.open(reference_path) as source_image:
                portrait = ImageOps.contain(
                    source_image.convert("RGB"),
                    (cell_width, cell_height),
                    method=RESAMPLE,
                )
                portrait_canvas = Image.new("RGB", (cell_width, cell_height), color=(255, 255, 255))
                paste_x = (cell_width - portrait.width) // 2
                paste_y = (cell_height - portrait.height) // 2
                portrait_canvas.paste(portrait, (paste_x, paste_y))
                board.paste(portrait_canvas, (x, y))

            draw.rectangle(
                [x, y, x + cell_width, y + cell_height],
                outline=(220, 214, 206),
                width=2,
            )

            member = self.cast_by_id[char_id]
            draw.rectangle(
                [x, y + cell_height, x + cell_width, y + cell_height + label_height],
                fill=(255, 255, 255),
            )
            draw.text(
                (x + 14, y + cell_height + 12),
                member.name_en,
                fill=(32, 32, 32),
                font=font,
            )

        board.save(board_path)
        print(f"🧩 已生成 cast board: {board_path}")
        return board_path

    def _build_anchor_prompt(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        quality_feedback: str | None = None,
    ) -> str:
        reference_slot_lines = []
        for index, char_id in enumerate(shot.visible_characters, start=1):
            member = self.cast_by_id.get(char_id)
            if member is None:
                continue
            outfit_text = self._selected_clip_outfit(clip, char_id)
            reference_slot_lines.append(
                f"Visible person {index} = {member.name_en} ({char_id}). "
                f"Use the individual reference image labeled visible person {index} as the identity source. "
                "Match this person's face, age, gender, ethnicity, body type, hair, and signature accessories exactly"
                + (f"; locked outfit: {outfit_text}" if outfit_text else "")
                + "."
            )
        parts = [
            f"Create the opening keyframe for {shot.id}.",
            f"Exactly {len(shot.visible_characters)} named recurring characters are visible.",
            "Use album-style identity control: every visible recurring character is supplied as a separate individual reference image, and the generated frame must map each visible person slot to that exact reference.",
            "Do not average, merge, beautify, age-shift, ethnicity-shift, or replace any reference person. Do not drop signature glasses, baldness, hair shape, facial hair, body type, or other identity features.",
            "Render anonymous private people only. The faces must not resemble celebrities, public figures, famous people, or any recognizable real-person lookalike.",
            "Single coherent camera view only: no cutaway, no split-screen, no collage, no multi-panel layout, no picture-in-picture, no duplicated dashboard, no impossible car interior, no duplicated room geometry.",
            "The frame must be a plausible still from one real camera perspective that can become a short image-to-video shot.",
            f"Composition: {shot.composition}.",
            f"Camera language: {shot.camera_language}.",
            f"Blocking: {shot.blocking_notes}.",
            f"Motion budget hint: {shot.motion_budget}.",
            "All visible people must be adults. Do not depict children, teenagers, babies, toddlers, students, school gates, or playgrounds.",
            shot.anchor_image_prompt.strip(),
        ]

        if reference_slot_lines:
            parts.append(
                "Individual reference slot mapping. The following slot numbers correspond exactly to the individual reference images supplied after the scene image: "
                + " ".join(reference_slot_lines)
            )

        locked_outfits = []
        for char_id in shot.visible_characters:
            outfit_text = self._selected_clip_outfit(clip, char_id)
            if outfit_text:
                locked_outfits.append(f"{char_id}: {outfit_text}")
        if locked_outfits:
            parts.append(
                "Locked clip outfits for the visible recurring characters: "
                + " | ".join(locked_outfits)
                + "."
            )

        if shot.background_extras:
            parts.append(
                "Include only these unnamed background extras as secondary, non-featured people: "
                + "; ".join(extra.strip() for extra in shot.background_extras if extra.strip())
                + "."
            )
            parts.append("Background extras must remain visually generic, silent, and non-dominant.")
        else:
            parts.append("Do not add any extra unnamed people.")

        if shot.left_to_right_order:
            parts.append(
                "Left to right order: " + ", ".join(shot.left_to_right_order) + "."
            )

        if shot.visible_characters:
            parts.append(
                "Visible character ids in frame: " + ", ".join(shot.visible_characters) + "."
            )

        if self.default_negative_prompt:
            parts.append(f"Avoid: {self.default_negative_prompt}.")

        if quality_feedback:
            parts.append(
                "Previous candidate failed visual QA. Correct these issues in the regenerated image: "
                + quality_feedback.strip()
            )

        return " ".join(part for part in parts if part)

    def _build_mixed_contents(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        *,
        include_scene_reference: bool = True,
        include_cast_board: bool = False,
        include_all_visible_references: bool = True,
        include_focus_references: bool = False,
        max_focus_references: int = 0,
        quality_feedback: str | None = None,
    ) -> tuple[list, list[Image.Image], Path | None]:
        opened_images: list[Image.Image] = []
        contents: list = [self._build_anchor_prompt(clip, shot, quality_feedback=quality_feedback)]

        scene_refs = self.asset_manifest.get("scenes", {}).get(shot.scene_id, {}).get("references", [])
        scene_reference_path = Path(scene_refs[0]) if scene_refs else None
        if include_scene_reference and scene_reference_path and scene_reference_path.exists():
            scene_image = Image.open(scene_reference_path)
            opened_images.append(scene_image)
            contents.extend(
                [
                    "\nThis is the scene reference image.",
                    scene_image,
                    "End of scene reference.\n",
                ]
            )

        cast_board_path = self._build_cast_board(clip, shot)
        if include_cast_board and cast_board_path and cast_board_path.exists():
            cast_board_image = Image.open(cast_board_path)
            opened_images.append(cast_board_image)
            contents.extend(
                [
                    "\nThis is the cast board showing the exact people who must appear in frame.",
                    cast_board_image,
                    "End of cast board.\n",
                ]
            )

        visible_reference_ids = shot.visible_characters if include_all_visible_references else []
        for index, char_id in enumerate(visible_reference_ids, start=1):
            reference_path = self._reference_path_for_clip_character(clip, char_id)
            if reference_path and reference_path.exists():
                character_image = Image.open(reference_path)
                opened_images.append(character_image)
                member = self.cast_by_id[char_id]
                contents.extend(
                    [
                        f"\nThis is the individual identity and outfit reference of visible person {index}: {member.name_en} ({char_id}). Use this exact person for visible person {index} in the generated frame.",
                        character_image,
                        f"Visible person {index} reference end.\n",
                    ]
                )

        remaining_focus_ids = []
        if include_focus_references:
            remaining_focus_ids = [
                char_id for char_id in shot.focus_characters if char_id not in visible_reference_ids
            ][:max_focus_references]

        for char_id in remaining_focus_ids:
            reference_path = self._reference_path_for_clip_character(clip, char_id)
            if reference_path and reference_path.exists():
                focus_image = Image.open(reference_path)
                opened_images.append(focus_image)
                contents.extend(
                    [
                        f"\nThis is the face reference for {self.cast_by_id[char_id].name_en}.",
                        focus_image,
                        "End of face reference.\n",
                    ]
                )

        return contents, opened_images, cast_board_path

    def _content_variants_for_shot(self, shot: ShotPlan) -> list[tuple[str, dict]]:
        if self.anchor_fallback_mode == "retry_only":
            return [
                (
                    "album_style_scene_plus_individual_refs",
                    {
                        "include_scene_reference": True,
                        "include_cast_board": False,
                        "include_all_visible_references": True,
                        "include_focus_references": False,
                    },
                ),
                (
                    "album_style_individual_refs_only",
                    {
                        "include_scene_reference": False,
                        "include_cast_board": False,
                        "include_all_visible_references": True,
                        "include_focus_references": False,
                    },
                ),
                (
                    "album_style_scene_plus_individual_refs_retry",
                    {
                        "include_scene_reference": True,
                        "include_cast_board": False,
                        "include_all_visible_references": True,
                        "include_focus_references": False,
                    },
                ),
            ]

        dense_group = len(shot.visible_characters) >= 5
        variants: list[tuple[str, dict]] = []

        if dense_group:
            variants.extend(
                [
                    (
                        "album_style_full_mixed",
                        {
                            "include_scene_reference": True,
                            "include_cast_board": False,
                            "include_all_visible_references": True,
                            "include_focus_references": False,
                        },
                    ),
                    (
                        "album_style_individual_refs_only",
                        {
                            "include_scene_reference": False,
                            "include_cast_board": False,
                            "include_all_visible_references": True,
                            "include_focus_references": False,
                        },
                    ),
                ]
            )
        else:
            variants.extend(
                [
                    (
                        "album_style_full_mixed",
                        {
                            "include_scene_reference": True,
                            "include_cast_board": False,
                            "include_all_visible_references": True,
                            "include_focus_references": False,
                        },
                    ),
                    (
                        "album_style_individual_refs_only",
                        {
                            "include_scene_reference": False,
                            "include_cast_board": False,
                            "include_all_visible_references": True,
                            "include_focus_references": False,
                        },
                    ),
                ]
            )

        return variants

    def _extract_json_object(self, raw_text: str) -> dict[str, Any]:
        normalized = (raw_text or "").strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
            normalized = re.sub(r"\s*```$", "", normalized)

        candidates = [normalized]
        first = normalized.find("{")
        last = normalized.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidates.append(normalized[first : last + 1])

        for candidate in candidates:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Could not parse anchor quality-review JSON: {raw_text[:500]}")

    def _anchor_quality_prompt(self, clip: ClipPlan, shot: ShotPlan) -> str:
        cast_lines = []
        for char_id in shot.visible_characters:
            member = self.cast_by_id.get(char_id)
            outfit = self._selected_clip_outfit(clip, char_id)
            if member is None:
                cast_lines.append(f"- {char_id}: expected visible recurring adult")
                continue
            cast_lines.append(
                f"- {char_id} / {member.name_en}: {member.appearance_description}"
                + (f" Outfit: {outfit}" if outfit else "")
            )

        return f"""
You are a strict visual quality reviewer for an image-to-video anchor keyframe.
Decide whether the attached image is good enough to use as the opening frame for this shot.

Shot id: {shot.id}
Clip id: {clip.id}
Expected visible recurring character count: {len(shot.visible_characters)}
Expected visible recurring characters:
{chr(10).join(cast_lines) if cast_lines else "- none"}

Scene id: {shot.scene_id}
Composition: {shot.composition}
Camera language: {shot.camera_language}
Blocking: {shot.blocking_notes}
Left-to-right order: {", ".join(shot.left_to_right_order or [])}
Shot anchor prompt: {shot.anchor_image_prompt}

Quality priorities:
- Treat named character identity, primary task/action, scene, and single coherent camera view as the most important checks.
- Compare every visible person against the expected recurring character list. Age, gender, ethnicity, face shape, body type, baldness/hair, glasses, facial hair, and locked outfit must match well enough that the character is recognizable.
- Treat left-to-right order, exact gaze direction, and small hand pose issues as soft constraints unless they make the shot action wrong or confuse which character is doing the task.

Hard-fail if any of these are present:
- split-screen, collage, multi-panel, cutaway, picture-in-picture, or multiple camera views in one image
- impossible or duplicated spatial layout, especially duplicated dashboards, duplicated car interiors, warped rooms, or disconnected foreground/background planes
- the primary task/action is clearly wrong for the shot
- any named recurring character is replaced by a different-looking person, wrong age group, wrong gender, or wrong ethnicity
- a signature identity feature is missing or clearly wrong, including required glasses, baldness, distinctive hair shape, facial hair, or strongly distinctive body type
- a locked outfit is completely wrong for a named recurring character
- featured children/teenagers/minors
- unreadable or dominant recurring characters when the shot requires named people
- severe face/body deformation, duplicated main characters, or impossible object geometry

Do NOT fail solely for minor unnamed background extras. Only mention extras if they dominate the frame or confuse the main action.
Do NOT fail solely for left-to-right order, seating order, or exact gaze direction when identities, scene, primary task, and single-view composition are otherwise substantially correct; put those in minor_issues.

Return only JSON with this schema:
{{
  "approved": true or false,
  "score": number from 0 to 1,
  "blocking_issues": ["short issue", "..."],
  "minor_issues": ["short issue", "..."],
  "matches_scene": true or false,
  "matches_primary_action": true or false,
  "single_coherent_view": true or false,
  "regeneration_feedback": "one concise paragraph telling the image model exactly what to fix"
}}
""".strip()

    def _assess_anchor_candidate(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        candidate_path: Path,
        *,
        attempt: int | None = None,
        reused: bool = False,
    ) -> dict[str, Any]:
        if not self.anchor_quality_check_enabled:
            return {
                "approved": True,
                "score": 1.0,
                "skipped": True,
                "candidate_path": str(candidate_path),
            }

        prompt = self._anchor_quality_prompt(clip, shot)
        response = None
        last_error: Exception | None = None
        max_attempts = max(2, int(self.anchor_generation_retries or 1))
        for quality_api_attempt in range(1, max_attempts + 1):
            try:
                with Image.open(candidate_path) as image:
                    response = self._thread_quality_client().models.generate_content(
                        model=self.anchor_quality_model,
                        contents=[prompt, image],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                break
            except Exception as exc:
                last_error = exc
                if quality_api_attempt < max_attempts:
                    print(
                        f"   ⚠️  关键帧质检调用失败 "
                        f"({quality_api_attempt}/{max_attempts}): {exc}"
                    )
                    time.sleep(self.anchor_retry_wait_seconds)
                else:
                    raise

        if response is None:
            raise RuntimeError(f"关键帧质检未获得有效响应: {last_error}")

        self.api_usage_logger.record_response(
            response=response,
            operation="anchor_quality_check",
            model=self.anchor_quality_model,
            prompt=prompt,
            attempt=attempt,
            extra={
                "clip_id": clip.id,
                "shot_id": shot.id,
                "candidate_path": str(candidate_path),
                "reused": reused,
            },
        )

        parsed = self._extract_json_object(getattr(response, "text", "") or "")
        parsed["candidate_path"] = str(candidate_path)
        parsed["clip_id"] = clip.id
        parsed["shot_id"] = shot.id
        parsed["attempt"] = attempt
        parsed["reused"] = reused
        parsed["model_approved"] = parsed.get("approved")
        parsed["approved"] = self._anchor_quality_passes(parsed)
        return parsed

    def _anchor_quality_passes(self, result: dict[str, Any]) -> bool:
        approved = bool(result.get("approved"))
        try:
            score = float(result.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        blocking_issues = result.get("blocking_issues") or []
        single_view = result.get("single_coherent_view")
        primary_action = result.get("matches_primary_action")
        scene_ok = result.get("matches_scene")
        if self._has_hard_anchor_quality_issue(result):
            return False
        strict_pass = (
            approved
            and score >= self.anchor_quality_min_score
            and single_view is not False
            and primary_action is not False
            and scene_ok is not False
        )
        if strict_pass:
            return True

        return self._soft_rejection_is_acceptable(
            result,
            score=score,
            single_view=single_view,
            primary_action=primary_action,
            scene_ok=scene_ok,
            blocking_issues=blocking_issues,
        )

    def _anchor_quality_issue_text(self, result: dict[str, Any]) -> str:
        return " ".join(
            str(value)
            for key in ("blocking_issues", "minor_issues", "regeneration_feedback")
            for value in (
                result.get(key)
                if isinstance(result.get(key), list)
                else [result.get(key)]
            )
            if value
        ).lower()

    def _has_hard_anchor_quality_issue(self, result: dict[str, Any]) -> bool:
        text = self._anchor_quality_issue_text(result)
        hard_terms = (
            "identity mismatch",
            "character identity",
            "wrong identity",
            "wrong person",
            "different person",
            "does not match",
            "doesn't match",
            "not match the reference",
            "replaced by",
            "wrong age",
            "too young",
            "too old",
            "young man",
            "young woman",
            "wrong gender",
            "man instead of a woman",
            "woman instead of a man",
            "wrong ethnicity",
            "caucasian instead of east asian",
            "missing his signature",
            "missing her signature",
            "missing signature",
            "signature glasses",
            "missing glasses",
            "without glasses",
            "not wearing glasses",
            "wire-rimmed glasses",
            "baldness",
            "bald on top",
            "full head of hair",
            "facial hair",
            "body type",
            "completely incorrect outfit",
            "completely wrong outfit",
            "wrong outfit",
            "wrong clothing",
            "severe character",
        )
        return any(term in text for term in hard_terms)

    def _soft_rejection_is_acceptable(
        self,
        result: dict[str, Any],
        *,
        score: float,
        single_view: Any,
        primary_action: Any,
        scene_ok: Any,
        blocking_issues: list[Any],
    ) -> bool:
        if score < max(0.82, self.anchor_quality_min_score):
            return False
        if single_view is False or primary_action is False or scene_ok is False:
            return False

        text = self._anchor_quality_issue_text(result)
        hard_terms = (
            "split-screen",
            "collage",
            "multi-panel",
            "cutaway",
            "picture-in-picture",
            "impossible",
            "duplicated dashboard",
            "duplicated car",
            "warped",
            "disconnected",
            "primary task",
            "wrong task",
            "wrong action",
            "completely missing",
            "is missing from",
            "not visible",
            "absent",
            "wrong gender",
            "man instead of a woman",
            "woman instead of a man",
            "child",
            "teen",
            "minor",
            "severe",
            "deformation",
            "duplicated main",
        )
        if any(term in text for term in hard_terms) or self._has_hard_anchor_quality_issue(result):
            return False

        soft_terms = (
            "left-to-right",
            "left to right",
            "order",
            "seating",
            "seat",
            "gaze",
            "looking",
            "outfit",
            "wardrobe",
            "hair",
            "glasses",
            "hand",
            "gesture",
            "minor",
        )
        return bool(blocking_issues) and any(term in text for term in soft_terms)

    def _quality_feedback_text(self, result: dict[str, Any]) -> str:
        pieces = []
        for key in ("blocking_issues", "minor_issues"):
            values = result.get(key) or []
            pieces.extend(str(value).strip() for value in values if str(value).strip())
        feedback = str(result.get("regeneration_feedback") or "").strip()
        if feedback:
            pieces.append(feedback)
        if not pieces:
            pieces.append(
                "Create a simpler, more coherent single-camera keyframe that matches the shot action and blocking exactly."
            )
        return " ".join(pieces)[:1200]

    def _validate_existing_candidate(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        candidate_path: Path,
        selected_path: Path,
        quality_records: list[dict[str, Any]],
    ) -> bool:
        if not self.anchor_quality_check_enabled or not self.anchor_quality_check_existing:
            return True
        try:
            result = self._assess_anchor_candidate(
                clip,
                shot,
                candidate_path,
                reused=True,
            )
            quality_records.append(result)
            if self._anchor_quality_passes(result):
                print(
                    f"   ✅ 复用关键帧质量通过: score={float(result.get('score') or 0):.2f}"
                )
                return True

            print(
                "   ⚠️  旧关键帧质量未通过，删除并重生: "
                + self._quality_feedback_text(result)[:300]
            )
        except Exception as exc:
            print(f"   ⚠️  旧关键帧质检失败，删除并重生: {exc}")

        candidate_path.unlink(missing_ok=True)
        selected_path.unlink(missing_ok=True)
        return False

    def _generate_anchor_candidate(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        candidate_path: Path,
    ) -> tuple[Path | None, list[dict[str, Any]]]:
        cast_board_path: Path | None = None
        variants = self._content_variants_for_shot(shot)
        quality_records: list[dict[str, Any]] = []
        quality_feedback: str | None = None
        generated_anchor_count = 0
        max_anchor_generations = max(1, int(self.anchor_quality_max_attempts))
        last_error: Exception | None = None

        for variant_index, (variant_name, variant_kwargs) in enumerate(variants, start=1):
            if generated_anchor_count >= max_anchor_generations:
                break
            print(
                f"   🧪 尝试关键帧方案 {variant_index}/{len(variants)}: {variant_name}"
            )
            while generated_anchor_count < max_anchor_generations:
                quality_attempt = generated_anchor_count + 1
                contents, opened_images, current_cast_board_path = self._build_mixed_contents(
                    clip,
                    shot,
                    quality_feedback=quality_feedback,
                    **variant_kwargs,
                )
                cast_board_path = current_cast_board_path or cast_board_path
                try:
                    if quality_attempt > 1:
                        print(
                            f"   🔁 根据质检反馈重生关键帧 {quality_attempt}/{max_anchor_generations}"
                        )
                    generated_images = self.nanobanana.generate_with_mixed_content(
                        contents=contents,
                        aspect_ratio=self.aspect_ratio,
                        save_path=str(candidate_path),
                        max_attempts=self.anchor_generation_retries,
                        retry_wait_seconds=self.anchor_retry_wait_seconds,
                    )
                    if not (generated_images and candidate_path.exists()):
                        raise RuntimeError(
                            f"关键帧方案 {variant_name} 未返回任何图片。"
                        )
                    generated_anchor_count += 1

                    result = self._assess_anchor_candidate(
                        clip,
                        shot,
                        candidate_path,
                        attempt=quality_attempt,
                    )
                    quality_records.append(result)
                    if self._anchor_quality_passes(result):
                        print(
                            f"   ✅ 关键帧质检通过: score={float(result.get('score') or 0):.2f}"
                        )
                        return cast_board_path, quality_records

                    quality_feedback = self._quality_feedback_text(result)
                    print(
                        "   ⚠️  关键帧质检未通过: "
                        + quality_feedback[:300]
                    )
                    if generated_anchor_count < max_anchor_generations:
                        candidate_path.unlink(missing_ok=True)
                        time.sleep(2)
                        continue
                    print(
                        f"   ⚠️  已达到每个 shot 最多 {max_anchor_generations} 张 anchor image 上限，"
                        "保留最后一张候选并继续。"
                    )
                    return cast_board_path, quality_records
                except Exception as exc:
                    last_error = exc
                    print(f"   ⚠️  关键帧方案失败: {variant_name} - {exc}")
                    if generated_anchor_count >= max_anchor_generations:
                        if candidate_path.exists():
                            print(
                                f"   ⚠️  已达到每个 shot 最多 {max_anchor_generations} 张 anchor image 上限，"
                                "保留最后一张候选并继续。"
                            )
                            return cast_board_path, quality_records
                        raise RuntimeError(
                            f"Shot {shot.id} 已达到每个 shot 最多 {max_anchor_generations} 张 anchor image 上限，"
                            f"但没有可保留的候选图。最后错误: {exc}"
                        ) from exc
                    candidate_path.unlink(missing_ok=True)
                    if variant_index < len(variants):
                        if self.anchor_fallback_mode == "retry_only":
                            print("   ↩️  将继续使用同一套输入重试。")
                        else:
                            print("   ↩️  降低 mixed content 复杂度后继续尝试...")
                        time.sleep(2)
                        break
                    else:
                        raise RuntimeError(
                            f"Shot {shot.id} 的所有关键帧生成方案都失败了。最后错误: {exc}"
                        ) from exc
                finally:
                    for image in opened_images:
                        image.close()

        if last_error is not None:
            raise RuntimeError(
                f"Shot {shot.id} 未能在每个 shot 最多 {max_anchor_generations} 张 anchor image 上限内通过质检。"
                f"最后错误: {last_error}"
            ) from last_error
        return cast_board_path, quality_records

    def _generate_anchor_for_shot(self, clip: ClipPlan, shot: ShotPlan) -> tuple[str, dict[str, Any]]:
        with self._anchor_manifest_lock:
            self._invalidate_stale_shot_assets(clip, shot)
            previous_entry = self.anchor_manifest.get("shots", {}).get(shot.id, {})

        shot_dir = self.anchors_dir / clip.id / shot.id
        shot_dir.mkdir(parents=True, exist_ok=True)

        candidate_paths: list[str] = []
        quality_records: list[dict[str, Any]] = []
        selected_path = shot_dir / "selected.png"
        cast_board_path = self._build_cast_board(clip, shot)
        regenerated_after_quality_failure = False

        print(f"\n🖼️  Shot: {clip.id}/{shot.id}")
        print(f"   人数: {len(shot.visible_characters)}")
        print(f"   场景: {shot.scene_id}")

        for index in range(1, self.anchor_candidates_per_shot + 1):
            candidate_path = shot_dir / f"candidate_{index:02d}.png"
            if candidate_path.exists() and self.reuse_existing_assets:
                if self._validate_existing_candidate(
                    clip,
                    shot,
                    candidate_path,
                    selected_path,
                    quality_records,
                ):
                    print(f"   ♻️  复用关键帧候选: {candidate_path.name}")
                    candidate_paths.append(str(candidate_path))
                    continue
                regenerated_after_quality_failure = True

            current_cast_board_path, generated_quality_records = self._generate_anchor_candidate(
                clip,
                shot,
                candidate_path,
            )
            cast_board_path = current_cast_board_path or cast_board_path
            quality_records.extend(generated_quality_records)

            candidate_paths.append(str(candidate_path))

        if not candidate_paths:
            raise RuntimeError(f"Shot {shot.id} 没有生成出任何关键帧。")

        if not selected_path.exists() or not self.reuse_existing_assets:
            shutil.copyfile(candidate_paths[0], selected_path)

        selected_quality = None
        quality_records_for_entry = quality_records or list(
            previous_entry.get("anchor_quality_checks") or []
        )
        if quality_records:
            selected_quality = next(
                (
                    record
                    for record in reversed(quality_records)
                    if self._anchor_quality_passes(record)
                ),
                quality_records[-1],
            )
        elif isinstance(previous_entry, dict):
            selected_quality = previous_entry.get("selected_quality")

        total_anchor_generations = self._anchor_generation_count_from_records(
            quality_records_for_entry,
            candidate_paths,
        )
        if isinstance(previous_entry, dict):
            try:
                total_anchor_generations = max(
                    total_anchor_generations,
                    int(previous_entry.get("total_anchor_generations") or 0),
                )
            except (TypeError, ValueError):
                pass

        entry = {
            "anchor_generation_method": ANCHOR_GENERATION_METHOD,
            "clip_id": clip.id,
            "strategy": clip.strategy,
            "scene_id": shot.scene_id,
            "visible_characters": shot.visible_characters,
            "clip_character_outfits": {
                char_id: self._selected_clip_outfit(clip, char_id)
                for char_id in shot.visible_characters
                if self._selected_clip_outfit(clip, char_id)
            },
            "cast_board_path": str(cast_board_path) if cast_board_path else None,
            "candidates": candidate_paths,
            "selected_anchor": str(selected_path),
            "anchor_quality_checks": quality_records_for_entry,
            "selected_quality": selected_quality,
            "total_anchor_generations": total_anchor_generations,
            "quality_regenerated": regenerated_after_quality_failure
            or any(not self._anchor_quality_passes(record) for record in quality_records),
        }
        print(f"   ✅ 默认关键帧: {selected_path}")
        return shot.id, entry

    @staticmethod
    def _anchor_generation_count_from_records(
        quality_records: list[dict[str, Any]],
        candidate_paths: list[str] | None = None,
    ) -> int:
        generated_count = sum(
            1
            for record in quality_records
            if isinstance(record, dict) and not record.get("reused")
        )
        if generated_count:
            return generated_count
        return max(1, len(candidate_paths or []))

    @staticmethod
    def _anchor_generation_count_from_entry(entry: dict[str, Any] | None) -> int:
        if not isinstance(entry, dict):
            return 0
        try:
            explicit_total = int(entry.get("total_anchor_generations") or 0)
        except (TypeError, ValueError):
            explicit_total = 0
        if explicit_total > 0:
            return explicit_total
        return VideoPreproductionBuilder._anchor_generation_count_from_records(
            list(entry.get("anchor_quality_checks") or []),
            list(entry.get("candidates") or []),
        )

    def regenerate_anchor_for_shot(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        *,
        reason: str = "",
        previous_generation_count: int | None = None,
    ) -> dict[str, Any]:
        """Force-regenerate one shot anchor and persist the updated manifest."""
        old_entry = self.anchor_manifest.get("shots", {}).get(shot.id)
        previous_count = (
            previous_generation_count
            if previous_generation_count is not None
            else self._anchor_generation_count_from_entry(old_entry)
        )
        shot_dir = self.anchors_dir / clip.id / shot.id

        print(
            f"   🔁 重生 {clip.id}/{shot.id} 的 anchor"
            + (f"：{reason}" if reason else "")
        )
        with self._anchor_manifest_lock:
            self.anchor_manifest.setdefault("shots", {}).pop(shot.id, None)
            if shot_dir.exists():
                shutil.rmtree(shot_dir)

        shot_id, entry = self._generate_anchor_for_shot(clip, shot)
        new_count = self._anchor_generation_count_from_entry(entry)
        entry["total_anchor_generations"] = max(0, int(previous_count or 0)) + new_count
        entry["render_rescue_regenerated"] = True
        if reason:
            entry["render_rescue_reason"] = reason

        with self._anchor_manifest_lock:
            self.anchor_manifest.setdefault("shots", {})[shot_id] = entry
        self._save_preproduction_manifests()
        return entry

    def generate_anchor_images(self) -> None:
        print("\n" + "=" * 60)
        print("步骤 4/4: 生成视频关键帧")
        print("=" * 60)

        shot_tasks = [
            (clip, shot)
            for clip in self.series_bible.clips
            for shot in clip.shots
        ]
        if not shot_tasks:
            self._save_preproduction_manifests()
            self.api_usage_logger.write_summary()
            return

        workers = min(self.anchor_generation_workers, len(shot_tasks))
        if workers > 1:
            print(f"🚀 并发生成/复用 shot anchor: workers={workers}, total_shots={len(shot_tasks)}")

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_iter = iter(shot_tasks)
            pending = {}

            def submit_next() -> None:
                try:
                    clip, shot = next(task_iter)
                except StopIteration:
                    return
                future = executor.submit(self._generate_anchor_for_shot, clip, shot)
                pending[future] = (clip, shot)

            for _ in range(workers):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    clip, shot = pending.pop(future)
                    shot_id, entry = future.result()
                    completed += 1
                    with self._anchor_manifest_lock:
                        self.anchor_manifest["shots"][shot_id] = entry
                    # Persist incrementally so a long image-generation run can resume
                    # without discarding already-generated anchors after an API hang.
                    self._save_preproduction_manifests()
                    print(f"✅ Anchor 进度: {completed}/{len(shot_tasks)} ({clip.id}/{shot.id})")
                    submit_next()

        self._save_preproduction_manifests()
        self.api_usage_logger.write_summary()

    def run(self) -> None:
        self.generate_character_references()
        self.generate_scene_references()
        self.generate_anchor_images()


def load_anchor_manifest(output_root: str | Path) -> dict:
    path = Path(output_root) / "metadata" / "03_anchor_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到 anchor manifest: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
