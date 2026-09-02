from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini.libs import GeminiVideoProcessor
from project_config import get_video_api_key, get_video_model, load_video_config
from video_generator.api_usage import ApiUsageLogger
from video_generator.schemas import ClipPlan, SeriesBible, ShotPlan


VALID_VEO_DURATIONS = (4, 6, 8)


@dataclass
class RenderedSegment:
    path: Path
    video_ref: object | None = None


class ShotRenderSkipped(RuntimeError):
    """Raised when a shot is intentionally skipped after bounded retries."""


class VideoClipRenderer:
    """Render planned clips from prepared anchor images."""

    def __init__(
        self,
        series_bible: SeriesBible,
        output_root: str | Path,
        anchor_manifest: dict,
        config_path: str | None = None,
        video_render_workers: int | None = None,
    ) -> None:
        self.series_bible = series_bible
        self.output_root = Path(output_root)
        self.anchor_manifest = anchor_manifest
        self.config_path = config_path
        self.config = load_video_config(config_path)

        production_cfg = self.config.get("production", {}) if isinstance(
            self.config.get("production"), dict
        ) else {}
        prompting_cfg = (
            self.config.get("prompting", {}) if isinstance(self.config.get("prompting"), dict) else {}
        )

        self.aspect_ratio = str(production_cfg.get("aspect_ratio", "16:9"))
        self.resolution = str(production_cfg.get("resolution", "720p"))
        self.reuse_existing_assets = bool(production_cfg.get("reuse_existing_assets", True))
        self.default_shot_duration = int(production_cfg.get("shot_duration_seconds", 6))
        self.video_generation_retries = max(1, int(production_cfg.get("video_generation_retries", 2)))
        self.video_retry_wait_seconds = max(0, int(production_cfg.get("video_retry_wait_seconds", 10)))
        self.render_regenerate_anchor_on_rai = bool(
            production_cfg.get("render_regenerate_anchor_on_rai", True)
        )
        self.skip_failed_shots = bool(production_cfg.get("skip_failed_shots", True))
        self.anchor_quality_max_attempts = max(
            1,
            int(production_cfg.get("anchor_quality_max_attempts", 2)),
        )
        self.allowed_shot_durations = tuple(
            sorted(
                {
                    int(value)
                    for value in production_cfg.get(
                        "allowed_shot_durations_seconds", VALID_VEO_DURATIONS
                    )
                }
            )
        )
        self.person_generation_image_to_video = production_cfg.get(
            "person_generation_image_to_video", "allow_adult"
        )
        self.person_generation_extension = production_cfg.get(
            "person_generation_extension", "allow_all"
        )
        self.generate_audio = bool(production_cfg.get("generate_audio", False))
        self.video_render_workers = max(
            1,
            int(
                video_render_workers
                if video_render_workers is not None
                else production_cfg.get("video_render_workers", 1)
            ),
        )
        self.default_negative_prompt = (
            prompting_cfg.get("negative_prompt") if isinstance(prompting_cfg.get("negative_prompt"), str) else None
        )

        self.video_model = get_video_model() or "veo-3.1-fast-generate-preview"

        self.renders_dir = self.output_root / "renders"
        self.segments_dir = self.renders_dir / "segments"
        self.clips_dir = self.renders_dir / "clips"
        self.metadata_dir = self.output_root / "metadata"
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.render_manifest_path = self.metadata_dir / "04_render_manifest.json"
        self.render_manifest = self._load_manifest()

        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self._render_manifest_lock = threading.RLock()
        self._clip_skipped_shots: dict[str, list[dict[str, str]]] = {}
        self._clip_skipped_shots_lock = threading.RLock()
        self._processor_local = threading.local()
        self.processor = self._make_processor()
        self._processor_local.processor = self.processor
        self.clip_by_id = {clip.id: clip for clip in self.series_bible.clips}
        self.clip_by_shot_id = {
            shot.id: clip
            for clip in self.series_bible.clips
            for shot in clip.shots
        }

    def _make_processor(self) -> GeminiVideoProcessor:
        return GeminiVideoProcessor(
            genai.Client(api_key=get_video_api_key()),
            output_dir=str(self.renders_dir),
            usage_logger=self.api_usage_logger,
        )

    def _thread_processor(self) -> GeminiVideoProcessor:
        processor = getattr(self._processor_local, "processor", None)
        if processor is None:
            processor = self._make_processor()
            self._processor_local.processor = processor
        return processor

    def _load_manifest(self) -> dict:
        if self.render_manifest_path.exists():
            with open(self.render_manifest_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {"clips": {}}

    def _save_manifest(self) -> None:
        with open(self.render_manifest_path, "w", encoding="utf-8") as file:
            json.dump(self.render_manifest, file, indent=2, ensure_ascii=False)
        print(f"💾 已保存: {self.render_manifest_path}")

    def _anchor_signature_for_shot(self, shot: ShotPlan) -> dict:
        entry = self.anchor_manifest.get("shots", {}).get(shot.id, {})
        selected_anchor = entry.get("selected_anchor")
        if not selected_anchor:
            raise FileNotFoundError(f"Shot {shot.id} 缺少 selected anchor。")

        anchor_path = Path(selected_anchor)
        if not anchor_path.exists():
            raise FileNotFoundError(f"Shot {shot.id} 的 selected anchor 不存在: {anchor_path}")

        stat = anchor_path.stat()
        return {
            "selected_anchor": str(anchor_path),
            "anchor_generation_method": entry.get("anchor_generation_method"),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    def _anchor_signature_for_clip(self, clip: ClipPlan) -> dict[str, dict]:
        return {
            shot.id: self._anchor_signature_for_shot(shot)
            for shot in clip.shots
        }

    def _rendered_clip_is_current(self, clip: ClipPlan, final_clip_path: Path) -> bool:
        if not (self.reuse_existing_assets and final_clip_path.exists()):
            return False

        entry = self.render_manifest.get("clips", {}).get(clip.id, {})
        if entry.get("output_path") != str(final_clip_path):
            return False

        try:
            return entry.get("anchor_signature") == self._anchor_signature_for_clip(clip)
        except FileNotFoundError:
            return False

    def _discard_stale_clip_render_outputs(
        self,
        clip: ClipPlan,
        final_clip_path: Path,
        clip_segment_dir: Path,
    ) -> None:
        if final_clip_path.exists():
            final_clip_path.unlink(missing_ok=True)
        if clip_segment_dir.exists():
            shutil.rmtree(clip_segment_dir)
        with self._render_manifest_lock:
            if clip.id in self.render_manifest.get("clips", {}):
                self.render_manifest["clips"].pop(clip.id, None)
                self._save_manifest()

    def _selected_anchor_for(self, shot: ShotPlan) -> Path:
        entry = self.anchor_manifest.get("shots", {}).get(shot.id)
        if not entry or not entry.get("selected_anchor"):
            raise FileNotFoundError(f"Shot {shot.id} 缺少 selected anchor。")
        return Path(entry["selected_anchor"])

    def _anchor_candidates_for(self, shot: ShotPlan) -> list[Path]:
        entry = self.anchor_manifest.get("shots", {}).get(shot.id)
        if not entry:
            raise FileNotFoundError(f"Shot {shot.id} 缺少 anchor manifest 记录。")

        ordered_paths: list[Path] = []
        seen: set[str] = set()

        selected = entry.get("selected_anchor")
        if selected:
            selected_path = str(Path(selected))
            if selected_path not in seen:
                ordered_paths.append(Path(selected))
                seen.add(selected_path)

        for candidate in entry.get("candidates", []):
            candidate_path = str(Path(candidate))
            if candidate_path not in seen:
                ordered_paths.append(Path(candidate))
                seen.add(candidate_path)

        existing_paths = [path for path in ordered_paths if path.exists()]
        if not existing_paths:
            raise FileNotFoundError(f"Shot {shot.id} 没有可用的 anchor 图片。")
        return existing_paths

    def _normalize_duration(self, duration_seconds: int) -> int:
        if duration_seconds not in self.allowed_shot_durations:
            raise ValueError(
                f"Unsupported shot duration {duration_seconds}. "
                f"This workflow supports only {self.allowed_shot_durations}s shots."
            )
        return duration_seconds

    def _character_label(self, char_id: str) -> str:
        member = self._find_cast_member(char_id)
        if member is None:
            return "the person"
        return self._character_descriptor(member)

    def _find_cast_member(self, char_id: str):
        for member in self.series_bible.cast:
            if member.id == char_id:
                return member
        return None

    def _clip_for_shot(self, shot: ShotPlan) -> ClipPlan | None:
        return self.clip_by_shot_id.get(shot.id)

    def _selected_clip_outfit(self, clip: ClipPlan | None, char_id: str) -> str:
        if clip is not None:
            outfit_text = str((clip.clip_character_outfits or {}).get(char_id, "")).strip()
            if outfit_text:
                return outfit_text
        member = self._find_cast_member(char_id)
        if member is None:
            return ""
        wardrobe_options = [option.strip() for option in getattr(member, "wardrobe_options", []) if option.strip()]
        if wardrobe_options:
            return wardrobe_options[0]
        return str(getattr(member, "signature_outfit", "") or "").strip()

    def _selected_outfit_for_shot(self, shot: ShotPlan, char_id: str) -> str:
        return self._selected_clip_outfit(self._clip_for_shot(shot), char_id)

    def _character_label_with_outfit(self, shot: ShotPlan, char_id: str) -> str:
        label = self._character_label(char_id)
        outfit = self._selected_outfit_for_shot(shot, char_id)
        if outfit:
            return f"{label} wearing {outfit}"
        return label

    @staticmethod
    def _gender_noun(gender: str | None) -> str:
        lowered = (gender or "").strip().lower()
        if lowered == "male":
            return "man"
        if lowered == "female":
            return "woman"
        return "person"

    def _character_descriptor(self, member) -> str:
        text = " ".join(
            part
            for part in [
                getattr(member, "appearance_description", ""),
                getattr(member, "signature_outfit", ""),
                " ".join(getattr(member, "wardrobe_options", []) or []),
            ]
            if part
        ).lower()

        adjectives: list[str] = []
        if "older" in text or getattr(member, "age", 0) >= 60:
            adjectives.append("older")
        for token in ("stocky", "slender", "tall", "skinny", "plump", "athletic", "elegant"):
            if token in text and token not in adjectives:
                adjectives.append(token)
        if not adjectives and "gaunt" in text:
            adjectives.append("skinny")

        noun = self._gender_noun(getattr(member, "gender", None))
        feature = ""
        if "goatee" in text:
            feature = "with a goatee"
        elif "bald" in text:
            feature = "with a bald head"
        elif "glasses" in text:
            feature = "with glasses"
        elif "trench coat" in text:
            feature = "in a trench coat"
        elif "denim jacket" in text:
            feature = "in a denim jacket"
        elif "cardigan" in text:
            feature = "in a cardigan"
        elif "wire-rimmed" in text:
            feature = "with glasses"

        prefix = " ".join(adjectives[:2]).strip()
        if prefix:
            descriptor = f"the {prefix} {noun}"
        else:
            descriptor = f"the {noun}"
        if feature:
            descriptor = f"{descriptor} {feature}"
        return descriptor

    def _name_variants_for_member(self, member) -> list[str]:
        variants: list[str] = []
        full_name = getattr(member, "name_en", "").strip()
        if full_name:
            variants.append(full_name)
            parts = [part for part in re.split(r"\s+", full_name) if part]
            variants.extend(parts)
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            key = variant.lower()
            if key not in seen:
                deduped.append(variant)
                seen.add(key)
        return deduped

    def _speaker_descriptor(self, speaker_name: str, shot: ShotPlan | None = None) -> str | None:
        normalized = speaker_name.strip().lower()
        if not normalized:
            return None
        candidate_ids = set(getattr(shot, "visible_characters", []) or []) if shot is not None else None
        for member in self.series_bible.cast:
            if candidate_ids is not None and member.id not in candidate_ids:
                continue
            variants = [variant.lower() for variant in self._name_variants_for_member(member)]
            if normalized in variants:
                descriptor = self._character_descriptor(member)
                if shot is not None:
                    outfit = self._selected_outfit_for_shot(shot, member.id)
                    if outfit:
                        return f"{descriptor} wearing {outfit}"
                return descriptor
        return None

    def _anonymize_prompt_text(self, prompt: str, shot: ShotPlan | None = None) -> str:
        anonymized = prompt or ""
        # Veo can mistake ordinary fictional names for real-person references,
        # especially when the prompt mentions screens, calls, portraits, or faces.
        # Always scrub the entire recurring cast, not just visible characters.
        members = list(self.series_bible.cast)
        replacements: list[tuple[str, str]] = []
        for member in members:
            descriptor = self._character_descriptor(member)
            for variant in self._name_variants_for_member(member):
                replacements.append((variant, descriptor))

        replacements.sort(key=lambda item: len(item[0]), reverse=True)
        for name, descriptor in replacements:
            anonymized = re.sub(rf"\b{re.escape(name)}\b", descriptor, anonymized, flags=re.IGNORECASE)

        anonymized = re.sub(r"\s+", " ", anonymized).strip()
        return anonymized

    def _finalize_veo_runtime_prompt(self, prompt: str) -> str:
        """Last safety pass before sending any prompt to Veo."""
        finalized = self._anonymize_prompt_text(prompt, shot=None)
        finalized += (
            " Single coherent camera view only. No cutaway, no split-screen, "
            "no collage, no multi-panel layout, no picture-in-picture, "
            "no impossible car interior, no duplicated dashboard, no duplicated room geometry. "
            "Preserve the anchor image spatial layout as one plausible continuous shot."
        )
        return re.sub(r"\s+", " ", finalized).strip()

    def _build_character_appearance_block(self, shot: ShotPlan) -> str:
        """Build a text block describing each visible character's appearance for Veo."""
        descriptions = []
        clip = self._clip_for_shot(shot)
        for char_id in shot.visible_characters:
            member = self._find_cast_member(char_id)
            if member is None:
                continue
            role_info = member.role
            if (
                member.relation_to_protagonist
                and member.relation_to_protagonist != "self"
                and member.relation_to_protagonist.lower() != member.role.lower()
            ):
                role_info = f"{member.role}, {member.relation_to_protagonist}"
            current_outfit = self._selected_clip_outfit(clip, char_id)
            wardrobe_text = f"Current outfit cue: {current_outfit}. " if current_outfit else ""
            desc = (
                f"{self._character_descriptor(member)} ({role_info}): "
                f"{member.appearance_description}. "
                f"{wardrobe_text}"
            )
            descriptions.append(desc)
        if not descriptions:
            return ""
        return "Character appearances in this scene: " + " | ".join(descriptions)

    def _build_clip_outfit_lock_block(self, shot: ShotPlan) -> str:
        clip = self._clip_for_shot(shot)
        if clip is None:
            return ""
        entries: list[str] = []
        for char_id in shot.visible_characters:
            outfit_text = self._selected_clip_outfit(clip, char_id)
            if not outfit_text:
                continue
            entries.append(f"{self._character_label(char_id)} wears {outfit_text}")
        if not entries:
            return ""
        return "Locked clip outfits: " + " | ".join(entries) + "."

    def _ordered_visible_character_ids(self, shot: ShotPlan) -> list[str]:
        visible = list(shot.visible_characters or [])
        ordered: list[str] = []
        for char_id in shot.left_to_right_order or []:
            if char_id in visible and char_id not in ordered:
                ordered.append(char_id)
        for char_id in visible:
            if char_id not in ordered:
                ordered.append(char_id)
        return ordered

    @staticmethod
    def _screen_position_label(index: int) -> str:
        labels = {
            1: "leftmost",
            2: "second from left",
            3: "third from left",
            4: "fourth from left",
            5: "fifth from left",
            6: "sixth from left",
        }
        return labels.get(index, f"position {index} from left")

    def _sanitize_runtime_instruction(
        self,
        text: str,
        shot: ShotPlan,
        *,
        silent: bool = False,
        audio_safe: bool = False,
    ) -> str:
        cleaned = self._anonymize_prompt_text(str(text or "").strip(), shot=shot)
        if silent:
            cleaned = self._sanitize_silent_video_prompt(cleaned)
        elif audio_safe:
            cleaned = self._sanitize_audio_enabled_prompt(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _build_first_frame_identity_action_lock_block(
        self,
        shot: ShotPlan,
        *,
        silent: bool = False,
        audio_safe: bool = False,
    ) -> str:
        ordered_ids = self._ordered_visible_character_ids(shot)
        parts = [
            "Use the provided anchor image as the exact first-frame reference.",
            "Do not reinterpret, replace, add, remove, or reorder the recurring adults in that image.",
            "Preserve each person's identity, outfit, prop relationship, screen position, and camera-facing arrangement from the anchor image.",
        ]

        if ordered_ids:
            entries = [
                f"{self._screen_position_label(index)} = {self._character_label_with_outfit(shot, char_id)}"
                for index, char_id in enumerate(ordered_ids, start=1)
            ]
            parts.append(
                "In the anchor image, lock recurring adult identities by screen position: "
                + "; ".join(entries)
                + "."
            )

        action_bits: list[str] = []
        video_prompt = self._sanitize_runtime_instruction(
            shot.video_prompt,
            shot,
            silent=silent,
            audio_safe=audio_safe,
        )
        if video_prompt:
            action_bits.append(video_prompt)

        blocking_notes = self._sanitize_runtime_instruction(
            shot.blocking_notes,
            shot,
            silent=silent,
            audio_safe=audio_safe,
        )
        if blocking_notes:
            action_bits.append("Blocking and movement: " + blocking_notes + ".")

        secondary_actions = [
            self._sanitize_runtime_instruction(action, shot, silent=silent, audio_safe=audio_safe)
            for action in shot.secondary_actions or []
            if str(action).strip()
        ]
        secondary_actions = [action for action in secondary_actions if action]
        if secondary_actions:
            action_bits.append("Parallel actions: " + "; ".join(secondary_actions) + ".")

        if action_bits:
            parts.append("Action assignment for this shot: " + " ".join(action_bits))
            parts.append(
                "Keep each described action, gaze, gesture, prop interaction, and reaction attached to the same identified adult; do not swap actions between people."
            )

        parts.append("Animate only from this existing first-frame arrangement with small, natural motion.")
        return " ".join(part for part in parts if part)

    def _sanitize_silent_video_prompt(self, prompt: str) -> str:
        """Remove obvious sound-generation cues when the run is configured as visual-only."""
        replacements = {
            r"\blaughs loudly\b": "reacts with a broad grin and shaking shoulders",
            r"\blaughing loudly\b": "showing amusement with a broad grin and shaking shoulders",
            r"\blaughs\b": "shows amusement with a broad grin",
            r"\blaughing\b": "showing amusement with a broad grin",
            r"\bspeaks\b": "uses a subtle facial expression",
            r"\bspeaking\b": "using a subtle facial expression",
            r"\bsays\b": "responds with a small facial reaction",
            r"\bsaying\b": "responding with a small facial reaction",
            r"\btells an animated story\b": "acts out an animated story with expressive gestures",
            r"\btells a story\b": "uses expressive gestures as if sharing a story",
            r"\bfinishes his story\b": "finishes the gesture with a playful expression",
            r"\bfinishes her story\b": "finishes the gesture with a playful expression",
            r"\bmurmurs\b": "leans in with a subtle facial expression",
            r"\bwhispers\b": "leans in with a subtle facial expression",
            r"\bshouts\b": "reacts with emphatic body language",
            r"\bspoken dialogue\b": "facial expression",
            r"\bdialogue\b": "facial expression",
            r"\bvoices?\b": "facial reactions",
            r"\baudio\b": "visual motion",
            r"\bsound(?:track| effects?)?\b": "visual atmosphere",
            r"\baudible\b": "visible",
            r"\bmouth(?:s)?\b": "face",
        }
        sanitized = prompt
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r"\bmoves?\s+(?:his|her|their)?\s*face\s+subtly\s+without\s+visible\s+facial expression\b",
            "uses subtle facial expressions",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def _compose_visual_only_minimal_runtime_prompt(self, shot: ShotPlan) -> str:
        sanitized_video_prompt = self._sanitize_silent_video_prompt(
            self._anonymize_prompt_text(shot.video_prompt.strip(), shot=shot)
        )
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", sanitized_video_prompt)
            if sentence.strip()
        ]
        summary = " ".join(sentences[:2]) if sentences else sanitized_video_prompt
        parts = [
            summary,
            "Match the anchor image closely for identity, clothing, props, and spatial layout.",
            self._build_first_frame_identity_action_lock_block(shot, silent=True),
            "Use restrained visual acting: facial expressions, eye lines, hand gestures, posture, and body movement.",
            "Do not include captions, subtitles, signs with readable new text, or on-screen text overlays.",
        ]
        if shot.composition:
            parts.append(f"Use a clear {shot.composition.lower()}.")
        if shot.blocking_notes:
            parts.append(
                "Blocking: "
                + self._sanitize_silent_video_prompt(
                    self._anonymize_prompt_text(shot.blocking_notes.strip(), shot=shot)
                )
                + "."
            )
        if shot.visible_characters:
            parts.append(
                f"Keep all {len(shot.visible_characters)} recurring adults readable, stable, and distinct."
            )
        return " ".join(part for part in parts if part)

    def _compose_visual_only_tiny_runtime_prompt(self, shot: ShotPlan) -> str:
        parts = [
            "Animate the provided image as a realistic short shot.",
            "Preserve the same people, clothing, props, room layout, and camera angle from the anchor image.",
            self._build_first_frame_identity_action_lock_block(shot, silent=True),
            "Use only small natural body movement, eye lines, hand gestures, and posture changes.",
            "Do not add captions, subtitles, title cards, or text overlays.",
        ]
        if shot.composition:
            parts.append(f"Keep the framing as {shot.composition}.")
        if shot.visible_characters:
            parts.append(f"Keep all {len(shot.visible_characters)} recurring adults visible and distinct.")
        return " ".join(parts)

    def _sanitize_audio_enabled_prompt(self, prompt: str) -> str:
        """Soften phrases that often trigger Veo's native-audio filtering."""
        replacements = {
            r"\bexcitedly\b": "with quiet anticipation",
            r"\benthusiastically\b": "with gentle enthusiasm",
            r"\bhearty\b": "warm",
            r"\bloud, genuine laugh(?:ter)?\b": "brief warm laughter",
            r"\bloud laughter\b": "soft laughter",
            r"\bgenuine laughter\b": "warm laughter",
            r"\blaughing loudly\b": "laughing softly",
            r"\blaughs loudly\b": "laughs softly",
            r"\blaughs out loud\b": "lets out a brief laugh",
            r"\blaughs\b": "smiles and laughs softly",
            r"\blaughing\b": "smiling with soft laughter",
            r"\bchuckles\b": "smiles with a brief laugh",
            r"\bchuckling\b": "smiling with a brief laugh",
            r"\bgiggling\b": "softly laughing",
            r"\bgiggles\b": "laughs softly",
            r"\bshouts\b": "speaks",
            r"\bshouting\b": "speaking",
            r"\bloud\b": "soft",
            r"\benergetic voice\b": "calm speaking voice",
            r"\boverlapping voices\b": "soft shared reaction",
            r"\bcrowd reaction\b": "background ambience",
            r"\bcrowd\b": "background",
            r"\bcheering\b": "subtle celebration",
            r"\bcheers\b": "a soft toast",
            r"\bbustling\b": "warm ambient",
            r"\bchatter\b": "background ambience",
            r"\bmurmurs of agreement\b": "quiet agreement",
            r"\bFriendly laughter\b": "soft shared laughter",
            r"\bWarm laughter\b": "soft warm laughter",
        }
        sanitized = prompt.strip()
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def _soften_audio_cues(self, shot: ShotPlan, *, keep_dialogue: bool) -> list[str]:
        softened = []
        for cue in shot.audio_cues:
            text = self._sanitize_audio_enabled_prompt(cue.strip()).strip(" .")
            if not text:
                continue
            lowered = text.lower()
            if not keep_dialogue and any(
                token in lowered
                for token in ("voice", "speaking", "spoken", "toast", "laugh", "gigg", "chuckl")
            ):
                continue
            softened.append(text)

        if not softened:
            if "train" in shot.scene_id or "platform" in shot.scene_id:
                return ["quiet station ambience", "distant approaching train"]
            if "office" in shot.scene_id:
                return ["soft office room tone", "light ceramic mug handling"]
            if "restaurant" in shot.scene_id or "dinner" in shot.scene_id or "pantry" in shot.scene_id:
                return ["warm indoor ambience", "soft dish and cup sounds"]
            return ["soft natural ambience"]

        limit = 2 if keep_dialogue else 1
        return softened[:limit]

    def _soft_dialogue_prompt(self, shot: ShotPlan) -> str | None:
        if not shot.dialogue_lines:
            return None

        spoken_lines: list[str] = []
        for raw_line in shot.dialogue_lines[:2]:
            speaker, separator, content = raw_line.partition(":")
            text = content.strip() if separator else raw_line.strip()
            text = self._anonymize_prompt_text(text, shot=shot)
            text = self._sanitize_audio_enabled_prompt(text)
            text = text.replace("!", ".").strip()
            if not text:
                continue
            if len(text) > 90:
                text = text[:87].rstrip() + "..."

            if separator and speaker.strip():
                speaker_desc = self._speaker_descriptor(speaker.strip(), shot)
                if speaker_desc:
                    spoken_lines.append(
                        f'{speaker_desc} speaks this brief, soft line with visible mouth movement: "{text}"'
                    )
                    continue
            spoken_lines.append(f'A visible adult speaks this brief, soft line: "{text}"')

        if not spoken_lines:
            return None
        return (
            "Spoken dialogue and lip movement assignment: "
            + " Then ".join(spoken_lines)
            + ". Only the assigned speaker moves their mouth for each line; the other visible adults listen or react quietly"
        )

    def _audio_strategy_for_shot(self, shot: ShotPlan) -> str:
        return getattr(shot, "audio_strategy", "ambient_with_sfx") or "ambient_with_sfx"

    def _trimmed_dialogue_lines(self, shot: ShotPlan) -> list[str]:
        strategy = self._audio_strategy_for_shot(shot)
        if strategy in ("ambient_only", "ambient_with_sfx"):
            return []
        if strategy == "soft_single_line":
            return shot.dialogue_lines[:1]
        if strategy == "soft_dialogue":
            return shot.dialogue_lines[:2]
        return shot.dialogue_lines

    def _audio_risk_score(self, shot: ShotPlan) -> int:
        strategy = self._audio_strategy_for_shot(shot)
        score = 0
        if len(shot.visible_characters) >= 5:
            score += 2
        elif len(shot.visible_characters) >= 4:
            score += 1
        if len(shot.dialogue_lines) >= 2:
            score += 2
        elif len(shot.dialogue_lines) == 1:
            score += 1
        if strategy == "soft_dialogue":
            score += 1

        combined_text = " ".join(
            [
                shot.video_prompt,
                " ".join(shot.dialogue_lines),
                " ".join(shot.audio_cues),
            ]
        ).lower()
        risky_patterns = (
            "loud",
            "hearty",
            "crowd",
            "cheer",
            "laugh",
            "chuckl",
            "giggl",
            "shout",
            "overlapping voices",
            "chatter",
            "bustling",
        )
        if any(pattern in combined_text for pattern in risky_patterns):
            score += 2
        return score

    def _compose_audio_safe_runtime_prompt(self, shot: ShotPlan, *, keep_dialogue: bool) -> str:
        parts = [
            self._sanitize_audio_enabled_prompt(self._anonymize_prompt_text(shot.anchor_image_prompt.strip(), shot=shot)),
            self._sanitize_audio_enabled_prompt(self._anonymize_prompt_text(shot.video_prompt.strip(), shot=shot)),
            "Keep all visible people adult, realistic, and visually consistent with the anchor image.",
            self._build_first_frame_identity_action_lock_block(shot, audio_safe=True),
        ]
        if shot.composition:
            parts.append(f"Composition: {shot.composition}.")
        if shot.camera_language:
            parts.append(
                f"Use simple, natural camera behavior: {self._sanitize_audio_enabled_prompt(self._anonymize_prompt_text(shot.camera_language.strip(), shot=shot))}."
            )
        if len(shot.visible_characters) >= 4:
            parts.append(
                f"Keep all {len(shot.visible_characters)} visible characters readable and active without chaotic motion."
            )

        dialogue_prompt = self._soft_dialogue_prompt(shot) if keep_dialogue else None
        if dialogue_prompt:
            parts.append(dialogue_prompt + ".")
        else:
            parts.append("Do not emphasize spoken words. Keep any vocalization brief and low-key.")

        cues = self._soften_audio_cues(shot, keep_dialogue=keep_dialogue)
        if cues:
            parts.append("Audio stays subtle and realistic: " + "; ".join(cues) + ".")
        parts.append(
            "Avoid cheering, shouting, overlapping loud voices, exaggerated laughter, or dominant crowd noise."
        )
        return " ".join(part for part in parts if part)

    def _compose_audio_minimal_runtime_prompt(self, shot: ShotPlan) -> str:
        sanitized_video_prompt = self._sanitize_audio_enabled_prompt(
            self._anonymize_prompt_text(shot.video_prompt.strip(), shot=shot)
        )
        parts = [
            sanitized_video_prompt,
            "Match the anchor image closely for identity, clothing, props, and spatial layout.",
            self._build_first_frame_identity_action_lock_block(shot, audio_safe=True),
            "Keep every visible person adult, realistic, and stable in appearance.",
        ]
        if shot.composition:
            parts.append(f"Use a simple, readable {shot.composition.lower()}.")
        if shot.visible_characters:
            parts.append(
                f"Keep all {len(shot.visible_characters)} recurring adults readable in frame with no chaotic motion."
            )
        parts.append("Do not emphasize spoken dialogue. Keep any vocalization brief, soft, and secondary.")
        cues = self._soften_audio_cues(shot, keep_dialogue=False)
        if cues:
            parts.append("Use only subtle natural audio: " + "; ".join(cues[:1]) + ".")
        parts.append("Avoid cheering, shouting, crowd noise, overlapping voices, or dominant laughter.")
        return " ".join(part for part in parts if part)

    def _compose_audio_ultra_minimal_runtime_prompt(self, shot: ShotPlan) -> str:
        sanitized_video_prompt = self._sanitize_audio_enabled_prompt(
            self._anonymize_prompt_text(shot.video_prompt.strip(), shot=shot)
        )
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", sanitized_video_prompt)
            if sentence.strip()
        ]
        summary = " ".join(sentences[:3]) if sentences else sanitized_video_prompt
        summary = summary.replace("radiating warmth", "with a warm expression")
        summary = summary.replace("beams with pride", "smiles warmly")

        parts = [
            summary,
            f"Keep all {len(shot.visible_characters)} visible adults stable and visually consistent with the anchor image."
            if shot.visible_characters
            else "Keep all visible adults stable and visually consistent with the anchor image.",
            self._build_first_frame_identity_action_lock_block(shot, audio_safe=True),
            "Use simple, readable framing and gentle natural motion.",
        ]
        cues = self._soften_audio_cues(shot, keep_dialogue=False)
        if cues:
            parts.append("Use only subtle natural ambience: " + "; ".join(cues[:1]) + ".")
        else:
            parts.append("Use only subtle natural ambience.")
        parts.append("No spoken dialogue, no crowd noise, no cheering, no shouting, and no dominant laughter.")
        return " ".join(part for part in parts if part)

    def _build_runtime_prompt_variants(self, shot: ShotPlan) -> list[tuple[str, str]]:
        default_prompt = self._compose_runtime_video_prompt(shot)
        safe_dialogue_prompt = self._compose_audio_safe_runtime_prompt(shot, keep_dialogue=True)
        safe_ambience_prompt = self._compose_audio_safe_runtime_prompt(shot, keep_dialogue=False)
        minimal_prompt = self._compose_audio_minimal_runtime_prompt(shot)
        ultra_minimal_prompt = self._compose_audio_ultra_minimal_runtime_prompt(shot)
        strategy = self._audio_strategy_for_shot(shot)
        risk_score = self._audio_risk_score(shot)

        if not self.generate_audio:
            candidate_order = [
                ("visual_only", default_prompt),
                ("visual_only_minimal", self._compose_visual_only_minimal_runtime_prompt(shot)),
                ("visual_only_tiny", self._compose_visual_only_tiny_runtime_prompt(shot)),
            ]
            variants: list[tuple[str, str]] = []
            seen: set[str] = set()
            for label, prompt in candidate_order:
                prompt = self._finalize_veo_runtime_prompt(prompt)
                normalized = " ".join(prompt.split())
                if normalized not in seen:
                    variants.append((label, prompt))
                    seen.add(normalized)
            return variants

        if strategy in ("ambient_only", "ambient_with_sfx"):
            candidate_order = [
                ("audio_safe_minimal", minimal_prompt),
                ("audio_safe_ultra_minimal", ultra_minimal_prompt),
                ("audio_safe_ambience_only", safe_ambience_prompt),
                ("default", default_prompt),
                ("audio_safe_soft_dialogue", safe_dialogue_prompt),
            ]
        elif risk_score >= 4:
            candidate_order = [
                ("audio_safe_minimal", minimal_prompt),
                ("audio_safe_ultra_minimal", ultra_minimal_prompt),
                ("audio_safe_ambience_only", safe_ambience_prompt),
                ("audio_safe_soft_dialogue", safe_dialogue_prompt),
                ("default", default_prompt),
            ]
        elif risk_score >= 3:
            candidate_order = [
                ("audio_safe_ultra_minimal", ultra_minimal_prompt),
                ("audio_safe_soft_dialogue", safe_dialogue_prompt),
                ("audio_safe_minimal", minimal_prompt),
                ("audio_safe_ambience_only", safe_ambience_prompt),
                ("default", default_prompt),
            ]
        else:
            candidate_order = [
                ("default", default_prompt),
                ("audio_safe_soft_dialogue", safe_dialogue_prompt),
                ("audio_safe_ultra_minimal", ultra_minimal_prompt),
                ("audio_safe_minimal", minimal_prompt),
                ("audio_safe_ambience_only", safe_ambience_prompt),
            ]

        variants: list[tuple[str, str]] = []
        seen: set[str] = set()
        for label, prompt in candidate_order:
            prompt = self._finalize_veo_runtime_prompt(prompt)
            normalized = " ".join(prompt.split())
            if normalized not in seen:
                variants.append((label, prompt))
                seen.add(normalized)
        return variants

    def _latest_video_failure_is_audio_filtered(
        self,
        processor: GeminiVideoProcessor | None = None,
    ) -> bool:
        processor = processor or self._thread_processor()
        diagnostics = getattr(processor, "last_video_generation_diagnostics", {}) or {}
        if diagnostics.get("audio_filtered"):
            return True

        joined = " ".join(str(reason) for reason in diagnostics.get("rai_media_filtered_reasons", []))
        joined = f"{joined} {diagnostics.get('error', '')}".lower()
        return "audio" in joined and "could not create your video" in joined

    def _latest_video_failure_reason(
        self,
        processor: GeminiVideoProcessor | None = None,
    ) -> str:
        processor = processor or self._thread_processor()
        diagnostics = getattr(processor, "last_video_generation_diagnostics", {}) or {}
        reasons = [str(reason) for reason in diagnostics.get("rai_media_filtered_reasons", [])]
        if diagnostics.get("error"):
            reasons.append(str(diagnostics["error"]))
        if reasons:
            return " | ".join(reasons)[:500]
        return str(diagnostics.get("status") or "unknown video generation failure")

    def _latest_video_failure_needs_new_anchor(
        self,
        processor: GeminiVideoProcessor | None = None,
    ) -> bool:
        processor = processor or self._thread_processor()
        diagnostics = getattr(processor, "last_video_generation_diagnostics", {}) or {}
        if not diagnostics.get("rai_media_filtered_count"):
            return False
        if diagnostics.get("audio_filtered"):
            return False

        text = self._latest_video_failure_reason(processor).lower()
        image_terms = (
            "celebrity",
            "likeness",
            "public figure",
            "famous",
            "input image",
            "reference",
            "face",
            "person",
            "people",
        )
        return not text or any(term in text for term in image_terms)

    @staticmethod
    def _anchor_generation_count_for_entry(entry: dict[str, Any] | None) -> int:
        if not isinstance(entry, dict):
            return 0
        try:
            explicit_total = int(entry.get("total_anchor_generations") or 0)
        except (TypeError, ValueError):
            explicit_total = 0
        if explicit_total > 0:
            return explicit_total

        records = entry.get("anchor_quality_checks") or []
        generated_records = [
            record for record in records if isinstance(record, dict) and not record.get("reused")
        ]
        if generated_records:
            return len(generated_records)
        return max(1, len(entry.get("candidates") or []))

    def _anchor_generation_count_for_shot(self, shot: ShotPlan) -> int:
        return self._anchor_generation_count_for_entry(
            self.anchor_manifest.get("shots", {}).get(shot.id)
        )

    def _regenerate_anchor_for_failed_shot(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        *,
        reason: str,
    ) -> bool:
        previous_count = self._anchor_generation_count_for_shot(shot)
        if previous_count >= self.anchor_quality_max_attempts:
            print(
                f"   ⚠️  {clip.id}/{shot.id} 已达到 anchor 总生成次数上限 "
                f"{previous_count}/{self.anchor_quality_max_attempts}，不再重生 anchor。"
            )
            return False

        from video_generator.anchor_generator import VideoPreproductionBuilder

        builder = VideoPreproductionBuilder(
            series_bible=self.series_bible,
            output_root=self.output_root,
            config_path=self.config_path,
        )
        entry = builder.regenerate_anchor_for_shot(
            clip,
            shot,
            reason=reason,
            previous_generation_count=previous_count,
        )
        self.anchor_manifest.setdefault("shots", {})[shot.id] = entry
        print(
            f"   ✅ 已重生 anchor，累计生成次数 "
            f"{entry.get('total_anchor_generations')}/{self.anchor_quality_max_attempts}。"
        )
        return True

    def _video_has_audio_stream(self, path: Path) -> bool:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        return bool(result.stdout.strip())

    def _compose_runtime_video_prompt(self, shot: ShotPlan) -> str:
        video_prompt = self._anonymize_prompt_text(shot.video_prompt.strip(), shot=shot)
        if not self.generate_audio:
            video_prompt = self._sanitize_silent_video_prompt(video_prompt)
        elif self._audio_risk_score(shot) >= 3:
            video_prompt = self._sanitize_audio_enabled_prompt(video_prompt)
        parts = [video_prompt]

        parts.append(
            self._build_first_frame_identity_action_lock_block(
                shot,
                silent=not self.generate_audio,
                audio_safe=self.generate_audio and self._audio_risk_score(shot) >= 3,
            )
        )

        parts.append(
            "All visible people must be adults. Do not generate children, teenagers, babies, toddlers, students, school gates, or playground scenes."
        )

        if shot.composition:
            parts.append(f"Composition: {shot.composition}.")

        if shot.camera_language:
            parts.append(f"Camera language: {self._anonymize_prompt_text(shot.camera_language, shot=shot)}.")

        if shot.motion_budget:
            parts.append(f"Motion budget: {shot.motion_budget}.")

        if shot.background_extras:
            parts.append(
                "Unnamed background extras/passersby: "
                + "; ".join(extra.strip() for extra in shot.background_extras if extra.strip())
                + "."
            )
            parts.append(
                "Keep background extras secondary, generic, and visually quiet. They must not steal focus from the recurring cast."
            )

        if self.generate_audio:
            strategy = self._audio_strategy_for_shot(shot)
            trimmed_dialogue = self._trimmed_dialogue_lines(shot)
            if strategy in ("ambient_only", "ambient_with_sfx"):
                parts.append("Keep the soundtrack subtle and mostly environmental, with no spoken dialogue.")
            elif trimmed_dialogue:
                prompt_shot = shot.model_copy(update={"dialogue_lines": trimmed_dialogue})
                dialogue_prompt = self._soft_dialogue_prompt(prompt_shot)
                if dialogue_prompt:
                    parts.append(dialogue_prompt + ".")

            audio_cues = self._soften_audio_cues(
                shot,
                keep_dialogue=strategy in ("soft_single_line", "soft_dialogue"),
            )
            if audio_cues:
                parts.append("Use subtle, realistic audio only: " + "; ".join(audio_cues) + ".")
        elif not self.generate_audio:
            parts.append(
                "Pure visual motion only. Show emotions through facial expressions, gestures, posture, eye lines, and body movement. Do not include captions, subtitles, narration cards, or on-screen text overlays."
            )

        if len(shot.visible_characters) >= 4:
            parts.append(
                f"Keep all {len(shot.visible_characters)} visible characters active, readable, and distinct in frame."
            )
            parts.append(
                "No idle background people. Give the ensemble layered, simultaneous activity and visible reactions."
            )

        return " ".join(part for part in parts if part)

    def _probe_duration_seconds(self, path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _concat_videos(self, inputs: list[Path], output_path: Path) -> None:
        list_path = output_path.with_suffix(".concat.txt")
        with open(list_path, "w", encoding="utf-8") as file:
            for input_path in inputs:
                file.write(f"file '{input_path.resolve()}'\n")

        copy_command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(copy_command, capture_output=True, text=True)
        if result.returncode != 0:
            reencode_command = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-map",
                "0:v:0?",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(output_path),
            ]
            subprocess.run(reencode_command, check=True)

        list_path.unlink(missing_ok=True)

    def _render_shot_segment(
        self,
        clip: ClipPlan,
        shot: ShotPlan,
        segment_path: Path,
    ) -> RenderedSegment:
        duration = self._normalize_duration(shot.duration_seconds or self.default_shot_duration)
        negative_prompt = shot.negative_prompt or self.default_negative_prompt
        prompt_variants = self._build_runtime_prompt_variants(shot)
        last_error: RuntimeError | None = None
        processor = self._thread_processor()

        while True:
            anchor_candidates = self._anchor_candidates_for(shot)
            needs_anchor_regeneration = False
            regeneration_reason = ""

            for candidate_index, anchor_path in enumerate(anchor_candidates, start=1):
                print(f"\n🎞️  渲染 {shot.id} -> {segment_path.name}")
                if len(anchor_candidates) > 1:
                    print(
                        f"   🖼️  尝试 anchor {candidate_index}/{len(anchor_candidates)}: {anchor_path.name}"
                    )

                attempt_plan = prompt_variants[: self.video_generation_retries]
                while attempt_plan and len(attempt_plan) < self.video_generation_retries:
                    attempt_plan.append(attempt_plan[-1])

                for attempt, (prompt_label, runtime_prompt) in enumerate(attempt_plan, start=1):
                    if self.video_generation_retries > 1:
                        print(
                            f"   🚀 Veo 尝试 {attempt}/{self.video_generation_retries} "
                            f"(anchor={anchor_path.name}, prompt={prompt_label})"
                        )

                    try:
                        result_path, video_ref = processor.generate_video_from_image(
                            image_path=str(anchor_path),
                            prompt=runtime_prompt,
                            duration=duration,
                            aspect_ratio=self.aspect_ratio,
                            resolution=self.resolution,
                            save_path=str(segment_path),
                            model=self.video_model,
                            negative_prompt=negative_prompt,
                            calling_from=f"{clip.id}:{shot.id}:attempt_{attempt}",
                            person_generation=self.person_generation_image_to_video,
                            generate_audio=self.generate_audio,
                            return_video_ref=True,
                        )
                    except Exception as error:
                        result_path = None
                        video_ref = None
                        last_error = RuntimeError(
                            f"视频生成异常: {clip.id}/{shot.id} "
                            f"(anchor={anchor_path.name}, attempt={attempt}): {error}"
                        )
                        print(f"   ⚠️  {last_error}")
                    if result_path:
                        if self.generate_audio and not self._video_has_audio_stream(Path(result_path)):
                            print("   ⚠️  视频已生成，但没有音轨；视为失败并切换到更稳的 prompt / anchor。")
                            Path(result_path).unlink(missing_ok=True)
                        else:
                            return RenderedSegment(path=segment_path, video_ref=video_ref)

                    if (
                        self.render_regenerate_anchor_on_rai
                        and self._latest_video_failure_needs_new_anchor(processor)
                    ):
                        regeneration_reason = self._latest_video_failure_reason(processor)
                        needs_anchor_regeneration = True
                        last_error = RuntimeError(
                            f"Veo RAI 过滤疑似由 anchor/input image 触发: "
                            f"{clip.id}/{shot.id} (anchor={anchor_path.name}) - {regeneration_reason}"
                        )
                        print(
                            "   ⚠️  检测到非音频 RAI / reference image 过滤，"
                            "将优先重生 anchor，而不是继续消耗同一张图。"
                        )
                        break

                    if attempt < len(attempt_plan):
                        if self._latest_video_failure_is_audio_filtered(processor):
                            next_label = attempt_plan[attempt][0]
                            print(
                                f"   ⚠️  检测到音频过滤，下一次改用更稳的 prompt 版本: {next_label}。"
                            )
                        else:
                            print(
                                f"   ↩️  Veo 本次未生成可用视频，等待 {self.video_retry_wait_seconds}s 后重试一次。"
                            )
                        if self.video_retry_wait_seconds:
                            time.sleep(self.video_retry_wait_seconds)

                if needs_anchor_regeneration:
                    break

                last_error = RuntimeError(
                    f"视频生成失败: {clip.id}/{shot.id} (anchor={anchor_path.name})"
                )
                if candidate_index < len(anchor_candidates):
                    print("   ↩️  当前 anchor 失败，尝试同一 shot 的下一个 candidate。")

            if needs_anchor_regeneration and self._regenerate_anchor_for_failed_shot(
                clip,
                shot,
                reason=regeneration_reason,
            ):
                segment_path.unlink(missing_ok=True)
                continue
            break

        final_error = last_error or RuntimeError(f"视频生成失败: {clip.id}/{shot.id}")
        if self.skip_failed_shots:
            raise ShotRenderSkipped(str(final_error)) from final_error
        raise final_error

    def render_shot_based_clip(self, clip: ClipPlan) -> Path:
        final_clip_path = self.clips_dir / f"{clip.id}.mp4"
        clip_segment_dir = self.segments_dir / clip.id
        if self._rendered_clip_is_current(clip, final_clip_path):
            print(f"♻️  复用 shot-based clip: {final_clip_path}")
            return final_clip_path
        if final_clip_path.exists() or clip_segment_dir.exists():
            print(f"♻️  {clip.id} 的旧渲染不匹配当前 anchor 签名，将重渲染。")
            self._discard_stale_clip_render_outputs(clip, final_clip_path, clip_segment_dir)

        clip_segment_dir.mkdir(parents=True, exist_ok=True)
        segment_paths: list[Path] = []
        skipped_shots: list[dict[str, str]] = []

        for shot in clip.shots:
            segment_path = clip_segment_dir / f"{shot.id}.mp4"
            if segment_path.exists() and self.reuse_existing_assets:
                print(f"♻️  复用 segment: {segment_path}")
            else:
                try:
                    self._render_shot_segment(clip, shot, segment_path)
                except ShotRenderSkipped as error:
                    segment_path.unlink(missing_ok=True)
                    skipped_shots.append(
                        {
                            "shot_id": shot.id,
                            "reason": str(error)[:800],
                        }
                    )
                    print(
                        f"   ⏭️  已跳过 {clip.id}/{shot.id}，继续后续 shot/clip。原因: {error}"
                    )
                    continue
            if segment_path.exists():
                segment_paths.append(segment_path)

        with self._clip_skipped_shots_lock:
            if skipped_shots:
                self._clip_skipped_shots[clip.id] = skipped_shots
            else:
                self._clip_skipped_shots.pop(clip.id, None)

        if not segment_paths:
            raise RuntimeError(f"{clip.id} 的所有 shot 都失败/跳过，无法拼接 clip。")

        self._concat_videos(segment_paths, final_clip_path)
        return final_clip_path

    def _coalesce_extended_result(
        self,
        previous_path: Path,
        extension_result_path: Path,
        merged_output_path: Path,
    ) -> Path:
        previous_duration = self._probe_duration_seconds(previous_path)
        extension_duration = self._probe_duration_seconds(extension_result_path)

        if extension_duration > previous_duration + 1.0:
            shutil.copyfile(extension_result_path, merged_output_path)
            return merged_output_path

        self._concat_videos([previous_path, extension_result_path], merged_output_path)
        return merged_output_path

    def render_extend_clip(self, clip: ClipPlan) -> Path:
        final_clip_path = self.clips_dir / f"{clip.id}.mp4"
        clip_segment_dir = self.segments_dir / clip.id
        if self._rendered_clip_is_current(clip, final_clip_path):
            print(f"♻️  复用 extend clip: {final_clip_path}")
            return final_clip_path
        if final_clip_path.exists() or clip_segment_dir.exists():
            print(f"♻️  {clip.id} 的旧 extend 渲染不匹配当前 anchor 签名，将重渲染。")
            self._discard_stale_clip_render_outputs(clip, final_clip_path, clip_segment_dir)
        processor = self._thread_processor()

        clip_segment_dir.mkdir(parents=True, exist_ok=True)

        first_shot = clip.shots[0]
        initial_segment_path = clip_segment_dir / f"{first_shot.id}_initial.mp4"
        if initial_segment_path.exists() and self.reuse_existing_assets:
            print("♻️  extend 模式需要保留原始 Veo video 引用，初始片段将重新生成以继续续写。")

        current_segment = self._render_shot_segment(clip, first_shot, initial_segment_path)

        for index, shot in enumerate(clip.shots[1:], start=2):
            extension_result_path = clip_segment_dir / f"{shot.id}_extend_result.mp4"
            merged_output_path = clip_segment_dir / f"{clip.id}_progressive_{index:02d}.mp4"
            negative_prompt = shot.negative_prompt or self.default_negative_prompt
            prompt_variants = self._build_runtime_prompt_variants(shot)

            if (
                self.reuse_existing_assets
                and (extension_result_path.exists() or merged_output_path.exists())
            ):
                print(
                    "♻️  检测到旧的 extend 中间文件，但这类文件无法恢复 Veo 的运行时 video 引用。"
                    " 当前 clip 将从初始片段重新顺序续写，并覆盖这些中间文件。"
                )

            print(f"\n🔁 Extend {shot.id}")
            if current_segment.video_ref is None:
                raise RuntimeError(
                    "当前 extend 片段没有可续写的 Veo video 引用。"
                    "请删除该 extend clip 的旧中间文件后重跑，或从初始片段重新开始。"
                )

            extension_path = None
            next_video_ref = None
            attempt_plan = prompt_variants[: self.video_generation_retries]
            while attempt_plan and len(attempt_plan) < self.video_generation_retries:
                attempt_plan.append(attempt_plan[-1])

            for attempt, (prompt_label, runtime_prompt) in enumerate(attempt_plan, start=1):
                if self.video_generation_retries > 1:
                    print(
                        f"   🚀 Veo extend 尝试 {attempt}/{self.video_generation_retries} "
                        f"(prompt={prompt_label})"
                    )
                try:
                    extension_path, next_video_ref = processor.generate_video_extension(
                        video_ref=current_segment.video_ref,
                        prompt=runtime_prompt,
                        aspect_ratio=self.aspect_ratio,
                        resolution=self.resolution,
                        save_path=str(extension_result_path),
                        model=self.video_model,
                        negative_prompt=negative_prompt,
                        calling_from=f"{clip.id}:{shot.id}:extend:attempt_{attempt}",
                        person_generation=self.person_generation_extension,
                        generate_audio=self.generate_audio,
                    )
                except Exception as error:
                    extension_path = None
                    next_video_ref = None
                    print(
                        f"   ⚠️  视频续写异常: {clip.id}/{shot.id} "
                        f"(attempt={attempt}): {error}"
                    )
                if extension_path:
                    if self.generate_audio and not self._video_has_audio_stream(Path(extension_path)):
                        print("   ⚠️  Extend 视频已生成，但没有音轨；继续尝试更稳的 prompt。")
                        Path(extension_path).unlink(missing_ok=True)
                        extension_path = None
                        next_video_ref = None
                    else:
                        break
                if attempt < len(attempt_plan):
                    if self._latest_video_failure_is_audio_filtered(processor):
                        next_label = attempt_plan[attempt][0]
                        print(
                            f"   ⚠️  Extend 检测到音频过滤，下一次改用更稳的 prompt 版本: {next_label}。"
                        )
                    else:
                        print(
                            f"   ↩️  Veo extend 本次未生成可用视频，等待 {self.video_retry_wait_seconds}s 后重试一次。"
                        )
                    if self.video_retry_wait_seconds:
                        time.sleep(self.video_retry_wait_seconds)
            if not extension_path:
                raise RuntimeError(f"视频续写失败: {clip.id}/{shot.id}")
            merged_path = self._coalesce_extended_result(
                previous_path=current_segment.path,
                extension_result_path=extension_result_path,
                merged_output_path=merged_output_path,
            )
            current_segment = RenderedSegment(path=merged_path, video_ref=next_video_ref)

        shutil.copyfile(current_segment.path, final_clip_path)
        return final_clip_path

    def _render_one_clip(self, clip: ClipPlan) -> Path:
        print(f"\n🎬 渲染 {clip.id}: {clip.title} ({clip.strategy})")
        if clip.strategy == "shot_based":
            return self.render_shot_based_clip(clip)
        return self.render_extend_clip(clip)

    def _record_rendered_clip(self, clip: ClipPlan, final_clip_path: Path) -> None:
        with self._clip_skipped_shots_lock:
            skipped_shots = self._clip_skipped_shots.pop(clip.id, [])

        entry = {
            "status": "partial" if skipped_shots else "success",
            "strategy": clip.strategy,
            "title": clip.title,
            "output_path": str(final_clip_path),
            "anchor_signature": self._anchor_signature_for_clip(clip),
        }
        if skipped_shots:
            entry["skipped_shots"] = skipped_shots

        with self._render_manifest_lock:
            self.render_manifest["clips"][clip.id] = entry
            self._save_manifest()
        if skipped_shots:
            print(f"✅ Clip 已保存: {final_clip_path}（跳过 {len(skipped_shots)} 个 shot）")
        else:
            print(f"✅ Clip 已保存: {final_clip_path}")

    def _record_failed_clip(self, clip: ClipPlan, error: Exception) -> None:
        with self._clip_skipped_shots_lock:
            skipped_shots = self._clip_skipped_shots.pop(clip.id, [])
        entry: dict[str, Any] = {
            "status": "failed",
            "strategy": clip.strategy,
            "title": clip.title,
            "error": str(error)[:1200],
        }
        if skipped_shots:
            entry["skipped_shots"] = skipped_shots
        try:
            entry["anchor_signature"] = self._anchor_signature_for_clip(clip)
        except Exception:
            pass
        with self._render_manifest_lock:
            self.render_manifest["clips"][clip.id] = entry
            self._save_manifest()
        print(f"⚠️  Clip 渲染失败但已记录并继续: {clip.id} - {error}")

    def render_all(self) -> None:
        print("\n" + "=" * 60)
        print("渲染视频 Clip")
        print("=" * 60)

        clips = list(self.series_bible.clips)
        if not clips:
            self._save_manifest()
            self.api_usage_logger.write_summary()
            return

        workers = min(self.video_render_workers, len(clips))
        if workers > 1:
            print(f"🚀 并发渲染 clip: workers={workers}, total_clips={len(clips)}")

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            clip_iter = iter(clips)
            pending = {}

            def submit_next() -> None:
                try:
                    clip = next(clip_iter)
                except StopIteration:
                    return
                pending[executor.submit(self._render_one_clip, clip)] = clip

            for _ in range(workers):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    clip = pending.pop(future)
                    completed += 1
                    try:
                        final_clip_path = future.result()
                    except Exception as error:
                        self._record_failed_clip(clip, error)
                        print(f"⚠️  Clip 渲染进度: {completed}/{len(clips)} ({clip.id}, failed)")
                        submit_next()
                        continue

                    self._record_rendered_clip(clip, final_clip_path)
                    print(f"✅ Clip 渲染进度: {completed}/{len(clips)} ({clip.id})")
                    submit_next()

        self._save_manifest()
        self.api_usage_logger.write_summary()
