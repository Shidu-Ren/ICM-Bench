from __future__ import annotations

import json
import math
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import (
    DEFAULT_VIDEO_CONFIG_PATH,
    DEFAULT_VIDEO_OUTPUT_BASE_DIR,
    build_stable_named_output_root,
    build_staging_output_root,
    ensure_unique_path,
    get_google_api_key,
    get_text_model,
    load_video_config,
)
from video_generator.theme_schemas import (
    CharacterGroups,
    DistributionPlan,
    ProtagonistData,
    SceneGroups,
    SceneReference as AlbumSceneReference,
)
from video_generator.schemas import (
    AudioStrategy,
    CastMember,
    ClipPlan,
    ClipOutlineSet,
    ClipShotBlueprint,
    ReferencePhotoPrompt,
    SceneReference,
    SeriesBatchOutline,
    SeriesBible,
    SeriesShotBlueprints,
    SeriesShotPlan,
    SeriesWardrobePlan,
    ShotBlueprint,
)
from video_generator.api_usage import ApiUsageLogger, estimate_usage_cost, text_model_price, usage_metadata_dict
from video_generator.voice.reference_gen import ReferenceVoiceGenerator, SUPPORTED_PREBUILT_VOICES


DEFAULT_USER_PROMPT = (
    "生成一个东亚男生 Ryan 的 22 岁生日惊喜派对短片。"
    "朋友和家人想偷偷为他准备一个温暖、真实、电影感的生日惊喜。"
)
DEFAULT_TEXT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TEXT_FALLBACK_MODELS = ("gemini-3-flash-preview", "gemini-3.1-pro-preview")
DEFAULT_STRATEGIES = ["shot_based"]
DEFAULT_TIME_SPAN = "1_month"
DEFAULT_START_DATE = "2025-05-01"

TIME_SPAN_DAYS = {
    "1_week": 7,
    "2_weeks": 14,
    "1_month": 30,
    "3_months": 90,
    "6_months": 180,
    "1_year": 365,
}

OUTFIT_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "up",
    "with",
    "while",
    "exactly",
    "same",
    "adult",
    "person",
    "people",
    "man",
    "woman",
    "male",
    "female",
    "east",
    "asian",
    "old",
    "year",
    "years",
    "wearing",
    "wears",
    "worn",
    "outfit",
    "look",
    "looks",
    "style",
}

OUTLINE_SIMILARITY_STOPWORDS = {
    *OUTFIT_MATCH_STOPWORDS,
    "clip",
    "series",
    "group",
    "main",
    "memory",
    "fact",
    "relationship",
    "hook",
    "goal",
    "scene",
    "moment",
    "shows",
    "show",
    "reveals",
    "reveal",
    "recurring",
    "character",
    "characters",
    "friend",
    "friends",
    "family",
    "work",
    "home",
    "office",
    "city",
    "day",
    "night",
    "morning",
    "evening",
    "afternoon",
    "quiet",
    "warm",
    "together",
}


def _normalize_outfit_text(text: str | None) -> str:
    return str(text or "").strip()


def _style_tokens(text: str | None) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return {
        token
        for token in normalized.split()
        if len(token) > 1 and token not in OUTFIT_MATCH_STOPWORDS
    }


def _candidate_outfit_texts(member: CastMember) -> list[str]:
    candidates: list[str] = []
    source_outfits = list(member.wardrobe_options or [])
    if not source_outfits:
        source_outfits = [member.signature_outfit]
    for outfit_text in source_outfits:
        normalized = _normalize_outfit_text(outfit_text)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _best_matching_outfit_text(outfit_text: str, candidates: list[str]) -> str:
    if not candidates:
        return _normalize_outfit_text(outfit_text)

    normalized = _normalize_outfit_text(outfit_text)
    if not normalized:
        return candidates[0]

    for candidate in candidates:
        if normalized == candidate:
            return candidate

    outfit_tokens = _style_tokens(normalized)
    best_text = candidates[0]
    best_score = float("-inf")
    for index, candidate in enumerate(candidates):
        candidate_tokens = _style_tokens(candidate)
        score = len(outfit_tokens & candidate_tokens) * 2.0
        if candidate.lower() in normalized.lower() or normalized.lower() in candidate.lower():
            score += 4.0
        score -= index * 0.01
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text


def _clip_text_for_character(clip: ClipPlan, char_id: str) -> str:
    text_parts: list[str] = []
    for shot in clip.shots:
        if char_id not in shot.visible_characters:
            continue
        text_parts.extend(
            [
                shot.anchor_image_prompt,
                shot.video_prompt,
                shot.blocking_notes,
                " ".join(shot.secondary_actions),
            ]
        )
    return "\n".join(part for part in text_parts if part)


def _infer_clip_outfit_for_member(clip: ClipPlan, member: CastMember) -> str:
    existing = _normalize_outfit_text((clip.clip_character_outfits or {}).get(member.id, ""))
    candidates = _candidate_outfit_texts(member)
    if not candidates:
        return existing
    if existing:
        return _best_matching_outfit_text(existing, candidates)
    if len(candidates) == 1:
        return candidates[0]

    clip_text = _clip_text_for_character(clip, member.id)
    if not clip_text.strip():
        return candidates[0]

    clip_text_lower = clip_text.lower()
    clip_tokens = _style_tokens(clip_text)
    best_text = candidates[0]
    best_score = float("-inf")

    for index, outfit_text in enumerate(candidates):
        outfit_tokens = _style_tokens(outfit_text)
        overlap_score = len(clip_tokens & outfit_tokens) * 2.0
        phrase_bonus = 4.0 if outfit_text.lower() in clip_text_lower else 0.0
        score = overlap_score + phrase_bonus - (index * 0.01)
        if score > best_score:
            best_score = score
            best_text = outfit_text

    return best_text


def _backfill_series_bible_clip_outfits(series_bible: SeriesBible) -> SeriesBible:
    cast_by_id = {member.id: member for member in series_bible.cast}
    updated_clips: list[ClipPlan] = []
    changed = False

    for clip in series_bible.clips:
        visible_character_ids: list[str] = []
        seen_character_ids: set[str] = set()
        for shot in clip.shots:
            for char_id in shot.visible_characters:
                if char_id not in seen_character_ids:
                    visible_character_ids.append(char_id)
                    seen_character_ids.add(char_id)

        resolved_outfits: dict[str, str] = {}
        for char_id, outfit_text in (clip.clip_character_outfits or {}).items():
            normalized_char_id = str(char_id).strip()
            normalized_outfit = _normalize_outfit_text(outfit_text)
            if not normalized_char_id or not normalized_outfit:
                continue
            member = cast_by_id.get(normalized_char_id)
            if member is None:
                resolved_outfits[normalized_char_id] = normalized_outfit
                continue
            resolved_outfits[normalized_char_id] = _best_matching_outfit_text(
                normalized_outfit,
                _candidate_outfit_texts(member),
            )

        for char_id in visible_character_ids:
            if char_id in resolved_outfits:
                continue
            member = cast_by_id.get(char_id)
            if member is None:
                continue
            inferred_outfit = _infer_clip_outfit_for_member(clip, member)
            if inferred_outfit:
                resolved_outfits[char_id] = inferred_outfit

        if resolved_outfits != (clip.clip_character_outfits or {}):
            changed = True
            updated_clips.append(clip.model_copy(update={"clip_character_outfits": resolved_outfits}))
        else:
            updated_clips.append(clip)

    if not changed:
        return series_bible

    return series_bible.model_copy(update={"clips": updated_clips})


class VideoSeriesPlanner:
    """Generate multi-stage video metadata by reusing album-style planning steps."""

    def __init__(
        self,
        user_prompt: str,
        clip_count_min: int = 20,
        clip_count_max: int = 30,
        default_clip_strategy: str = "shot_based",
        strategies: list[str] | None = None,
        target_runtime_seconds_min: int = 10,
        target_runtime_seconds_max: int = 60,
        target_total_runtime_seconds: int | None = None,
        target_total_runtime_tolerance_seconds: int | None = None,
        target_core_cast_size_min: int = 4,
        target_core_cast_size_max: int = 6,
        max_visible_characters: int = 6,
        allow_background_extras: bool = True,
        max_background_extras_per_shot: int = 3,
        shot_duration_seconds: int = 6,
        allowed_shot_durations_seconds: list[int] | None = None,
        adult_cast_only: bool = True,
        preferred_min_visible_characters: int = 3,
        preferred_max_visible_characters: int = 6,
        min_three_to_six_shot_fraction: float = 0.7,
        min_five_to_six_shot_fraction: float = 0.2,
        target_one_to_two_shot_fraction: float = 0.2,
        max_single_shot_fraction: float = 0.15,
        max_zero_or_one_shot_fraction: float = 0.05,
        max_empty_shot_fraction: float = 0.1,
        target_average_visible_characters: float = 4.0,
        minimum_dense_shots_per_clip: int = 1,
        minimum_dialogue_beats_per_clip: int = 1,
        dialogue_shots_per_clip_min: int = 1,
        dialogue_shots_per_clip_max: int = 2,
        dialogue_max_lines_per_clip: int = 4,
        dialogue_min_words_per_line: int = 1,
        dialogue_max_words_per_4s_shot: int = 12,
        dialogue_max_words_per_6s_shot: int = 18,
        dialogue_max_total_words_per_4s_shot: int = 14,
        dialogue_max_total_words_per_6s_shot: int = 24,
        minimum_secondary_actions_per_dense_shot: int = 2,
        minimum_distinct_speakers_per_clip: int = 1,
        shot_blueprint_mode: str = "balanced_auto",
        clip_outline_batch_size: int = 25,
        clip_outline_context_mode: str = "previous_outlines",
        clip_outline_similarity_check_enabled: bool = False,
        clip_outline_similarity_threshold: float = 0.30,
        clip_outline_max_previous_examples: int = 500,
        wardrobe_options_max: int = 5,
        time_span: str = DEFAULT_TIME_SPAN,
        start_date: str = DEFAULT_START_DATE,
        planning_reference_count: int = 24,
        workspace_name: str | None = None,
        output_root: str | Path | None = None,
        config_snapshot: dict | None = None,
        stage_text_models: dict[str, str] | None = None,
    ) -> None:
        self.user_prompt = user_prompt
        self.clip_count_min = clip_count_min
        self.clip_count_max = clip_count_max
        self.clip_count_target = max(
            self.clip_count_min,
            min(self.clip_count_max, round((self.clip_count_min + self.clip_count_max) / 2)),
        )
        self.default_clip_strategy = default_clip_strategy
        self.strategies = self._resolve_strategies(self.clip_count_target, strategies or [default_clip_strategy] or DEFAULT_STRATEGIES)
        self.target_runtime_seconds_min = target_runtime_seconds_min
        self.target_runtime_seconds_max = target_runtime_seconds_max
        self.target_total_runtime_seconds = target_total_runtime_seconds
        self.target_total_runtime_tolerance_seconds = target_total_runtime_tolerance_seconds
        self.target_core_cast_size_min = target_core_cast_size_min
        self.target_core_cast_size_max = target_core_cast_size_max
        self.max_visible_characters = max_visible_characters
        self.allow_background_extras = allow_background_extras
        self.max_background_extras_per_shot = max_background_extras_per_shot
        self.shot_duration_seconds = shot_duration_seconds
        self.allowed_shot_durations_seconds = tuple(
            sorted(set(int(value) for value in (allowed_shot_durations_seconds or [4, 6, 8])))
        )
        self.adult_cast_only = adult_cast_only
        self.preferred_min_visible_characters = preferred_min_visible_characters
        self.preferred_max_visible_characters = preferred_max_visible_characters
        self.min_three_to_six_shot_fraction = float(min_three_to_six_shot_fraction)
        self.min_five_to_six_shot_fraction = float(min_five_to_six_shot_fraction)
        self.target_one_to_two_shot_fraction = float(target_one_to_two_shot_fraction)
        self.max_single_shot_fraction = float(max_single_shot_fraction)
        self.max_zero_or_one_shot_fraction = float(max_zero_or_one_shot_fraction)
        self.max_empty_shot_fraction = float(max_empty_shot_fraction)
        self.target_average_visible_characters = float(target_average_visible_characters)
        self.minimum_dense_shots_per_clip = minimum_dense_shots_per_clip
        self.minimum_dialogue_beats_per_clip = minimum_dialogue_beats_per_clip
        self.dialogue_shots_per_clip_min = max(0, int(dialogue_shots_per_clip_min))
        self.dialogue_shots_per_clip_max = max(
            self.dialogue_shots_per_clip_min,
            int(dialogue_shots_per_clip_max),
        )
        self.dialogue_max_lines_per_clip = max(0, int(dialogue_max_lines_per_clip))
        self.dialogue_min_words_per_line = max(0, int(dialogue_min_words_per_line))
        self.dialogue_max_words_per_4s_shot = max(0, int(dialogue_max_words_per_4s_shot))
        self.dialogue_max_words_per_6s_shot = max(0, int(dialogue_max_words_per_6s_shot))
        self.dialogue_max_total_words_per_4s_shot = max(0, int(dialogue_max_total_words_per_4s_shot))
        self.dialogue_max_total_words_per_6s_shot = max(0, int(dialogue_max_total_words_per_6s_shot))
        self.minimum_secondary_actions_per_dense_shot = minimum_secondary_actions_per_dense_shot
        self.minimum_distinct_speakers_per_clip = minimum_distinct_speakers_per_clip
        self.shot_blueprint_mode = shot_blueprint_mode
        self.clip_outline_batch_size = max(1, int(clip_outline_batch_size))
        self.clip_outline_context_mode = str(clip_outline_context_mode or "none").strip().lower()
        self.clip_outline_similarity_check_enabled = bool(clip_outline_similarity_check_enabled)
        self.clip_outline_similarity_threshold = max(0.0, float(clip_outline_similarity_threshold))
        self.clip_outline_max_previous_examples = max(0, int(clip_outline_max_previous_examples))
        self.wardrobe_options_max = max(1, min(5, int(wardrobe_options_max)))
        self._wardrobe_options_by_character_id: dict[str, list[str]] = {}
        self.time_span = time_span
        self.start_date = start_date
        self.planning_reference_count = planning_reference_count
        self.workspace_name = workspace_name.strip() if isinstance(workspace_name, str) and workspace_name.strip() else None
        self.config_snapshot = config_snapshot or {}
        self.stage_text_models = {
            str(key).strip(): str(value).strip()
            for key, value in (stage_text_models or {}).items()
            if str(key).strip() and isinstance(value, str) and value.strip()
        }

        self.output_root = Path(output_root) if output_root else build_staging_output_root(
            base_dir=DEFAULT_VIDEO_OUTPUT_BASE_DIR
        )
        self.metadata_dir = self.output_root / "metadata"
        self.debug_dir = self.output_root / "debug"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.client = genai.Client(api_key=get_google_api_key())
        self.text_model = get_text_model() or DEFAULT_TEXT_MODEL
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.usage_records = self.api_usage_logger.records
        self.video_meta_prompts_dir = Path(__file__).resolve().parent / "meta_prompts"
        self._final_output_root_locked = False

        print("✅ 视频规划器初始化成功")
        print(f"📝 用户 Prompt: {self.user_prompt}")
        print(f"🎬 目标 Clip 数量范围: {self.clip_count_min}-{self.clip_count_max}")
        if self.target_total_runtime_seconds:
            print(f"⏱️  目标总时长: {self.target_total_runtime_seconds / 60:.2f} 分钟")
        print(f"🗓️  时间范围: {self.start_date} -> {self.end_date}")
        print(f"📁 本次输出目录(暂存): {self.output_root}")
        if self.stage_text_models:
            print(f"🧠 阶段模型覆盖: {self.stage_text_models}")

    @property
    def end_date(self) -> str:
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        days = TIME_SPAN_DAYS.get(self.time_span, 30)
        return (start + timedelta(days=days)).strftime("%Y-%m-%d")

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "VideoSeriesPlanner":
        config = load_video_config(config_path)
        series_cfg = config.get("series", {}) if isinstance(config.get("series"), dict) else {}
        production_cfg = config.get("production", {}) if isinstance(config.get("production"), dict) else {}
        timeline_cfg = config.get("timeline", {}) if isinstance(config.get("timeline"), dict) else {}

        if "clip_count" in series_cfg:
            clip_count_min = clip_count_max = int(series_cfg.get("clip_count", 4))
        else:
            clip_count_min = int(series_cfg.get("clip_count_min", 20))
            clip_count_max = int(series_cfg.get("clip_count_max", 30))
        if clip_count_min > clip_count_max:
            raise ValueError("series.clip_count_min cannot exceed series.clip_count_max.")

        default_clip_strategy = str(series_cfg.get("default_clip_strategy", "shot_based")).strip() or "shot_based"
        clip_count_target = round((clip_count_min + clip_count_max) / 2)
        strategies = [default_clip_strategy] * max(1, clip_count_target)

        if "target_runtime_seconds_per_clip" in series_cfg:
            target_runtime_seconds_min = target_runtime_seconds_max = int(
                series_cfg.get("target_runtime_seconds_per_clip", 24)
            )
        else:
            target_runtime_seconds_min = int(series_cfg.get("target_runtime_seconds_per_clip_min", 10))
            target_runtime_seconds_max = int(series_cfg.get("target_runtime_seconds_per_clip_max", 60))
        if target_runtime_seconds_min > target_runtime_seconds_max:
            raise ValueError(
                "series.target_runtime_seconds_per_clip_min cannot exceed series.target_runtime_seconds_per_clip_max."
            )
        target_total_runtime_seconds = None
        target_total_runtime_tolerance_seconds = None
        if "target_total_runtime_minutes" in series_cfg:
            target_total_runtime_seconds = round(float(series_cfg.get("target_total_runtime_minutes", 0)) * 60)
            target_total_runtime_tolerance_seconds = round(
                float(series_cfg.get("target_total_runtime_tolerance_minutes", 5)) * 60
            )

        if "target_core_cast_size" in series_cfg:
            target_core_cast_size_min = target_core_cast_size_max = int(
                series_cfg.get("target_core_cast_size", 6)
            )
        else:
            target_core_cast_size_min = int(series_cfg.get("target_core_cast_size_min", 4))
            target_core_cast_size_max = int(series_cfg.get("target_core_cast_size_max", 6))
        if target_core_cast_size_min > target_core_cast_size_max:
            raise ValueError(
                "series.target_core_cast_size_min cannot exceed series.target_core_cast_size_max."
            )

        shot_duration_seconds = int(production_cfg.get("shot_duration_seconds", 6))
        allowed_shot_durations_seconds = [
            int(value) for value in production_cfg.get("allowed_shot_durations_seconds", [4, 6, 8])
        ]
        runtime_midpoint = round((target_runtime_seconds_min + target_runtime_seconds_max) / 2)
        derived_shot_count = max(
            clip_count_target * max(2, round(runtime_midpoint / max(shot_duration_seconds, 1))),
            24,
        )

        return cls(
            user_prompt=config.get("user_prompt") or DEFAULT_USER_PROMPT,
            clip_count_min=clip_count_min,
            clip_count_max=clip_count_max,
            default_clip_strategy=default_clip_strategy,
            strategies=[str(strategy) for strategy in strategies],
            target_runtime_seconds_min=target_runtime_seconds_min,
            target_runtime_seconds_max=target_runtime_seconds_max,
            target_total_runtime_seconds=target_total_runtime_seconds,
            target_total_runtime_tolerance_seconds=target_total_runtime_tolerance_seconds,
            target_core_cast_size_min=target_core_cast_size_min,
            target_core_cast_size_max=target_core_cast_size_max,
            max_visible_characters=int(series_cfg.get("max_visible_characters", 6)),
            allow_background_extras=bool(series_cfg.get("allow_background_extras", True)),
            max_background_extras_per_shot=int(series_cfg.get("max_background_extras_per_shot", 3)),
            shot_duration_seconds=shot_duration_seconds,
            allowed_shot_durations_seconds=allowed_shot_durations_seconds,
            adult_cast_only=bool(series_cfg.get("adult_cast_only", True)),
            preferred_min_visible_characters=int(series_cfg.get("preferred_min_visible_characters", 3)),
            preferred_max_visible_characters=int(series_cfg.get("preferred_max_visible_characters", 6)),
            min_three_to_six_shot_fraction=float(series_cfg.get("min_three_to_six_shot_fraction", 0.7)),
            min_five_to_six_shot_fraction=float(series_cfg.get("min_five_to_six_shot_fraction", 0.2)),
            target_one_to_two_shot_fraction=float(series_cfg.get("target_one_to_two_shot_fraction", 0.2)),
            max_single_shot_fraction=float(series_cfg.get("max_single_shot_fraction", 0.15)),
            max_zero_or_one_shot_fraction=float(series_cfg.get("max_zero_or_one_shot_fraction", 0.05)),
            max_empty_shot_fraction=float(series_cfg.get("max_empty_shot_fraction", 0.1)),
            target_average_visible_characters=float(series_cfg.get("target_average_visible_characters", 4.0)),
            minimum_dense_shots_per_clip=int(series_cfg.get("minimum_dense_shots_per_clip", 1)),
            minimum_dialogue_beats_per_clip=int(series_cfg.get("minimum_dialogue_beats_per_clip", 1)),
            dialogue_shots_per_clip_min=int(series_cfg.get("dialogue_shots_per_clip_min", 1)),
            dialogue_shots_per_clip_max=int(series_cfg.get("dialogue_shots_per_clip_max", 2)),
            dialogue_max_lines_per_clip=int(series_cfg.get("dialogue_max_lines_per_clip", 4)),
            dialogue_min_words_per_line=int(series_cfg.get("dialogue_min_words_per_line", 1)),
            dialogue_max_words_per_4s_shot=int(series_cfg.get("dialogue_max_words_per_4s_shot", 12)),
            dialogue_max_words_per_6s_shot=int(series_cfg.get("dialogue_max_words_per_6s_shot", 18)),
            dialogue_max_total_words_per_4s_shot=int(series_cfg.get("dialogue_max_total_words_per_4s_shot", 14)),
            dialogue_max_total_words_per_6s_shot=int(series_cfg.get("dialogue_max_total_words_per_6s_shot", 24)),
            minimum_secondary_actions_per_dense_shot=int(
                series_cfg.get("minimum_secondary_actions_per_dense_shot", 2)
            ),
            minimum_distinct_speakers_per_clip=int(
                series_cfg.get("minimum_distinct_speakers_per_clip", 1)
            ),
            shot_blueprint_mode=str(series_cfg.get("shot_blueprint_mode", "balanced_auto")).strip()
            or "balanced_auto",
            clip_outline_batch_size=int(series_cfg.get("clip_outline_batch_size", 25)),
            clip_outline_context_mode=str(series_cfg.get("clip_outline_context_mode", "previous_outlines")),
            clip_outline_similarity_check_enabled=bool(
                series_cfg.get("clip_outline_similarity_check_enabled", False)
            ),
            clip_outline_similarity_threshold=float(series_cfg.get("clip_outline_similarity_threshold", 0.30)),
            clip_outline_max_previous_examples=int(series_cfg.get("clip_outline_max_previous_examples", 500)),
            wardrobe_options_max=int(series_cfg.get("wardrobe_options_max", 5)),
            time_span=str(timeline_cfg.get("time_span", DEFAULT_TIME_SPAN)),
            start_date=str(timeline_cfg.get("start_date", DEFAULT_START_DATE)),
            planning_reference_count=int(
                timeline_cfg.get("planning_reference_count", derived_shot_count)
            ),
            workspace_name=config.get("workspace_name"),
            config_snapshot=config,
            stage_text_models=config.get("text_models", {}) if isinstance(config.get("text_models"), dict) else {},
        )

    def _text_model_for_stage(self, stage: str | None = None) -> str:
        if stage and stage in self.stage_text_models:
            return self.stage_text_models[stage]
        return self.text_model

    def _text_model_sequence(self, selected_model: str) -> list[str]:
        configured_fallbacks = self.config_snapshot.get("text_model_fallbacks", [])
        if isinstance(configured_fallbacks, str):
            configured_fallbacks = [configured_fallbacks]
        fallback_models = [
            str(model).strip()
            for model in configured_fallbacks
            if isinstance(model, str) and model.strip()
        ] or list(DEFAULT_TEXT_FALLBACK_MODELS)

        sequence: list[str] = []
        for model in [selected_model, *fallback_models, DEFAULT_TEXT_MODEL]:
            if model and model not in sequence:
                sequence.append(model)
        return sequence

    def _validate_clip_runtime(self, clip_runtime_seconds: int) -> None:
        if not (self.target_runtime_seconds_min <= clip_runtime_seconds <= self.target_runtime_seconds_max):
            raise ValueError(
                f"Clip target_runtime_seconds={clip_runtime_seconds} is outside the configured range "
                f"{self.target_runtime_seconds_min}-{self.target_runtime_seconds_max}."
            )
        duration_gcd = 0
        for duration in self.allowed_shot_durations_seconds:
            duration_gcd = math.gcd(duration_gcd, duration)
        if duration_gcd and clip_runtime_seconds % duration_gcd != 0:
            raise ValueError(
                f"Clip target_runtime_seconds={clip_runtime_seconds} cannot be represented by "
                f"allowed shot durations {self.allowed_shot_durations_seconds}."
            )

    def _validate_series_runtime(self, clips: list[Any], *, label: str) -> None:
        if not self.target_total_runtime_seconds:
            return
        total_runtime = sum(int(getattr(clip, "target_runtime_seconds", 0)) for clip in clips)
        tolerance = self.target_total_runtime_tolerance_seconds or 0
        minimum = self.target_total_runtime_seconds - tolerance
        maximum = self.target_total_runtime_seconds + tolerance
        if not (minimum <= total_runtime <= maximum):
            raise ValueError(
                f"{label} total target runtime {total_runtime}s is outside configured range "
                f"{minimum}-{maximum}s."
            )

    def _validate_shot_duration(self, shot_id: str, duration_seconds: int) -> None:
        if duration_seconds not in self.allowed_shot_durations_seconds:
            raise ValueError(
                f"{shot_id} duration_seconds={duration_seconds}; expected one of {self.allowed_shot_durations_seconds}."
            )

    def _validate_visible_character_count(self, shot_id: str, visible_characters: list[str]) -> None:
        if len(visible_characters) > self.max_visible_characters:
            raise ValueError(
                f"{shot_id} has {len(visible_characters)} visible recurring character(s); "
                f"maximum allowed is {self.max_visible_characters}."
            )

    def _report_shot_size_distribution(self, series_plan: SeriesShotPlan) -> None:
        total = 0
        three_to_six = 0
        five_to_six = 0
        one_to_two = 0
        single = 0
        empty = 0
        visible_total = 0
        for clip in series_plan.clips:
            for shot in clip.shots:
                total += 1
                count = len(shot.visible_characters)
                visible_total += count
                if 3 <= count <= 6:
                    three_to_six += 1
                if 5 <= count <= 6:
                    five_to_six += 1
                if 1 <= count <= 2:
                    one_to_two += 1
                if count == 1:
                    single += 1
                if count == 0:
                    empty += 1

        if not total:
            return

        three_to_six_fraction = three_to_six / total
        five_to_six_fraction = five_to_six / total
        one_to_two_fraction = one_to_two / total
        single_fraction = single / total
        zero_or_one_fraction = (single + empty) / total
        empty_fraction = empty / total
        average_visible = visible_total / total

        print(
            "📊 Shot size distribution: "
            f"3-6={three_to_six_fraction:.2f}, "
            f"5-6={five_to_six_fraction:.2f}, "
            f"1-2={one_to_two_fraction:.2f}, "
            f"single={single_fraction:.2f}, "
            f"empty={empty_fraction:.2f}, "
            f"avg_visible={average_visible:.2f}"
        )

        warnings: list[str] = []
        if three_to_six_fraction < self.min_three_to_six_shot_fraction:
            warnings.append(
                f"3-6人镜头占比偏低({three_to_six_fraction:.2f} < {self.min_three_to_six_shot_fraction:.2f})"
            )
        if five_to_six_fraction < self.min_five_to_six_shot_fraction:
            warnings.append(
                f"5-6人镜头占比偏低({five_to_six_fraction:.2f} < {self.min_five_to_six_shot_fraction:.2f})"
            )
        if abs(one_to_two_fraction - self.target_one_to_two_shot_fraction) > 0.12:
            warnings.append(
                f"1-2人镜头占比与目标偏差较大({one_to_two_fraction:.2f} vs {self.target_one_to_two_shot_fraction:.2f})"
            )
        if single_fraction > self.max_single_shot_fraction:
            warnings.append(
                f"单人镜头占比偏高({single_fraction:.2f} > {self.max_single_shot_fraction:.2f})"
            )
        if zero_or_one_fraction > self.max_zero_or_one_shot_fraction:
            warnings.append(
                f"0-1人镜头占比偏高({zero_or_one_fraction:.2f} > {self.max_zero_or_one_shot_fraction:.2f})"
            )
        if empty_fraction > self.max_empty_shot_fraction:
            warnings.append(
                f"空镜头占比偏高({empty_fraction:.2f} > {self.max_empty_shot_fraction:.2f})"
            )
        if abs(average_visible - self.target_average_visible_characters) > 0.6:
            warnings.append(
                f"平均出镜人数偏离目标({average_visible:.2f} vs {self.target_average_visible_characters:.2f})"
            )
        for warning in warnings:
            print(f"⚠️  {warning}")

    def _validate_clip_shot_runtime(self, clip: Any) -> None:
        planned_runtime = sum(shot.duration_seconds for shot in clip.shots)
        if planned_runtime != clip.target_runtime_seconds:
            raise ValueError(
                f"{clip.id} shot durations sum to {planned_runtime}s, "
                f"but target_runtime_seconds is {clip.target_runtime_seconds}s."
            )

    def _validate_clip_count(self, clip_total: int) -> None:
        if not (self.clip_count_min <= clip_total <= self.clip_count_max):
            raise ValueError(
                f"Planned clip count {clip_total} is outside the configured range "
                f"{self.clip_count_min}-{self.clip_count_max}."
            )

    def _validate_cast_size(self, cast_size: int) -> None:
        if not (self.target_core_cast_size_min <= cast_size <= self.target_core_cast_size_max):
            raise ValueError(
                f"Recurring cast size {cast_size} is outside the configured range "
                f"{self.target_core_cast_size_min}-{self.target_core_cast_size_max}."
            )

    @staticmethod
    def _resolve_strategies_from_config(series_cfg: dict[str, Any], clip_count: int) -> list[str]:
        strategy_counts = series_cfg.get("strategy_counts")
        if isinstance(strategy_counts, dict) and strategy_counts:
            shot_based_count = int(strategy_counts.get("shot_based", 0) or 0)
            extend_count = int(strategy_counts.get("extend", 0) or 0)
            if shot_based_count < 0 or extend_count < 0:
                raise ValueError("strategy_counts cannot be negative.")

            strategies = (["shot_based"] * shot_based_count) + (["extend"] * extend_count)
            if not strategies:
                raise ValueError("strategy_counts must request at least one clip.")
            if clip_count and clip_count != len(strategies):
                raise ValueError(
                    "series.clip_count must equal the total of series.strategy_counts "
                    f"({len(strategies)}), got {clip_count}."
                )
            return strategies

        strategies = series_cfg.get("strategies") or DEFAULT_STRATEGIES
        if not isinstance(strategies, list):
            strategies = DEFAULT_STRATEGIES
        return VideoSeriesPlanner._resolve_strategies(clip_count, [str(strategy) for strategy in strategies])

    @staticmethod
    def _resolve_strategies(clip_count: int, strategies: list[str]) -> list[str]:
        normalized = [str(strategy).strip() for strategy in strategies if str(strategy).strip()]
        if not normalized:
            normalized = DEFAULT_STRATEGIES.copy()

        if len(normalized) == clip_count:
            return normalized

        if len(normalized) > clip_count:
            return normalized[:clip_count]

        resolved = normalized.copy()
        index = 0
        while len(resolved) < clip_count:
            resolved.append(normalized[index % len(normalized)])
            index += 1
        return resolved

    def _load_meta_prompt(self, prompt_file: str) -> str:
        with open(self.video_meta_prompts_dir / prompt_file, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _subset_catalog(catalog: list[dict[str, Any]], allowed_ids: list[str]) -> list[dict[str, Any]]:
        allowed_set = set(allowed_ids)
        return [entry for entry in catalog if entry.get("id") in allowed_set]

    @staticmethod
    def _extract_json_candidates(raw_text: str) -> list[str]:
        cleaned = (raw_text or "").strip()
        if not cleaned:
            return []

        normalized = (
            cleaned.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        candidates: list[str] = [normalized]

        fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", normalized, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(match.strip() for match in fence_matches if match.strip())

        first_object = normalized.find("{")
        last_object = normalized.rfind("}")
        if first_object != -1 and last_object != -1 and last_object > first_object:
            candidates.append(normalized[first_object : last_object + 1].strip())

        first_array = normalized.find("[")
        last_array = normalized.rfind("]")
        if first_array != -1 and last_array != -1 and last_array > first_array:
            candidates.append(normalized[first_array : last_array + 1].strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    @staticmethod
    def _try_load_json_candidate(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            trimmed = re.sub(r",(\s*[}\]])", r"\1", candidate)
            return json.loads(trimmed)

    def _load_json_from_response_text(self, raw_text: str) -> Any:
        last_error: Exception | None = None
        for candidate in self._extract_json_candidates(raw_text):
            try:
                return self._try_load_json_candidate(candidate)
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise ValueError("模型返回为空，无法提取 JSON。")

    @staticmethod
    def _normalize_payload_for_schema(payload: Any) -> Any:
        if isinstance(payload, list) and len(payload) == 1:
            return payload[0]
        return payload

    def _repair_json_response_text(self, raw_text: str) -> str:
        if not raw_text.strip():
            raise ValueError("模型返回为空，无法修复 JSON。")
        repair_prompt = (
            "You are repairing malformed JSON produced by a planning model.\n"
            "Return only strict JSON. Do not add markdown fences or explanations.\n"
            "Keep the content as intact as possible while making it valid JSON.\n\n"
            "Malformed JSON:\n"
            f"{raw_text}"
        )
        repair_model = self._text_model_sequence(self.text_model)[0]
        response = self.client.models.generate_content(
            model=repair_model,
            contents=[repair_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )
        self._record_api_usage(
            response=response,
            operation="json_repair",
            model=repair_model,
            prompt=repair_prompt,
            attempt=None,
            schema_class=None,
        )
        return getattr(response, "text", "") or ""

    def _coerce_response_to_schema(
        self,
        response,
        schema_class: type[BaseModel],
    ) -> BaseModel:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                parsed = self._normalize_payload_for_schema(parsed)
                return schema_class.model_validate(parsed)
            except Exception as parsed_exc:
                raw_text = getattr(response, "text", "") or ""
                if not raw_text.strip():
                    raise ValueError(f"Structured response parsed as invalid/empty object: {parsed_exc}") from parsed_exc

        raw_text = getattr(response, "text", "") or ""
        try:
            json_data = self._load_json_from_response_text(raw_text)
        except Exception:
            repaired_text = self._repair_json_response_text(raw_text)
            json_data = self._load_json_from_response_text(repaired_text)
        json_data = self._normalize_payload_for_schema(json_data)
        return schema_class.model_validate(json_data)

    @staticmethod
    def _response_debug_text(response: Any) -> str:
        if response is None:
            return ""
        parts: list[str] = []
        text = getattr(response, "text", "") or ""
        if text:
            parts.append("TEXT:\n" + text)
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                parts.append("PARSED:\n" + json.dumps(parsed, indent=2, ensure_ascii=False, default=str))
            except Exception:
                parts.append(f"PARSED:\n{parsed}")
        candidates = getattr(response, "candidates", None) or []
        for index, candidate in enumerate(candidates, start=1):
            finish_reason = getattr(candidate, "finish_reason", None)
            safety_ratings = getattr(candidate, "safety_ratings", None)
            parts.append(f"CANDIDATE {index} finish_reason={finish_reason} safety_ratings={safety_ratings}")
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None) if content is not None else None
            if candidate_parts:
                for part in candidate_parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(f"CANDIDATE {index} PART TEXT:\n{part_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _schema_contract(schema_class: type[BaseModel]) -> str:
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2, ensure_ascii=False)
        return (
            "\n\nSTRICT OUTPUT CONTRACT:\n"
            "- Return only one valid JSON object. Do not include markdown fences or explanations.\n"
            "- The JSON object must validate against the following schema exactly.\n"
            "- Include every required property, even when the value is an empty list, empty object, null, or a short string.\n"
            "- Do not invent alternative field names or move fields into a different nesting level.\n"
            "- All string content should be in English unless a field explicitly asks for Chinese.\n\n"
            "JSON schema:\n"
            f"{schema_json}"
        )

    def _save_metadata_text(self, data: str, filename: str) -> None:
        filepath = self.metadata_dir / filename
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(data)

    def _save_metadata_model(self, data: BaseModel, filename: str) -> None:
        filepath = self.metadata_dir / filename
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(data.model_dump_json(indent=2, exclude_none=True))
        print(f"💾 已保存: {filepath}")

    def _save_metadata_json(self, data: dict[str, Any], filename: str) -> None:
        filepath = self.metadata_dir / filename
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"💾 已保存: {filepath}")

    def _append_api_usage_record(self, record: dict[str, Any]) -> None:
        self.api_usage_logger.append_record(record)

    @staticmethod
    def _usage_metadata_dict(response: Any) -> dict[str, Any]:
        return usage_metadata_dict(response)

    @staticmethod
    def _text_model_price(model: str, prompt_tokens: int) -> tuple[float, float]:
        return text_model_price(model, prompt_tokens)

    def _estimate_usage_cost(self, model: str, usage: dict[str, Any]) -> dict[str, Any]:
        return estimate_usage_cost(model, usage)

    def _record_api_usage(
        self,
        *,
        response: Any,
        operation: str,
        model: str,
        prompt: str,
        attempt: int | None = None,
        schema_class: type[BaseModel] | None = None,
    ) -> None:
        self.api_usage_logger.record_response(
            response=response,
            operation=operation,
            model=model,
            prompt=prompt,
            attempt=attempt,
            schema=schema_class.__name__ if schema_class is not None else None,
        )

    def _write_api_usage_summary(self) -> None:
        self.api_usage_logger.write_summary()

    def _save_config_snapshot(self) -> None:
        filepath = self.metadata_dir / "00_video_config_snapshot.json"
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.config_snapshot, file, indent=2, ensure_ascii=False)
        print(f"💾 已保存配置快照: {filepath}")

    def _finalize_output_root(self, protagonist_name: str) -> None:
        if self._final_output_root_locked:
            return

        stable_name = self.workspace_name or protagonist_name
        final_root = build_stable_named_output_root(
            name=stable_name,
            base_dir=DEFAULT_VIDEO_OUTPUT_BASE_DIR,
        )
        final_root = ensure_unique_path(final_root)
        final_root.mkdir(parents=True, exist_ok=True)

        for child in self.output_root.iterdir():
            target = final_root / child.name
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(child), str(target))
            else:
                if target.exists():
                    target.unlink()
                shutil.move(str(child), str(target))

        try:
            self.output_root.rmdir()
        except OSError:
            pass

        self.output_root = final_root
        self.metadata_dir = self.output_root / "metadata"
        self.debug_dir = self.output_root / "debug"
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.usage_records = self.api_usage_logger.records
        self._final_output_root_locked = True
        print(f"📁 视频输出目录已锁定: {self.output_root}")

    def _generate_with_schema(
        self,
        prompt: str,
        schema_class: type[BaseModel],
        temperature: float = 0.9,
        max_retries: int = 3,
        model: str | None = None,
        operation: str = "generate_with_schema",
        allow_model_fallback: bool = True,
    ) -> BaseModel:
        import time

        base_prompt = prompt + self._schema_contract(schema_class)
        selected_model = model or self.text_model
        model_sequence = (
            self._text_model_sequence(selected_model)
            if allow_model_fallback
            else [selected_model]
        )

        last_error_text = ""
        last_exception: Exception | None = None
        for model_index, current_model in enumerate(model_sequence):
            if model_index > 0:
                print(
                    f"\n🛟 {schema_class.__name__} 当前模型连续 "
                    f"{max_retries} 次失败，切换到 {current_model} 继续生成。"
                )

            for attempt in range(max_retries):
                response = None
                try:
                    attempt_prompt = base_prompt
                    if last_error_text:
                        attempt_prompt += (
                            "\n\nREPAIR NOTE FROM PREVIOUS ATTEMPT:\n"
                            f"{last_error_text}\n"
                            "Fix only the schema/JSON issues and return a complete corrected JSON object."
                        )
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=[attempt_prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=max(0.3, temperature - 0.2 * attempt),
                            max_output_tokens=65536,
                        ),
                    )
                    self._record_api_usage(
                        response=response,
                        operation=operation,
                        model=current_model,
                        prompt=attempt_prompt,
                        attempt=attempt + 1,
                        schema_class=schema_class,
                    )
                    return self._coerce_response_to_schema(response, schema_class)
                except Exception as exc:
                    last_exception = exc
                    last_error_text = str(exc)
                    print(
                        f"\n⚠️  视频规划生成失败 "
                        f"(model={current_model}, 尝试 {attempt + 1}/{max_retries}): {exc}"
                    )
                    self.debug_dir.mkdir(parents=True, exist_ok=True)
                    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", current_model)
                    safe_operation = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation)
                    debug_path = self.debug_dir / (
                        f"{safe_operation}_{schema_class.__name__}_{safe_model}_attempt_{attempt + 1}.txt"
                    )
                    try:
                        with open(debug_path, "w", encoding="utf-8") as file:
                            file.write(str(exc))
                            file.write("\n\n")
                            file.write(self._response_debug_text(response))
                        print(f"   已保存调试信息到: {debug_path}")
                    except Exception:
                        pass

                    if attempt < max_retries - 1:
                        wait_seconds = 2 ** attempt
                        time.sleep(wait_seconds)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("无法生成视频元数据。")

    def _flatten_character_groups(self, character_groups: CharacterGroups) -> list[dict[str, Any]]:
        groups = character_groups.character_groups
        flattened: list[dict[str, Any]] = []
        for group_name in ("core_family", "close_friends", "colleagues", "acquaintances", "other"):
            for member in getattr(groups, group_name, []):
                base_wardrobe = member.wardrobe_options or [member.typical_clothing]
                flattened.append(
                    {
                        "id": member.id,
                        "name_en": member.name_en,
                        "name_cn": member.name_cn,
                        "group": group_name,
                        "relation_to_protagonist": member.relation_to_protagonist,
                        "age": member.age,
                        "gender": member.gender,
                        "appearance_description": member.appearance_description,
                        "typical_clothing": member.typical_clothing,
                        "wardrobe_options": self._wardrobe_options_for_character(member.id, base_wardrobe),
                        "personality_brief": member.personality_brief,
                    }
                )
        return flattened

    def _dedupe_wardrobe_options(self, outfits: list[str], *, limit: int | None = None) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for outfit in outfits:
            text = _normalize_outfit_text(outfit)
            if not text:
                continue
            key = re.sub(r"\s+", " ", text.lower())
            if key in seen:
                continue
            cleaned.append(text)
            seen.add(key)
            if limit is not None and len(cleaned) >= limit:
                break
        return cleaned

    def _wardrobe_options_for_character(self, char_id: str, base_options: list[str]) -> list[str]:
        override = self._wardrobe_options_by_character_id.get(str(char_id).strip())
        if override:
            return self._dedupe_wardrobe_options(override, limit=self.wardrobe_options_max)
        return self._dedupe_wardrobe_options(base_options, limit=self.wardrobe_options_max)

    def _base_protagonist_wardrobe_options(self, protagonist_data: ProtagonistData) -> list[str]:
        return self._dedupe_wardrobe_options(
            [
                protagonist_data.clothing_styles.casual_daily,
                protagonist_data.clothing_styles.work_attire,
                protagonist_data.clothing_styles.outdoor_activities,
                protagonist_data.clothing_styles.travel_outfit,
                protagonist_data.clothing_styles.formal_wear or "",
            ],
            limit=self.wardrobe_options_max,
        )

    def _set_wardrobe_plan(self, wardrobe_plan: SeriesWardrobePlan | None) -> None:
        if wardrobe_plan is None:
            self._wardrobe_options_by_character_id = {}
            return
        self._wardrobe_options_by_character_id = {
            character.character_id: self._dedupe_wardrobe_options(
                character.wardrobe_options,
                limit=self.wardrobe_options_max,
            )
            for character in wardrobe_plan.characters
        }

    def _flatten_scene_groups(self, scene_groups: SceneGroups) -> list[dict[str, Any]]:
        groups = scene_groups.scene_groups
        flattened: list[dict[str, Any]] = []
        for group_name in ("high_frequency", "medium_frequency", "low_frequency"):
            for scene in getattr(groups, group_name, []):
                flattened.append(
                    {
                        "id": scene.id,
                        "name_en": scene.name_en,
                        "name_cn": scene.name_cn,
                        "group": group_name,
                        "category": scene.category,
                        "frequency": scene.frequency,
                        "description": scene.description,
                        "lighting": scene.lighting,
                        "mood": scene.mood,
                    }
                )
        return flattened

    @staticmethod
    def _scene_title_from_id(scene_id: str) -> str:
        words = [word for word in re.split(r"[_\\-]+", str(scene_id).strip()) if word]
        return " ".join(word.capitalize() for word in words) or "One-Off Scene"

    @staticmethod
    def _scene_category_from_id(scene_id: str, context_text: str) -> str:
        text = f"{scene_id} {context_text}".lower()
        if any(word in text for word in ("office", "meeting", "desk", "work", "tech", "cowork", "colleague")):
            return "work"
        if any(word in text for word in ("travel", "airport", "hotel", "station", "europe", "trip", "souvenir")):
            return "travel"
        if any(word in text for word in ("home", "apartment", "kitchen", "living", "balcony", "bedroom", "entryway")):
            return "home"
        if any(word in text for word in ("park", "street", "riverside", "outdoor", "promenade", "subway")):
            return "outdoor"
        return "social"

    def _make_dynamic_scene_reference(self, scene_id: str, outline_clip: Any) -> AlbumSceneReference:
        title = self._scene_title_from_id(scene_id)
        context_parts = [
            getattr(outline_clip, "title", ""),
            getattr(outline_clip, "logline", ""),
            getattr(outline_clip, "story_purpose", ""),
            " ".join(getattr(outline_clip, "outline_beats", []) or []),
            " ".join(getattr(outline_clip, "memory_facts", []) or []),
        ]
        context_text = re.sub(r"\s+", " ", " ".join(str(part) for part in context_parts if part)).strip()
        category = self._scene_category_from_id(scene_id, context_text)
        short_context = context_text[:260] if context_text else "A specific one-off location requested by the clip outline."
        description = (
            f"A low-frequency one-off location for clip {getattr(outline_clip, 'id', 'unknown')}: {short_context} "
            "The setting should be realistic, adult-only, visually readable, and specific enough to support "
            "a concrete life-album memory without becoming a recurring location."
        )
        lighting = "naturalistic location-appropriate lighting"
        mood = "grounded, specific, documentary-like"
        background_prompt = (
            f"Realistic cinematic background for {title}. {description} "
            "Warm documentary life-album style, clear composition, adult-only environment, no children, "
            "no readable brand logos, no text overlays, no watermark."
        )
        return AlbumSceneReference(
            id=str(scene_id),
            name_en=title,
            name_cn=title,
            category=category,
            frequency="low",
            description=description,
            lighting=lighting,
            mood=mood,
            background_prompt=background_prompt,
        )

    def _expand_scene_groups_for_clip_outline(
        self,
        scene_groups_data: SceneGroups,
        clip_outline: ClipOutlineSet,
    ) -> SceneGroups:
        existing_scene_ids = {entry["id"] for entry in self._flatten_scene_groups(scene_groups_data)}
        recurring_scene_ids = set(clip_outline.recurring_scene_ids or [])
        added_scenes: list[AlbumSceneReference] = []
        for outline_clip in clip_outline.clips:
            for scene_id in getattr(outline_clip, "primary_scene_ids", []) or []:
                scene_id = str(scene_id).strip()
                if not scene_id or scene_id in existing_scene_ids:
                    continue
                scene_ref = self._make_dynamic_scene_reference(scene_id, outline_clip)
                scene_groups_data.scene_groups.low_frequency.append(scene_ref)
                existing_scene_ids.add(scene_id)
                added_scenes.append(scene_ref)

        if not added_scenes:
            return scene_groups_data

        summary = scene_groups_data.summary
        summary.high_frequency_count = len(scene_groups_data.scene_groups.high_frequency)
        summary.medium_frequency_count = len(scene_groups_data.scene_groups.medium_frequency)
        summary.low_frequency_count = len(scene_groups_data.scene_groups.low_frequency)
        summary.total_scenes = (
            summary.high_frequency_count
            + summary.medium_frequency_count
            + summary.low_frequency_count
        )
        self._save_metadata_model(scene_groups_data, "04_scene_groups.json")
        self._save_metadata_json(
            {
                "added_scene_count": len(added_scenes),
                "added_scene_ids": [scene.id for scene in added_scenes],
                "note": (
                    "These low-frequency scenes were dynamically created from clip primary_scene_ids "
                    "that were not present in the original scene catalog. recurring_scene_ids remain "
                    "the stable reusable scene pool."
                ),
            },
            "04c_dynamic_scene_expansion.json",
        )
        print(
            "🧩 动态补充 one-off scenes: "
            f"{len(added_scenes)} ({', '.join(scene.id for scene in added_scenes[:8])})"
        )
        missing_recurring = sorted(scene_id for scene_id in recurring_scene_ids if scene_id not in existing_scene_ids)
        if missing_recurring:
            raise ValueError(f"Recurring scene ids missing from expanded scene catalog: {missing_recurring}")
        return scene_groups_data

    def _is_minor_like_character(self, *, age: int | None, relation: str | None, role_text: str | None = None) -> bool:
        if age is not None and age < 18:
            return True
        text = " ".join(part for part in (relation or "", role_text or "") if part).lower()
        minor_terms = (
            "child",
            "children",
            "son",
            "daughter",
            "teen",
            "teenager",
            "minor",
            "student",
            "school",
            "kid",
            "baby",
            "toddler",
        )
        return any(term in text for term in minor_terms)

    def _remove_minor_like_people(
        self,
        protagonist_data: ProtagonistData,
        character_groups: CharacterGroups | None = None,
    ) -> None:
        if not self.adult_cast_only:
            return

        protagonist_data.family = [
            member
            for member in protagonist_data.family
            if not self._is_minor_like_character(
                age=member.age,
                relation=member.relation,
                role_text=member.brief_description,
            )
        ]

        if character_groups is None:
            return

        groups = character_groups.character_groups
        for group_name in ("core_family", "close_friends", "colleagues", "acquaintances", "other"):
            people = getattr(groups, group_name, [])
            filtered_people = [
                member
                for member in people
                if not self._is_minor_like_character(
                    age=member.age,
                    relation=member.relation_to_protagonist,
                    role_text=member.personality_brief,
                )
            ]
            setattr(groups, group_name, filtered_people)

        character_groups.summary.core_family_count = len(groups.core_family)
        character_groups.summary.close_friends_count = len(groups.close_friends)
        character_groups.summary.colleagues_count = len(groups.colleagues)
        character_groups.summary.other_count = len(groups.acquaintances) + len(groups.other)
        character_groups.summary.total_characters = (
            len(groups.core_family)
            + len(groups.close_friends)
            + len(groups.colleagues)
            + len(groups.acquaintances)
            + len(groups.other)
        )

    def _build_character_catalog(
        self,
        protagonist_data: ProtagonistData,
        character_groups: CharacterGroups,
    ) -> list[dict[str, Any]]:
        catalog = [
            {
                "id": "protagonist",
                "name_en": protagonist_data.basic_info.name_en,
                "name_cn": protagonist_data.basic_info.name_cn,
                "group": "protagonist",
                "relation_to_protagonist": "self",
                "age": protagonist_data.basic_info.age,
                "gender": protagonist_data.basic_info.gender,
                "appearance_description": protagonist_data.appearance.detailed_description,
                "typical_clothing": protagonist_data.clothing_styles.casual_daily,
                "wardrobe_options": self._wardrobe_options_for_character(
                    "protagonist",
                    self._base_protagonist_wardrobe_options(protagonist_data),
                ),
                "personality_brief": ", ".join(protagonist_data.personality_traits[:4]),
            }
        ]
        catalog.extend(self._flatten_character_groups(character_groups))
        return catalog

    @staticmethod
    def _default_outfit_from_character_catalog_entry(entry: dict[str, Any] | None) -> str:
        if not entry:
            return ""
        wardrobe_options = [
            str(option).strip()
            for option in entry.get("wardrobe_options", []) or []
            if str(option).strip()
        ]
        if wardrobe_options:
            return wardrobe_options[0]
        return str(entry.get("typical_clothing", "") or "").strip()

    @staticmethod
    def _catalog_outfit_options(entry: dict[str, Any] | None) -> list[str]:
        if not entry:
            return []
        wardrobe_options = [
            str(option).strip()
            for option in entry.get("wardrobe_options", []) or []
            if str(option).strip()
        ]
        if wardrobe_options:
            return wardrobe_options
        fallback = str(entry.get("typical_clothing", "") or "").strip()
        return [fallback] if fallback else []

    def _resolve_clip_character_outfits(
        self,
        clip: ClipPlan,
        character_catalog: list[dict[str, Any]],
    ) -> ClipPlan:
        catalog_by_id = {entry["id"]: entry for entry in character_catalog}
        selected_outfits = {
            str(char_id).strip(): str(outfit_text).strip()
            for char_id, outfit_text in (clip.clip_character_outfits or {}).items()
            if str(char_id).strip() and str(outfit_text).strip()
        }

        used_character_ids: list[str] = []
        seen_character_ids: set[str] = set()
        for shot in clip.shots:
            for char_id in shot.visible_characters:
                if char_id not in seen_character_ids:
                    used_character_ids.append(char_id)
                    seen_character_ids.add(char_id)

        resolved_outfits: dict[str, str] = {}
        for char_id in used_character_ids:
            options = self._catalog_outfit_options(catalog_by_id.get(char_id))
            outfit_text = selected_outfits.get(char_id, "")
            if outfit_text:
                resolved_outfits[char_id] = _best_matching_outfit_text(outfit_text, options)
                continue
            default_outfit = options[0] if options else self._default_outfit_from_character_catalog_entry(catalog_by_id.get(char_id))
            if default_outfit:
                resolved_outfits[char_id] = default_outfit

        return clip.model_copy(update={"clip_character_outfits": resolved_outfits})

    def _reference_prompts_from_album(self, prompts: list[Any]) -> list[ReferencePhotoPrompt]:
        references: list[ReferencePhotoPrompt] = []
        for prompt in prompts:
            references.append(
                ReferencePhotoPrompt(
                    photo_type=getattr(prompt, "photo_type", "portrait_front"),
                    prompt=getattr(prompt, "prompt", ""),
                    aspect_ratio=getattr(prompt, "aspect_ratio", "3:4") or "3:4",
                )
            )
        return references

    def _voice_brief(self, gender: str, personality_text: str) -> str:
        gender_label = "male" if str(gender).lower() == "male" else "female"
        personality_hint = personality_text.split(",")[0].strip() if personality_text else "natural"
        return f"A {gender_label} voice with a {personality_hint} conversational tone."

    def _build_cast_lookup(
        self,
        protagonist_data: ProtagonistData,
        character_groups: CharacterGroups,
    ) -> dict[str, CastMember]:
        cast_lookup: dict[str, CastMember] = {
            "protagonist": CastMember(
                id="protagonist",
                name_en=protagonist_data.basic_info.name_en,
                name_cn=protagonist_data.basic_info.name_cn,
                is_protagonist=True,
                role="protagonist",
                relation_to_protagonist="self",
                age=protagonist_data.basic_info.age,
                gender=protagonist_data.basic_info.gender,
                appearance_description=protagonist_data.appearance.detailed_description,
                signature_outfit=protagonist_data.clothing_styles.casual_daily,
                wardrobe_options=self._wardrobe_options_for_character(
                    "protagonist",
                    self._base_protagonist_wardrobe_options(protagonist_data),
                ),
                personality_brief=", ".join(protagonist_data.personality_traits[:5]),
                voice_brief=self._voice_brief(
                    protagonist_data.basic_info.gender,
                    ", ".join(protagonist_data.personality_traits[:5]),
                ),
                reference_photo_prompts=self._reference_prompts_from_album(protagonist_data.reference_photos),
            )
        }

        groups = character_groups.character_groups
        for group_name in ("core_family", "close_friends", "colleagues", "acquaintances", "other"):
            for member in getattr(groups, group_name, []):
                cast_lookup[member.id] = CastMember(
                    id=member.id,
                    name_en=member.name_en,
                    name_cn=member.name_cn,
                    is_protagonist=False,
                    role=member.relation_to_protagonist,
                    relation_to_protagonist=member.relation_to_protagonist,
                    age=member.age,
                    gender=member.gender,
                    appearance_description=member.appearance_description,
                    signature_outfit=member.typical_clothing,
                    wardrobe_options=self._wardrobe_options_for_character(
                        member.id,
                        member.wardrobe_options or [member.typical_clothing],
                    ),
                    personality_brief=member.personality_brief,
                    voice_brief=self._voice_brief(member.gender, member.personality_brief),
                    reference_photo_prompts=self._reference_prompts_from_album(member.reference_photos),
                )
        return cast_lookup

    def _voice_config(self) -> dict[str, Any]:
        voice_cfg = self.config_snapshot.get("voice", {})
        return voice_cfg if isinstance(voice_cfg, dict) else {}

    @staticmethod
    def _clean_supported_voice_name(voice_name: Any) -> str:
        cleaned = str(voice_name or "").strip()
        return cleaned if cleaned.lower() in SUPPORTED_PREBUILT_VOICES else ""

    def _configured_voice_pool(self, key: str, fallback: list[str]) -> list[str]:
        raw_pool = self._voice_config().get(key, fallback)
        if not isinstance(raw_pool, list):
            raw_pool = fallback
        cleaned: list[str] = []
        for voice_name in raw_pool:
            supported = self._clean_supported_voice_name(voice_name)
            if supported and supported not in cleaned:
                cleaned.append(supported)
        return cleaned or fallback

    def _load_existing_voice_assignments(self) -> dict[str, str]:
        manifest_path = self.metadata_dir / "05_voice_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_assignments = data.get("voice_assignments") or {}
        if not isinstance(raw_assignments, dict):
            return {}
        assignments: dict[str, str] = {}
        for char_id, voice_name in raw_assignments.items():
            supported = self._clean_supported_voice_name(voice_name)
            if supported:
                assignments[str(char_id)] = supported
        return assignments

    def _assign_recurring_cast_voices(self, cast_members: list[CastMember]) -> dict[str, str]:
        voice_cfg = self._voice_config()
        assignments: dict[str, str] = {}

        for source in (self._load_existing_voice_assignments(), voice_cfg.get("voice_assignments") or {}):
            if not isinstance(source, dict):
                continue
            for char_id, voice_name in source.items():
                supported = self._clean_supported_voice_name(voice_name)
                if supported:
                    assignments[str(char_id)] = supported

        used_voices = set(assignments.values())
        male_pool = [
            voice
            for voice in self._configured_voice_pool("default_male_voices", ["Puck", "Charon", "Fenrir", "Orus", "Leda"])
            if voice not in used_voices
        ]
        female_pool = [
            voice
            for voice in self._configured_voice_pool(
                "default_female_voices",
                ["Kore", "Aoede", "Zephyr", "Autonoe", "Callirrhoe"],
            )
            if voice not in used_voices
        ]
        all_pool = self._configured_voice_pool(
            "default_male_voices",
            ["Puck", "Charon", "Fenrir", "Orus", "Leda"],
        ) + self._configured_voice_pool(
            "default_female_voices",
            ["Kore", "Aoede", "Zephyr", "Autonoe", "Callirrhoe"],
        )

        for member in cast_members:
            if member.id in assignments:
                continue
            gender = str(member.gender).strip().lower()
            if gender == "female" and female_pool:
                voice_name = female_pool.pop(0)
            elif male_pool:
                voice_name = male_pool.pop(0)
            elif female_pool:
                voice_name = female_pool.pop(0)
            else:
                voice_name = all_pool[len(assignments) % len(all_pool)]
            assignments[member.id] = voice_name
        return assignments

    def _audition_candidate_voices_for_member(
        self,
        member: CastMember,
        used_voices: set[str],
        candidate_count: int,
    ) -> list[str]:
        gender = str(member.gender).strip().lower()
        if gender == "female":
            pool = self._configured_voice_pool(
                "default_female_voices",
                ["Kore", "Aoede", "Zephyr", "Autonoe", "Callirrhoe"],
            )
        else:
            pool = self._configured_voice_pool(
                "default_male_voices",
                ["Puck", "Charon", "Fenrir", "Orus", "Leda"],
            )

        if not pool:
            return []
        start_index = sum(ord(char) for char in member.id) % len(pool)
        rotated = [pool[(start_index + offset) % len(pool)] for offset in range(len(pool))]
        preferred = [voice for voice in rotated if voice not in used_voices]
        candidates = preferred[:candidate_count]
        for voice in rotated:
            if len(candidates) >= candidate_count:
                break
            if voice not in candidates:
                candidates.append(voice)
        return candidates

    @staticmethod
    def _age_band(age: int) -> str:
        if age >= 60:
            return "older adult"
        if age >= 45:
            return "mature adult"
        if age >= 35:
            return "middle-aged adult"
        return "adult"

    def _audition_script_for_member(self, member: CastMember) -> str:
        relation = str(member.relation_to_protagonist or member.role or "").lower()
        if member.is_protagonist:
            return "I saved the review notes. Let's leave after tea."
        if "wife" in relation or "spouse" in relation or "partner" in relation:
            return "Your blue folder is packed. Let's leave after tea."
        if "mother" in relation:
            return "Eat first, then show me your travel photos."
        if "father" in relation:
            return "Check the tire pressure before the mountain road."
        if "colleague" in relation:
            return "Your cache patch finally stopped waking us up."
        if "friend" in relation:
            return "I still remember our rainy road-trip detour."
        return "Keep this note with the tickets for later."

    def _voice_audition_style_prompt(self, member: CastMember) -> str:
        return (
            f"{member.voice_brief} The speaker is a {self._age_band(int(member.age))} "
            f"{member.gender} adult, {member.relation_to_protagonist} to the protagonist. "
            "Speak naturally, conversationally, and only say the transcript."
        )

    def _fallback_voice_audition_choice(
        self,
        candidates: list[str],
        used_voices: set[str],
    ) -> str:
        for voice_name in candidates:
            if voice_name not in used_voices:
                return voice_name
        return candidates[0] if candidates else "Puck"

    def _judge_voice_audition(
        self,
        member: CastMember,
        candidate_paths: dict[str, Path],
    ) -> dict[str, Any]:
        voice_cfg = self._voice_config()
        judge_model = str(voice_cfg.get("audition_judge_model") or self._text_model_for_stage("protagonist"))
        prompt = (
            "You are selecting the best prebuilt TTS voice for a recurring video character.\n"
            "Listen to each candidate audio sample and choose the voice that best matches the character's "
            "gender, approximate age band, relationship role, and personality. Prefer naturalness and "
            "distinctiveness. Return strict JSON only with fields: winner_voice, reason, scores. "
            "scores must be an array of objects with voice_name, score from 0 to 1, and reason.\n\n"
            f"Character id: {member.id}\n"
            f"Name: {member.name_en}\n"
            f"Age: {member.age} ({self._age_band(int(member.age))})\n"
            f"Gender: {member.gender}\n"
            f"Relation/role: {member.relation_to_protagonist}\n"
            f"Personality: {member.personality_brief}\n"
            f"Voice brief: {member.voice_brief}\n"
        )
        parts = [types.Part.from_text(text=prompt)]
        for voice_name, path in candidate_paths.items():
            parts.append(types.Part.from_text(text=f"Candidate voice_name: {voice_name}"))
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="audio/wav"))
        response = self.client.models.generate_content(
            model=judge_model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        self._record_api_usage(
            response=response,
            operation="voice_audition_judge",
            model=judge_model,
            prompt=prompt,
            attempt=None,
            schema_class=None,
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        return self._load_json_from_response_text(getattr(response, "text", "") or "")

    def _audition_recurring_cast_voices(
        self,
        cast_members: list[CastMember],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        voice_cfg = self._voice_config()
        candidate_count = max(1, min(5, int(voice_cfg.get("audition_candidates_per_character", 3))))
        reuse_existing = bool(voice_cfg.get("audition_reuse_existing", True))
        tts_model = str(voice_cfg.get("tts_model", "gemini-3.1-flash-tts-preview"))
        explicit_assignments = {
            str(char_id): self._clean_supported_voice_name(voice_name)
            for char_id, voice_name in (voice_cfg.get("voice_assignments") or {}).items()
            if self._clean_supported_voice_name(voice_name)
        }
        existing_assignments = self._load_existing_voice_assignments()
        assignments: dict[str, str] = {}
        audition_records: dict[str, Any] = {}
        used_voices: set[str] = set()
        audition_dir = self.output_root / "assets" / "voices" / "auditions"
        ref_generator = ReferenceVoiceGenerator(
            api_key=get_google_api_key(),
            output_dir=self.output_root / "assets" / "voices",
            tts_model=tts_model,
            voice_assignments={},
            male_voices=self._configured_voice_pool(
                "default_male_voices",
                ["Puck", "Charon", "Fenrir", "Orus", "Leda"],
            ),
            female_voices=self._configured_voice_pool(
                "default_female_voices",
                ["Kore", "Aoede", "Zephyr", "Autonoe", "Callirrhoe"],
            ),
        )

        for member in cast_members:
            manual_voice = explicit_assignments.get(member.id)
            if manual_voice:
                assignments[member.id] = manual_voice
                used_voices.add(manual_voice)
                audition_records[member.id] = {
                    "selected_voice": manual_voice,
                    "selection_method": "manual_config",
                    "reason": "voice.voice_assignments explicitly configured this character.",
                }
                continue

            if reuse_existing and member.id in existing_assignments:
                existing_voice = existing_assignments[member.id]
                assignments[member.id] = existing_voice
                used_voices.add(existing_voice)
                audition_records[member.id] = {
                    "selected_voice": existing_voice,
                    "selection_method": "existing_manifest",
                    "reason": "Reusing existing voice assignment.",
                }
                continue

            candidates = self._audition_candidate_voices_for_member(member, used_voices, candidate_count)
            if not candidates:
                fallback = "Puck"
                assignments[member.id] = fallback
                used_voices.add(fallback)
                audition_records[member.id] = {
                    "selected_voice": fallback,
                    "selection_method": "fallback_no_candidates",
                }
                continue

            member_dir = audition_dir / member.id
            member_dir.mkdir(parents=True, exist_ok=True)
            script = self._audition_script_for_member(member)
            style_prompt = self._voice_audition_style_prompt(member)
            candidate_paths: dict[str, Path] = {}
            for voice_name in candidates:
                sample_path = member_dir / f"{voice_name}.wav"
                if not (reuse_existing and ref_generator._is_valid_reference_audio(sample_path)):
                    ref_generator.synthesize_dialogue_line(
                        text=script,
                        voice_name=voice_name,
                        save_path=sample_path,
                        style_prompt=style_prompt,
                    )
                candidate_paths[voice_name] = sample_path

            try:
                judge_result = self._judge_voice_audition(member, candidate_paths)
                if not isinstance(judge_result, dict):
                    raise ValueError("Voice audition judge did not return a JSON object.")
                winner = self._clean_supported_voice_name(judge_result.get("winner_voice"))
                if winner not in candidate_paths:
                    winner = self._fallback_voice_audition_choice(candidates, used_voices)
                selection_method = "gemini_audio_judge"
            except Exception as exc:
                judge_result = {"reason": f"Judge failed: {exc}", "scores": []}
                winner = self._fallback_voice_audition_choice(candidates, used_voices)
                selection_method = "fallback_after_judge_failure"

            assignments[member.id] = winner
            used_voices.add(winner)
            audition_records[member.id] = {
                "selected_voice": winner,
                "selection_method": selection_method,
                "script": script,
                "candidate_paths": {voice: str(path) for voice, path in candidate_paths.items()},
                "judge_result": judge_result,
                "age_band": self._age_band(int(member.age)),
            }

        ApiUsageLogger(self.metadata_dir).write_summary()
        return assignments, audition_records

    def _save_voice_manifest_for_recurring_cast(
        self,
        protagonist_data: ProtagonistData,
        character_groups_data: CharacterGroups,
        recurring_character_ids: list[str],
    ) -> dict[str, str]:
        cast_lookup = self._build_cast_lookup(protagonist_data, character_groups_data)
        cast_members = [
            cast_lookup[char_id]
            for char_id in recurring_character_ids
            if char_id in cast_lookup
        ]
        voice_cfg = self._voice_config()
        audition_enabled = bool(voice_cfg.get("audition_enabled", False))
        if audition_enabled:
            assignments, audition_records = self._audition_recurring_cast_voices(cast_members)
        else:
            assignments = self._assign_recurring_cast_voices(cast_members)
            audition_records = {}
        existing_manifest: dict[str, Any] = {}
        manifest_path = self.metadata_dir / "05_voice_manifest.json"
        if manifest_path.exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing_manifest = {}
        existing_auditions = existing_manifest.get("auditions") or {}
        if isinstance(existing_auditions, dict):
            for char_id, record in list(audition_records.items()):
                existing_record = existing_auditions.get(char_id)
                if (
                    isinstance(existing_record, dict)
                    and isinstance(record, dict)
                    and record.get("selection_method") == "existing_manifest"
                ):
                    audition_records[char_id] = existing_record
        existing_reference_paths = existing_manifest.get("reference_paths") or {}
        if not isinstance(existing_reference_paths, dict):
            existing_reference_paths = {}
        manifest = {
            "tts_model": voice_cfg.get("tts_model", "gemini-3.1-flash-tts-preview"),
            "voice_assignments": assignments,
            "reference_paths": existing_reference_paths,
            "audition_enabled": audition_enabled,
            "auditions": audition_records,
            "characters": [
                {
                    "character_id": member.id,
                    "name_en": member.name_en,
                    "name_cn": member.name_cn,
                    "age": member.age,
                    "age_band": self._age_band(int(member.age)),
                    "gender": member.gender,
                    "relation_to_protagonist": member.relation_to_protagonist,
                    "voice_name": assignments.get(member.id),
                    "voice_brief": member.voice_brief,
                }
                for member in cast_members
            ],
        }
        self._save_metadata_json(manifest, "05_voice_manifest.json")
        return assignments

    def _build_scene_lookup(self, scene_groups: SceneGroups) -> dict[str, SceneReference]:
        scene_lookup: dict[str, SceneReference] = {}
        groups = scene_groups.scene_groups
        for group_name in ("high_frequency", "medium_frequency", "low_frequency"):
            for scene in getattr(groups, group_name, []):
                scene_lookup[scene.id] = SceneReference(
                    id=scene.id,
                    name_en=scene.name_en,
                    name_cn=scene.name_cn,
                    description=scene.description,
                    lighting=scene.lighting,
                    mood=scene.mood,
                    background_prompt=scene.background_prompt,
                    aspect_ratio="16:9",
                )
        return scene_lookup

    def generate_protagonist(self) -> ProtagonistData:
        print("\n" + "=" * 60)
        print("步骤 1/9: 生成主角数据")
        print("=" * 60)

        template = self._load_meta_prompt("01_protagonist.txt")
        prompt = template.format(user_prompt=self.user_prompt)
        self._save_metadata_text(prompt, "00_01_protagonist_prompt.txt")

        protagonist_data = self._generate_with_schema(
            prompt=prompt,
            schema_class=ProtagonistData,
            temperature=1.0,
            model=self._text_model_for_stage("protagonist"),
            operation="protagonist",
        )
        self._remove_minor_like_people(protagonist_data)
        self._save_metadata_model(protagonist_data, "01_protagonist.json")
        self._finalize_output_root(protagonist_data.basic_info.name_en)
        return protagonist_data

    def generate_distribution(self, protagonist_data: ProtagonistData) -> DistributionPlan:
        print("\n" + "=" * 60)
        print("步骤 2/9: 生成时间分布计划")
        print("=" * 60)

        template = self._load_meta_prompt("02_distribution.txt")
        prompt = template.format(
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            total_photos=self.planning_reference_count,
            time_span=self.time_span,
            start_date=self.start_date,
        )
        self._save_metadata_text(prompt, "00_02_distribution_prompt.txt")

        distribution_data = self._generate_with_schema(
            prompt=prompt,
            schema_class=DistributionPlan,
            temperature=0.8,
            model=self._text_model_for_stage("distribution"),
            operation="distribution",
        )
        self._save_metadata_model(distribution_data, "02_distribution.json")
        return distribution_data

    def generate_character_groups(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
    ) -> CharacterGroups:
        print("\n" + "=" * 60)
        print("步骤 3/9: 生成人物组数据")
        print("=" * 60)

        template = self._load_meta_prompt("03_character_groups.txt")
        prompt = template.format(
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            total_photos=self.planning_reference_count,
        )
        self._save_metadata_text(prompt, "00_03_character_groups_prompt.txt")

        character_groups_data = self._generate_with_schema(
            prompt=prompt,
            schema_class=CharacterGroups,
            temperature=0.9,
            model=self._text_model_for_stage("character_groups"),
            operation="character_groups",
        )
        self._remove_minor_like_people(protagonist_data, character_groups_data)
        self._save_metadata_model(character_groups_data, "03_character_groups.json")
        return character_groups_data

    def generate_scene_groups(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
    ) -> SceneGroups:
        print("\n" + "=" * 60)
        print("步骤 4/9: 生成场景组数据")
        print("=" * 60)

        template = self._load_meta_prompt("04_scene_groups.txt")
        prompt = template.format(
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            total_photos=self.planning_reference_count,
        )
        self._save_metadata_text(prompt, "00_04_scene_groups_prompt.txt")

        scene_groups_data = self._generate_with_schema(
            prompt=prompt,
            schema_class=SceneGroups,
            temperature=0.7,
            model=self._text_model_for_stage("scene_groups"),
            operation="scene_groups",
        )
        self._save_metadata_model(scene_groups_data, "04_scene_groups.json")
        return scene_groups_data

    @staticmethod
    def _season_for_date(date_text: str) -> str:
        month = int(date_text.split("-", 2)[1])
        if month in (12, 1, 2):
            return "winter"
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        return "autumn"

    def _planned_clip_schedule(self, clip_count: int) -> list[dict[str, Any]]:
        width = 3 if clip_count >= 100 else 2
        duration_cycle = [8, 10, 12, 14, 16, 10, 12, 14, 12, 12]
        runtimes = [
            max(self.target_runtime_seconds_min, min(self.target_runtime_seconds_max, duration_cycle[index % len(duration_cycle)]))
            for index in range(clip_count)
        ]

        if self.target_total_runtime_seconds:
            target_total = self.target_total_runtime_seconds
            current_total = sum(runtimes)
            step = 2
            guard = clip_count * 20
            cursor = 0
            while current_total != target_total and guard > 0:
                index = cursor % clip_count
                if current_total < target_total and runtimes[index] + step <= self.target_runtime_seconds_max:
                    runtimes[index] += step
                    current_total += step
                elif current_total > target_total and runtimes[index] - step >= self.target_runtime_seconds_min:
                    runtimes[index] -= step
                    current_total -= step
                cursor += 1
                guard -= 1
            if current_total != target_total:
                raise ValueError(
                    f"Unable to build a clip runtime schedule totaling {target_total}s; got {current_total}s."
                )

        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        total_days = max(0, (end - start).days)
        time_windows = [
            "early morning",
            "morning",
            "late morning",
            "afternoon",
            "late afternoon",
            "evening",
            "night",
        ]
        date_counts: dict[str, int] = {}
        schedule: list[dict[str, Any]] = []
        denominator = max(1, clip_count - 1)

        for index, runtime in enumerate(runtimes, start=1):
            day_offset = round((index - 1) * total_days / denominator)
            clip_date = (start + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            date_instance = date_counts.get(clip_date, 0)
            date_counts[clip_date] = date_instance + 1
            schedule.append(
                {
                    "id": f"clip_{index:0{width}d}",
                    "target_runtime_seconds": runtime,
                    "clip_date": clip_date,
                    "clip_time_window": time_windows[date_instance % len(time_windows)],
                    "season": self._season_for_date(clip_date),
                }
            )

        return schedule

    @staticmethod
    def _format_clip_schedule(schedule: list[dict[str, Any]]) -> str:
        return json.dumps(schedule, indent=2, ensure_ascii=False)

    def _planned_batch_schedule(self, clip_schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks = [
            clip_schedule[index : index + self.clip_outline_batch_size]
            for index in range(0, len(clip_schedule), self.clip_outline_batch_size)
        ]
        return [
            {
                "batch_index": index,
                "clip_id_start": chunk[0]["id"],
                "clip_id_end": chunk[-1]["id"],
                "start_date": chunk[0]["clip_date"],
                "end_date": chunk[-1]["clip_date"],
                "target_clip_count": len(chunk),
                "target_runtime_seconds": sum(int(entry["target_runtime_seconds"]) for entry in chunk),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]

    @staticmethod
    def _format_batch_schedule(schedule: list[dict[str, Any]]) -> str:
        return json.dumps(schedule, indent=2, ensure_ascii=False)

    def _validate_batch_outline(
        self,
        batch_outline: SeriesBatchOutline,
        expected_batch_schedule: list[dict[str, Any]],
        character_catalog: list[dict[str, Any]],
        scene_catalog: list[dict[str, Any]],
    ) -> None:
        self._validate_cast_size(batch_outline.target_core_cast_size)
        if len(batch_outline.batches) != len(expected_batch_schedule):
            raise ValueError(
                f"Expected {len(expected_batch_schedule)} batch outlines, got {len(batch_outline.batches)}."
            )

        character_ids = {entry.get("id") for entry in character_catalog}
        scene_ids = {entry.get("id") for entry in scene_catalog}

        unknown_characters = [char_id for char_id in batch_outline.recurring_character_ids if char_id not in character_ids]
        if unknown_characters:
            raise ValueError(f"Batch outline selected unknown recurring_character_ids: {unknown_characters}.")
        unknown_scenes = [scene_id for scene_id in batch_outline.recurring_scene_ids if scene_id not in scene_ids]
        if unknown_scenes:
            raise ValueError(f"Batch outline selected unknown recurring_scene_ids: {unknown_scenes}.")

        expected_by_index = {entry["batch_index"]: entry for entry in expected_batch_schedule}
        actual_indexes = [batch.batch_index for batch in batch_outline.batches]
        expected_indexes = [entry["batch_index"] for entry in expected_batch_schedule]
        if actual_indexes != expected_indexes:
            raise ValueError(f"Batch indexes must be exactly {expected_indexes}, got {actual_indexes}.")

        for batch in batch_outline.batches:
            expected = expected_by_index[batch.batch_index]
            for field_name in (
                "clip_id_start",
                "clip_id_end",
                "start_date",
                "end_date",
                "target_clip_count",
                "target_runtime_seconds",
            ):
                actual_value = getattr(batch, field_name)
                expected_value = expected[field_name]
                if actual_value != expected_value:
                    raise ValueError(
                        f"Batch {batch.batch_index} {field_name} must be {expected_value}, got {actual_value}."
                    )

            bad_focus_characters = [
                char_id for char_id in batch.character_focus_ids
                if char_id not in batch_outline.recurring_character_ids
            ]
            if bad_focus_characters:
                raise ValueError(
                    f"Batch {batch.batch_index} uses character_focus_ids outside recurring cast: {bad_focus_characters}."
                )
            bad_focus_scenes = [scene_id for scene_id in batch.scene_focus_ids if scene_id not in scene_ids]
            if bad_focus_scenes:
                raise ValueError(
                    f"Batch {batch.batch_index} uses unknown scene_focus_ids: {bad_focus_scenes}."
                )

    def generate_batch_outline(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_groups_data: CharacterGroups,
        scene_groups_data: SceneGroups,
    ) -> SeriesBatchOutline:
        print("\n" + "=" * 60)
        print("步骤 5/9: 生成全局 Batch Outline")
        print("=" * 60)

        clip_schedule = self._planned_clip_schedule(self.clip_count_target)
        batch_schedule = self._planned_batch_schedule(clip_schedule)
        character_catalog = self._build_character_catalog(protagonist_data, character_groups_data)
        scene_catalog = self._flatten_scene_groups(scene_groups_data)
        template = self._load_meta_prompt("04b_batch_outline.txt", source="video")
        prompt = template.format(
            target_core_cast_size_min=self.target_core_cast_size_min,
            target_core_cast_size_max=self.target_core_cast_size_max,
            user_prompt=self.user_prompt,
            start_date=self.start_date,
            end_date=self.end_date,
            time_span=self.time_span,
            batch_schedule_json=self._format_batch_schedule(batch_schedule),
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            character_catalog_json=json.dumps(character_catalog, indent=2, ensure_ascii=False),
            scene_catalog_json=json.dumps(scene_catalog, indent=2, ensure_ascii=False),
            dialogue_max_lines_per_clip=self.dialogue_max_lines_per_clip,
        )
        self._save_metadata_text(prompt, "00_04b_batch_outline_prompt.txt")
        batch_outline = self._generate_with_schema(
            prompt=prompt,
            schema_class=SeriesBatchOutline,
            temperature=0.65,
            model=self._text_model_for_stage("batch_outline"),
            operation="batch_outline",
        )
        self._validate_batch_outline(
            batch_outline,
            batch_schedule,
            character_catalog,
            scene_catalog,
        )
        self._save_metadata_model(batch_outline, "04b_batch_outline.json")
        return batch_outline

    def _validate_wardrobe_plan(
        self,
        wardrobe_plan: SeriesWardrobePlan,
        recurring_character_ids: list[str],
        character_catalog: list[dict[str, Any]],
    ) -> None:
        expected_ids = set(recurring_character_ids)
        actual_ids = {character.character_id for character in wardrobe_plan.characters}
        if actual_ids != expected_ids:
            raise ValueError(
                "Wardrobe plan must cover exactly the recurring characters. "
                f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}"
            )
        catalog_by_id = {entry.get("id"): entry for entry in character_catalog}
        for character in wardrobe_plan.characters:
            if len(character.wardrobe_options) > self.wardrobe_options_max:
                raise ValueError(
                    f"{character.character_id} has {len(character.wardrobe_options)} outfits; "
                    f"max is {self.wardrobe_options_max}."
                )
            if not catalog_by_id.get(character.character_id):
                raise ValueError(f"Wardrobe plan references unknown character id {character.character_id}.")

    def _summarize_wardrobe_plan(self, wardrobe_plan: SeriesWardrobePlan) -> dict[str, Any]:
        counts = {
            character.character_id: len(character.wardrobe_options)
            for character in wardrobe_plan.characters
        }
        unique_outfits = sum(counts.values())
        return {
            "project_title": wardrobe_plan.project_title,
            "max_outfits_per_character": wardrobe_plan.max_outfits_per_character,
            "characters": len(wardrobe_plan.characters),
            "total_character_outfit_slots": unique_outfits,
            "outfit_counts_by_character": counts,
        }

    def _load_existing_wardrobe_plan(self, metadata_dir: Path) -> SeriesWardrobePlan | None:
        wardrobe_path = metadata_dir / "04c_wardrobe_plan.json"
        if not wardrobe_path.exists():
            return None
        wardrobe_plan = SeriesWardrobePlan.model_validate_json(
            wardrobe_path.read_text(encoding="utf-8")
        )
        self._set_wardrobe_plan(wardrobe_plan)
        print(f"👕 已加载全局 Wardrobe Plan: {wardrobe_path}")
        return wardrobe_plan

    def generate_wardrobe_plan(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_groups_data: CharacterGroups,
        batch_outline: SeriesBatchOutline,
    ) -> SeriesWardrobePlan:
        print("\n" + "=" * 60)
        print("步骤 6/9: 生成全局 Wardrobe Plan")
        print("=" * 60)

        character_catalog = self._subset_catalog(
            self._build_character_catalog(protagonist_data, character_groups_data),
            batch_outline.recurring_character_ids,
        )
        template = self._load_meta_prompt("04c_wardrobe_plan.txt", source="video")
        prompt = template.format(
            max_outfits_per_character=self.wardrobe_options_max,
            user_prompt=self.user_prompt,
            start_date=self.start_date,
            end_date=self.end_date,
            time_span=self.time_span,
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            character_catalog_json=json.dumps(character_catalog, indent=2, ensure_ascii=False),
            batch_outline_json=batch_outline.model_dump_json(indent=2),
        )
        self._save_metadata_text(prompt, "00_04c_wardrobe_plan_prompt.txt")
        wardrobe_plan = self._generate_with_schema(
            prompt=prompt,
            schema_class=SeriesWardrobePlan,
            temperature=0.45,
            model=self._text_model_for_stage("wardrobe_plan"),
            operation="wardrobe_plan",
        )
        self._validate_wardrobe_plan(
            wardrobe_plan,
            batch_outline.recurring_character_ids,
            character_catalog,
        )
        self._set_wardrobe_plan(wardrobe_plan)
        self._save_metadata_model(wardrobe_plan, "04c_wardrobe_plan.json")
        self._save_metadata_json(
            self._summarize_wardrobe_plan(wardrobe_plan),
            "04d_wardrobe_plan_summary.json",
        )
        return wardrobe_plan

    def _load_or_generate_wardrobe_plan(
        self,
        metadata_dir: Path,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_groups_data: CharacterGroups,
        batch_outline: SeriesBatchOutline,
    ) -> SeriesWardrobePlan:
        existing = self._load_existing_wardrobe_plan(metadata_dir)
        if existing is not None:
            return existing
        return self.generate_wardrobe_plan(
            protagonist_data,
            distribution_data,
            character_groups_data,
            batch_outline,
        )

    def _validate_clip_outline_chunk(
        self,
        outline: ClipOutlineSet,
        expected_schedule: list[dict[str, Any]],
        expected_recurring_character_ids: list[str] | None,
    ) -> None:
        if len(outline.clips) != len(expected_schedule):
            raise ValueError(
                f"Expected {len(expected_schedule)} clips in outline chunk, got {len(outline.clips)}."
            )

        expected_ids = [entry["id"] for entry in expected_schedule]
        actual_ids = [clip.id for clip in outline.clips]
        if actual_ids != expected_ids:
            raise ValueError(f"Clip ids must be exactly {expected_ids}, got {actual_ids}.")

        expected_by_id = {entry["id"]: entry for entry in expected_schedule}
        for clip in outline.clips:
            expected_entry = expected_by_id[clip.id]
            expected_runtime = expected_entry["target_runtime_seconds"]
            if clip.target_runtime_seconds != expected_runtime:
                raise ValueError(
                    f"{clip.id} target_runtime_seconds must be {expected_runtime}, got {clip.target_runtime_seconds}."
                )
            if clip.clip_date != expected_entry["clip_date"]:
                raise ValueError(
                    f"{clip.id} clip_date must be {expected_entry['clip_date']}, got {clip.clip_date}."
                )
            if clip.clip_time_window != expected_entry["clip_time_window"]:
                raise ValueError(
                    f"{clip.id} clip_time_window must be {expected_entry['clip_time_window']}, got {clip.clip_time_window}."
                )
            if clip.season != expected_entry["season"]:
                raise ValueError(f"{clip.id} season must be {expected_entry['season']}, got {clip.season}.")
            if clip.strategy != self.default_clip_strategy:
                raise ValueError(
                    f"{clip.id} returned strategy={clip.strategy}, expected {self.default_clip_strategy}."
                )
            self._validate_clip_runtime(clip.target_runtime_seconds)

        if expected_recurring_character_ids is not None:
            if set(outline.recurring_character_ids) != set(expected_recurring_character_ids):
                raise ValueError(
                    "Chunk must reuse the exact recurring_character_ids from chunk 1: "
                    f"{expected_recurring_character_ids}; got {outline.recurring_character_ids}."
                )

    def _generate_clip_outline_chunked(
        self,
        template: str,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_catalog: list[dict[str, Any]],
        scene_catalog: list[dict[str, Any]],
        batch_outline: SeriesBatchOutline | None = None,
    ) -> ClipOutlineSet:
        schedule = self._planned_clip_schedule(self.clip_count_target)
        chunks = [
            schedule[index : index + self.clip_outline_batch_size]
            for index in range(0, len(schedule), self.clip_outline_batch_size)
        ]
        print(
            f"🧩 Clip 大纲采用分批生成: {len(schedule)} clips, "
            f"{len(chunks)} batches, batch_size={self.clip_outline_batch_size}"
        )

        assembled_clips: list[Any] = []
        first_outline: ClipOutlineSet | None = None
        recurring_character_ids: list[str] | None = (
            list(batch_outline.recurring_character_ids) if batch_outline is not None else None
        )
        recurring_scene_ids: list[str] = (
            list(batch_outline.recurring_scene_ids) if batch_outline is not None else []
        )
        batch_outline_by_index = {
            batch.batch_index: batch
            for batch in (batch_outline.batches if batch_outline is not None else [])
        }

        for chunk_index, chunk_schedule in enumerate(chunks, start=1):
            start_id = chunk_schedule[0]["id"]
            end_id = chunk_schedule[-1]["id"]
            start_date = chunk_schedule[0]["clip_date"]
            end_date = chunk_schedule[-1]["clip_date"]
            print(f"🧩 生成 Clip 大纲 batch {chunk_index}/{len(chunks)}: {start_id}-{end_id}")
            batch_brief = batch_outline_by_index.get(chunk_index)
            scope_lines = [
                f"Generate only batch {chunk_index}/{len(chunks)} of the full series.",
                f"Return exactly {len(chunk_schedule)} clips.",
                "This batch is only an engineering partition, not a story unit, chapter, act, or mini-series.",
                "Use exactly this global clip schedule, including id, target_runtime_seconds, clip_date, clip_time_window, and season:",
                self._format_clip_schedule(chunk_schedule),
                "Every returned clip must copy its clip_date, clip_time_window, season, and target_runtime_seconds exactly from this schedule.",
                "Do not include clips outside this schedule.",
                f"This batch covers only {start_date} to {end_date} within the global timeline; do not restart the configured life arc inside this batch.",
                "Do not create a batch opening, batch climax, batch ending, final montage, bookend, conclusion, or 'series ends' moment unless the scheduled clip is the actual final clip of the full series.",
                "These clips are independent life-album memories within the same configured series; preserve identity and relationship continuity.",
            ]
            if batch_brief is not None:
                scope_lines.extend(
                    [
                        "Use this Batch Outline brief as the local planning map for this batch:",
                        json.dumps(batch_brief.model_dump(), indent=2, ensure_ascii=False),
                        "The Batch Outline may permit a few local story threads, but do not make every clip part of one serialized plot.",
                        "Use independent_memory_targets to keep standalone memories mixed into the batch.",
                        "Respect active_major_events and forbidden_timeline_conflicts exactly. Do not schedule setup, active travel/event scenes, or aftermath outside this batch's date range and local event phase.",
                    ]
                )
            if recurring_character_ids is not None:
                scope_lines.extend(
                    [
                        "Reuse exactly these recurring_character_ids from the first batch:",
                        json.dumps(recurring_character_ids, ensure_ascii=False),
                        "Do not add or remove recurring characters in later batches.",
                    ]
                )
            previous_outline_context = self._previous_clip_outline_context(assembled_clips)
            if previous_outline_context["previous_clip_count"]:
                scope_lines.extend(
                    [
                        "Previous clip outlines are provided below as anti-repetition context.",
                        "Use them to avoid repeating titles, event functions, scene-character combinations, memory facts, relationship facts, and continuity hooks.",
                        "New clips may reuse recurring people and places, but must add new information payloads.",
                    ]
                )
            outline_generation_scope = "\n".join(scope_lines)
            chunk_runtime_seconds = sum(int(entry["target_runtime_seconds"]) for entry in chunk_schedule)

            last_error: Exception | None = None
            chunk_outline: ClipOutlineSet | None = None
            for attempt in range(1, 3):
                repair_note = ""
                if attempt > 1 and last_error is not None:
                    repair_note = (
                        "\n\nIMPORTANT REPAIR NOTE:\n"
                        f"Fix this batch outline issue and return corrected JSON: {last_error}"
                    )
                prompt = (
                    template.format(
                        clip_count_min=len(chunk_schedule),
                        clip_count_max=len(chunk_schedule),
                        default_clip_strategy=self.default_clip_strategy,
                        target_core_cast_size_min=self.target_core_cast_size_min,
                        target_core_cast_size_max=self.target_core_cast_size_max,
                        target_runtime_seconds_per_clip_min=self.target_runtime_seconds_min,
                        target_runtime_seconds_per_clip_max=self.target_runtime_seconds_max,
                        target_total_runtime_minutes=round(chunk_runtime_seconds / 60, 2),
                        target_total_runtime_tolerance_minutes=0,
                        allowed_shot_durations=", ".join(str(value) for value in self.allowed_shot_durations_seconds),
                        allow_background_extras=str(self.allow_background_extras).lower(),
                        max_background_extras_per_shot=self.max_background_extras_per_shot,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        time_span=self.time_span,
                        outline_generation_scope=outline_generation_scope,
                        user_prompt=self.user_prompt,
                        protagonist_json=protagonist_data.model_dump_json(indent=2),
                        distribution_json=distribution_data.model_dump_json(indent=2),
                        character_catalog_json=json.dumps(character_catalog, indent=2, ensure_ascii=False),
                        scene_catalog_json=json.dumps(scene_catalog, indent=2, ensure_ascii=False),
                        previous_clip_outline_context_json=json.dumps(
                            previous_outline_context,
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )
                    + repair_note
                )
                self._save_metadata_text(prompt, f"00_05_clip_outline_batch_{chunk_index:03d}_attempt_{attempt}.txt")
                candidate = self._generate_with_schema(
                    prompt=prompt,
                    schema_class=ClipOutlineSet,
                    temperature=0.75 if attempt == 1 else 0.55,
                    model=self._text_model_for_stage("clip_outline"),
                    operation="clip_outline",
                )
                try:
                    self._validate_clip_outline_chunk(candidate, chunk_schedule, recurring_character_ids)
                    if self.clip_outline_similarity_check_enabled:
                        similarity_errors = self._outline_similarity_errors(candidate.clips, assembled_clips)
                        if similarity_errors:
                            raise ValueError(
                                "New outline batch repeats previous clip content too closely. "
                                f"Examples: {similarity_errors[:5]}"
                            )
                except Exception as exc:
                    last_error = exc
                    continue
                chunk_outline = candidate
                break

            if chunk_outline is None:
                raise last_error or RuntimeError(f"Failed to generate clip outline batch {chunk_index}.")
            if first_outline is None:
                first_outline = chunk_outline
                if recurring_character_ids is None:
                    recurring_character_ids = chunk_outline.recurring_character_ids
            assembled_clips.extend(chunk_outline.clips)
            for scene_id in chunk_outline.recurring_scene_ids:
                if scene_id not in recurring_scene_ids:
                    recurring_scene_ids.append(scene_id)

        if first_outline is None or recurring_character_ids is None:
            raise RuntimeError("Clip outline chunking produced no batches.")

        clip_outline = ClipOutlineSet(
            project_title=first_outline.project_title,
            premise=first_outline.premise,
            visual_style=first_outline.visual_style,
            protagonist_id=first_outline.protagonist_id,
            target_core_cast_size=first_outline.target_core_cast_size,
            start_date=first_outline.start_date or self.start_date,
            end_date=first_outline.end_date or self.end_date,
            time_span=first_outline.time_span or self.time_span,
            continuity_rules=first_outline.continuity_rules,
            recurring_character_ids=recurring_character_ids,
            recurring_scene_ids=recurring_scene_ids,
            clips=assembled_clips,
        )
        self._validate_clip_count(len(clip_outline.clips))
        self._validate_cast_size(clip_outline.target_core_cast_size)
        self._validate_series_runtime(clip_outline.clips, label="Clip outline")
        self._save_metadata_model(clip_outline, "05_clip_outline.json")
        return clip_outline

    def _previous_clip_outline_context(self, clips: list[Any]) -> dict[str, Any]:
        if not clips or self.clip_outline_context_mode in {"", "none", "off", "false"}:
            return {
                "mode": "none",
                "previous_clip_count": 0,
                "clips": [],
                "overused_titles": [],
                "overused_scene_character_combos": [],
                "overused_memory_patterns": [],
            }

        selected_clips = clips[-self.clip_outline_max_previous_examples :] if self.clip_outline_max_previous_examples else clips
        title_counts = Counter(str(clip.title).strip() for clip in clips if str(clip.title).strip())
        combo_counts: Counter[str] = Counter()
        memory_pattern_counts: Counter[str] = Counter()
        compact_clips: list[dict[str, Any]] = []

        for clip in selected_clips:
            scene_ids = list(getattr(clip, "primary_scene_ids", []) or [])
            character_ids = list(getattr(clip, "key_character_ids", []) or [])
            combo_key = "|".join(sorted(scene_ids)) + " :: " + "|".join(sorted(character_ids))
            if combo_key.strip(" |:"):
                combo_counts[combo_key] += 1

            facts = (
                list(getattr(clip, "memory_facts", []) or [])
                + list(getattr(clip, "relationship_facts", []) or [])
                + list(getattr(clip, "continuity_hooks", []) or [])
            )
            for fact in facts:
                pattern = self._outline_similarity_text({"text": fact})
                if pattern:
                    memory_pattern_counts[pattern] += 1

            compact_clips.append(
                {
                    "id": clip.id,
                    "title": clip.title,
                    "logline": clip.logline,
                    "date": clip.clip_date,
                    "runtime": clip.target_runtime_seconds,
                    "scene_ids": scene_ids,
                    "key_character_ids": character_ids,
                    "outline_beats": self._trim_list(getattr(clip, "outline_beats", []) or [], 3, 180),
                    "memory_facts": self._trim_list(getattr(clip, "memory_facts", []) or [], 3, 180),
                    "relationship_facts": self._trim_list(getattr(clip, "relationship_facts", []) or [], 3, 180),
                    "continuity_hooks": self._trim_list(getattr(clip, "continuity_hooks", []) or [], 3, 180),
                    "dialogue_goals": self._trim_list(getattr(clip, "dialogue_goals", []) or [], 2, 160),
                }
            )

        return {
            "mode": self.clip_outline_context_mode,
            "instruction": (
                "Use these previous outlines as anti-repetition context. New clips may reuse people "
                "and places, but they must not repeat the same event function, relationship reveal, "
                "memory fact, or continuity hook."
            ),
            "previous_clip_count": len(clips),
            "included_previous_clip_count": len(compact_clips),
            "overused_titles": [
                {"title": title, "count": count}
                for title, count in title_counts.most_common(30)
                if count > 1
            ],
            "overused_scene_character_combos": [
                {"combo": combo, "count": count}
                for combo, count in combo_counts.most_common(30)
                if count > 1
            ],
            "overused_memory_patterns": [
                {"pattern": pattern, "count": count}
                for pattern, count in memory_pattern_counts.most_common(30)
                if count > 1
            ],
            "clips": compact_clips,
        }

    @staticmethod
    def _trim_list(values: list[Any], limit: int, max_chars: int) -> list[str]:
        trimmed: list[str] = []
        for value in values[:limit]:
            text = re.sub(r"\s+", " ", str(value)).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[: max_chars - 3].rstrip() + "..."
            trimmed.append(text)
        return trimmed

    @staticmethod
    def _outline_similarity_text(value: Any) -> str:
        if isinstance(value, dict):
            raw = " ".join(str(part) for part in value.values())
        else:
            raw = str(value)
        text = raw.lower()
        words = [
            word for word in re.findall(r"[a-z][a-z0-9_]+", text)
            if len(word) > 2 and word not in OUTLINE_SIMILARITY_STOPWORDS
        ]
        return " ".join(words[:40])

    def _outline_token_set(self, clip: Any) -> set[str]:
        text = " ".join(
            [
                getattr(clip, "title", ""),
                getattr(clip, "logline", ""),
                " ".join(getattr(clip, "outline_beats", []) or []),
                " ".join(getattr(clip, "memory_facts", []) or []),
                " ".join(getattr(clip, "relationship_facts", []) or []),
                " ".join(getattr(clip, "continuity_hooks", []) or []),
                " ".join(getattr(clip, "dialogue_goals", []) or []),
            ]
        ).lower()
        return {
            word for word in re.findall(r"[a-z][a-z0-9_]+", text)
            if len(word) > 2 and word not in OUTLINE_SIMILARITY_STOPWORDS
        }

    def _outline_similarity_errors(self, candidate_clips: list[Any], previous_clips: list[Any]) -> list[str]:
        if not previous_clips or self.clip_outline_similarity_threshold <= 0:
            return []
        previous_tokens = [(clip, self._outline_token_set(clip)) for clip in previous_clips]
        errors: list[str] = []
        for clip in candidate_clips:
            candidate_tokens = self._outline_token_set(clip)
            if not candidate_tokens:
                continue
            for previous_clip, tokens in previous_tokens:
                if not tokens:
                    continue
                similarity = len(candidate_tokens & tokens) / max(1, len(candidate_tokens | tokens))
                same_title = str(clip.title).strip().lower() == str(previous_clip.title).strip().lower()
                if same_title or similarity >= self.clip_outline_similarity_threshold:
                    errors.append(
                        f"{clip.id} is too similar to previous {previous_clip.id}: "
                        f"similarity={similarity:.2f}, title={clip.title!r} vs {previous_clip.title!r}."
                    )
                    break
        return errors

    def generate_clip_outline(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_groups_data: CharacterGroups,
        scene_groups_data: SceneGroups,
        batch_outline: SeriesBatchOutline | None = None,
    ) -> ClipOutlineSet:
        print("\n" + "=" * 60)
        print("步骤 7/9: 生成 Clip 大纲")
        print("=" * 60)

        template = self._load_meta_prompt("05_clip_outline.txt", source="video")
        character_catalog = self._build_character_catalog(protagonist_data, character_groups_data)
        scene_catalog = self._flatten_scene_groups(scene_groups_data)

        if self.clip_count_target > self.clip_outline_batch_size:
            return self._generate_clip_outline_chunked(
                template=template,
                protagonist_data=protagonist_data,
                distribution_data=distribution_data,
                character_catalog=character_catalog,
                scene_catalog=scene_catalog,
                batch_outline=batch_outline,
            )

        prompt = template.format(
            clip_count_min=self.clip_count_min,
            clip_count_max=self.clip_count_max,
            default_clip_strategy=self.default_clip_strategy,
            target_core_cast_size_min=self.target_core_cast_size_min,
            target_core_cast_size_max=self.target_core_cast_size_max,
            target_runtime_seconds_per_clip_min=self.target_runtime_seconds_min,
            target_runtime_seconds_per_clip_max=self.target_runtime_seconds_max,
            target_total_runtime_minutes=(
                round(self.target_total_runtime_seconds / 60, 2)
                if self.target_total_runtime_seconds
                else "not fixed"
            ),
            target_total_runtime_tolerance_minutes=(
                round((self.target_total_runtime_tolerance_seconds or 0) / 60, 2)
                if self.target_total_runtime_seconds
                else "not fixed"
            ),
            allowed_shot_durations=", ".join(str(value) for value in self.allowed_shot_durations_seconds),
            allow_background_extras=str(self.allow_background_extras).lower(),
            max_background_extras_per_shot=self.max_background_extras_per_shot,
            start_date=self.start_date,
            end_date=self.end_date,
            time_span=self.time_span,
            outline_generation_scope=(
                "Generate the complete clip outline in one response. "
                "Use normal sequential clip ids such as clip_01, clip_02, and so on."
            ),
            user_prompt=self.user_prompt,
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            character_catalog_json=json.dumps(character_catalog, indent=2, ensure_ascii=False),
            scene_catalog_json=json.dumps(scene_catalog, indent=2, ensure_ascii=False),
            previous_clip_outline_context_json=json.dumps(
                self._previous_clip_outline_context([]),
                indent=2,
                ensure_ascii=False,
            ),
        )
        self._save_metadata_text(prompt, "00_05_clip_outline_prompt.txt")

        clip_outline = self._generate_with_schema(
            prompt=prompt,
            schema_class=ClipOutlineSet,
            temperature=0.8,
            model=self._text_model_for_stage("clip_outline"),
            operation="clip_outline",
        )
        self._validate_clip_count(len(clip_outline.clips))
        self._validate_cast_size(clip_outline.target_core_cast_size)
        for clip in clip_outline.clips:
            if clip.strategy != self.default_clip_strategy:
                raise ValueError(
                    f"Clip outline returned strategy={clip.strategy}, expected {self.default_clip_strategy}."
                )
            self._validate_clip_runtime(clip.target_runtime_seconds)
        self._validate_series_runtime(clip_outline.clips, label="Clip outline")
        self._save_metadata_model(clip_outline, "05_clip_outline.json")
        return clip_outline

    def _summarize_blueprint_progress(
        self,
        blueprints: list[ClipShotBlueprint],
        recurring_character_ids: list[str],
    ) -> dict[str, Any]:
        shot_count_by_size = {str(size): 0 for size in range(0, self.max_visible_characters + 1)}
        audio_strategy_counts = {
            "ambient_only": 0,
            "ambient_with_sfx": 0,
            "soft_single_line": 0,
            "soft_dialogue": 0,
        }
        group_profile_counts = {
            "dense_social": 0,
            "mixed": 0,
            "intimate": 0,
        }
        character_requirements = {char_id: 0 for char_id in recurring_character_ids}
        total_shots = 0
        total_visible = 0

        for clip_blueprint in blueprints:
            group_profile_counts[clip_blueprint.group_profile] = (
                group_profile_counts.get(clip_blueprint.group_profile, 0) + 1
            )
            for shot in clip_blueprint.shots:
                total_shots += 1
                total_visible += shot.target_character_count
                shot_count_by_size[str(shot.target_character_count)] += 1
                audio_strategy_counts[shot.audio_strategy] = audio_strategy_counts.get(shot.audio_strategy, 0) + 1
                for char_id in shot.required_character_ids:
                    if char_id in character_requirements:
                        character_requirements[char_id] += 1

        def safe_fraction(numerator: int) -> float:
            return round(numerator / total_shots, 3) if total_shots else 0.0

        one_to_two = shot_count_by_size["1"] + shot_count_by_size["2"]
        three_to_six = sum(shot_count_by_size[str(size)] for size in range(3, 7))
        five_to_six = shot_count_by_size["5"] + shot_count_by_size["6"]
        zero_or_one = shot_count_by_size["0"] + shot_count_by_size["1"]

        return {
            "completed_clips": len(blueprints),
            "total_shots": total_shots,
            "shot_count_by_size": shot_count_by_size,
            "fractions": {
                "one_to_two": safe_fraction(one_to_two),
                "three_to_six": safe_fraction(three_to_six),
                "five_to_six": safe_fraction(five_to_six),
                "zero_or_one": safe_fraction(zero_or_one),
            },
            "average_target_visible_characters": round(total_visible / total_shots, 3) if total_shots else 0.0,
            "audio_strategy_counts": audio_strategy_counts,
            "group_profile_counts": group_profile_counts,
            "required_character_shot_counts": character_requirements,
        }

    def _validate_clip_blueprint(
        self,
        outline_clip: Any,
        clip_blueprint: ClipShotBlueprint,
        recurring_character_ids: list[str],
    ) -> None:
        if clip_blueprint.clip_id != outline_clip.id:
            raise ValueError(
                f"Shot blueprint returned clip id {clip_blueprint.clip_id}; expected {outline_clip.id}."
            )
        if clip_blueprint.target_runtime_seconds != outline_clip.target_runtime_seconds:
            raise ValueError(
                f"{outline_clip.id} blueprint runtime {clip_blueprint.target_runtime_seconds} "
                f"does not match outline runtime {outline_clip.target_runtime_seconds}."
            )
        allowed_character_ids = set(recurring_character_ids)
        for shot in clip_blueprint.shots:
            if not shot.id.startswith(f"{outline_clip.id}_shot_"):
                raise ValueError(f"{outline_clip.id} blueprint returned unexpected shot id {shot.id}.")
            self._validate_shot_duration(shot.id, shot.duration_seconds)
            if shot.target_character_count > self.max_visible_characters:
                raise ValueError(
                    f"{shot.id} target_character_count={shot.target_character_count} exceeds "
                    f"configured max_visible_characters={self.max_visible_characters}."
                )
            unknown_required = [
                char_id for char_id in shot.required_character_ids if char_id not in allowed_character_ids
            ]
            if unknown_required:
                raise ValueError(f"{shot.id} blueprint uses unknown required_character_ids: {unknown_required}.")

    def _validate_clip_plan_against_blueprint(
        self,
        clip_plan: ClipPlan,
        clip_blueprint: ClipShotBlueprint,
    ) -> None:
        if len(clip_plan.shots) != len(clip_blueprint.shots):
            raise ValueError(
                f"{clip_plan.id} returned {len(clip_plan.shots)} shots, expected {len(clip_blueprint.shots)} from blueprint."
            )

        for shot, blueprint in zip(clip_plan.shots, clip_blueprint.shots):
            if shot.id != blueprint.id:
                raise ValueError(f"{clip_plan.id} shot id mismatch: {shot.id} vs {blueprint.id}.")
            if shot.shot_index != blueprint.shot_index:
                raise ValueError(
                    f"{shot.id} shot_index mismatch: {shot.shot_index} vs {blueprint.shot_index}."
                )
            if shot.duration_seconds != blueprint.duration_seconds:
                raise ValueError(
                    f"{shot.id} duration mismatch: {shot.duration_seconds} vs {blueprint.duration_seconds}."
                )
            if len(shot.visible_characters) != blueprint.target_character_count:
                raise ValueError(
                    f"{shot.id} visible character count mismatch: "
                    f"{len(shot.visible_characters)} vs {blueprint.target_character_count}."
                )
            missing_required = [
                char_id for char_id in blueprint.required_character_ids if char_id not in shot.visible_characters
            ]
            if missing_required:
                raise ValueError(f"{shot.id} is missing required characters: {missing_required}.")
            if shot.audio_strategy != blueprint.audio_strategy:
                raise ValueError(
                    f"{shot.id} audio_strategy mismatch: {shot.audio_strategy} vs {blueprint.audio_strategy}."
                )

    @staticmethod
    def _generic_dialogue_phrases() -> list[str]:
        return [
            "perfect weather",
            "great angle",
            "colors are",
            "stunning",
            "looks amazing",
            "light is perfect",
            "hold still",
        ]

    @staticmethod
    def _low_information_dialogue_phrases() -> set[str]:
        return {
            "ok",
            "okay",
            "yes",
            "no",
            "sure",
            "thanks",
            "thank you",
            "nice",
            "great",
            "good",
            "looks good",
            "sounds good",
            "i agree",
            "no problem",
            "morning",
            "good morning",
            "good night",
        }

    def _generic_dialogue_lines_for_clip(self, clip: ClipPlan) -> list[dict[str, str]]:
        generic_lines: list[dict[str, str]] = []
        generic_phrases = self._generic_dialogue_phrases()
        for shot in clip.shots:
            for line in shot.dialogue_lines:
                lower_line = str(line).lower()
                if any(phrase in lower_line for phrase in generic_phrases):
                    generic_lines.append({"shot_id": shot.id, "line": str(line)})
        return generic_lines

    @staticmethod
    def _dialogue_line_count_for_clip(clip: ClipPlan) -> int:
        return sum(len(shot.dialogue_lines) for shot in clip.shots)

    @staticmethod
    def _malformed_dialogue_lines_for_clip(clip: ClipPlan) -> list[dict[str, str]]:
        malformed: list[dict[str, str]] = []
        for shot in clip.shots:
            for line in shot.dialogue_lines:
                text = str(line).strip()
                speaker = text.split(":", 1)[0].strip() if ":" in text else ""
                if ":" not in text or not speaker or len(speaker.split()) > 4:
                    malformed.append({"shot_id": shot.id, "line": text})
        return malformed

    @staticmethod
    def _dialogue_text_without_speaker(line: str) -> str:
        text = str(line).strip()
        return text.split(":", 1)[1].strip() if ":" in text else text

    @classmethod
    def _dialogue_word_count(cls, line: str) -> int:
        text = cls._dialogue_text_without_speaker(line)
        return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))

    def _dialogue_limits_for_duration(self, duration_seconds: int) -> tuple[int, int]:
        if duration_seconds <= 4:
            return self.dialogue_max_words_per_4s_shot, self.dialogue_max_total_words_per_4s_shot
        return self.dialogue_max_words_per_6s_shot, self.dialogue_max_total_words_per_6s_shot

    def _dialogue_budget_violations_for_clip(self, clip: ClipPlan) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        total_clip_lines = sum(len(shot.dialogue_lines) for shot in clip.shots)
        if self.dialogue_max_lines_per_clip and total_clip_lines > self.dialogue_max_lines_per_clip:
            violations.append(
                {
                    "clip_id": clip.id,
                    "reason": "too_many_clip_dialogue_lines",
                    "count": total_clip_lines,
                    "limit": self.dialogue_max_lines_per_clip,
                }
            )

        low_info_phrases = self._low_information_dialogue_phrases()
        for shot in clip.shots:
            lines = [str(line).strip() for line in shot.dialogue_lines if str(line).strip()]
            if not lines:
                continue
            if shot.audio_strategy in ("ambient_only", "ambient_with_sfx"):
                violations.append(
                    {
                        "shot_id": shot.id,
                        "reason": "dialogue_in_ambient_shot",
                        "audio_strategy": shot.audio_strategy,
                    }
                )
            if shot.audio_strategy == "soft_single_line" and len(lines) > 1:
                violations.append(
                    {
                        "shot_id": shot.id,
                        "reason": "too_many_lines_for_soft_single_line",
                        "count": len(lines),
                        "limit": 1,
                    }
                )
            if shot.audio_strategy == "soft_dialogue" and len(lines) > 2:
                violations.append(
                    {
                        "shot_id": shot.id,
                        "reason": "too_many_lines_for_soft_dialogue",
                        "count": len(lines),
                        "limit": 2,
                    }
                )

            max_words_per_line, max_total_words = self._dialogue_limits_for_duration(shot.duration_seconds)
            total_words = 0
            for line in lines:
                words = self._dialogue_word_count(line)
                total_words += words
                spoken_text = self._dialogue_text_without_speaker(line).strip()
                normalized = re.sub(r"[^a-z0-9 ]+", "", spoken_text.lower()).strip()
                normalized = re.sub(r"\s+", " ", normalized)
                if self.dialogue_min_words_per_line and words < self.dialogue_min_words_per_line:
                    violations.append(
                        {
                            "shot_id": shot.id,
                            "reason": "dialogue_line_too_short",
                            "line": line,
                            "words": words,
                            "minimum": self.dialogue_min_words_per_line,
                        }
                    )
                if max_words_per_line and words > max_words_per_line:
                    violations.append(
                        {
                            "shot_id": shot.id,
                            "reason": "dialogue_line_too_long",
                            "line": line,
                            "words": words,
                            "limit": max_words_per_line,
                        }
                    )
                if normalized in low_info_phrases:
                    violations.append(
                        {
                            "shot_id": shot.id,
                            "reason": "low_information_dialogue",
                            "line": line,
                        }
                    )

            if max_total_words and total_words > max_total_words:
                violations.append(
                    {
                        "shot_id": shot.id,
                        "reason": "shot_dialogue_too_long",
                        "words": total_words,
                        "limit": max_total_words,
                        "duration_seconds": shot.duration_seconds,
                    }
                )

        return violations

    def _minimum_dialogue_lines_for_clip(self, clip_blueprint: ClipShotBlueprint) -> int:
        if self.minimum_dialogue_beats_per_clip <= 0:
            return 0
        dialogue_capable = [
            shot
            for shot in clip_blueprint.shots
            if shot.audio_strategy in ("soft_single_line", "soft_dialogue")
        ]
        if not dialogue_capable:
            return 0
        return min(self.minimum_dialogue_beats_per_clip, len(dialogue_capable))

    def _balanced_shot_durations(self, target_runtime_seconds: int) -> list[int]:
        """Prefer compact 4s evidence beats, using one 6s beat when needed to hit odd multiples."""

        allowed = set(self.allowed_shot_durations_seconds)
        if target_runtime_seconds in allowed and target_runtime_seconds <= 6:
            return [target_runtime_seconds]

        if not {4, 6}.issubset(allowed):
            durations: list[int] = []
            remaining = target_runtime_seconds
            for duration in sorted(allowed, reverse=True):
                while remaining >= duration:
                    durations.append(duration)
                    remaining -= duration
            if remaining != 0:
                raise ValueError(
                    f"Cannot compose {target_runtime_seconds}s from allowed durations {sorted(allowed)}."
                )
            return sorted(durations)

        if target_runtime_seconds < 4 or target_runtime_seconds % 2 != 0:
            raise ValueError(f"target_runtime_seconds={target_runtime_seconds} cannot be built from 4/6s shots.")

        if target_runtime_seconds % 4 == 0:
            return [4] * (target_runtime_seconds // 4)

        # 6 + N*4 covers 6, 10, 14, 18, ... while keeping 4s as the dominant beat.
        return [4] * ((target_runtime_seconds - 6) // 4) + [6]

    def _balanced_target_character_count(self, global_shot_index: int) -> int:
        dense_cycle = [3, 4, 3, 5, 4, 2, 4, 3, 5, 4, 3, 2, 4, 3, 5]
        target_count = dense_cycle[global_shot_index % len(dense_cycle)]
        return max(0, min(self.max_visible_characters, target_count))

    def _dialogue_slot_indexes_for_clip(self, durations: list[int], outline_clip: Any | None = None) -> set[int]:
        if not durations or self.dialogue_shots_per_clip_max <= 0:
            return set()

        min_slots = min(len(durations), self.dialogue_shots_per_clip_min)
        max_slots = min(len(durations), self.dialogue_shots_per_clip_max)
        text = self._clip_context_text(outline_clip) if outline_clip is not None else ""
        clip_id = str(getattr(outline_clip, "id", "") or "")
        bucket = sum(ord(char) for char in f"{clip_id}:{text[:120]}") % 100
        quiet_terms = (
            "quiet",
            "silent",
            "museum",
            "gallery",
            "train window",
            "commute",
            "walking",
            "photo",
            "photograph",
            "landscape",
            "balcony",
        )
        speech_terms = (
            "meeting",
            "debug",
            "planning",
            "plan",
            "dinner",
            "restaurant",
            "family",
            "parent",
            "wife",
            "friend",
            "colleague",
            "teasing",
            "recalling",
            "promise",
            "advice",
            "handoff",
            "decision",
        )
        quiet_context = any(term in text for term in quiet_terms)
        speech_context = any(term in text for term in speech_terms)
        target_slots = min_slots
        if max_slots > target_slots:
            if speech_context and (not quiet_context or bucket < 70):
                target_slots = max(target_slots, 1)
            elif not quiet_context and bucket < 55:
                target_slots = max(target_slots, 1)
            if target_slots and max_slots >= 2 and len(durations) >= 3 and speech_context and bucket % 4 == 0:
                target_slots = 2
        target_slots = max(min_slots, min(max_slots, target_slots))
        if target_slots <= 0:
            return set()

        slots: list[int] = []

        for index, duration in enumerate(durations, start=1):
            if duration >= 6:
                slots.append(index)
        if 1 not in slots:
            slots.insert(0, 1)
        if len(durations) >= 3:
            middle_index = (len(durations) + 1) // 2
            if middle_index not in slots:
                slots.append(middle_index)
        if len(durations) >= 2 and len(slots) < min_slots:
            if len(durations) not in slots:
                slots.append(len(durations))

        for index in range(1, len(durations) + 1):
            if len(slots) >= target_slots:
                break
            if index not in slots:
                slots.append(index)

        return set(slots[:target_slots])

    @staticmethod
    def _ordered_unique(values: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    def _select_required_characters_for_blueprint(
        self,
        target_count: int,
        recurring_character_ids: list[str],
        key_character_ids: list[str],
        compatible_character_ids: list[str],
        character_usage: dict[str, int],
        shot_index: int,
    ) -> list[str]:
        if target_count <= 0:
            return []

        recurring_ids = self._ordered_unique(recurring_character_ids)
        compatible_ids = [
            char_id
            for char_id in self._ordered_unique(compatible_character_ids)
            if char_id in recurring_ids
        ]
        key_ids = [
            char_id
            for char_id in self._ordered_unique(key_character_ids)
            if char_id in recurring_ids and (not compatible_ids or char_id in compatible_ids)
        ]
        fill_pool = compatible_ids or recurring_ids
        selected: list[str] = []

        # Keep the protagonist visible often, but not as the only automatic center of every shot.
        if (
            "protagonist" in recurring_ids
            and ("protagonist" in key_ids or shot_index % 4 != 2)
            and len(selected) < target_count
        ):
            selected.append("protagonist")

        for char_id in sorted(key_ids, key=lambda item: (character_usage.get(item, 0), item)):
            if len(selected) >= target_count:
                break
            if char_id not in selected:
                selected.append(char_id)

        for char_id in sorted(fill_pool, key=lambda item: (character_usage.get(item, 0), item)):
            if len(selected) >= target_count:
                break
            if char_id not in selected:
                selected.append(char_id)

        return selected[:target_count]

    @staticmethod
    def _clip_context_text(outline_clip: Any) -> str:
        parts = [
            getattr(outline_clip, "title", ""),
            getattr(outline_clip, "logline", ""),
            getattr(outline_clip, "story_purpose", ""),
            " ".join(getattr(outline_clip, "outline_beats", []) or []),
            " ".join(getattr(outline_clip, "memory_facts", []) or []),
            " ".join(getattr(outline_clip, "relationship_facts", []) or []),
            " ".join(getattr(outline_clip, "dialogue_goals", []) or []),
            " ".join(getattr(outline_clip, "primary_scene_ids", []) or []),
        ]
        return " ".join(str(part) for part in parts if part).lower()

    def _compatible_character_ids_for_clip(
        self,
        outline_clip: Any,
        recurring_character_ids: list[str],
        character_catalog_by_id: dict[str, dict[str, Any]],
    ) -> list[str]:
        text = self._clip_context_text(outline_clip)
        key_ids = set(getattr(outline_clip, "key_character_ids", []) or [])
        work_terms = (
            "office",
            "work",
            "coding",
            "debug",
            "database",
            "software",
            "module",
            "meeting",
            "client",
            "launch",
            "screen",
            "keyboard",
        )
        family_terms = (
            "parent",
            "mother",
            "father",
            "festival",
            "family",
            "dinner",
            "home",
            "tea",
            "mid-autumn",
            "dragon boat",
        )
        travel_social_terms = (
            "travel",
            "trip",
            "restaurant",
            "cafe",
            "walk",
            "hike",
            "park",
            "water town",
            "museum",
            "friends",
            "celebration",
        )
        commute_terms = ("commute", "subway", "train station", "metro")
        home_terms = ("home_", "home ", "apartment", "kitchen", "living room", "balcony", "study desk")
        office_terms = ("office_", "office ", "meeting room", "desk tech park", "pantry")

        is_work_context = any(term in text for term in work_terms)
        is_commute_context = any(term in text for term in commute_terms)
        is_home_context = any(term in text for term in home_terms)
        if is_work_context or any(term in text for term in office_terms):
            allowed_groups = {"protagonist", "colleagues"}
        elif is_commute_context:
            allowed_groups = {"protagonist", "core_family", "close_friends", "colleagues"}
        elif any(term in text for term in family_terms):
            allowed_groups = {"protagonist", "core_family", "close_friends"}
        elif is_home_context:
            allowed_groups = {"protagonist", "core_family", "close_friends"}
        elif any(term in text for term in travel_social_terms):
            allowed_groups = {"protagonist", "core_family", "close_friends", "colleagues"}
        else:
            allowed_groups = {"protagonist", "core_family", "close_friends", "colleagues"}

        compatible: list[str] = []
        for char_id in recurring_character_ids:
            entry = character_catalog_by_id.get(char_id, {})
            group = str(entry.get("group", "") or "")
            relation = str(entry.get("relation_to_protagonist", "") or "").lower()
            is_spouse = "wife" in relation or "husband" in relation or "spouse" in relation or "partner" in relation
            is_parent_or_elder = "mother" in relation or "father" in relation or "parent" in relation
            if is_commute_context and is_parent_or_elder and not any(term in text for term in ("airport", "travel", "trip", "holiday")):
                continue
            if group in allowed_groups or (is_spouse and not is_work_context):
                compatible.append(char_id)
        return self._ordered_unique(compatible)

    def _validate_clip_plan_role_location(
        self,
        clip_plan: ClipPlan,
        outline_clip: Any,
        character_catalog_by_id: dict[str, dict[str, Any]],
    ) -> None:
        compatible_ids = set(
            self._compatible_character_ids_for_clip(
                outline_clip,
                list(character_catalog_by_id.keys()),
                character_catalog_by_id,
            )
        )
        if not compatible_ids:
            return
        for shot in clip_plan.shots:
            incompatible = [char_id for char_id in shot.visible_characters if char_id not in compatible_ids]
            if incompatible:
                raise ValueError(
                    f"{shot.id} uses role-location incompatible characters {incompatible}. "
                    f"Allowed for this clip context: {sorted(compatible_ids)}."
                )

    def _build_balanced_clip_blueprint(
        self,
        outline_clip: Any,
        recurring_character_ids: list[str],
        character_catalog_by_id: dict[str, dict[str, Any]],
        character_usage: dict[str, int],
        global_shot_start_index: int,
    ) -> tuple[ClipShotBlueprint, int]:
        durations = self._balanced_shot_durations(outline_clip.target_runtime_seconds)
        shot_blueprints: list[ShotBlueprint] = []
        key_character_ids = list(getattr(outline_clip, "key_character_ids", []) or [])
        compatible_character_ids = self._compatible_character_ids_for_clip(
            outline_clip,
            recurring_character_ids,
            character_catalog_by_id,
        )
        dialogue_slot_indexes = self._dialogue_slot_indexes_for_clip(durations, outline_clip)

        for local_index, duration in enumerate(durations, start=1):
            global_index = global_shot_start_index + local_index - 1
            target_count = self._balanced_target_character_count(global_index)
            compatible_count = len(compatible_character_ids) or len(recurring_character_ids)
            if target_count > compatible_count:
                target_count = max(3, compatible_count) if compatible_count >= 3 else compatible_count
            if target_count > len(recurring_character_ids):
                target_count = len(recurring_character_ids)
            required_character_ids = self._select_required_characters_for_blueprint(
                target_count=target_count,
                recurring_character_ids=recurring_character_ids,
                key_character_ids=key_character_ids,
                compatible_character_ids=compatible_character_ids,
                character_usage=character_usage,
                shot_index=global_index,
            )
            for char_id in required_character_ids:
                character_usage[char_id] = character_usage.get(char_id, 0) + 1

            if local_index in dialogue_slot_indexes:
                audio_strategy: AudioStrategy = (
                    "soft_dialogue"
                    if duration >= 6 or (target_count >= 4 and local_index != 1)
                    else "soft_single_line"
                )
            else:
                audio_strategy = "ambient_with_sfx"

            shot_blueprints.append(
                ShotBlueprint(
                    id=f"{outline_clip.id}_shot_{local_index:02d}",
                    shot_index=local_index,
                    duration_seconds=duration,
                    target_character_count=target_count,
                    required_character_ids=required_character_ids,
                    audio_strategy=audio_strategy,
                    rationale=(
                        "Programmatic balanced blueprint: compact story beat with "
                        f"{target_count} recurring adult characters and a low-risk audio strategy. "
                        "If audio_strategy permits speech, use relationship-specific dialogue rather than filler."
                    ),
                )
            )

        average_count = (
            sum(shot.target_character_count for shot in shot_blueprints) / len(shot_blueprints)
            if shot_blueprints
            else 0
        )
        group_profile = "dense_social" if average_count >= 4 else "mixed"

        return (
            ClipShotBlueprint(
                clip_id=outline_clip.id,
                target_runtime_seconds=outline_clip.target_runtime_seconds,
                group_profile=group_profile,
                shots=shot_blueprints,
            ),
            global_shot_start_index + len(shot_blueprints),
        )

    def _generate_balanced_shot_blueprints(
        self,
        clip_outline: ClipOutlineSet,
        character_catalog: list[dict[str, Any]],
    ) -> SeriesShotBlueprints:
        recurring_character_ids = clip_outline.recurring_character_ids
        character_catalog_by_id = {entry.get("id"): entry for entry in character_catalog}
        character_usage = {char_id: 0 for char_id in recurring_character_ids}
        planned_blueprints: list[ClipShotBlueprint] = []
        global_shot_index = 0

        for index, outline_clip in enumerate(clip_outline.clips, start=1):
            print(f"🧭 自动生成均衡 Shot 蓝图: {outline_clip.id} ({index}/{len(clip_outline.clips)})")
            clip_blueprint, global_shot_index = self._build_balanced_clip_blueprint(
                outline_clip=outline_clip,
                recurring_character_ids=recurring_character_ids,
                character_catalog_by_id=character_catalog_by_id,
                character_usage=character_usage,
                global_shot_start_index=global_shot_index,
            )
            self._validate_clip_blueprint(outline_clip, clip_blueprint, recurring_character_ids)
            planned_blueprints.append(clip_blueprint)

        blueprints = SeriesShotBlueprints(
            project_title=clip_outline.project_title,
            protagonist_id=clip_outline.protagonist_id,
            recurring_character_ids=recurring_character_ids,
            clips=planned_blueprints,
        )
        self._save_metadata_model(blueprints, "05b_shot_blueprints.json")
        self._save_metadata_json(
            self._summarize_blueprint_progress(planned_blueprints, recurring_character_ids),
            "05c_shot_blueprint_summary.json",
        )
        return blueprints

    def generate_shot_blueprints(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_groups_data: CharacterGroups,
        scene_groups_data: SceneGroups,
        clip_outline: ClipOutlineSet,
    ) -> SeriesShotBlueprints:
        print("\n" + "=" * 60)
        print("步骤 8/9: 生成 Shot 蓝图")
        print("=" * 60)

        character_catalog = self._subset_catalog(
            self._build_character_catalog(protagonist_data, character_groups_data),
            clip_outline.recurring_character_ids,
        )
        full_scene_catalog = self._flatten_scene_groups(scene_groups_data)
        if self.shot_blueprint_mode == "balanced_auto":
            print("🧭 使用程序化均衡 Shot 蓝图：优先 3-6 人、多数 4s、少量 6s。")
            return self._generate_balanced_shot_blueprints(clip_outline, character_catalog)

        template = self._load_meta_prompt("06_shot_blueprint_clip.txt", source="video")
        scene_catalog = self._subset_catalog(
            full_scene_catalog,
            clip_outline.recurring_scene_ids,
        )
        planned_blueprints: list[ClipShotBlueprint] = []

        shared_prompt_values = dict(
            preferred_min_visible_characters=self.preferred_min_visible_characters,
            preferred_max_visible_characters=self.preferred_max_visible_characters,
            min_three_to_six_shot_fraction=self.min_three_to_six_shot_fraction,
            min_five_to_six_shot_fraction=self.min_five_to_six_shot_fraction,
            target_one_to_two_shot_fraction=self.target_one_to_two_shot_fraction,
            max_zero_or_one_shot_fraction=self.max_zero_or_one_shot_fraction,
            target_average_visible_characters=self.target_average_visible_characters,
            max_visible_characters=self.max_visible_characters,
            allow_background_extras=str(self.allow_background_extras).lower(),
            max_background_extras_per_shot=self.max_background_extras_per_shot,
            allowed_shot_durations=", ".join(str(value) for value in self.allowed_shot_durations_seconds),
            user_prompt=self.user_prompt,
            project_title=clip_outline.project_title,
            premise=clip_outline.premise,
            protagonist_id=clip_outline.protagonist_id,
            recurring_character_ids_json=json.dumps(clip_outline.recurring_character_ids, ensure_ascii=False),
            recurring_scene_ids_json=json.dumps(clip_outline.recurring_scene_ids, ensure_ascii=False),
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            character_catalog_json=json.dumps(character_catalog, indent=2, ensure_ascii=False),
            scene_catalog_json=json.dumps(scene_catalog, indent=2, ensure_ascii=False),
            dialogue_max_lines_per_clip=self.dialogue_max_lines_per_clip,
        )

        for index, outline_clip in enumerate(clip_outline.clips, start=1):
            print(f"🧭 生成 Shot 蓝图: {outline_clip.id} ({index}/{len(clip_outline.clips)})")
            progress_summary = self._summarize_blueprint_progress(planned_blueprints, clip_outline.recurring_character_ids)
            allowed_clip_scene_catalog = self._subset_catalog(
                full_scene_catalog,
                outline_clip.primary_scene_ids or clip_outline.recurring_scene_ids,
            )
            planned_clip_blueprint = None
            last_error: Exception | None = None

            for clip_attempt in range(1, 3):
                repair_note = ""
                if clip_attempt > 1 and last_error is not None:
                    repair_note = (
                        "\n\nIMPORTANT REPAIR NOTE:\n"
                        "Your previous blueprint attempt violated the requested structure. "
                        f"Fix this specific issue and return corrected JSON: {last_error}"
                    )
                prompt = (
                    template.format(
                        **shared_prompt_values,
                        clip_index=index,
                        total_clips=len(clip_outline.clips),
                        clip_id=outline_clip.id,
                        clip_target_runtime_seconds=outline_clip.target_runtime_seconds,
                        existing_blueprint_progress_json=json.dumps(progress_summary, indent=2, ensure_ascii=False),
                        clip_outline_json=json.dumps(outline_clip.model_dump(), indent=2, ensure_ascii=False),
                        allowed_clip_scene_catalog_json=json.dumps(
                            allowed_clip_scene_catalog,
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )
                    + repair_note
                )
                self._save_metadata_text(prompt, f"00_06_blueprint_{outline_clip.id}_attempt_{clip_attempt}.txt")

                partial_blueprint_set = self._generate_with_schema(
                    prompt=prompt,
                    schema_class=SeriesShotBlueprints,
                    temperature=0.6 if clip_attempt == 1 else 0.45,
                    model=self._text_model_for_stage("shot_blueprint"),
                    operation="shot_blueprint",
                )
                if len(partial_blueprint_set.clips) != 1:
                    last_error = ValueError(
                        f"{outline_clip.id} blueprint generation returned {len(partial_blueprint_set.clips)} clips; expected exactly 1."
                    )
                    continue
                clip_blueprint = partial_blueprint_set.clips[0]
                try:
                    self._validate_clip_blueprint(
                        outline_clip,
                        clip_blueprint,
                        clip_outline.recurring_character_ids,
                    )
                except Exception as exc:
                    last_error = exc
                    continue
                planned_clip_blueprint = clip_blueprint
                break

            if planned_clip_blueprint is None:
                raise last_error or RuntimeError(f"Failed to generate shot blueprint for {outline_clip.id}.")
            planned_blueprints.append(planned_clip_blueprint)

        blueprints = SeriesShotBlueprints(
            project_title=clip_outline.project_title,
            protagonist_id=clip_outline.protagonist_id,
            recurring_character_ids=clip_outline.recurring_character_ids,
            clips=planned_blueprints,
        )
        self._save_metadata_model(blueprints, "05b_shot_blueprints.json")
        self._save_metadata_json(
            self._summarize_blueprint_progress(planned_blueprints, clip_outline.recurring_character_ids),
            "05c_shot_blueprint_summary.json",
        )
        return blueprints

    def generate_series_shot_plan(
        self,
        protagonist_data: ProtagonistData,
        distribution_data: DistributionPlan,
        character_groups_data: CharacterGroups,
        scene_groups_data: SceneGroups,
        clip_outline: ClipOutlineSet,
        shot_blueprints: SeriesShotBlueprints,
        max_workers: int = 1,
    ) -> SeriesShotPlan:
        print("\n" + "=" * 60)
        print("步骤 9/9: 生成详细 Shot Plan")
        print("=" * 60)
        max_workers = max(1, int(max_workers or 1))
        if max_workers > 1:
            print(f"⚡ Shot Plan 并发模式: {max_workers} workers")

        template = self._load_meta_prompt("06_shot_plan_clip.txt", source="video")
        character_catalog = self._subset_catalog(
            self._build_character_catalog(protagonist_data, character_groups_data),
            clip_outline.recurring_character_ids,
        )
        full_scene_catalog = self._flatten_scene_groups(scene_groups_data)
        scene_catalog = self._subset_catalog(
            full_scene_catalog,
            clip_outline.recurring_scene_ids,
        )
        planned_clips: list[ClipPlan] = []

        shared_prompt_values = dict(
            allowed_shot_durations=", ".join(str(value) for value in self.allowed_shot_durations_seconds),
            preferred_min_visible_characters=self.preferred_min_visible_characters,
            preferred_max_visible_characters=self.preferred_max_visible_characters,
            min_three_to_six_shot_fraction=self.min_three_to_six_shot_fraction,
            min_five_to_six_shot_fraction=self.min_five_to_six_shot_fraction,
            target_one_to_two_shot_fraction=self.target_one_to_two_shot_fraction,
            max_zero_or_one_shot_fraction=self.max_zero_or_one_shot_fraction,
            target_average_visible_characters=self.target_average_visible_characters,
            max_single_shot_fraction=self.max_single_shot_fraction,
            max_empty_shot_fraction=self.max_empty_shot_fraction,
            allow_background_extras=str(self.allow_background_extras).lower(),
            max_background_extras_per_shot=self.max_background_extras_per_shot,
            project_title=clip_outline.project_title,
            premise=clip_outline.premise,
            visual_style=clip_outline.visual_style,
            protagonist_id=clip_outline.protagonist_id,
            target_core_cast_size=clip_outline.target_core_cast_size,
            start_date=clip_outline.start_date or self.start_date,
            end_date=clip_outline.end_date or self.end_date,
            time_span=clip_outline.time_span or self.time_span,
            continuity_rules_json=json.dumps(clip_outline.continuity_rules, indent=2, ensure_ascii=False),
            recurring_character_ids_json=json.dumps(clip_outline.recurring_character_ids, ensure_ascii=False),
            recurring_scene_ids_json=json.dumps(clip_outline.recurring_scene_ids, ensure_ascii=False),
            user_prompt=self.user_prompt,
            protagonist_json=protagonist_data.model_dump_json(indent=2),
            distribution_json=distribution_data.model_dump_json(indent=2),
            character_catalog_json=json.dumps(character_catalog, indent=2, ensure_ascii=False),
            scene_catalog_json=json.dumps(scene_catalog, indent=2, ensure_ascii=False),
            dialogue_max_lines_per_clip=self.dialogue_max_lines_per_clip,
        )
        blueprint_by_clip_id = {clip_blueprint.clip_id: clip_blueprint for clip_blueprint in shot_blueprints.clips}
        outline_index_by_id = {
            outline_clip.id: index
            for index, outline_clip in enumerate(clip_outline.clips, start=1)
        }
        planned_by_index: dict[int, ClipPlan] = {}
        partial_filename = "06_shot_plan_partial.json"
        partial_path = self.metadata_dir / partial_filename
        if partial_path.exists():
            try:
                partial_plan = SeriesShotPlan.model_validate_json(partial_path.read_text(encoding="utf-8"))
                for partial_clip in partial_plan.clips:
                    partial_index = outline_index_by_id.get(partial_clip.id)
                    if partial_index is not None:
                        planned_by_index[partial_index] = partial_clip
                if planned_by_index:
                    print(
                        f"↩️  已加载 Shot Plan partial checkpoint: "
                        f"{len(planned_by_index)}/{len(clip_outline.clips)} clips"
                    )
            except Exception as exc:
                print(f"⚠️  忽略无法读取的 Shot Plan partial checkpoint: {exc}")
                planned_by_index = {}

        def save_partial_checkpoint(force: bool = False) -> None:
            if not planned_by_index:
                return
            if not force and len(planned_by_index) % 10 != 0:
                return
            partial_plan = SeriesShotPlan(
                project_title=clip_outline.project_title,
                premise=clip_outline.premise,
                visual_style=clip_outline.visual_style,
                protagonist_id=clip_outline.protagonist_id,
                target_core_cast_size=clip_outline.target_core_cast_size,
                start_date=clip_outline.start_date,
                end_date=clip_outline.end_date,
                time_span=clip_outline.time_span,
                continuity_rules=clip_outline.continuity_rules,
                recurring_character_ids=clip_outline.recurring_character_ids,
                recurring_scene_ids=clip_outline.recurring_scene_ids,
                clips=[
                    planned_by_index[index]
                    for index in sorted(planned_by_index)
                ],
            )
            self._save_metadata_model(partial_plan, partial_filename)

        def generate_one_clip(index: int, outline_clip: Any) -> ClipPlan:
            clip_blueprint = blueprint_by_clip_id.get(outline_clip.id)
            if clip_blueprint is None:
                raise ValueError(f"Missing shot blueprint for clip {outline_clip.id}.")
            print(f"🎞️  生成 Shot Plan: {outline_clip.id} ({index}/{len(clip_outline.clips)})")
            allowed_clip_scene_catalog = self._subset_catalog(
                full_scene_catalog,
                outline_clip.primary_scene_ids or clip_outline.recurring_scene_ids,
            )
            compatible_character_ids = self._compatible_character_ids_for_clip(
                outline_clip,
                clip_outline.recurring_character_ids,
                {entry.get("id"): entry for entry in character_catalog},
            )
            planned_clip = None
            last_error: Exception | None = None
            primary_shot_plan_model = self._text_model_for_stage("shot_plan")
            shot_plan_models = self._text_model_sequence(primary_shot_plan_model)

            for model_index, shot_plan_model in enumerate(shot_plan_models):
                if model_index > 0:
                    print(
                        f"🛟 {outline_clip.id} 使用 {primary_shot_plan_model} 连续 3 次失败，"
                        f"切换到 {shot_plan_model} 只重试当前 clip。"
                    )

                for clip_attempt in range(1, 4):
                    repair_note = ""
                    if last_error is not None:
                        repair_note = (
                            "\n\nIMPORTANT REPAIR NOTE:\n"
                            "Your previous attempt violated the shot blueprint or schema. "
                            f"Fix this specific issue and return corrected JSON: {last_error}"
                        )
                    prompt = (
                        template.format(
                            **shared_prompt_values,
                            clip_id=outline_clip.id,
                            clip_target_runtime_seconds=outline_clip.target_runtime_seconds,
                            clip_outline_json=json.dumps(outline_clip.model_dump(), indent=2, ensure_ascii=False),
                            clip_shot_blueprint_json=json.dumps(clip_blueprint.model_dump(), indent=2, ensure_ascii=False),
                            compatible_character_ids_json=json.dumps(
                                compatible_character_ids,
                                indent=2,
                                ensure_ascii=False,
                            ),
                            allowed_clip_scene_catalog_json=json.dumps(
                                allowed_clip_scene_catalog,
                                indent=2,
                                ensure_ascii=False,
                            ),
                        )
                        + repair_note
                    )
                    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", shot_plan_model)
                    self._save_metadata_text(
                        prompt,
                        f"00_07_shot_plan_{outline_clip.id}_{safe_model}_attempt_{clip_attempt}.txt",
                    )

                    try:
                        partial_plan = self._generate_with_schema(
                            prompt=prompt,
                            schema_class=SeriesShotPlan,
                            temperature=0.75 if clip_attempt == 1 else 0.55,
                            max_retries=1,
                            model=shot_plan_model,
                            operation="shot_plan",
                            allow_model_fallback=False,
                        )
                    except Exception as exc:
                        last_error = exc
                        continue

                    if len(partial_plan.clips) != 1:
                        last_error = ValueError(
                            f"{outline_clip.id} partial shot plan returned {len(partial_plan.clips)} clips; expected exactly 1."
                        )
                        continue
                    candidate_clip = partial_plan.clips[0]
                    if candidate_clip.id != outline_clip.id:
                        last_error = ValueError(
                            f"Partial shot plan returned clip id {candidate_clip.id}; expected {outline_clip.id}."
                        )
                        continue
                    try:
                        self._validate_clip_plan_against_blueprint(candidate_clip, clip_blueprint)
                        self._validate_clip_plan_role_location(
                            candidate_clip,
                            outline_clip,
                            {entry.get("id"): entry for entry in character_catalog},
                        )
                    except Exception as exc:
                        last_error = exc
                        continue
                    malformed_dialogue = self._malformed_dialogue_lines_for_clip(candidate_clip)
                    if malformed_dialogue:
                        last_error = ValueError(
                            "Every dialogue line must start with the named speaker followed by a colon, "
                            f"for example 'Chen Wei: ...'. Fix these lines: {malformed_dialogue[:3]}"
                        )
                        continue
                    generic_lines = self._generic_dialogue_lines_for_clip(candidate_clip)
                    if generic_lines:
                        last_error = ValueError(
                            "Generic filler dialogue detected; rewrite with relationship-specific dialogue: "
                            f"{generic_lines[:3]}"
                        )
                        continue
                    dialogue_count = self._dialogue_line_count_for_clip(candidate_clip)
                    minimum_dialogue_lines = self._minimum_dialogue_lines_for_clip(clip_blueprint)
                    if dialogue_count < minimum_dialogue_lines:
                        last_error = ValueError(
                            f"Dialogue is too sparse for this clip: got {dialogue_count} lines, "
                            f"expected at least {minimum_dialogue_lines}. Add short, low-risk, "
                            "relationship-specific dialogue in the soft_single_line/soft_dialogue shots."
                        )
                        continue
                    dialogue_budget_violations = self._dialogue_budget_violations_for_clip(candidate_clip)
                    if dialogue_budget_violations:
                        last_error = ValueError(
                            "Dialogue violates the 4s/6s TTS budget or is too low-information. "
                            "Rewrite spoken lines as moderate, evidence-carrying micro-dialogue: "
                            f"{dialogue_budget_violations[:4]}"
                        )
                        continue
                    candidate_clip = self._resolve_clip_character_outfits(candidate_clip, character_catalog)
                    planned_clip = candidate_clip
                    break

                if planned_clip is not None:
                    break

            if planned_clip is None:
                raise last_error or RuntimeError(f"Failed to generate shot plan for {outline_clip.id}.")
            return planned_clip

        if max_workers == 1:
            for index, outline_clip in enumerate(clip_outline.clips, start=1):
                if index in planned_by_index:
                    print(f"↩️  跳过已完成 Shot Plan: {outline_clip.id} ({index}/{len(clip_outline.clips)})")
                    continue
                planned_by_index[index] = generate_one_clip(index, outline_clip)
                save_partial_checkpoint()
        else:
            pending_items = [
                (index, outline_clip)
                for index, outline_clip in enumerate(clip_outline.clips, start=1)
                if index not in planned_by_index
            ]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index: dict[Any, int] = {}
                item_cursor = 0

                def submit_next() -> None:
                    nonlocal item_cursor
                    if item_cursor >= len(pending_items):
                        return
                    next_index, next_outline_clip = pending_items[item_cursor]
                    item_cursor += 1
                    future_to_index[executor.submit(generate_one_clip, next_index, next_outline_clip)] = next_index

                for _ in range(min(max_workers, len(pending_items))):
                    submit_next()

                while future_to_index:
                    done, _ = wait(future_to_index, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = future_to_index.pop(future)
                        try:
                            planned_by_index[index] = future.result()
                        except Exception:
                            save_partial_checkpoint(force=True)
                            raise
                        print(
                            f"✅ Shot Plan 完成: {planned_by_index[index].id} "
                            f"({len(planned_by_index)}/{len(clip_outline.clips)})"
                        )
                        save_partial_checkpoint()
                        submit_next()
        planned_clips = [
            planned_by_index[index]
            for index in range(1, len(clip_outline.clips) + 1)
        ]
        save_partial_checkpoint(force=True)

        series_plan = SeriesShotPlan(
            project_title=clip_outline.project_title,
            premise=clip_outline.premise,
            visual_style=clip_outline.visual_style,
            protagonist_id=clip_outline.protagonist_id,
            target_core_cast_size=clip_outline.target_core_cast_size,
            start_date=clip_outline.start_date,
            end_date=clip_outline.end_date,
            time_span=clip_outline.time_span,
            continuity_rules=clip_outline.continuity_rules,
            recurring_character_ids=clip_outline.recurring_character_ids,
            recurring_scene_ids=clip_outline.recurring_scene_ids,
            clips=planned_clips,
        )

        self._validate_clip_count(len(series_plan.clips))
        self._validate_cast_size(series_plan.target_core_cast_size)
        self._validate_series_runtime(series_plan.clips, label="Shot plan")
        for clip in series_plan.clips:
            if clip.strategy != self.default_clip_strategy:
                raise ValueError(
                    f"Shot plan returned strategy={clip.strategy}, expected {self.default_clip_strategy}."
                )
            self._validate_clip_runtime(clip.target_runtime_seconds)
            for shot in clip.shots:
                self._validate_shot_duration(shot.id, shot.duration_seconds)
                self._validate_visible_character_count(shot.id, shot.visible_characters)
                if len(shot.background_extras) > self.max_background_extras_per_shot:
                    raise ValueError(
                        f"{shot.id} uses {len(shot.background_extras)} background extras, "
                        f"exceeding configured limit {self.max_background_extras_per_shot}."
                    )
            self._validate_clip_shot_runtime(clip)
        self._report_shot_size_distribution(series_plan)
        self._save_metadata_model(series_plan, "06_shot_plan.json")
        return series_plan

    def _assemble_series_bible(
        self,
        protagonist_data: ProtagonistData,
        character_groups_data: CharacterGroups,
        scene_groups_data: SceneGroups,
        series_plan: SeriesShotPlan,
    ) -> SeriesBible:
        cast_lookup = self._build_cast_lookup(protagonist_data, character_groups_data)
        scene_lookup = self._build_scene_lookup(scene_groups_data)

        used_character_ids = list(series_plan.recurring_character_ids)
        extra_character_ids: list[str] = []
        for clip in series_plan.clips:
            for shot in clip.shots:
                for char_id in shot.visible_characters:
                    if char_id not in used_character_ids:
                        extra_character_ids.append(char_id)

        used_scene_ids = list(series_plan.recurring_scene_ids)
        for clip in series_plan.clips:
            for scene_id in clip.scene_ids:
                if scene_id not in used_scene_ids:
                    used_scene_ids.append(scene_id)
            for shot in clip.shots:
                if shot.scene_id not in used_scene_ids:
                    used_scene_ids.append(shot.scene_id)

        if extra_character_ids:
            raise ValueError(
                "Shot plan used character ids outside recurring_character_ids: "
                f"{sorted(set(extra_character_ids))}"
            )
        missing_chars = [char_id for char_id in used_character_ids if char_id not in cast_lookup]
        if missing_chars:
            raise ValueError(f"Series plan references unknown character ids: {missing_chars}")

        missing_scenes = [scene_id for scene_id in used_scene_ids if scene_id not in scene_lookup]
        if missing_scenes:
            raise ValueError(f"Series plan references unknown scene ids: {missing_scenes}")

        return SeriesBible(
            project_title=series_plan.project_title,
            premise=series_plan.premise,
            visual_style=series_plan.visual_style,
            protagonist_id=series_plan.protagonist_id,
            target_core_cast_size=series_plan.target_core_cast_size,
            start_date=self.start_date,
            end_date=self.end_date,
            time_span=self.time_span,
            continuity_rules=series_plan.continuity_rules,
            cast=[cast_lookup[char_id] for char_id in used_character_ids],
            scenes=[scene_lookup[scene_id] for scene_id in used_scene_ids],
            clips=series_plan.clips,
        )

    def _build_metadata_quality_report(self, series_bible: SeriesBible) -> dict[str, Any]:
        shot_duration_counts: dict[str, int] = {}
        clip_duration_counts: dict[str, int] = {}
        visible_counts: dict[str, int] = {}
        dialogue_speaker_counts: dict[str, int] = {}
        generic_dialogue_lines: list[dict[str, str]] = []
        dialogue_budget_violations: list[dict[str, Any]] = []
        dialogue_word_counts: list[int] = []
        clips_missing_memory_facts: list[str] = []
        shots_missing_evidence_facts: list[str] = []
        generic_phrases = self._generic_dialogue_phrases()

        total_runtime = 0
        total_shots = 0
        total_visible = 0
        one_to_two = 0
        three_to_six = 0

        for clip in series_bible.clips:
            total_runtime += clip.target_runtime_seconds
            clip_duration_counts[str(clip.target_runtime_seconds)] = (
                clip_duration_counts.get(str(clip.target_runtime_seconds), 0) + 1
            )
            if not clip.memory_facts:
                clips_missing_memory_facts.append(clip.id)
            dialogue_budget_violations.extend(self._dialogue_budget_violations_for_clip(clip))
            for shot in clip.shots:
                total_shots += 1
                count = len(shot.visible_characters)
                total_visible += count
                visible_counts[str(count)] = visible_counts.get(str(count), 0) + 1
                shot_duration_counts[str(shot.duration_seconds)] = (
                    shot_duration_counts.get(str(shot.duration_seconds), 0) + 1
                )
                if 1 <= count <= 2:
                    one_to_two += 1
                if 3 <= count <= 6:
                    three_to_six += 1
                if not shot.evidence_facts:
                    shots_missing_evidence_facts.append(shot.id)
                for line in shot.dialogue_lines:
                    speaker = str(line).split(":", 1)[0].strip() if ":" in str(line) else "unknown"
                    dialogue_speaker_counts[speaker] = dialogue_speaker_counts.get(speaker, 0) + 1
                    dialogue_word_counts.append(self._dialogue_word_count(str(line)))
                    lower_line = str(line).lower()
                    if any(phrase in lower_line for phrase in generic_phrases):
                        generic_dialogue_lines.append({"shot_id": shot.id, "line": str(line)})

        return {
            "clips": len(series_bible.clips),
            "shots": total_shots,
            "total_runtime_seconds": total_runtime,
            "total_runtime_minutes": round(total_runtime / 60, 2),
            "clip_duration_counts": clip_duration_counts,
            "shot_duration_counts": shot_duration_counts,
            "visible_character_counts": visible_counts,
            "fractions": {
                "one_to_two": round(one_to_two / total_shots, 3) if total_shots else 0,
                "three_to_six": round(three_to_six / total_shots, 3) if total_shots else 0,
            },
            "average_visible_characters": round(total_visible / total_shots, 3) if total_shots else 0,
            "dialogue_speaker_counts": dialogue_speaker_counts,
            "dialogue_word_counts": {
                "count": len(dialogue_word_counts),
                "min": min(dialogue_word_counts) if dialogue_word_counts else 0,
                "max": max(dialogue_word_counts) if dialogue_word_counts else 0,
                "average": round(sum(dialogue_word_counts) / len(dialogue_word_counts), 2)
                if dialogue_word_counts
                else 0,
            },
            "clips_missing_memory_facts": clips_missing_memory_facts,
            "shots_missing_evidence_facts": shots_missing_evidence_facts,
            "generic_dialogue_lines": generic_dialogue_lines,
            "dialogue_budget_violations": dialogue_budget_violations,
        }

    def generate_all(self) -> SeriesBible:
        print("\n" + "🎬" * 30)
        print("开始生成视频元数据（分阶段规划）")
        print("🎬" * 30)

        self._save_config_snapshot()

        protagonist_data = self.generate_protagonist()
        distribution_data = self.generate_distribution(protagonist_data)
        character_groups_data = self.generate_character_groups(protagonist_data, distribution_data)
        scene_groups_data = self.generate_scene_groups(protagonist_data, distribution_data)
        batch_outline = self.generate_batch_outline(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
        )
        self._save_voice_manifest_for_recurring_cast(
            protagonist_data,
            character_groups_data,
            batch_outline.recurring_character_ids,
        )
        self.generate_wardrobe_plan(
            protagonist_data,
            distribution_data,
            character_groups_data,
            batch_outline,
        )
        clip_outline = self.generate_clip_outline(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            batch_outline,
        )
        scene_groups_data = self._expand_scene_groups_for_clip_outline(
            scene_groups_data,
            clip_outline,
        )
        shot_blueprints = self.generate_shot_blueprints(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
        )
        series_plan = self.generate_series_shot_plan(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
            shot_blueprints,
        )

        series_bible = self._assemble_series_bible(
            protagonist_data,
            character_groups_data,
            scene_groups_data,
            series_plan,
        )
        self._save_metadata_model(series_bible, "07_series_bible.json")
        self._save_metadata_model(series_bible, "01_series_bible.json")

        summary = {
            "project_title": series_bible.project_title,
            "timeline": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "time_span": self.time_span,
            },
            "counts": {
                "recurring_cast": len(series_bible.cast),
                "scenes": len(series_bible.scenes),
                "clips": len(series_bible.clips),
                "shots": sum(len(clip.shots) for clip in series_bible.clips),
            },
        }
        self._save_metadata_json(summary, "00_video_summary.json")
        self._save_metadata_json(
            self._build_metadata_quality_report(series_bible),
            "09_metadata_quality_report.json",
        )
        self._write_api_usage_summary()

        print("\n" + "=" * 60)
        print("✅ 视频元数据生成完成")
        print("=" * 60)
        print(f"📁 元数据保存在: {self.metadata_dir.absolute()}")
        print(f"📁 本次输出根目录: {self.output_root.absolute()}")
        return series_bible

    def generate_from_existing_batch_outline(
        self,
        source_output_root: str | Path,
        shot_plan_workers: int = 1,
    ) -> SeriesBible:
        source_root = Path(source_output_root).expanduser().resolve()
        metadata_dir = source_root / "metadata"
        if not metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

        required_files = [
            "01_protagonist.json",
            "02_distribution.json",
            "03_character_groups.json",
            "04_scene_groups.json",
            "04b_batch_outline.json",
        ]
        missing = [filename for filename in required_files if not (metadata_dir / filename).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot resume from {source_root}; missing required metadata files: {missing}"
            )

        self.output_root = source_root
        self.metadata_dir = metadata_dir
        self.debug_dir = source_root / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.usage_records = self.api_usage_logger.records
        self._final_output_root_locked = True

        print("\n" + "🎬" * 30)
        print("复用已有 Batch Outline，继续生成后续 metadata")
        print("🎬" * 30)
        print(f"📁 复用输出目录: {self.output_root}")

        protagonist_data = ProtagonistData.model_validate_json(
            (metadata_dir / "01_protagonist.json").read_text(encoding="utf-8")
        )
        distribution_data = DistributionPlan.model_validate_json(
            (metadata_dir / "02_distribution.json").read_text(encoding="utf-8")
        )
        character_groups_data = CharacterGroups.model_validate_json(
            (metadata_dir / "03_character_groups.json").read_text(encoding="utf-8")
        )
        scene_groups_data = SceneGroups.model_validate_json(
            (metadata_dir / "04_scene_groups.json").read_text(encoding="utf-8")
        )
        batch_outline = SeriesBatchOutline.model_validate_json(
            (metadata_dir / "04b_batch_outline.json").read_text(encoding="utf-8")
        )
        self._save_voice_manifest_for_recurring_cast(
            protagonist_data,
            character_groups_data,
            batch_outline.recurring_character_ids,
        )
        self._load_or_generate_wardrobe_plan(
            metadata_dir,
            protagonist_data,
            distribution_data,
            character_groups_data,
            batch_outline,
        )

        clip_outline = self.generate_clip_outline(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            batch_outline,
        )
        scene_groups_data = self._expand_scene_groups_for_clip_outline(
            scene_groups_data,
            clip_outline,
        )
        shot_blueprints = self.generate_shot_blueprints(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
        )
        series_plan = self.generate_series_shot_plan(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
            shot_blueprints,
            max_workers=shot_plan_workers,
        )

        series_bible = self._assemble_series_bible(
            protagonist_data,
            character_groups_data,
            scene_groups_data,
            series_plan,
        )
        self._save_metadata_model(series_bible, "07_series_bible.json")
        self._save_metadata_model(series_bible, "01_series_bible.json")
        summary = {
            "project_title": series_bible.project_title,
            "timeline": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "time_span": self.time_span,
            },
            "counts": {
                "recurring_cast": len(series_bible.cast),
                "scenes": len(series_bible.scenes),
                "clips": len(series_bible.clips),
                "shots": sum(len(clip.shots) for clip in series_bible.clips),
            },
            "resumed_from_output_root": str(source_root),
            "resume_stage": "clip-outline",
        }
        self._save_metadata_json(summary, "00_video_summary.json")
        self._save_metadata_json(
            self._build_metadata_quality_report(series_bible),
            "09_metadata_quality_report.json",
        )
        self._write_api_usage_summary()

        print("\n" + "=" * 60)
        print("✅ 已从 Batch Outline 继续生成完整 metadata")
        print("=" * 60)
        print(f"📁 元数据保存在: {self.metadata_dir.absolute()}")
        return series_bible

    def generate_clip_outline_from_existing_batch_outline(
        self,
        source_output_root: str | Path,
    ) -> ClipOutlineSet:
        source_root = Path(source_output_root).expanduser().resolve()
        metadata_dir = source_root / "metadata"
        if not metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

        required_files = [
            "01_protagonist.json",
            "02_distribution.json",
            "03_character_groups.json",
            "04_scene_groups.json",
            "04b_batch_outline.json",
        ]
        missing = [filename for filename in required_files if not (metadata_dir / filename).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot resume from {source_root}; missing required metadata files: {missing}"
            )

        self.output_root = source_root
        self.metadata_dir = metadata_dir
        self.debug_dir = source_root / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.usage_records = self.api_usage_logger.records
        self._final_output_root_locked = True

        print("\n" + "🎬" * 30)
        print("复用已有 Batch Outline，只生成 Clip Outline")
        print("🎬" * 30)
        print(f"📁 复用输出目录: {self.output_root}")

        protagonist_data = ProtagonistData.model_validate_json(
            (metadata_dir / "01_protagonist.json").read_text(encoding="utf-8")
        )
        distribution_data = DistributionPlan.model_validate_json(
            (metadata_dir / "02_distribution.json").read_text(encoding="utf-8")
        )
        character_groups_data = CharacterGroups.model_validate_json(
            (metadata_dir / "03_character_groups.json").read_text(encoding="utf-8")
        )
        scene_groups_data = SceneGroups.model_validate_json(
            (metadata_dir / "04_scene_groups.json").read_text(encoding="utf-8")
        )
        batch_outline = SeriesBatchOutline.model_validate_json(
            (metadata_dir / "04b_batch_outline.json").read_text(encoding="utf-8")
        )
        self._save_voice_manifest_for_recurring_cast(
            protagonist_data,
            character_groups_data,
            batch_outline.recurring_character_ids,
        )
        self._load_or_generate_wardrobe_plan(
            metadata_dir,
            protagonist_data,
            distribution_data,
            character_groups_data,
            batch_outline,
        )

        clip_outline = self.generate_clip_outline(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            batch_outline,
        )
        self._expand_scene_groups_for_clip_outline(
            scene_groups_data,
            clip_outline,
        )
        self._write_api_usage_summary()

        print("\n🧭 已完成 Clip Outline，按要求停止。")
        print(f"📁 本次输出目录: {self.output_root}")
        print(f"  - Clip Outline: {self.metadata_dir / '05_clip_outline.json'}")
        print(f"  - API Usage: {self.metadata_dir / '00_api_usage_summary.json'}")
        return clip_outline

    def generate_from_existing_clip_outline(
        self,
        source_output_root: str | Path,
        shot_plan_workers: int = 1,
    ) -> SeriesBible:
        source_root = Path(source_output_root).expanduser().resolve()
        metadata_dir = source_root / "metadata"
        if not metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

        required_files = [
            "01_protagonist.json",
            "02_distribution.json",
            "03_character_groups.json",
            "04_scene_groups.json",
            "05_clip_outline.json",
        ]
        missing = [filename for filename in required_files if not (metadata_dir / filename).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot resume from {source_root}; missing required metadata files: {missing}"
            )

        self.output_root = source_root
        self.metadata_dir = metadata_dir
        self.debug_dir = source_root / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.usage_records = self.api_usage_logger.records
        self._final_output_root_locked = True

        print("\n" + "🎬" * 30)
        print("复用已有 Clip Outline，继续生成 Shot Blueprint / Shot Plan / Series Bible")
        print("🎬" * 30)
        print(f"📁 复用输出目录: {self.output_root}")

        protagonist_data = ProtagonistData.model_validate_json(
            (metadata_dir / "01_protagonist.json").read_text(encoding="utf-8")
        )
        distribution_data = DistributionPlan.model_validate_json(
            (metadata_dir / "02_distribution.json").read_text(encoding="utf-8")
        )
        character_groups_data = CharacterGroups.model_validate_json(
            (metadata_dir / "03_character_groups.json").read_text(encoding="utf-8")
        )
        scene_groups_data = SceneGroups.model_validate_json(
            (metadata_dir / "04_scene_groups.json").read_text(encoding="utf-8")
        )
        clip_outline = ClipOutlineSet.model_validate_json(
            (metadata_dir / "05_clip_outline.json").read_text(encoding="utf-8")
        )
        self._save_voice_manifest_for_recurring_cast(
            protagonist_data,
            character_groups_data,
            clip_outline.recurring_character_ids,
        )
        self._load_existing_wardrobe_plan(metadata_dir)

        shot_blueprints = self.generate_shot_blueprints(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
        )
        series_plan = self.generate_series_shot_plan(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
            shot_blueprints,
            max_workers=shot_plan_workers,
        )

        series_bible = self._assemble_series_bible(
            protagonist_data,
            character_groups_data,
            scene_groups_data,
            series_plan,
        )
        self._save_metadata_model(series_bible, "07_series_bible.json")
        self._save_metadata_model(series_bible, "01_series_bible.json")
        summary = {
            "project_title": series_bible.project_title,
            "timeline": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "time_span": self.time_span,
            },
            "counts": {
                "recurring_cast": len(series_bible.cast),
                "scenes": len(series_bible.scenes),
                "clips": len(series_bible.clips),
                "shots": sum(len(clip.shots) for clip in series_bible.clips),
            },
            "resumed_from_output_root": str(source_root),
            "resume_stage": "existing-clip-outline",
        }
        self._save_metadata_json(summary, "00_video_summary.json")
        self._save_metadata_json(
            self._build_metadata_quality_report(series_bible),
            "09_metadata_quality_report.json",
        )
        self._write_api_usage_summary()

        print("\n" + "=" * 60)
        print("✅ 已从现有 Clip Outline 继续生成完整 metadata")
        print("=" * 60)
        print(f"📁 元数据保存在: {self.metadata_dir.absolute()}")
        return series_bible

    def generate_from_existing_shot_blueprints(
        self,
        source_output_root: str | Path,
        shot_plan_workers: int = 1,
    ) -> SeriesBible:
        source_root = Path(source_output_root).expanduser().resolve()
        metadata_dir = source_root / "metadata"
        if not metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

        required_files = [
            "01_protagonist.json",
            "03_character_groups.json",
            "04_scene_groups.json",
            "05_clip_outline.json",
            "05b_shot_blueprints.json",
        ]
        missing = [filename for filename in required_files if not (metadata_dir / filename).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot resume shot-plan from {source_root}; missing required metadata files: {missing}"
            )

        self.output_root = source_root
        self.metadata_dir = metadata_dir
        self.debug_dir = source_root / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.api_usage_logger = ApiUsageLogger(self.metadata_dir)
        self.usage_records = self.api_usage_logger.records
        self._final_output_root_locked = True

        print("\n" + "🎬" * 30)
        print("复用已有 Clip Outline / Shot Blueprints，继续生成 Shot Plan")
        print("🎬" * 30)
        print(f"📁 复用输出目录: {self.output_root}")

        protagonist_data = ProtagonistData.model_validate_json(
            (metadata_dir / "01_protagonist.json").read_text(encoding="utf-8")
        )
        character_groups_data = CharacterGroups.model_validate_json(
            (metadata_dir / "03_character_groups.json").read_text(encoding="utf-8")
        )
        scene_groups_data = SceneGroups.model_validate_json(
            (metadata_dir / "04_scene_groups.json").read_text(encoding="utf-8")
        )
        clip_outline = ClipOutlineSet.model_validate_json(
            (metadata_dir / "05_clip_outline.json").read_text(encoding="utf-8")
        )
        self._save_voice_manifest_for_recurring_cast(
            protagonist_data,
            character_groups_data,
            clip_outline.recurring_character_ids,
        )
        shot_blueprints = SeriesShotBlueprints.model_validate_json(
            (metadata_dir / "05b_shot_blueprints.json").read_text(encoding="utf-8")
        )

        distribution_path = metadata_dir / "02_distribution.json"
        if distribution_path.exists():
            distribution_data = DistributionPlan.model_validate_json(
                distribution_path.read_text(encoding="utf-8")
            )
        else:
            raise FileNotFoundError(
                f"Cannot resume shot-plan from {source_root}; missing required metadata file: 02_distribution.json"
            )
        self._load_existing_wardrobe_plan(metadata_dir)

        series_plan = self.generate_series_shot_plan(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
            clip_outline,
            shot_blueprints,
            max_workers=shot_plan_workers,
        )

        series_bible = self._assemble_series_bible(
            protagonist_data,
            character_groups_data,
            scene_groups_data,
            series_plan,
        )
        self._save_metadata_model(series_bible, "07_series_bible.json")
        self._save_metadata_model(series_bible, "01_series_bible.json")
        summary = {
            "project_title": series_bible.project_title,
            "timeline": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "time_span": self.time_span,
            },
            "counts": {
                "recurring_cast": len(series_bible.cast),
                "scenes": len(series_bible.scenes),
                "clips": len(series_bible.clips),
                "shots": sum(len(clip.shots) for clip in series_bible.clips),
            },
            "resumed_from_output_root": str(source_root),
            "resume_stage": "shot-plan",
            "shot_plan_workers": max(1, int(shot_plan_workers or 1)),
        }
        self._save_metadata_json(summary, "00_video_summary.json")
        self._save_metadata_json(
            self._build_metadata_quality_report(series_bible),
            "09_metadata_quality_report.json",
        )
        self._write_api_usage_summary()

        print("\n" + "=" * 60)
        print("✅ 已从 Shot Blueprints 继续生成完整 metadata")
        print("=" * 60)
        print(f"📁 元数据保存在: {self.metadata_dir.absolute()}")
        return series_bible

    def generate_until_batch_outline(self) -> SeriesBatchOutline:
        print("\n" + "🎬" * 30)
        print("开始生成视频元数据（停在 Batch Outline）")
        print("🎬" * 30)

        self._save_config_snapshot()

        protagonist_data = self.generate_protagonist()
        distribution_data = self.generate_distribution(protagonist_data)
        character_groups_data = self.generate_character_groups(protagonist_data, distribution_data)
        scene_groups_data = self.generate_scene_groups(protagonist_data, distribution_data)
        batch_outline = self.generate_batch_outline(
            protagonist_data,
            distribution_data,
            character_groups_data,
            scene_groups_data,
        )
        self._save_voice_manifest_for_recurring_cast(
            protagonist_data,
            character_groups_data,
            batch_outline.recurring_character_ids,
        )
        self._write_api_usage_summary()

        print("\n" + "=" * 60)
        print("✅ Batch Outline 生成完成")
        print("=" * 60)
        print(f"📁 元数据保存在: {self.metadata_dir.absolute()}")
        print(f"📁 本次输出根目录: {self.output_root.absolute()}")
        return batch_outline


def load_series_bible(output_root: str | Path) -> SeriesBible:
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(
            "未找到指定的视频输出目录: "
            f"{root}\n"
            "如果这是一个新的 workspace，请不要使用 --skip-planning；"
            "先直接运行 `python video_generator/pipeline.py` 或 `--metadata-only`。"
        )
    candidate_paths = [
        root / "metadata" / "07_series_bible.json",
        root / "metadata" / "01_series_bible.json",
    ]
    for path in candidate_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                series_bible = SeriesBible.model_validate_json(file.read())
                patched_series_bible = _backfill_series_bible_clip_outfits(series_bible)
                if patched_series_bible != series_bible:
                    print("🧵 已为旧 metadata 自动回填 clip-level outfits，后续 anchor 将按 clip 固定服装生成。")
                return patched_series_bible
    raise FileNotFoundError(
        "未找到视频系列规划文件。\n"
        f"期望存在其一: {candidate_paths[0]} 或 {candidate_paths[1]}\n"
        "如果这是一个新的 workspace，请不要使用 --skip-planning；"
        "先运行 `python video_generator/pipeline.py --metadata-only` 或直接完整跑一次。"
    )


def load_video_prompt_config(config_path: str | None = None) -> dict:
    return load_video_config(config_path or str(DEFAULT_VIDEO_CONFIG_PATH))
