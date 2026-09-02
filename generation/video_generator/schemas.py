from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ClipStrategy = Literal["shot_based", "extend"]
AudioStrategy = Literal[
    "ambient_only",
    "ambient_with_sfx",
    "soft_single_line",
    "soft_dialogue",
]


class ReferencePhotoPrompt(BaseModel):
    """Prompt for a reusable character reference image."""

    photo_type: str = Field(description="Reference photo type, for example portrait_front_smile.")
    prompt: str = Field(description="English prompt for generating this reference image.")
    aspect_ratio: str = Field(default="1:1", description="Image aspect ratio, usually 1:1 or 3:4.")


class CastMember(BaseModel):
    """Recurring on-camera character."""

    id: str = Field(description="Stable character id. Protagonist must use 'protagonist'.")
    name_en: str = Field(description="English display name.")
    name_cn: Optional[str] = Field(default=None, description="Chinese name if applicable.")
    is_protagonist: bool = Field(default=False, description="Whether this is the protagonist.")
    role: str = Field(description="Short role, for example birthday boy, mother, best friend.")
    relation_to_protagonist: str = Field(description="Relationship to the protagonist.")
    age: int = Field(description="Character age.")
    gender: str = Field(description="Gender label.")
    appearance_description: str = Field(description="Detailed visual description for consistency.")
    signature_outfit: str = Field(description="Default outfit used as a fallback if a clip-specific outfit is not described.")
    wardrobe_options: List[str] = Field(
        default_factory=list,
        description="Up to five complete reusable outfit descriptions. Adjacent continuous shots in a clip should keep the same selected outfit."
    )
    personality_brief: str = Field(description="Short personality summary.")
    voice_brief: str = Field(description="How this person sounds when speaking.")
    reference_photo_prompts: List[ReferencePhotoPrompt] = Field(
        default_factory=list,
        description="Reusable reference prompts, usually one front-facing portrait in the current video workflow."
    )


class SceneReference(BaseModel):
    """Reusable scene location."""

    id: str = Field(description="Scene id like scene_001.")
    name_en: str = Field(description="English scene name.")
    name_cn: Optional[str] = Field(default=None, description="Chinese scene name.")
    description: str = Field(description="Detailed physical description of the scene.")
    lighting: str = Field(description="Lighting plan for this scene.")
    mood: str = Field(description="Emotional tone of this scene.")
    background_prompt: str = Field(description="English prompt for generating a reusable background image.")
    aspect_ratio: str = Field(default="16:9", description="Preferred aspect ratio.")


class ShotPlan(BaseModel):
    """Single motion beat that can be rendered as a short video segment."""

    id: str = Field(description="Stable shot id like clip_01_shot_01.")
    shot_index: int = Field(description="1-based order within the clip.")
    beat_title: str = Field(description="Short descriptive shot title.")
    purpose: str = Field(description="Why this shot exists in the scene.")
    date: Optional[str] = Field(default=None, description="Planned calendar date in YYYY-MM-DD.")
    time: Optional[str] = Field(default=None, description="Planned clock time in HH:MM.")
    day_of_week: Optional[str] = Field(default=None, description="Day of week name.")
    time_of_day: Optional[str] = Field(default=None, description="morning, afternoon, evening, or night.")
    season: Optional[str] = Field(default=None, description="spring, summer, autumn, or winter.")
    weather: Optional[str] = Field(default=None, description="Weather context if relevant.")
    occasion: Optional[str] = Field(default=None, description="Daily life, birthday setup, surprise reveal, etc.")
    scene_id: str = Field(description="Scene id used for this shot.")
    visible_characters: List[str] = Field(
        default_factory=list,
        description="Exactly which recurring characters are on camera."
    )
    background_extras: List[str] = Field(
        default_factory=list,
        description="Brief descriptions of unnamed background extras or passersby visible in the shot."
    )
    focus_characters: List[str] = Field(
        default_factory=list,
        description="One or two characters who should be visually emphasized."
    )
    target_character_count: int = Field(
        default=0,
        description="Derived count of visible_characters."
    )
    composition: str = Field(description="Shot composition such as wide shot or medium close-up.")
    camera_language: str = Field(description="Camera behavior, lens feeling, and movement intent.")
    blocking_notes: str = Field(description="Concrete staging notes including placement and movement.")
    left_to_right_order: List[str] = Field(
        default_factory=list,
        description="Left-to-right order when three or more characters share the frame."
    )
    motion_budget: str = Field(description="How simple or complex the motion is allowed to be.")
    secondary_actions: List[str] = Field(
        default_factory=list,
        description="Parallel actions performed by non-focus visible characters."
    )
    evidence_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete continuity facts established by this shot, "
            "such as identity, relationship, preference, shared history, work detail, plan, or object state."
        ),
    )
    dialogue_lines: List[str] = Field(
        default_factory=list,
        description="Short lines of dialogue or vocal beats, if any."
    )
    audio_cues: List[str] = Field(
        default_factory=list,
        description="Ambient sound, SFX, crowd reactions, or vocal texture cues."
    )
    audio_strategy: AudioStrategy = Field(
        default="ambient_with_sfx",
        description="How aggressively this shot should use native generated audio."
    )
    anchor_image_prompt: str = Field(description="English prompt for generating the anchor still image.")
    video_prompt: str = Field(description="English prompt for generating motion from the anchor.")
    negative_prompt: Optional[str] = Field(default=None, description="Optional per-shot negative prompt.")
    duration_seconds: int = Field(default=6, description="Veo segment duration. Must be exactly 4, 6, or 8 seconds.")

    @model_validator(mode="after")
    def validate_shot(self) -> "ShotPlan":
        self.target_character_count = len(self.visible_characters)

        if len(self.visible_characters) > 6:
            raise ValueError("This prototype workflow supports up to 6 visible recurring characters per shot.")
        if len(self.background_extras) > 4:
            raise ValueError("This prototype workflow supports at most 4 unnamed background extras per shot.")
        if self.duration_seconds not in (4, 6, 8):
            raise ValueError("duration_seconds must be exactly one of 4, 6, or 8 for Veo rendering.")

        visible_set = set(self.visible_characters)
        if self.focus_characters:
            known_focus = [char_id for char_id in self.focus_characters if char_id in visible_set]
            if known_focus:
                self.focus_characters = known_focus
            elif self.visible_characters:
                self.focus_characters = self.visible_characters[:1]

        if self.left_to_right_order:
            known_order = [char_id for char_id in self.left_to_right_order if char_id in visible_set]
            if known_order:
                self.left_to_right_order = known_order
            elif len(self.left_to_right_order) == len(self.visible_characters):
                self.left_to_right_order = list(self.visible_characters)
            else:
                self.left_to_right_order = []

        return self

class ShotBlueprint(BaseModel):
    """Lightweight per-shot control plan generated before the full shot prompts."""

    id: str = Field(description="Stable shot id like clip_01_shot_01.")
    shot_index: int = Field(description="1-based order within the clip.")
    duration_seconds: int = Field(description="Veo segment duration. Must be exactly 4, 6, or 8 seconds.")
    target_character_count: int = Field(description="Desired number of visible recurring characters in this shot.")
    required_character_ids: List[str] = Field(
        default_factory=list,
        description="Recurring characters that must appear in this shot."
    )
    audio_strategy: AudioStrategy = Field(
        default="ambient_with_sfx",
        description="Preferred native-audio strategy for this shot."
    )
    rationale: str = Field(description="Short explanation of why this shot size and audio strategy fit the beat.")

    @model_validator(mode="after")
    def validate_blueprint(self) -> "ShotBlueprint":
        if self.duration_seconds not in (4, 6, 8):
            raise ValueError("duration_seconds must be exactly one of 4, 6, or 8.")
        if not (0 <= self.target_character_count <= 6):
            raise ValueError("target_character_count must be between 0 and 6.")
        if len(self.required_character_ids) > self.target_character_count:
            raise ValueError("required_character_ids cannot exceed target_character_count.")
        return self


class ClipShotBlueprint(BaseModel):
    """Shot-count and audio-density skeleton for a single clip."""

    clip_id: str = Field(description="Clip id like clip_01.")
    target_runtime_seconds: int = Field(description="Desired runtime for this clip.")
    group_profile: Literal["dense_social", "mixed", "intimate"] = Field(
        description="High-level social density for this clip."
    )
    shots: List[ShotBlueprint] = Field(default_factory=list, description="Ordered shot blueprints.")

    @model_validator(mode="after")
    def validate_clip_blueprint(self) -> "ClipShotBlueprint":
        if not self.shots:
            raise ValueError("Each clip blueprint must contain at least one shot.")
        total_runtime = sum(shot.duration_seconds for shot in self.shots)
        if total_runtime != self.target_runtime_seconds:
            raise ValueError(
                f"Shot blueprint durations sum to {total_runtime}s, expected {self.target_runtime_seconds}s."
            )
        expected_indexes = list(range(1, len(self.shots) + 1))
        actual_indexes = [shot.shot_index for shot in self.shots]
        if actual_indexes != expected_indexes:
            raise ValueError("Shot blueprint indexes must be consecutive starting from 1.")
        return self


class SeriesShotBlueprints(BaseModel):
    """Series-level set of per-clip shot blueprints."""

    project_title: str = Field(description="Human-readable project title.")
    protagonist_id: str = Field(default="protagonist", description="Stable protagonist character id.")
    recurring_character_ids: List[str] = Field(default_factory=list, description="Selected recurring cast ids.")
    clips: List[ClipShotBlueprint] = Field(default_factory=list, description="Per-clip shot blueprints.")

    @model_validator(mode="after")
    def validate_blueprints(self) -> "SeriesShotBlueprints":
        if self.protagonist_id not in self.recurring_character_ids:
            raise ValueError("protagonist_id must be included in recurring_character_ids.")
        if not self.clips:
            raise ValueError("SeriesShotBlueprints expects at least one clip blueprint.")
        return self


class ClipPlan(BaseModel):
    """One output clip in the prototype workflow."""

    id: str = Field(description="Clip id like clip_01.")
    title: str = Field(description="Clip title.")
    strategy: ClipStrategy = Field(description="shot_based or extend.")
    logline: str = Field(description="One-sentence clip summary.")
    continuity_from_previous: Optional[str] = Field(
        default=None,
        description="Optional album-level continuity from the previous clip, or a brief note explaining the time jump/new memory."
    )
    continuity_to_next: Optional[str] = Field(
        default=None,
        description="Optional album-level handoff into the next clip, if any."
    )
    target_runtime_seconds: int = Field(description="Desired total runtime of this clip.")
    clip_date: Optional[str] = Field(default=None, description="Primary date for this clip.")
    clip_time_window: Optional[str] = Field(default=None, description="Primary time window for this clip.")
    season: Optional[str] = Field(default=None, description="Seasonal context for this clip.")
    scene_ids: List[str] = Field(default_factory=list, description="Scenes used in this clip.")
    newly_introduced_characters: List[str] = Field(
        default_factory=list,
        description="Characters who make an important first appearance in this clip."
    )
    contains_full_group_moment: bool = Field(
        default=False,
        description="Whether the full recurring cast appears together in this clip."
    )
    is_intro_clip: bool = Field(
        default=False,
        description="Whether this clip is a non-evidence calibration/intro clip placed before the story timeline.",
    )
    memory_facts: List[str] = Field(
        default_factory=list,
        description="Clip-level facts that should remain consistent across the story."
    )
    relationship_facts: List[str] = Field(
        default_factory=list,
        description="Clip-level relationship signals or social dynamics made explicit in this clip."
    )
    continuity_hooks: List[str] = Field(
        default_factory=list,
        description="Story details that later clips can revisit or develop."
    )
    clip_character_outfits: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map from recurring character id to the exact outfit description selected for this clip. "
            "Use one fixed outfit per character for a continuous clip moment so adjacent shots can stay consistent."
        ),
    )
    shots: List[ShotPlan] = Field(default_factory=list, description="Ordered shot or beat list.")

    @model_validator(mode="after")
    def validate_clip(self) -> "ClipPlan":
        if not self.shots:
            raise ValueError("Each clip must contain at least one shot.")
        normalized_outfits: Dict[str, str] = {}
        for char_id, outfit_text in (self.clip_character_outfits or {}).items():
            key = str(char_id).strip()
            value = str(outfit_text).strip()
            if key and value:
                normalized_outfits[key] = value
        self.clip_character_outfits = normalized_outfits
        return self


class SeriesBible(BaseModel):
    """Full pre-production plan for the prototype video workflow."""

    project_title: str = Field(description="Human-readable project title.")
    premise: str = Field(description="Short story premise.")
    visual_style: str = Field(description="Overall visual direction.")
    protagonist_id: str = Field(default="protagonist", description="Stable protagonist character id.")
    target_core_cast_size: int = Field(default=6, description="Target count of recurring on-camera characters.")
    start_date: Optional[str] = Field(default=None, description="Series timeline start date.")
    end_date: Optional[str] = Field(default=None, description="Series timeline end date.")
    time_span: Optional[str] = Field(default=None, description="Configured planning time span.")
    continuity_rules: List[str] = Field(default_factory=list, description="Global continuity constraints.")
    cast: List[CastMember] = Field(default_factory=list, description="Recurring on-camera cast.")
    scenes: List[SceneReference] = Field(default_factory=list, description="Reusable scene library.")
    clips: List[ClipPlan] = Field(default_factory=list, description="Ordered clip plans for the series.")

    @model_validator(mode="after")
    def validate_series(self) -> "SeriesBible":
        if len(self.cast) != self.target_core_cast_size:
            raise ValueError(
                f"Recurring cast size must equal target_core_cast_size ({self.target_core_cast_size}) "
                "for this prototype workflow."
            )

        cast_ids = {member.id for member in self.cast}
        if self.protagonist_id not in cast_ids:
            raise ValueError("protagonist_id must exist in cast.")

        scene_ids = {scene.id for scene in self.scenes}
        if not scene_ids:
            raise ValueError("At least one reusable scene is required.")

        if not self.clips:
            raise ValueError("This workflow expects at least one planned clip.")

        for clip in self.clips:
            unknown_outfit_ids = [char_id for char_id in clip.clip_character_outfits if char_id not in cast_ids]
            if unknown_outfit_ids:
                raise ValueError(
                    f"Clip {clip.id} references unknown clip_character_outfits ids {unknown_outfit_ids}."
                )
            for shot in clip.shots:
                if shot.scene_id not in scene_ids:
                    raise ValueError(f"Shot {shot.id} references unknown scene_id {shot.scene_id}.")

                missing_characters = [char_id for char_id in shot.visible_characters if char_id not in cast_ids]
                if missing_characters:
                    raise ValueError(f"Shot {shot.id} references unknown characters {missing_characters}.")

        return self


class ClipOutline(BaseModel):
    """Higher-level clip story outline before detailed shot planning."""

    id: str = Field(description="Clip id like clip_01.")
    title: str = Field(description="Clip title.")
    strategy: ClipStrategy = Field(description="shot_based or extend.")
    logline: str = Field(description="One-sentence clip summary.")
    story_purpose: str = Field(description="Narrative purpose of the clip.")
    continuity_from_previous: str = Field(description="Optional album-level continuity from the previous clip, or a short note describing the time jump/new memory.")
    continuity_to_next: str = Field(description="Optional album-level handoff into the next clip, if any.")
    target_runtime_seconds: int = Field(description="Desired total runtime of this clip.")
    clip_date: str = Field(description="Primary date for this clip in YYYY-MM-DD.")
    clip_time_window: str = Field(description="Primary time window such as evening or late afternoon.")
    season: str = Field(description="Season for this clip.")
    primary_scene_ids: List[str] = Field(default_factory=list, description="Main scenes used in this clip.")
    key_character_ids: List[str] = Field(default_factory=list, description="Recurring characters essential to this clip.")
    newly_introduced_characters: List[str] = Field(default_factory=list, description="Characters meaningfully introduced in this clip.")
    contains_full_group_moment: bool = Field(default=False, description="Whether this clip contains a full ensemble moment.")
    outline_beats: List[str] = Field(default_factory=list, description="High-level story beats in order.")
    memory_facts: List[str] = Field(default_factory=list, description="Specific facts this clip should establish for later continuity.")
    relationship_facts: List[str] = Field(default_factory=list, description="Specific relationship cues this clip should reveal.")
    continuity_hooks: List[str] = Field(default_factory=list, description="Story details that later clips can revisit or develop.")
    dialogue_goals: List[str] = Field(default_factory=list, description="What kinds of dialogue should happen in the clip.")
    audio_palette: List[str] = Field(default_factory=list, description="Ambient and sonic goals for the clip.")


class ClipOutlineSet(BaseModel):
    """Series-level story outline before detailed shot planning."""

    project_title: str = Field(description="Human-readable project title.")
    premise: str = Field(description="Short story premise.")
    visual_style: str = Field(description="Overall visual direction.")
    protagonist_id: str = Field(default="protagonist", description="Stable protagonist character id.")
    target_core_cast_size: int = Field(default=6, description="Target count of recurring on-camera characters.")
    start_date: Optional[str] = Field(default=None, description="Series timeline start date.")
    end_date: Optional[str] = Field(default=None, description="Series timeline end date.")
    time_span: Optional[str] = Field(default=None, description="Configured planning time span.")
    continuity_rules: List[str] = Field(default_factory=list, description="Global continuity constraints.")
    recurring_character_ids: List[str] = Field(default_factory=list, description="Selected recurring cast ids for the series.")
    recurring_scene_ids: List[str] = Field(default_factory=list, description="Selected recurring scene ids for the series.")
    clips: List[ClipOutline] = Field(default_factory=list, description="Ordered clip outlines.")

    @model_validator(mode="after")
    def validate_outline(self) -> "ClipOutlineSet":
        if self.protagonist_id not in self.recurring_character_ids:
            raise ValueError("protagonist_id must be included in recurring_character_ids.")
        if len(self.recurring_character_ids) != self.target_core_cast_size:
            raise ValueError(
                f"recurring_character_ids must contain exactly {self.target_core_cast_size} ids."
            )
        if not self.clips:
            raise ValueError("ClipOutlineSet expects at least one clip.")
        return self


class BatchOutline(BaseModel):
    """Global planning brief for one API-generation batch of clip outlines."""

    batch_index: int = Field(description="1-based batch index.")
    clip_id_start: str = Field(description="First clip id covered by this batch.")
    clip_id_end: str = Field(description="Last clip id covered by this batch.")
    start_date: str = Field(description="First scheduled clip date in this batch, YYYY-MM-DD.")
    end_date: str = Field(description="Last scheduled clip date in this batch, YYYY-MM-DD.")
    target_clip_count: int = Field(description="Exact number of clips in this batch.")
    target_runtime_seconds: int = Field(description="Total target runtime for clips in this batch.")
    batch_role: str = Field(description="What this batch contributes to the full life-album series.")
    time_period_focus: str = Field(description="Seasonal and calendar focus for this batch.")
    character_focus_ids: List[str] = Field(default_factory=list, description="Recurring characters to emphasize.")
    scene_focus_ids: List[str] = Field(default_factory=list, description="Recurring scenes to emphasize.")
    allowed_story_threads: List[str] = Field(
        default_factory=list,
        description="Optional short story threads that may span a few clips in this batch."
    )
    active_major_events: List[str] = Field(
        default_factory=list,
        description=(
            "Major series-level events that are allowed to be active in this batch, "
            "for example travel_planning, europe_trip, post_trip_souvenirs, or launch_crunch."
        ),
    )
    forbidden_timeline_conflicts: List[str] = Field(
        default_factory=list,
        description=(
            "Specific timeline mistakes this batch must avoid, such as booking a trip after it already happened "
            "or showing travel scenes outside the travel window."
        ),
    )
    independent_memory_targets: List[str] = Field(
        default_factory=list,
        description="Independent memory/event types this batch should include to avoid over-serialized storytelling."
    )
    continuity_from_previous: str = Field(description="How this batch follows prior batches without becoming a chapter opening.")
    continuity_to_next: str = Field(description="How this batch hands off later time without becoming a finale.")
    anti_repetition_notes: List[str] = Field(default_factory=list, description="Specific things to avoid repeating.")
    continuity_focus: List[str] = Field(
        default_factory=list,
        description="Memory, relationship, and recurring-story details this batch should establish."
    )


class SeriesBatchOutline(BaseModel):
    """Series-level batch map used before generating per-clip outlines."""

    project_title: str = Field(description="Human-readable project title.")
    premise: str = Field(description="Short story premise.")
    visual_style: str = Field(description="Overall visual direction.")
    protagonist_id: str = Field(default="protagonist", description="Stable protagonist character id.")
    target_core_cast_size: int = Field(default=6, description="Target count of recurring on-camera characters.")
    start_date: Optional[str] = Field(default=None, description="Series timeline start date.")
    end_date: Optional[str] = Field(default=None, description="Series timeline end date.")
    time_span: Optional[str] = Field(default=None, description="Configured planning time span.")
    continuity_rules: List[str] = Field(default_factory=list, description="Global continuity constraints.")
    recurring_character_ids: List[str] = Field(default_factory=list, description="Selected recurring cast ids for the series.")
    recurring_scene_ids: List[str] = Field(default_factory=list, description="Selected recurring scene ids for the series.")
    batches: List[BatchOutline] = Field(default_factory=list, description="Ordered batch planning briefs.")

    @model_validator(mode="after")
    def validate_batch_outline(self) -> "SeriesBatchOutline":
        if self.protagonist_id not in self.recurring_character_ids:
            raise ValueError("protagonist_id must be included in recurring_character_ids.")
        if len(self.recurring_character_ids) != self.target_core_cast_size:
            raise ValueError(
                f"recurring_character_ids must contain exactly {self.target_core_cast_size} ids."
            )
        if not self.batches:
            raise ValueError("SeriesBatchOutline expects at least one batch.")
        return self


class CharacterWardrobePlan(BaseModel):
    """Expanded reusable wardrobe for one recurring character."""

    character_id: str = Field(description="Recurring character id, for example protagonist or char_001.")
    character_name: str = Field(description="Human-readable character name.")
    role_context: str = Field(description="Short reminder of this person's social role and common contexts.")
    wardrobe_options: List[str] = Field(
        description=(
            "One to five complete English outfit descriptions. Each item must be a reusable full outfit, "
            "not a one-off shot-specific variation."
        )
    )
    selection_guidance: List[str] = Field(
        default_factory=list,
        description="Brief guidance for choosing these outfits by home/work/travel/social/seasonal context.",
    )

    @model_validator(mode="after")
    def validate_character_wardrobe(self) -> "CharacterWardrobePlan":
        cleaned: list[str] = []
        seen: set[str] = set()
        for outfit in self.wardrobe_options:
            text = str(outfit).strip()
            normalized = re.sub(r"\s+", " ", text.lower())
            if text and normalized not in seen:
                cleaned.append(text)
                seen.add(normalized)
        if not cleaned:
            raise ValueError("Each character wardrobe must include at least one outfit.")
        if len(cleaned) > 5:
            raise ValueError("Each character wardrobe can include at most five outfits.")
        self.wardrobe_options = cleaned
        return self


class SeriesWardrobePlan(BaseModel):
    """Series-level reusable wardrobe plan generated after batch outline."""

    project_title: str = Field(description="Human-readable project title.")
    premise: str = Field(description="Short story premise.")
    protagonist_id: str = Field(default="protagonist", description="Stable protagonist character id.")
    recurring_character_ids: List[str] = Field(default_factory=list, description="Recurring cast ids covered by this wardrobe.")
    max_outfits_per_character: int = Field(default=5, description="Maximum outfit slots per recurring character.")
    characters: List[CharacterWardrobePlan] = Field(default_factory=list, description="Reusable wardrobe for each recurring character.")

    @model_validator(mode="after")
    def validate_wardrobe_plan(self) -> "SeriesWardrobePlan":
        if self.max_outfits_per_character > 5:
            raise ValueError("max_outfits_per_character cannot exceed five.")
        expected_ids = set(self.recurring_character_ids)
        character_ids = [character.character_id for character in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("Wardrobe plan character ids must be unique.")
        missing = sorted(expected_ids - set(character_ids))
        extra = sorted(set(character_ids) - expected_ids)
        if missing:
            raise ValueError(f"Wardrobe plan is missing recurring character ids: {missing}")
        if extra:
            raise ValueError(f"Wardrobe plan includes non-recurring character ids: {extra}")
        for character in self.characters:
            if len(character.wardrobe_options) > self.max_outfits_per_character:
                raise ValueError(
                    f"{character.character_id} has {len(character.wardrobe_options)} outfits, "
                    f"exceeding max_outfits_per_character={self.max_outfits_per_character}."
                )
        return self


class SeriesShotPlan(BaseModel):
    """Detailed recurring cast/scene selection plus shot plans before final assembly."""

    project_title: str = Field(description="Human-readable project title.")
    premise: str = Field(description="Short story premise.")
    visual_style: str = Field(description="Overall visual direction.")
    protagonist_id: str = Field(default="protagonist", description="Stable protagonist character id.")
    target_core_cast_size: int = Field(default=6, description="Target count of recurring on-camera characters.")
    start_date: Optional[str] = Field(default=None, description="Series timeline start date.")
    end_date: Optional[str] = Field(default=None, description="Series timeline end date.")
    time_span: Optional[str] = Field(default=None, description="Configured planning time span.")
    continuity_rules: List[str] = Field(default_factory=list, description="Global continuity constraints.")
    recurring_character_ids: List[str] = Field(default_factory=list, description="Selected recurring cast ids for the series.")
    recurring_scene_ids: List[str] = Field(default_factory=list, description="Selected recurring scene ids for the series.")
    clips: List[ClipPlan] = Field(default_factory=list, description="Ordered clips with detailed shots.")

    @model_validator(mode="after")
    def validate_plan(self) -> "SeriesShotPlan":
        if self.protagonist_id not in self.recurring_character_ids:
            raise ValueError("protagonist_id must be included in recurring_character_ids.")
        if len(self.recurring_character_ids) != self.target_core_cast_size:
            raise ValueError(
                f"recurring_character_ids must contain exactly {self.target_core_cast_size} ids."
            )
        if not self.clips:
            raise ValueError("SeriesShotPlan expects at least one clip.")
        return self
