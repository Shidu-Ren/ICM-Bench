#!/usr/bin/env python3
"""
长视频生成流水线
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import (
    DEFAULT_VIDEO_CONFIG_PATH,
    DEFAULT_VIDEO_OUTPUT_BASE_DIR,
    find_latest_output_root,
    load_video_config,
)
def run_preflight(config_path: str | None = None) -> None:
    """Validate a production config without making API calls."""

    config = load_video_config(config_path)
    series_cfg = config.get("series", {}) if isinstance(config.get("series"), dict) else {}
    production_cfg = config.get("production", {}) if isinstance(config.get("production"), dict) else {}
    timeline_cfg = config.get("timeline", {}) if isinstance(config.get("timeline"), dict) else {}
    voice_cfg = config.get("voice", {}) if isinstance(config.get("voice"), dict) else {}
    text_models = config.get("text_models", {}) if isinstance(config.get("text_models"), dict) else {}
    errors: list[str] = []

    clip_count_min = int(series_cfg.get("clip_count_min", series_cfg.get("clip_count", 0)))
    clip_count_max = int(series_cfg.get("clip_count_max", series_cfg.get("clip_count", 0)))
    if clip_count_min <= 0 or clip_count_max <= 0 or clip_count_min > clip_count_max:
        errors.append("series.clip_count_min/max must be positive and ordered.")

    if float(series_cfg.get("target_total_runtime_minutes", 0)) <= 0:
        errors.append("series.target_total_runtime_minutes must be positive.")
    if not str(timeline_cfg.get("start_date", "")).strip():
        errors.append("timeline.start_date is required.")
    if not str(timeline_cfg.get("time_span", "")).strip():
        errors.append("timeline.time_span is required.")

    dialogue_min = int(series_cfg.get("dialogue_shots_per_clip_min", 0))
    dialogue_max = int(series_cfg.get("dialogue_shots_per_clip_max", 0))
    if dialogue_min < 0 or dialogue_max < dialogue_min:
        errors.append("series.dialogue_shots_per_clip_min/max must be non-negative and ordered.")

    allowed_durations = production_cfg.get("allowed_shot_durations_seconds", [4, 6, 8])
    if not isinstance(allowed_durations, list) or not allowed_durations:
        errors.append("production.allowed_shot_durations_seconds must be a non-empty list.")
    elif any(int(value) not in {4, 6, 8} for value in allowed_durations):
        errors.append("Veo shot durations must be selected from 4, 6, or 8 seconds.")

    if voice_cfg.get("enabled", False) and not str(voice_cfg.get("tts_model", "")).strip():
        errors.append("voice.tts_model is required when voice.enabled=true.")
    if voice_cfg.get("enabled", False) and voice_cfg.get("audition_enabled", False):
        audition_count = int(voice_cfg.get("audition_candidates_per_character", 0))
        if not (1 <= audition_count <= 5):
            errors.append("voice.audition_candidates_per_character should be between 1 and 5.")
        if not str(voice_cfg.get("audition_judge_model", "")).strip():
            errors.append("voice.audition_judge_model is required when audition_enabled=true.")

    required_stages = [
        "protagonist",
        "distribution",
        "character_groups",
        "scene_groups",
        "batch_outline",
        "clip_outline",
        "shot_blueprint",
        "shot_plan",
    ]
    if text_models:
        missing_text_models = [stage for stage in required_stages if not text_models.get(stage)]
        if missing_text_models:
            errors.append(f"text_models is present but missing stages: {missing_text_models}.")

    if errors:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(errors))

    print("✅ Preflight OK")
    print(f"  workspace_name: {config.get('workspace_name')}")
    print(f"  clips: {clip_count_min}-{clip_count_max}")
    print(f"  target runtime: {series_cfg.get('target_total_runtime_minutes')} min")
    print(f"  timeline: {timeline_cfg.get('start_date')} / {timeline_cfg.get('time_span')}")
    print(
        "  dialogue: "
        f"{dialogue_min}-{dialogue_max} speech shots/clip, "
        f"{series_cfg.get('dialogue_max_lines_per_clip', 'default')} lines/clip max"
    )


def run_pipeline(
    config_path: str | None = None,
    skip_planning: bool = False,
    resume_from_output: str | None = None,
    resume_stage: str | None = None,
    batch_outline_only: bool = False,
    clip_outline_only: bool = False,
    metadata_only: bool = False,
    anchors_only: bool = False,
    skip_voice: bool = False,
    output_root_arg: str | None = None,
    include_clips: list[str] | None = None,
    shot_plan_workers: int = 1,
    video_render_workers: int | None = None,
) -> None:
    from video_generator.anchor_generator import VideoPreproductionBuilder, load_anchor_manifest
    from video_generator.planner import VideoSeriesPlanner, load_series_bible
    from video_generator.renderer import VideoClipRenderer

    print("\n" + "🎬" * 30)
    print("视频生成流水线")
    print("🎬" * 30)

    output_root: Path | None = None
    series_bible = None

    if resume_from_output:
        if skip_planning:
            raise ValueError("--resume-from-output 不能和 --skip-planning 一起使用。")
        if batch_outline_only:
            raise ValueError("--resume-from-output 不能和 --batch-outline-only 一起使用。")
        normalized_stage = (resume_stage or "clip-outline").strip().lower()
        if normalized_stage not in {
            "clip-outline",
            "clip_outline",
            "shot-blueprint",
            "shot_blueprint",
            "shot-blueprints",
            "shot_blueprints",
            "shot-plan",
            "shot_plan",
        }:
            raise ValueError(
                "--resume-stage 当前支持 clip-outline、shot-blueprint 或 shot-plan。"
            )
        planner = VideoSeriesPlanner.from_config(config_path=config_path)
        if clip_outline_only:
            if normalized_stage not in {"clip-outline", "clip_outline"}:
                raise ValueError("--clip-outline-only 只能和 --resume-stage clip-outline 一起使用。")
            planner.generate_clip_outline_from_existing_batch_outline(resume_from_output)
            return
        if normalized_stage in {"shot-plan", "shot_plan"}:
            series_bible = planner.generate_from_existing_shot_blueprints(
                resume_from_output,
                shot_plan_workers=shot_plan_workers,
            )
        elif normalized_stage in {"shot-blueprint", "shot_blueprint", "shot-blueprints", "shot_blueprints"}:
            series_bible = planner.generate_from_existing_clip_outline(
                resume_from_output,
                shot_plan_workers=shot_plan_workers,
            )
        else:
            series_bible = planner.generate_from_existing_batch_outline(
                resume_from_output,
                shot_plan_workers=shot_plan_workers,
            )
        output_root = planner.output_root
    elif not skip_planning:
        planner = VideoSeriesPlanner.from_config(config_path=config_path)
        if clip_outline_only:
            raise ValueError("--clip-outline-only 需要配合 --resume-from-output 使用。")
        if batch_outline_only:
            planner.generate_until_batch_outline()
            output_root = planner.output_root
            print("\n🧭 已完成 Batch Outline，按要求停止。")
            print(f"📁 本次输出目录: {output_root}")
            print(f"  - Batch Outline: {output_root / 'metadata' / '04b_batch_outline.json'}")
            print(f"  - API Usage: {output_root / 'metadata' / '00_api_usage_summary.json'}")
            return
        series_bible = planner.generate_all()
        output_root = planner.output_root
    else:
        if batch_outline_only:
            raise ValueError("--batch-outline-only 不能和 --skip-planning 一起使用。")
        if clip_outline_only:
            raise ValueError("--clip-outline-only 不能和 --skip-planning 一起使用。")
        if output_root_arg:
            output_root = Path(output_root_arg).expanduser().resolve()
        else:
            output_root = find_latest_output_root(
                base_dir=DEFAULT_VIDEO_OUTPUT_BASE_DIR,
                marker_filename="01_series_bible.json",
            )
        print(f"📁 复用最近一次视频输出目录: {output_root}")
        series_bible = load_series_bible(output_root)

    if include_clips:
        include_set = set(include_clips)
        original_count = len(series_bible.clips)
        filtered_clips = [clip for clip in series_bible.clips if clip.id in include_set]
        missing = [clip_id for clip_id in include_clips if clip_id not in {clip.id for clip in series_bible.clips}]
        if not filtered_clips:
            raise ValueError(
                f"--include-clips did not match any clip ids. Requested: {include_clips}"
            )
        if missing:
            print(f"⚠️  以下 clip id 未命中当前 series_bible，将忽略: {missing}")
        series_bible = series_bible.model_copy(update={"clips": filtered_clips})
        print(
            f"🎯 已启用 clip 子集渲染: {len(filtered_clips)}/{original_count} "
            f"({', '.join(clip.id for clip in filtered_clips)})"
        )

    if metadata_only:
        print("\n📝 已完成导演规划，按要求停止在 metadata 阶段。")
        return

    preproduction = VideoPreproductionBuilder(
        series_bible=series_bible,
        output_root=output_root,
        config_path=config_path,
    )
    preproduction.run()

    if anchors_only:
        print("\n🖼️  已完成人物/场景/关键帧生成，按要求停止在 anchors 阶段。")
        return

    renderer = VideoClipRenderer(
        series_bible=series_bible,
        output_root=output_root,
        anchor_manifest=load_anchor_manifest(output_root),
        config_path=config_path,
        video_render_workers=video_render_workers,
    )
    renderer.render_all()

    # Voice consistency processing (optional)
    video_config = {}
    try:
        from project_config import load_video_config as _load_vc
        video_config = _load_vc(config_path)
    except Exception:
        pass

    voice_cfg = video_config.get("voice", {})
    if isinstance(voice_cfg, dict) and voice_cfg.get("enabled", False) and not skip_voice:
        from video_generator.voice.pipeline import VoiceConsistencyPipeline

        voice_pipeline = VoiceConsistencyPipeline(
            series_bible=series_bible,
            output_root=output_root,
            config_path=config_path,
        )
        voice_pipeline.process_all_clips()

    print("\n" + "=" * 60)
    print("✅ 视频流水线完成")
    print("=" * 60)
    print(f"📁 本次输出目录: {output_root}")
    print(f"  - 元数据: {output_root / 'metadata'}")
    print(f"  - API Usage: {output_root / 'metadata' / '00_api_usage_summary.json'}")
    print(f"  - 参考图/关键帧: {output_root / 'assets'}")
    print(f"  - 最终 Clip: {output_root / 'renders' / 'clips'}")
    if isinstance(voice_cfg, dict) and voice_cfg.get("enabled", False) and not skip_voice:
        voice_output_name = voice_cfg.get("output_dir_name") or "clips_voiced"
        print(f"  - 声纹一致 Clip: {output_root / 'renders' / str(voice_output_name)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="独立视频生成流水线：先做导演规划，再做 anchor image，最后渲染一组 clips。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_VIDEO_CONFIG_PATH),
        help="视频 YAML 配置文件路径",
    )
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="跳过导演规划，复用最近一次 video_runs 输出目录中的 01_series_bible.json",
    )
    parser.add_argument(
        "--resume-from-output",
        type=str,
        help="复用已有 video_runs 输出目录中的 01-04b metadata，并从指定阶段继续生成",
    )
    parser.add_argument(
        "--resume-stage",
        type=str,
        default="clip-outline",
        help="从已有输出目录继续的阶段；支持 clip-outline、shot-blueprint 或 shot-plan",
    )
    parser.add_argument(
        "--shot-plan-workers",
        type=int,
        default=1,
        help="生成 detailed shot plan 时的并发 worker 数；建议 2-4，过高可能触发限流",
    )
    parser.add_argument(
        "--batch-outline-only",
        action="store_true",
        help="只生成到全局 Batch Outline，不继续生成 clip outline、shot plan、图片或视频",
    )
    parser.add_argument(
        "--clip-outline-only",
        action="store_true",
        help="从已有 Batch Outline 继续，只生成到 Clip Outline，不继续生成 shot blueprint、shot plan、图片或视频",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只生成导演规划，不生成图片和视频",
    )
    parser.add_argument(
        "--anchors-only",
        action="store_true",
        help="生成导演规划、人物参考图、场景参考图和关键帧，但不渲染最终视频",
    )
    parser.add_argument(
        "--skip-voice",
        action="store_true",
        help="跳过 voice consistency/TTS 后处理，只渲染基础视频 clips。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只做本地配置/prompt 检查，不调用 API。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        help="显式指定要复用的现有视频输出目录；通常与 --skip-planning 一起使用",
    )
    parser.add_argument(
        "--include-clips",
        type=str,
        help="仅处理指定 clip id，逗号分隔，例如 clip_01,clip_02,clip_05",
    )
    parser.add_argument(
        "--video-render-workers",
        type=int,
        default=None,
        help="覆盖 production.video_render_workers，用于并发渲染 Veo clip。",
    )

    args = parser.parse_args()

    if args.preflight_only:
        run_preflight(args.config)
        return

    run_pipeline(
        config_path=args.config,
        skip_planning=args.skip_planning,
        resume_from_output=args.resume_from_output,
        resume_stage=args.resume_stage,
        batch_outline_only=args.batch_outline_only,
        clip_outline_only=args.clip_outline_only,
        metadata_only=args.metadata_only,
        anchors_only=args.anchors_only,
        skip_voice=args.skip_voice,
        output_root_arg=args.output_root,
        include_clips=[
            part.strip() for part in (args.include_clips or "").split(",") if part.strip()
        ] or None,
        shot_plan_workers=args.shot_plan_workers,
        video_render_workers=args.video_render_workers,
    )


if __name__ == "__main__":
    main()
