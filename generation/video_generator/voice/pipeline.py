"""Voice Consistency Pipeline — script-first Gemini TTS redubbing.

The current workflow intentionally treats the shot metadata as the source of
truth for "who says what". Instead of trying to infer identities from raw
audio, we:

1. Extract the clip audio
2. Detect approximate speech regions with VAD
3. Align metadata dialogue lines to those regions (with a shot-timing fallback)
4. Generate per-line Gemini TTS using fixed per-character voices
5. Duck the original audio bed during dialogue windows
6. Overlay the synthesized dialogue while preserving ambience/SFX
7. Mux the new audio back to the clip
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import soundfile as sf
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import get_google_api_key, load_video_config
from video_generator.api_usage import ApiUsageLogger
from video_generator.schemas import SeriesBible
from video_generator.voice.converter import VoiceConverter
from video_generator.voice.mixer import AudioMixer
from video_generator.voice.reference_gen import ReferenceVoiceGenerator
from video_generator.voice.separator import AudioSeparator, SeparationResult
from video_generator.voice.speaker_mapper import SpeakerCharacterMapper
from video_generator.voice.vad import VoiceActivityDetector


class VoiceConsistencyPipeline:
    """Script-guided Gemini TTS dialogue replacement pipeline."""

    def __init__(
        self,
        series_bible: SeriesBible,
        output_root: str | Path,
        config_path: str | None = None,
        include_clip_ids: set[str] | None = None,
        force: bool = False,
        voice_method: str | None = None,
        voice_device: str | None = None,
        diffusion_steps: int | None = None,
        voice_output_dir_name: str | None = None,
        voice_work_dir_name: str | None = None,
        voice_workers: int | None = None,
        semantic_shortening: bool | None = None,
        semantic_shortening_model: str | None = None,
    ) -> None:
        self.series_bible = series_bible
        self.output_root = Path(output_root)
        self.config = load_video_config(config_path)
        self.include_clip_ids = set(include_clip_ids or [])
        self.force = force

        voice_cfg = self.config.get("voice", {})
        if not isinstance(voice_cfg, dict):
            voice_cfg = {}

        self.voice_method = str(voice_method or voice_cfg.get("method", "gemini_tts_redub"))
        self.voice_device = str(voice_device or voice_cfg.get("device", "cuda:0"))
        seed_vc_value = os.environ.get("SEED_VC_DIR") or voice_cfg.get(
            "seed_vc_dir", "external/seed-vc"
        )
        seed_vc_path = Path(str(seed_vc_value)).expanduser()
        if not seed_vc_path.is_absolute():
            seed_vc_path = PROJECT_ROOT / seed_vc_path
        self.seed_vc_dir = str(seed_vc_path.resolve())
        self.demucs_model = str(voice_cfg.get("demucs_model", "htdemucs_ft"))
        self.redub_background_mode = str(voice_cfg.get("redub_background_mode", "silent")).lower()
        self.dialogue_duck_db = float(voice_cfg.get("dialogue_duck_db", 18.0))
        self.dialogue_fade_seconds = float(voice_cfg.get("dialogue_fade_seconds", 0.08))
        self.min_line_duration_seconds = float(voice_cfg.get("min_line_duration_seconds", 0.9))
        self.max_tts_stretch_ratio = float(voice_cfg.get("max_tts_stretch_ratio", 1.45))
        self.min_tts_stretch_ratio = float(voice_cfg.get("min_tts_stretch_ratio", 0.72))
        self.allow_unbounded_tts_speedup = bool(voice_cfg.get("allow_unbounded_tts_speedup", True))
        self.strict_tts_fit = bool(voice_cfg.get("strict_tts_fit", False))
        self.max_tts_truncation_seconds = max(
            0.0,
            float(voice_cfg.get("max_tts_truncation_seconds", 0.35)),
        )
        self.shorten_overlong_dialogue = bool(voice_cfg.get("shorten_overlong_dialogue", True))
        self.semantic_shortening_enabled = bool(
            semantic_shortening
            if semantic_shortening is not None
            else voice_cfg.get("semantic_shortening_enabled", False)
        )
        self.semantic_shortening_model = str(
            semantic_shortening_model
            or voice_cfg.get("semantic_shortening_model")
            or "gemini-3.1-flash-preview"
        )
        self.tts_shortening_max_rounds = max(0, int(voice_cfg.get("tts_shortening_max_rounds", 2)))
        self.tts_estimated_words_per_second = max(
            0.8,
            float(voice_cfg.get("tts_estimated_words_per_second", 1.75)),
        )
        self.tts_shortening_min_words = max(1, int(voice_cfg.get("tts_shortening_min_words", 3)))
        self.tts_shortening_buffer_seconds = max(
            0.0,
            float(voice_cfg.get("tts_shortening_buffer_seconds", 0.15)),
        )
        self.conversion_min_segment_seconds = float(
            voice_cfg.get("conversion_min_segment_seconds", self.min_line_duration_seconds)
        )
        self.conversion_segment_padding_seconds = float(
            voice_cfg.get("conversion_segment_padding_seconds", 0.12)
        )
        self.conversion_source = str(voice_cfg.get("conversion_source", "script_tts"))
        self.strict_voice_conversion = bool(voice_cfg.get("strict_voice_conversion", True))
        self.voice_workers = max(
            1,
            int(
                voice_workers
                if voice_workers is not None
                else voice_cfg.get("voice_workers", voice_cfg.get("workers", 1))
            ),
        )
        self.default_extra_voices = [
            str(voice)
            for voice in voice_cfg.get(
                "default_extra_voices",
                ["Schedar", "Rasalgethi", "Iapetus", "Charon"],
            )
        ]
        self.seed_vc_diffusion_steps = int(
            diffusion_steps if diffusion_steps is not None else voice_cfg.get("diffusion_steps", 25)
        )
        self.seed_vc_length_adjust = float(voice_cfg.get("length_adjust", 1.0))
        self.seed_vc_inference_cfg_rate = float(voice_cfg.get("inference_cfg_rate", 0.7))

        self.vad = VoiceActivityDetector(
            min_speech_duration_ms=int(voice_cfg.get("min_speech_duration_ms", 220)),
            min_silence_duration_ms=int(voice_cfg.get("min_silence_duration_ms", 260)),
        )
        self.mapper = SpeakerCharacterMapper(series_bible=series_bible)
        self.mixer = AudioMixer(
            vocals_loudness_lufs=float(voice_cfg.get("vocals_loudness_lufs", -16)),
            background_loudness_lufs=float(voice_cfg.get("background_loudness_lufs", -26)),
        )
        self.ref_generator = ReferenceVoiceGenerator(
            api_key=get_google_api_key(),
            output_dir=self.output_root / "assets" / "voices",
            tts_model=str(voice_cfg.get("tts_model", "gemini-3.1-flash-tts-preview")),
            voice_assignments=voice_cfg.get("voice_assignments") or {},
            male_voices=voice_cfg.get("default_male_voices"),
            female_voices=voice_cfg.get("default_female_voices"),
        )
        self.semantic_client = genai.Client(api_key=get_google_api_key())
        self._semantic_client_local = threading.local()
        self._semantic_client_local.client = self.semantic_client
        self.api_usage_logger = ApiUsageLogger(self.output_root / "metadata")

        self.clips_dir = self.output_root / "renders" / "clips"
        self.voice_work_dir = self.output_root / str(
            voice_work_dir_name or voice_cfg.get("work_dir_name") or "voice_work"
        )
        output_dir_name = voice_output_dir_name or voice_cfg.get("output_dir_name")
        if not output_dir_name:
            output_dir_name = (
                "clips_voice_converted"
                if self._is_seed_vc_conversion_mode()
                else "clips_voiced"
            )
        self.voice_output_dir = self.output_root / "renders" / str(output_dir_name)
        srt_dir_name = voice_cfg.get("srt_dir_name") or f"{output_dir_name}_srt"
        self.voice_srt_dir = self.output_root / "subtitles" / str(srt_dir_name)
        self.voice_work_dir.mkdir(parents=True, exist_ok=True)
        self.voice_output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_srt_dir.mkdir(parents=True, exist_ok=True)

        self.reuse_existing = bool(
            self.config.get("production", {}).get("reuse_existing_assets", True)
        )
        self._voice_assignment_lock = threading.RLock()
        self.cast_by_id = {member.id: member for member in self.series_bible.cast}

    def _thread_semantic_client(self):
        client = getattr(self._semantic_client_local, "client", None)
        if client is None:
            client = genai.Client(api_key=get_google_api_key())
            self._semantic_client_local.client = client
        return client

    def process_all_clips(self) -> None:
        """Main entry point."""
        print("\n" + "🎙️ " * 20)
        print("Voice Consistency Pipeline")
        print("🎙️ " * 20)
        print(f"🧭 Mode: {self.voice_method}")

        if self._is_seed_vc_conversion_mode():
            voice_assignments, reference_paths = self._prepare_voice_references()
        else:
            voice_assignments = self._prepare_voice_assignments()
            reference_paths = {}

        if not voice_assignments:
            print("⚠️  No voice assignments available. Skipping voice processing.")
            return

        target_clips = [
            clip
            for clip in self.series_bible.clips
            if not self.include_clip_ids or clip.id in self.include_clip_ids
        ]
        failed_clip_ids: list[str] = []

        workers = min(self.voice_workers, max(1, len(target_clips)))
        if workers > 1:
            print(f"🚀 Concurrent voice redub: workers={workers}, total_clips={len(target_clips)}")

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            clip_iter = iter(target_clips)
            pending = {}

            def submit_next() -> None:
                try:
                    clip = next(clip_iter)
                except StopIteration:
                    return
                pending[executor.submit(
                    self._process_voice_clip,
                    clip,
                    voice_assignments,
                    reference_paths,
                )] = clip

            for _ in range(workers):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    clip = pending.pop(future)
                    clip_id, ok = future.result()
                    completed += 1
                    if not ok:
                        failed_clip_ids.append(clip_id)
                    print(f"✅ Voice 进度: {completed}/{len(target_clips)} ({clip.id})")
                    submit_next()

        print("\n" + "=" * 60)
        print("✅ Voice Consistency Pipeline complete")
        print("=" * 60)
        print(f"📁 Original clips: {self.clips_dir}")
        print(f"📁 Voiced clips:   {self.voice_output_dir}")

        if failed_clip_ids:
            raise RuntimeError(
                "Voice processing failed for clip(s): " + ", ".join(failed_clip_ids)
            )

    def _process_voice_clip(
        self,
        clip,
        voice_assignments: dict[str, str],
        reference_paths: dict[str, Path],
    ) -> tuple[str, bool]:
        clip_path = self.clips_dir / f"{clip.id}.mp4"
        output_path = self.voice_output_dir / f"{clip.id}.mp4"

        if self._should_reuse_voiced_clip(clip.id, output_path):
            print(f"\n♻️  Reusing voiced clip: {clip.id}")
            return clip.id, True

        if not clip_path.exists():
            print(f"\n⚠️  Clip not found, skipping: {clip_path}")
            return clip.id, True

        print(f"\n{'=' * 60}")
        print(f"🎬 Processing clip: {clip.id} ({clip.title})")
        print(f"{'=' * 60}")

        try:
            if self._is_seed_vc_conversion_mode():
                self._process_single_clip_conversion(
                    clip=clip,
                    clip_path=clip_path,
                    output_path=output_path,
                    voice_assignments=voice_assignments,
                    reference_paths=reference_paths,
                )
            else:
                self._process_single_clip_tts(
                    clip=clip,
                    clip_path=clip_path,
                    output_path=output_path,
                    voice_assignments=voice_assignments,
                )
            self._write_render_manifest(clip.id, clip_path, output_path)
            return clip.id, True
        except Exception as exc:
            print(f"❌ Voice processing failed for {clip.id}: {exc}")
            import traceback

            traceback.print_exc()
            return clip.id, False

    def _is_seed_vc_conversion_mode(self) -> bool:
        return self.voice_method in {
            "seed_vc",
            "seed_vc_conversion",
            "local_voice_conversion",
            "voice_conversion",
        }

    def _process_single_clip_tts(
        self,
        clip,
        clip_path: Path,
        output_path: Path,
        voice_assignments: dict[str, str],
    ) -> None:
        """Replace scripted dialogue with Gemini TTS while preserving ambience."""
        work_dir = self.voice_work_dir / clip.id
        work_dir.mkdir(parents=True, exist_ok=True)

        raw_audio = work_dir / "raw_audio.wav"
        video_duration = AudioMixer.video_duration_seconds(clip_path)
        use_silent_redub = self.redub_background_mode in {
            "silent",
            "none",
            "tts_only",
            "script_only",
        }

        if use_silent_redub:
            AudioMixer.create_silent_audio(video_duration, raw_audio)
            vad_segments = []
            print(
                f"   🔇 Redub background mode '{self.redub_background_mode}'; "
                f"using a {video_duration:.2f}s silent bed and script-timed dialogue windows."
            )
        elif AudioMixer.has_audio_track(clip_path):
            AudioMixer.extract_audio(clip_path, raw_audio)
            print("\n📌 Step 1: Voice activity detection")
            vad_segments = self.vad.detect(raw_audio)
        else:
            AudioMixer.create_silent_audio(video_duration, raw_audio)
            vad_segments = []
            print(
                f"   🔇 No audio track in video; using a {video_duration:.2f}s silent bed "
                "and script-timed dialogue windows."
            )

        print("\n📌 Step 2: Script-guided dialogue alignment")
        aligned_dialogue = self.mapper.align_dialogue_segments(
            vad_segments=vad_segments,
            clip=clip,
            output_dir=work_dir,
        )

        if not aligned_dialogue:
            print("   No scripted dialogue aligned for this clip. Copying original clip.")
            import shutil

            shutil.copyfile(clip_path, output_path)
            return

        print("\n📌 Step 3: Gemini TTS synthesis")
        tts_dir = work_dir / "tts_lines"
        tts_dir.mkdir(parents=True, exist_ok=True)

        synthesized_segments: list[tuple[float, float, Path]] = []
        synthesis_manifest: list[dict[str, object]] = []

        for index, segment in enumerate(aligned_dialogue):
            char_id = segment["char_id"]
            voice_name = self._voice_for_dialogue_character(char_id, voice_assignments)
            if not voice_name:
                print(f"   ⏭️  Skipping line {index} ({char_id}): no assigned Gemini voice")
                continue

            member = self.cast_by_id.get(char_id)
            style_prompt = None
            if member is not None:
                style_prompt = (
                    f"{getattr(member, 'voice_brief', '').strip()} "
                    "Keep the line natural, low-key, and conversational. "
                    "Do not add narration, stage directions, or extra words."
                ).strip()

            raw_line_path = tts_dir / f"line_{index:03d}_{char_id}_raw.wav"
            fitted_line_path = tts_dir / f"line_{index:03d}_{char_id}_fit.wav"
            target_duration = max(
                self.min_line_duration_seconds,
                float(segment["end"]) - float(segment["start"]),
            )
            original_text = str(segment["text"])
            tts_text, shorten_meta = self._synthesize_dialogue_with_timing_guard(
                text=original_text,
                voice_name=voice_name,
                raw_line_path=raw_line_path,
                target_duration=target_duration,
                style_prompt=style_prompt,
                clip=clip,
                segment=segment,
            )
            self._fit_dialogue_to_window(
                source_path=raw_line_path,
                output_path=fitted_line_path,
                target_duration=target_duration,
            )

            synthesized_segments.append((float(segment["start"]), float(segment["end"]), fitted_line_path))
            synthesis_manifest.append(
                {
                    "index": index,
                    "shot_id": segment.get("shot_id"),
                    "char_id": char_id,
                    "character_name": member.name_en if member else char_id,
                    "voice_name": voice_name,
                    "text": tts_text,
                    "original_text": original_text,
                    "text_shortened": tts_text != original_text,
                    "shortening": shorten_meta,
                    "start": segment["start"],
                    "end": segment["end"],
                    "path": str(fitted_line_path),
                }
            )

        if not synthesized_segments:
            print("   No dialogue segments synthesized. Copying original clip.")
            import shutil

            shutil.copyfile(clip_path, output_path)
            return

        manifest_path = work_dir / "tts_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"segments": synthesis_manifest}, f, indent=2, ensure_ascii=False)
        self._write_tts_srt(synthesis_manifest, work_dir / "dialogue.srt", include_speaker=True)
        self._write_tts_srt(
            synthesis_manifest,
            self.voice_srt_dir / f"{clip.id}.srt",
            include_speaker=True,
        )

        print("\n📌 Step 4: Build dialogue track")
        dialogue_track = work_dir / "dialogue_track.wav"
        self.mixer.assemble_dialogue_track(
            base_audio_path=raw_audio,
            dialogue_segments=synthesized_segments,
            output_path=dialogue_track,
        )

        if use_silent_redub:
            print("\n📌 Step 5: Use silent bed")
            background_bed = raw_audio
        else:
            print("\n📌 Step 5: Duck original bed under dialogue")
            ducked_bed = work_dir / "ducked_bed.wav"
            self.mixer.duck_audio(
                audio_path=raw_audio,
                segments=[(start, end) for start, end, _ in synthesized_segments],
                output_path=ducked_bed,
                reduction_db=self.dialogue_duck_db,
                fade_seconds=self.dialogue_fade_seconds,
            )
            background_bed = ducked_bed

        print("\n📌 Step 6: Mix dialogue over bed")
        mixed_audio = work_dir / "mixed_audio.wav"
        self.mixer.mix_audio(
            vocals_path=dialogue_track,
            background_path=background_bed,
            output_path=mixed_audio,
        )

        print("\n📌 Step 7: Mux audio to video")
        self.mixer.mux_audio_to_video(
            video_path=clip_path,
            audio_path=mixed_audio,
            output_path=output_path,
        )

        print(f"\n✅ Voiced clip saved: {output_path}")

    def _synthesize_dialogue_with_timing_guard(
        self,
        *,
        text: str,
        voice_name: str,
        raw_line_path: Path,
        target_duration: float,
        style_prompt: str | None,
        clip=None,
        segment: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Generate TTS, shortening locally when the raw line cannot fit."""
        current_text = self._clean_dialogue_text_for_tts(text)
        original_text = current_text
        allowed_raw_duration = max(
            target_duration * self.max_tts_stretch_ratio - self.tts_shortening_buffer_seconds,
            target_duration,
        )
        hard_allowed_raw_duration = max(
            target_duration * self.max_tts_stretch_ratio,
            target_duration,
        )
        metadata: dict[str, object] = {
            "target_duration": round(target_duration, 3),
            "max_allowed_raw_duration": round(allowed_raw_duration, 3),
            "hard_allowed_raw_duration": round(hard_allowed_raw_duration, 3),
            "rounds": [],
        }

        max_rounds = self.tts_shortening_max_rounds if self.shorten_overlong_dialogue else 0
        for round_index in range(max_rounds + 1):
            try:
                self.ref_generator.synthesize_dialogue_line(
                    text=current_text,
                    voice_name=voice_name,
                    save_path=raw_line_path,
                    style_prompt=style_prompt,
                )
            except Exception as exc:
                if metadata["rounds"] and raw_line_path.exists():
                    previous_round = metadata["rounds"][-1]
                    metadata["final_raw_duration"] = previous_round["raw_duration"]
                    metadata["note"] = (
                        "shortened TTS failed; reused previous successful audio before final time-fit"
                    )
                    print(
                        "   ⚠️  Shortened TTS failed; reusing previous successful audio "
                        f"for final fit: {exc}"
                    )
                    return str(previous_round["text"]), metadata
                raise
            raw_duration = self._audio_duration_seconds(raw_line_path)
            metadata["rounds"].append(
                {
                    "round": round_index,
                    "text": current_text,
                    "raw_duration": round(raw_duration, 3),
                }
            )
            if (
                self.allow_unbounded_tts_speedup
                and round_index > 0
                and current_text != original_text
            ):
                metadata["final_raw_duration"] = round(raw_duration, 3)
                metadata["note"] = (
                    "shortened once; final ffmpeg fit may use unbounded speed-up "
                    "to preserve the complete line"
                )
                return current_text, metadata
            if raw_duration <= allowed_raw_duration or (
                round_index > 0 and raw_duration <= hard_allowed_raw_duration
            ):
                metadata["final_raw_duration"] = round(raw_duration, 3)
                if raw_duration > allowed_raw_duration:
                    metadata["note"] = "shortened locally; final ffmpeg fit absorbs small overrun"
                return current_text, metadata

            if round_index >= max_rounds:
                break

            if self.semantic_shortening_enabled:
                shortened = self._semantic_shorten_dialogue_text(
                    current_text,
                    target_duration=target_duration,
                    raw_duration=raw_duration,
                    round_index=round_index + 1,
                    clip=clip,
                    segment=segment,
                )
                metadata.setdefault("semantic_shortening", True)
            else:
                shortened = self._shorten_dialogue_text(
                    current_text,
                    target_duration=target_duration,
                    raw_duration=raw_duration,
                    round_index=round_index + 1,
                )
            if shortened == current_text:
                break
            print(
                "   ✂️  TTS raw line is too long "
                f"({raw_duration:.2f}s for {target_duration:.2f}s window); "
                f"shortening: \"{current_text}\" -> \"{shortened}\""
            )
            current_text = shortened

        metadata["final_raw_duration"] = round(self._audio_duration_seconds(raw_line_path), 3)
        if current_text != original_text:
            metadata["note"] = (
                "shortened before final time-fit; final ffmpeg fit preserves the complete line"
            )
        else:
            metadata["note"] = "not shortened; final fit preserves the complete line"
        return current_text, metadata

    def _semantic_shorten_dialogue_text(
        self,
        text: str,
        *,
        target_duration: float,
        raw_duration: float,
        round_index: int,
        clip=None,
        segment: dict[str, object] | None = None,
    ) -> str:
        cleaned = self._clean_dialogue_text_for_tts(text)
        fallback = self._shorten_dialogue_text(
            cleaned,
            target_duration=target_duration,
            raw_duration=raw_duration,
            round_index=round_index,
        )
        if not cleaned:
            return cleaned

        budget = self._word_budget_for_tts_window(
            target_duration=target_duration,
            raw_duration=raw_duration,
            current_word_count=self._word_count(cleaned),
            round_index=round_index,
        )
        # Give Gemini a little room; TTS duration, not word count alone, is the real guard.
        max_words = max(self.tts_shortening_min_words, min(max(budget + 1, 5), 10))
        shot = self._shot_for_segment(clip, segment)
        speaker_name = ""
        if segment:
            member = self.cast_by_id.get(str(segment.get("char_id") or ""))
            speaker_name = member.name_en if member else str(segment.get("char_id") or "")

        context_lines = []
        if clip is not None:
            context_lines.append(f"Clip: {getattr(clip, 'id', '')} - {getattr(clip, 'title', '')}")
        if shot is not None:
            context_lines.extend(
                [
                    f"Shot: {getattr(shot, 'id', '')}",
                    f"Beat: {getattr(shot, 'beat_title', '')}",
                    f"Purpose: {getattr(shot, 'purpose', '')}",
                    f"Blocking: {getattr(shot, 'blocking_notes', '')}",
                    "Evidence: " + " | ".join(getattr(shot, "evidence_facts", []) or []),
                ]
            )

        prompt = f"""
Rewrite this single spoken dialogue line so it fits a short {target_duration:.2f}s video shot.

Rules:
- Preserve the meaning and scene relevance.
- Keep the same speaker; do not add a speaker name.
- Natural conversational English, adult everyday speech.
- One short sentence or fragment, maximum {max_words} words.
- Do not end with an incomplete phrase.
- Do not add stage directions, narration, quotation marks, or explanations.

Speaker: {speaker_name}
Original line: {cleaned}
Current raw TTS duration: {raw_duration:.2f}s
Target dialogue window: {target_duration:.2f}s
Context:
{chr(10).join(context_lines)}

Return only JSON: {{"text": "short line"}}
""".strip()

        try:
            response = self._thread_semantic_client().models.generate_content(
                model=self.semantic_shortening_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            self.api_usage_logger.record_response(
                response=response,
                operation="semantic_tts_shortening",
                model=self.semantic_shortening_model,
                prompt=prompt,
                attempt=round_index,
                extra={
                    "clip_id": getattr(clip, "id", None),
                    "shot_id": segment.get("shot_id") if segment else None,
                    "target_duration": target_duration,
                    "raw_duration": raw_duration,
                },
            )
            parsed = json.loads(getattr(response, "text", "") or "{}")
            candidate = self._normalize_shortened_dialogue(str(parsed.get("text") or ""))
            if self._semantic_shortening_candidate_ok(candidate, max_words=max_words):
                return candidate
            print(
                "   ⚠️  Semantic shortening produced invalid text; "
                f"falling back locally: {candidate!r}"
            )
        except Exception as exc:
            self.api_usage_logger.record_failure(
                operation="semantic_tts_shortening",
                model=self.semantic_shortening_model,
                error=exc,
                prompt=prompt,
                attempt=round_index,
                extra={
                    "clip_id": getattr(clip, "id", None),
                    "shot_id": segment.get("shot_id") if segment else None,
                },
            )
            print(f"   ⚠️  Semantic shortening failed; falling back locally: {exc}")
        return fallback

    def _shot_for_segment(self, clip, segment: dict[str, object] | None):
        if clip is None or not segment:
            return None
        shot_id = str(segment.get("shot_id") or "")
        for shot in getattr(clip, "shots", []) or []:
            if getattr(shot, "id", "") == shot_id:
                return shot
        return None

    def _semantic_shortening_candidate_ok(self, text: str, *, max_words: int) -> bool:
        if not text or len(text) > 140:
            return False
        if "\n" in text or ":" in text:
            return False
        if self._word_count(text) > max_words + 2:
            return False
        weak_tail = {
            "a", "an", "the", "and", "or", "but", "so", "if", "with", "to",
            "for", "from", "of", "in", "on", "at", "by", "than", "as",
            "into", "when", "while", "because", "that", "this", "these",
            "those", "have", "has", "had", "haven't", "need", "needs",
            "can", "could", "would", "should", "will", "is", "are",
        }
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
        return not words or words[-1].lower() not in weak_tail

    @staticmethod
    def _audio_duration_seconds(path: Path) -> float:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)

    def _clean_dialogue_text_for_tts(self, text: str) -> str:
        cleaned = " ".join(str(text).strip().split())
        names = [member.name_en for member in self.series_bible.cast]
        names.extend(member.name_en.split()[0] for member in self.series_bible.cast if member.name_en)
        name_pattern = "|".join(re.escape(name) for name in sorted(set(names), key=len, reverse=True))
        if name_pattern:
            cleaned = re.sub(
                rf"(?<!\w)(?:{name_pattern})\s*:\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        return cleaned.strip("\"' \u201c\u201d")

    def _shorten_dialogue_text(
        self,
        text: str,
        *,
        target_duration: float,
        raw_duration: float,
        round_index: int,
    ) -> str:
        cleaned = self._clean_dialogue_text_for_tts(text)
        if not cleaned:
            return cleaned

        rewritten = self._normalize_shortened_dialogue(
            self._remove_dialogue_filler(cleaned)
        )

        allowed_raw_duration = max(
            target_duration * self.max_tts_stretch_ratio - self.tts_shortening_buffer_seconds,
            target_duration,
        )
        estimated_budget = max(
            self.tts_shortening_min_words,
            int(allowed_raw_duration * self.tts_estimated_words_per_second),
        )
        if rewritten != cleaned and self._word_count(rewritten) <= estimated_budget:
            return self._normalize_shortened_dialogue(rewritten)

        budget = self._word_budget_for_tts_window(
            target_duration=target_duration,
            raw_duration=raw_duration,
            current_word_count=self._word_count(rewritten),
            round_index=round_index,
        )
        if self._word_count(rewritten) <= budget:
            return self._normalize_shortened_dialogue(rewritten)
        return self._truncate_to_word_budget(rewritten, budget)

    def _word_budget_for_tts_window(
        self,
        *,
        target_duration: float,
        raw_duration: float,
        current_word_count: int,
        round_index: int,
    ) -> int:
        allowed_raw_duration = max(
            target_duration * self.max_tts_stretch_ratio - self.tts_shortening_buffer_seconds,
            target_duration,
        )
        ratio_budget = int(current_word_count * allowed_raw_duration / max(raw_duration, 0.01) * 0.92)
        estimated_budget = int(allowed_raw_duration * self.tts_estimated_words_per_second)
        budget = min(ratio_budget, estimated_budget)
        if round_index > 1:
            budget = int(budget * 0.85)
        return max(self.tts_shortening_min_words, budget)

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))

    def _remove_dialogue_filler(self, text: str) -> str:
        names = {member.name_en for member in self.series_bible.cast if member.name_en}
        names.update(name.split()[0] for name in list(names) if name.split())
        if names:
            name_pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
            text = re.sub(
                rf"^(?:{name_pattern})\s*,\s+",
                "",
                text,
                flags=re.IGNORECASE,
            )
        replacements = [
            (r"\breally\b", ""),
            (r"\bactually\b", ""),
            (r"\bdefinitely\b", ""),
            (r"\bsurprisingly\b", ""),
            (r"\bspecific\b", ""),
            (r"\bjust\b", ""),
            (r"\bquite\b", ""),
            (r"\balready\b", ""),
            (r"\bfor a change\b", ""),
            (r"\bby Friday\b", "Friday"),
            (r"\bsystem architecture\b", "architecture"),
            (r"\bthe new feature\b", "the feature"),
            (r"\bnew app\b", "app"),
            (r"\bmake sure\b", "check"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return text.strip()

    @staticmethod
    def _normalize_shortened_dialogue(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        for index, char in enumerate(text):
            if char.isalpha():
                return text[:index] + char.upper() + text[index + 1 :]
        return text

    def _truncate_to_word_budget(self, text: str, budget: int) -> str:
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", text)
        kept_words = 0
        output: list[str] = []
        for token in words:
            if re.match(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*$", token):
                if kept_words >= budget:
                    break
                kept_words += 1
            output.append(token)
        output = self._trim_incomplete_dialogue_tail(output)
        shortened = " ".join(output)
        shortened = re.sub(r"\s+([,.!?;:])", r"\1", shortened).strip(" ,;:")
        if shortened and shortened[-1] not in ".!?":
            question_starters = {"can", "could", "did", "do", "does", "is", "are", "was", "were", "will", "would"}
            first_word = re.match(r"[A-Za-z0-9]+", shortened)
            shortened += "?" if first_word and first_word.group(0).lower() in question_starters else "."
        return self._normalize_shortened_dialogue(shortened) or text

    @staticmethod
    def _trim_incomplete_dialogue_tail(tokens: list[str]) -> list[str]:
        weak_endings = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "so",
            "if",
            "with",
            "to",
            "for",
            "from",
            "of",
            "in",
            "on",
            "at",
            "by",
            "than",
            "as",
            "into",
            "over",
            "under",
            "when",
            "while",
            "since",
            "because",
            "that",
            "this",
            "these",
            "those",
            "my",
            "your",
            "our",
            "his",
            "her",
            "their",
            "i",
            "i'm",
            "you",
            "we",
            "they",
            "he",
            "she",
            "it",
            "can",
            "could",
            "would",
            "should",
            "will",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "haven't",
            "hasn't",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "can't",
            "couldn't",
            "shouldn't",
            "need",
            "needs",
            "li",
            "ming",
            "wang",
            "lin",
            "chen",
            "tao",
            "sarah",
            "wu",
            "zhang",
            "hua",
            "jian",
            "mom",
            "dad",
            "ma",
        }
        punctuation = {",", ";", ":"}
        trimmed = list(tokens)
        while trimmed:
            tail = trimmed[-1]
            if tail in punctuation:
                trimmed.pop()
                continue
            if re.match(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*$", tail) and tail.lower() in weak_endings:
                trimmed.pop()
                continue
            break
        return trimmed

    def _voice_for_dialogue_character(
        self,
        char_id: str,
        voice_assignments: dict[str, str],
    ) -> str | None:
        with self._voice_assignment_lock:
            voice_name = voice_assignments.get(char_id)
            if voice_name or not char_id.startswith("extra_") or not self.default_extra_voices:
                return voice_name

            index = sum(ord(char) for char in char_id) % len(self.default_extra_voices)
            voice_name = self.default_extra_voices[index]
            voice_assignments[char_id] = voice_name
            self.ref_generator.save_voice_manifest(assignments=voice_assignments, reference_paths={})
            return voice_name

    def _process_single_clip_conversion(
        self,
        clip,
        clip_path: Path,
        output_path: Path,
        voice_assignments: dict[str, str],
        reference_paths: dict[str, Path],
    ) -> None:
        """Separate native audio, convert scripted vocal windows with Seed-VC, then remix."""
        work_dir = self.voice_work_dir / clip.id
        work_dir.mkdir(parents=True, exist_ok=True)

        raw_audio = work_dir / "raw_audio.wav"
        has_native_audio = AudioMixer.has_audio_track(clip_path)
        if has_native_audio:
            AudioMixer.extract_audio(clip_path, raw_audio)
            raw_duration = AudioMixer.audio_duration_seconds(raw_audio)

            print("\n📌 Step 1: Separate vocals/background with Demucs")
            separator = AudioSeparator(
                model=self.demucs_model,
                device=self.voice_device,
                output_dir=work_dir / "demucs",
                python_executable=sys.executable,
            )
            separated = separator.separate(raw_audio)

            print("\n📌 Step 2: Voice activity detection on isolated vocals")
            vad_segments = self.vad.detect(separated.vocals_path)
        else:
            raw_duration = AudioMixer.video_duration_seconds(clip_path)
            AudioMixer.create_silent_audio(raw_duration, raw_audio)
            separated = SeparationResult(vocals_path=raw_audio, background_path=raw_audio)
            vad_segments = []
            print(
                f"   🔇 No audio track in video; using a {raw_duration:.2f}s silent bed "
                "and script-timed dialogue windows."
            )

        print("\n📌 Step 3: Script-guided dialogue alignment")
        aligned_dialogue = self.mapper.align_dialogue_segments(
            vad_segments=vad_segments,
            clip=clip,
            output_dir=work_dir,
        )

        if not aligned_dialogue:
            print("   No scripted dialogue aligned for this clip. Copying original clip.")
            import shutil

            shutil.copyfile(clip_path, output_path)
            return

        print("\n📌 Step 4: Seed-VC voice conversion per scripted speaker")
        source_dir = work_dir / "source_vocal_segments"
        tts_source_dir = work_dir / "source_tts_segments"
        converted_dir = work_dir / "seed_vc_segments"
        fitted_dir = work_dir / "converted_vocal_segments"
        source_dir.mkdir(parents=True, exist_ok=True)
        tts_source_dir.mkdir(parents=True, exist_ok=True)
        converted_dir.mkdir(parents=True, exist_ok=True)
        fitted_dir.mkdir(parents=True, exist_ok=True)

        converter = VoiceConverter(
            seed_vc_dir=self.seed_vc_dir,
            device=self.voice_device,
            diffusion_steps=self.seed_vc_diffusion_steps,
            length_adjust=self.seed_vc_length_adjust,
            inference_cfg_rate=self.seed_vc_inference_cfg_rate,
            python_executable=sys.executable,
            allow_fallback=not self.strict_voice_conversion,
        )

        converted_segments: list[tuple[float, float, Path]] = []
        conversion_manifest: list[dict[str, object]] = []

        for index, segment in enumerate(aligned_dialogue):
            char_id = str(segment["char_id"])
            reference_path = reference_paths.get(char_id)
            if not reference_path or not Path(reference_path).exists():
                message = f"No reference voice available for {char_id}"
                if self.strict_voice_conversion:
                    raise RuntimeError(message)
                print(f"   ⚠️  {message}; skipping line {index}")
                continue

            target_start = float(segment["start"])
            target_end = float(segment["end"])
            extract_start, extract_end = self._expanded_audio_window(
                start=target_start,
                end=target_end,
                total_duration=raw_duration,
                minimum_duration=self.conversion_min_segment_seconds,
                padding=self.conversion_segment_padding_seconds,
            )

            source_segment = source_dir / f"line_{index:03d}_{char_id}_source.wav"
            tts_source = tts_source_dir / f"line_{index:03d}_{char_id}_script.wav"
            converted_raw = converted_dir / f"line_{index:03d}_{char_id}_seedvc.wav"
            converted_fit = fitted_dir / f"line_{index:03d}_{char_id}_fit.wav"

            conversion_source = self.conversion_source
            if conversion_source == "native_vocals" and not has_native_audio:
                print("   🔇 Native vocals requested, but the video is silent; using script TTS source.")
                conversion_source = "script_tts"

            if conversion_source == "native_vocals":
                AudioMixer.extract_segment(
                    audio_path=separated.vocals_path,
                    start=extract_start,
                    end=extract_end,
                    output_path=source_segment,
                )
                if AudioMixer.is_effectively_silent(source_segment):
                    print("   ⚠️  Vocal stem slice is near-silent; using raw audio slice for this line.")
                    AudioMixer.extract_segment(
                        audio_path=raw_audio,
                        start=extract_start,
                        end=extract_end,
                        output_path=source_segment,
                    )
            else:
                voice_name = voice_assignments.get(char_id)
                if not voice_name:
                    message = f"No source TTS voice assignment available for {char_id}"
                    if self.strict_voice_conversion:
                        raise RuntimeError(message)
                    print(f"   ⚠️  {message}; skipping line {index}")
                    continue

                member = self.cast_by_id.get(char_id)
                style_prompt = None
                if member is not None:
                    style_prompt = (
                        f"{getattr(member, 'voice_brief', '').strip()} "
                        "Speak only the scripted line. Do not add narration, stage directions, or extra words."
                    ).strip()
                if self.ref_generator._is_valid_reference_audio(tts_source):
                    print(f"   ♻️  Reusing source TTS line: {tts_source.name}")
                else:
                    self.ref_generator.synthesize_dialogue_line(
                        text=str(segment.get("text") or ""),
                        voice_name=voice_name,
                        save_path=tts_source,
                        style_prompt=style_prompt,
                    )
                source_segment = tts_source

            converter.convert(
                source_path=source_segment,
                reference_path=reference_path,
                output_path=converted_raw,
            )

            target_duration = max(
                self.min_line_duration_seconds,
                target_end - target_start,
            )
            self._fit_dialogue_to_window(
                source_path=converted_raw,
                output_path=converted_fit,
                target_duration=target_duration,
            )

            converted_segments.append((target_start, target_end, converted_fit))
            member = self.cast_by_id.get(char_id)
            conversion_manifest.append(
                {
                    "index": index,
                    "shot_id": segment.get("shot_id"),
                    "char_id": char_id,
                    "character_name": member.name_en if member else char_id,
                    "text": segment.get("text"),
                    "target_start": target_start,
                    "target_end": target_end,
                    "extract_start": extract_start,
                    "extract_end": extract_end,
                    "conversion_source": conversion_source,
                    "source_path": str(source_segment),
                    "reference_path": str(reference_path),
                    "converted_path": str(converted_fit),
                }
            )

        if not converted_segments:
            message = "No dialogue segments converted."
            if self.strict_voice_conversion:
                raise RuntimeError(message)
            print(f"   {message} Copying original clip.")
            import shutil

            shutil.copyfile(clip_path, output_path)
            return

        manifest_path = work_dir / "seed_vc_conversion_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"segments": conversion_manifest}, f, indent=2, ensure_ascii=False)

        print("\n📌 Step 5: Assemble converted vocal timeline")
        converted_vocals = work_dir / "converted_vocals.wav"
        self.mixer.assemble_dialogue_track(
            base_audio_path=raw_audio,
            dialogue_segments=converted_segments,
            output_path=converted_vocals,
        )

        print("\n📌 Step 6: Mix converted vocals with separated background")
        mixed_audio = work_dir / "mixed_audio.wav"
        self.mixer.mix_audio(
            vocals_path=converted_vocals,
            background_path=separated.background_path,
            output_path=mixed_audio,
        )

        print("\n📌 Step 7: Mux audio to video")
        self.mixer.mux_audio_to_video(
            video_path=clip_path,
            audio_path=mixed_audio,
            output_path=output_path,
        )

        print(f"\n✅ Voice-converted clip saved: {output_path}")

    def _prepare_voice_assignments(self) -> dict[str, str]:
        """Load or create the fixed per-character Gemini voice assignment map."""
        manifest_path = self.output_root / "metadata" / "05_voice_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                assignments = data.get("voice_assignments") or {}
                if isinstance(assignments, dict) and assignments:
                    self.ref_generator.save_voice_manifest(assignments=assignments, reference_paths={})
                    return {str(k): str(v) for k, v in assignments.items()}
            except Exception:
                pass

        assignments = self.ref_generator.assign_voices(self.series_bible.cast)
        self.ref_generator.save_voice_manifest(assignments=assignments, reference_paths={})
        return assignments

    def _prepare_voice_references(self) -> tuple[dict[str, str], dict[str, Path]]:
        """Load or generate the target reference voice WAV for each character."""
        manifest_path = self.output_root / "metadata" / "05_voice_manifest.json"
        assignments: dict[str, str] = {}
        reference_paths: dict[str, Path] = {}

        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_assignments = data.get("voice_assignments") or {}
                raw_refs = data.get("reference_paths") or {}
                if isinstance(raw_assignments, dict):
                    assignments = {str(k): str(v) for k, v in raw_assignments.items()}
                if isinstance(raw_refs, dict):
                    reference_paths = {
                        str(k): Path(str(v)).expanduser()
                        for k, v in raw_refs.items()
                        if str(v).strip()
                    }
            except Exception:
                assignments = {}
                reference_paths = {}

        if assignments:
            self.ref_generator.voice_assignments = assignments

        required_ids = {member.id for member in self.series_bible.cast}
        usable_refs = {
            char_id: path
            for char_id, path in reference_paths.items()
            if char_id in required_ids and self.ref_generator._is_valid_reference_audio(path)
        }

        if required_ids.issubset(set(usable_refs)) and assignments:
            self.ref_generator.save_voice_manifest(
                assignments=assignments,
                reference_paths=usable_refs,
            )
            return assignments, usable_refs

        generated_refs = self.ref_generator.generate_all(
            self.series_bible.cast,
            reuse_existing=self.reuse_existing,
        )
        assignments = self.ref_generator.assign_voices(self.series_bible.cast)
        return assignments, generated_refs

    def _expanded_audio_window(
        self,
        start: float,
        end: float,
        total_duration: float,
        minimum_duration: float,
        padding: float,
    ) -> tuple[float, float]:
        """Expand a dialogue window enough for stable voice conversion."""
        start = max(0.0, start - max(0.0, padding))
        end = min(total_duration, end + max(0.0, padding))
        duration = end - start
        if duration >= minimum_duration:
            return start, end

        extra = max(0.0, minimum_duration - duration)
        start = max(0.0, start - extra / 2)
        end = min(total_duration, end + extra / 2)

        duration = end - start
        if duration < minimum_duration and start <= 0.0:
            end = min(total_duration, minimum_duration)
        elif duration < minimum_duration and end >= total_duration:
            start = max(0.0, total_duration - minimum_duration)

        return start, end

    @classmethod
    def _write_tts_srt(
        cls,
        segments: list[dict[str, object]],
        srt_path: Path,
        *,
        include_speaker: bool,
    ) -> int:
        blocks: list[str] = []
        for index, segment in enumerate(segments, start=1):
            text = cls._clean_srt_text(str(segment.get("text") or ""))
            if not text:
                continue
            if include_speaker:
                speaker = cls._clean_srt_text(
                    str(segment.get("character_name") or segment.get("char_id") or "")
                )
                if speaker:
                    text = f"{speaker}: {text}"
            blocks.append(
                "\n".join(
                    [
                        str(len(blocks) + 1),
                        (
                            f"{cls._srt_timestamp(float(segment['start']))} --> "
                            f"{cls._srt_timestamp(float(segment['end']))}"
                        ),
                        text,
                    ]
                )
            )

        srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
        return len(blocks)

    @staticmethod
    def _srt_timestamp(seconds: float) -> str:
        millis = int(round(max(0.0, seconds) * 1000))
        hours, millis = divmod(millis, 3_600_000)
        minutes, millis = divmod(millis, 60_000)
        secs, millis = divmod(millis, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _clean_srt_text(text: str) -> str:
        return " ".join(str(text).replace("\n", " ").split())

    def _fit_dialogue_to_window(
        self,
        source_path: Path,
        output_path: Path,
        target_duration: float,
    ) -> Path:
        """Time-fit a TTS line to its allocated dialogue window."""
        data, rate = sf.read(str(source_path))
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        current_duration = len(data) / float(rate)
        if current_duration <= 0 or target_duration <= 0:
            sf.write(str(output_path), data, rate)
            return output_path

        stretch_rate = current_duration / target_duration
        if self.allow_unbounded_tts_speedup and stretch_rate > 1.0:
            stretch_rate = max(stretch_rate, 1e-6)
        else:
            stretch_rate = min(
                max(stretch_rate, self.min_tts_stretch_ratio),
                self.max_tts_stretch_ratio,
            )
        tempo_path = output_path.with_suffix(".tempo.wav")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-filter:a",
            self._atempo_filter(stretch_rate),
            str(tempo_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"TTS time-fit failed: {result.stderr[:300]}")

        stretched, stretched_rate = sf.read(str(tempo_path))
        tempo_path.unlink(missing_ok=True)
        if stretched.ndim == 1:
            stretched = stretched.reshape(-1, 1)
        if stretched_rate != rate:
            stretched = self._resample_audio_array(
                stretched,
                orig_sr=stretched_rate,
                target_sr=rate,
            )

        target_samples = max(1, int(target_duration * rate))
        if len(stretched) > target_samples:
            excess_seconds = (len(stretched) - target_samples) / float(rate)
            if self.allow_unbounded_tts_speedup:
                if excess_seconds > self.max_tts_truncation_seconds:
                    print(
                        "   ⚙️  TTS line still exceeds the window after tempo fitting "
                        f"({excess_seconds:.2f}s): {source_path.name}; "
                        "precision-resampling the full line instead of trimming."
                    )
                stretched = self._resize_audio_to_samples(stretched, target_samples)
            else:
                message = (
                    f"TTS line remains {excess_seconds:.2f}s longer than the dialogue window "
                    f"after max allowed speed-up ({self.max_tts_stretch_ratio:.2f}x): {source_path.name}"
                )
                if self.strict_tts_fit and excess_seconds > self.max_tts_truncation_seconds:
                    raise RuntimeError(message)
                if excess_seconds > self.max_tts_truncation_seconds:
                    print(f"   ⚠️  {message}; trimming to preserve clip timing.")
                stretched = stretched[:target_samples]
        elif len(stretched) < target_samples:
            pad = np.zeros((target_samples - len(stretched), stretched.shape[1]), dtype=np.float32)
            stretched = np.concatenate([stretched, pad], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), stretched, rate)
        return output_path

    @staticmethod
    def _atempo_filter(rate: float) -> str:
        """Build an ffmpeg atempo chain, keeping each factor in the stable 0.5-2.0 range."""
        rate = max(rate, 1e-6)
        factors: list[float] = []
        while rate > 2.0:
            factors.append(2.0)
            rate /= 2.0
        while rate < 0.5:
            factors.append(0.5)
            rate /= 0.5
        factors.append(rate)
        return ",".join(f"atempo={factor:.6f}" for factor in factors)

    @staticmethod
    def _resize_audio_to_samples(data: np.ndarray, target_samples: int) -> np.ndarray:
        """Resample the whole line to an exact sample count instead of cutting off the tail."""
        target_samples = max(1, int(target_samples))
        if len(data) == target_samples:
            return data
        if len(data) <= 1:
            return np.resize(data, (target_samples, data.shape[1]))

        source_positions = np.arange(len(data), dtype=np.float64)
        target_positions = np.linspace(0, len(data) - 1, target_samples, dtype=np.float64)
        channels = [
            np.interp(target_positions, source_positions, data[:, channel])
            for channel in range(data.shape[1])
        ]
        return np.stack(channels, axis=1).astype(data.dtype, copy=False)

    def _resample_audio_array(
        self,
        audio_data: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Lightweight linear resample to avoid extra runtime deps."""
        if orig_sr == target_sr:
            return audio_data

        orig_len = len(audio_data)
        if orig_len == 0:
            return audio_data

        target_len = max(1, int(round(orig_len * float(target_sr) / float(orig_sr))))
        orig_positions = np.linspace(0.0, 1.0, num=orig_len, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_len, endpoint=False)

        channels = []
        for channel_idx in range(audio_data.shape[1]):
            channels.append(
                np.interp(target_positions, orig_positions, audio_data[:, channel_idx]).astype(np.float32)
            )
        return np.stack(channels, axis=1)

    def _manifest_path_for_clip(self, clip_id: str) -> Path:
        return self.voice_work_dir / clip_id / "voice_render_manifest.json"

    def _should_reuse_voiced_clip(self, clip_id: str, output_path: Path) -> bool:
        """Reuse only if the clip was generated by the same voice method."""
        if self.force or not (self.reuse_existing and output_path.exists()):
            return False

        source_clip_path = self.clips_dir / f"{clip_id}.mp4"
        if not source_clip_path.exists():
            return False

        manifest_path = self._manifest_path_for_clip(clip_id)
        if not manifest_path.exists():
            return False

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        return (
            data.get("voice_method") == self.voice_method
            and data.get("source_clip_path") == str(source_clip_path)
            and data.get("source_clip_mtime_ns") == source_clip_path.stat().st_mtime_ns
            and data.get("source_clip_size") == source_clip_path.stat().st_size
            and data.get("output_path") == str(output_path)
            and data.get("voice_signature") == self._voice_manifest_signature()
        )

    def _voice_manifest_signature(self) -> dict[str, object]:
        """Config fields that materially affect voice render output."""
        return {
            "voice_method": self.voice_method,
            "conversion_source": self.conversion_source,
            "demucs_model": self.demucs_model,
            "seed_vc_dir": self.seed_vc_dir,
            "diffusion_steps": self.seed_vc_diffusion_steps,
            "length_adjust": self.seed_vc_length_adjust,
            "inference_cfg_rate": self.seed_vc_inference_cfg_rate,
            "min_line_duration_seconds": self.min_line_duration_seconds,
            "max_tts_stretch_ratio": self.max_tts_stretch_ratio,
            "min_tts_stretch_ratio": self.min_tts_stretch_ratio,
            "allow_unbounded_tts_speedup": self.allow_unbounded_tts_speedup,
            "strict_tts_fit": self.strict_tts_fit,
            "max_tts_truncation_seconds": self.max_tts_truncation_seconds,
            "shorten_overlong_dialogue": self.shorten_overlong_dialogue,
            "tts_shortening_max_rounds": self.tts_shortening_max_rounds,
            "tts_estimated_words_per_second": self.tts_estimated_words_per_second,
            "tts_shortening_min_words": self.tts_shortening_min_words,
            "tts_shortening_buffer_seconds": self.tts_shortening_buffer_seconds,
            "semantic_shortening_enabled": self.semantic_shortening_enabled,
            "semantic_shortening_model": self.semantic_shortening_model,
            "voice_device": self.voice_device,
        }

    def _write_render_manifest(
        self,
        clip_id: str,
        clip_path: Path,
        output_path: Path,
    ) -> None:
        manifest_path = self._manifest_path_for_clip(clip_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        clip_stat = clip_path.stat()
        payload = {
            "voice_method": self.voice_method,
            "voice_signature": self._voice_manifest_signature(),
            "source_clip_path": str(clip_path),
            "source_clip_mtime_ns": clip_stat.st_mtime_ns,
            "source_clip_size": clip_stat.st_size,
            "output_path": str(output_path),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Standalone entry point for the voice pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice Consistency Pipeline: process rendered clips for consistent character voices."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to video_config.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Path to the video output root directory",
    )
    parser.add_argument(
        "--include-clips",
        type=str,
        default=None,
        help="Comma-separated clip ids to process, for example clip_002,clip_003.",
    )
    parser.add_argument(
        "--voice-method",
        type=str,
        default=None,
        help="Override voice.method, for example seed_vc_conversion.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override voice.device, for example cuda:1 or cpu.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Override Seed-VC diffusion steps for this run.",
    )
    parser.add_argument(
        "--voice-output-dir-name",
        type=str,
        default=None,
        help="Override voice.output_dir_name for this run.",
    )
    parser.add_argument(
        "--voice-work-dir-name",
        type=str,
        default=None,
        help="Override voice work directory name for this run.",
    )
    parser.add_argument(
        "--voice-workers",
        type=int,
        default=None,
        help="Override voice.voice_workers for this run.",
    )
    parser.add_argument(
        "--semantic-shortening",
        action="store_true",
        help="Use Gemini text rewriting before local fallback when TTS lines are too long.",
    )
    parser.add_argument(
        "--semantic-shortening-model",
        type=str,
        default=None,
        help="Gemini text model for semantic TTS shortening.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render voice output even if a matching manifest already exists.",
    )
    args = parser.parse_args()

    from video_generator.planner import load_series_bible

    output_root = Path(args.output_root).expanduser().resolve()
    series_bible = load_series_bible(output_root)
    include_clip_ids = None
    if args.include_clips:
        include_clip_ids = {
            clip_id.strip()
            for clip_id in args.include_clips.split(",")
            if clip_id.strip()
        }

    pipeline = VoiceConsistencyPipeline(
        series_bible=series_bible,
        output_root=output_root,
        config_path=args.config,
        include_clip_ids=include_clip_ids,
        force=args.force,
        voice_method=args.voice_method,
        voice_device=args.device,
        diffusion_steps=args.diffusion_steps,
        voice_output_dir_name=args.voice_output_dir_name,
        voice_work_dir_name=args.voice_work_dir_name,
        voice_workers=args.voice_workers,
        semantic_shortening=args.semantic_shortening,
        semantic_shortening_model=args.semantic_shortening_model,
    )
    pipeline.process_all_clips()


if __name__ == "__main__":
    main()
