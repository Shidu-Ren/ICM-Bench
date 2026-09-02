"""Generate reference voice samples for each character using Gemini TTS.

Each character gets a fixed voice_name (e.g., Puck, Kore) and a ~30s
reference audio file that Seed-VC will use as the target voice for
conversion, ensuring consistent voice identity across all clips.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from video_generator.api_usage import ApiUsageLogger


# Preset voice pools for automatic assignment
MALE_VOICES = ["Puck", "Charon", "Fenrir", "Orus", "Leda"]
FEMALE_VOICES = ["Kore", "Aoede", "Zephyr", "Autonoe", "Callirrhoe"]
SUPPORTED_PREBUILT_VOICES = {
    "achernar",
    "achird",
    "algenib",
    "algieba",
    "alnilam",
    "aoede",
    "autonoe",
    "callirrhoe",
    "charon",
    "despina",
    "enceladus",
    "erinome",
    "fenrir",
    "gacrux",
    "iapetus",
    "kore",
    "laomedeia",
    "leda",
    "orus",
    "puck",
    "pulcherrima",
    "rasalgethi",
    "sadachbia",
    "sadaltager",
    "schedar",
    "sulafat",
    "umbriel",
    "vindemiatrix",
    "zephyr",
    "zubenelgenubi",
}

# Reference text template — each character says a self-intro to build
# a ~30s sample with varied intonation
REFERENCE_SCRIPT_TEMPLATE = """
Hello, my name is {name}. I'm {age} years old.
Today is a wonderful day. I feel really happy to be here with everyone.
Sometimes I get excited, and other times I feel calm and reflective.
Let me tell you a bit about myself. I enjoy spending time with my friends and family.
Life has its ups and downs, but I always try to stay positive.
When I'm surprised, I can't help but gasp and laugh.
When I'm touched, my voice gets a little softer and warmer.
Thank you for listening. This is how I sound when I speak naturally.
"""


class ReferenceVoiceGenerator:
    """Generate and manage per-character reference voice files."""

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path,
        tts_model: str = "gemini-3.1-flash-tts-preview",
        voice_assignments: dict[str, str] | None = None,
        male_voices: list[str] | None = None,
        female_voices: list[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self._client_local = threading.local()
        self._client_local.client = self.client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts_model = tts_model
        self.voice_assignments = dict(voice_assignments or {})
        self.male_voices = list(male_voices or MALE_VOICES)
        self.female_voices = list(female_voices or FEMALE_VOICES)
        self.api_usage_logger = ApiUsageLogger(self.output_dir.parent.parent / "metadata")

    def _thread_client(self):
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = genai.Client(api_key=self.api_key)
            self._client_local.client = client
        return client

    def assign_voices(self, cast: list) -> dict[str, str]:
        """
        Assign a fixed voice_name to each cast member.

        Uses explicit voice_assignments from config first, then assigns
        remaining voices based on gender.

        Returns:
            dict mapping character_id to voice_name.
        """
        assignments = {}
        for char_id, voice_name in self.voice_assignments.items():
            if self._is_supported_voice(voice_name):
                assignments[char_id] = voice_name
            else:
                print(f"⚠️  Unsupported Gemini voice '{voice_name}' for {char_id}; reassigning.")
        used_voices = set(assignments.values())

        male_pool = [v for v in self.male_voices if v not in used_voices]
        female_pool = [v for v in self.female_voices if v not in used_voices]

        for member in cast:
            if member.id in assignments:
                continue

            gender = str(member.gender).lower()
            if gender == "female" and female_pool:
                voice = female_pool.pop(0)
            elif male_pool:
                voice = male_pool.pop(0)
            elif female_pool:
                voice = female_pool.pop(0)
            else:
                # Cycle through all voices
                all_voices = self.male_voices + self.female_voices
                idx = len(assignments) % len(all_voices)
                voice = all_voices[idx]

            assignments[member.id] = voice

        return assignments

    def _is_supported_voice(self, voice_name: str) -> bool:
        return str(voice_name).strip().lower() in SUPPORTED_PREBUILT_VOICES

    def save_voice_manifest(
        self,
        assignments: dict[str, str],
        reference_paths: dict[str, Path] | None = None,
    ) -> Path:
        """Persist the fixed voice assignment manifest for downstream pipelines."""
        manifest_path = self.output_dir.parent.parent / "metadata" / "05_voice_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    manifest.update(existing)
            except Exception:
                manifest = {}

        existing_refs = manifest.get("reference_paths") or {}
        if not isinstance(existing_refs, dict):
            existing_refs = {}
        merged_refs = {
            str(char_id): str(path)
            for char_id, path in existing_refs.items()
            if str(char_id).strip() and str(path).strip()
        }
        merged_refs.update(
            {
                char_id: str(path)
                for char_id, path in (reference_paths or {}).items()
            }
        )

        manifest["tts_model"] = self.tts_model
        manifest["voice_assignments"] = assignments
        manifest["reference_paths"] = merged_refs
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Voice manifest saved: {manifest_path}")
        return manifest_path

    def _generate_speech(
        self,
        text: str,
        voice_name: str,
        style_prompt: str | None = None,
        save_path: Path | None = None,
    ) -> bytes:
        """Generate speech audio using Gemini TTS."""
        response = None
        audio_part = None
        prompt_variants = self._build_tts_prompt_variants(text=text, style_prompt=style_prompt)
        for attempt in range(1, 9):
            full_prompt = prompt_variants[min(attempt - 1, len(prompt_variants) - 1)]
            try:
                response = self._thread_client().models.generate_content(
                    model=self.tts_model,
                    contents=full_prompt,
                    config={
                        "response_modalities": ["AUDIO"],
                        "speech_config": types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name,
                                )
                            )
                        ),
                    },
                )
                audio_part = self._extract_audio_part(response)
                self.api_usage_logger.record_response(
                    response=response,
                    operation="gemini_tts",
                    model=self.tts_model,
                    prompt=full_prompt,
                    attempt=attempt,
                    extra={
                        "voice_name": voice_name,
                        "save_path": str(save_path) if save_path is not None else None,
                    },
                )
                break
            except Exception as exc:
                self.api_usage_logger.record_failure(
                    operation="gemini_tts",
                    model=self.tts_model,
                    prompt=full_prompt,
                    attempt=attempt,
                    error=exc,
                    extra={
                        "voice_name": voice_name,
                        "save_path": str(save_path) if save_path is not None else None,
                    },
                )
                if attempt >= 8:
                    raise
                wait_seconds = min(2 * attempt, 8)
                print(
                    f"   ⚠️  Gemini TTS request failed on attempt {attempt}/8: {exc}. "
                    f"Retrying in {wait_seconds}s."
                )
                time.sleep(wait_seconds)

        if response is None or audio_part is None:
            raise RuntimeError("Gemini TTS returned no response.")

        # Extract audio data
        audio_data = audio_part.inline_data.data
        mime_type = getattr(audio_part.inline_data, "mime_type", None)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_reference_audio(
                audio_data=audio_data,
                mime_type=mime_type,
                save_path=save_path,
            )
            print(f"   💾 Saved: {save_path}")

        return audio_data

    @staticmethod
    def _extract_audio_part(response):
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None and getattr(inline_data, "data", None):
                    return part
        raise RuntimeError("Gemini TTS returned no audio candidate.")

    def _build_tts_prompt_variants(
        self,
        *,
        text: str,
        style_prompt: str | None,
    ) -> list[str]:
        transcript = text.strip()
        variants: list[str] = []
        if style_prompt:
            clean_style = " ".join(style_prompt.strip().split())
            variants.append(
                "Read this transcript aloud as natural dialogue. Produce audio only. "
                f"Voice direction: {clean_style}\nTranscript: {transcript}"
            )
            variants.append(f"[{clean_style}] {transcript}")
        variants.append(
            "Read aloud exactly this transcript. Produce audio only, with no extra words.\n"
            f"{transcript}"
        )
        variants.append(transcript)

        unique: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            if variant and variant not in seen:
                unique.append(variant)
                seen.add(variant)
        return unique or [transcript]

    def synthesize_dialogue_line(
        self,
        text: str,
        voice_name: str,
        save_path: Path,
        style_prompt: str | None = None,
    ) -> Path:
        """Generate a single dialogue line as a standard PCM WAV."""
        self._generate_speech(
            text=text,
            voice_name=voice_name,
            style_prompt=style_prompt,
            save_path=save_path,
        )
        return save_path

    def _save_reference_audio(
        self,
        audio_data: bytes,
        mime_type: Optional[str],
        save_path: Path,
    ) -> None:
        """Persist Gemini TTS audio as a standard PCM WAV."""
        if mime_type and mime_type.lower().startswith("audio/l16"):
            sample_rate = self._extract_sample_rate(mime_type) or 24000
            channels = self._extract_channel_count(mime_type) or 1
            with wave.open(str(save_path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            return

        temp_path = save_path.with_suffix(save_path.suffix + ".raw")
        with open(temp_path, "wb") as f:
            f.write(audio_data)
        self._normalize_reference_audio(temp_path, save_path)

    def _extract_sample_rate(self, mime_type: str) -> Optional[int]:
        """Parse sample rate from Gemini's inline audio mime type."""
        for part in mime_type.split(";"):
            part = part.strip().lower()
            if part.startswith("rate="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _extract_channel_count(self, mime_type: str) -> Optional[int]:
        """Parse channel count from Gemini's inline audio mime type."""
        for part in mime_type.split(";"):
            part = part.strip().lower()
            if part.startswith("channels="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _normalize_reference_audio(self, source_path: Path, output_path: Path) -> None:
        """Convert Gemini TTS bytes into a standard PCM WAV for downstream VC tools."""
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        source_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to normalize Gemini TTS reference audio: {result.stderr[:300]}"
            )

    def _is_valid_reference_audio(self, path: Path) -> bool:
        """Return True if a reference voice file is a readable non-empty WAV."""
        if not path.exists() or path.stat().st_size <= 44:
            return False

        try:
            with wave.open(str(path), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            return (
                frame_count > 0
                and sample_rate > 0
                and channels in {1, 2}
                and sample_width in {1, 2, 3, 4}
            )
        except wave.Error:
            return False

    def generate_all(
        self,
        cast: list,
        reuse_existing: bool = True,
    ) -> dict[str, Path]:
        """
        Generate reference voice files for all cast members.

        Args:
            cast: List of CastMember objects from SeriesBible.
            reuse_existing: If True, skip generation for existing files.

        Returns:
            dict mapping character_id to reference voice file path.
        """
        print("\n" + "=" * 60)
        print("🎙️  Generating character reference voices")
        print("=" * 60)

        assignments = self.assign_voices(cast)
        self.voice_assignments = assignments
        reference_paths: dict[str, Path] = {}

        for member in cast:
            voice_name = assignments.get(member.id, "Puck")
            ref_path = self.output_dir / f"{member.id}_reference.wav"

            if ref_path.exists() and reuse_existing:
                if self._is_valid_reference_audio(ref_path):
                    print(f"♻️  Reusing reference voice: {member.name_en} ({voice_name})")
                    reference_paths[member.id] = ref_path
                    continue

                print(f"♻️  Existing reference invalid, regenerating: {member.name_en} ({voice_name})")
                ref_path.unlink(missing_ok=True)

            print(f"\n🎙️  {member.name_en} → voice: {voice_name}")

            script = REFERENCE_SCRIPT_TEMPLATE.format(
                name=member.name_en,
                age=member.age,
            )
            style = member.voice_brief if hasattr(member, "voice_brief") else None

            try:
                self._generate_speech(
                    text=script,
                    voice_name=voice_name,
                    style_prompt=style,
                    save_path=ref_path,
                )
                reference_paths[member.id] = ref_path
            except Exception as exc:
                print(f"   ⚠️  TTS failed for {member.name_en}: {exc}")
                continue

        self.save_voice_manifest(assignments=assignments, reference_paths=reference_paths)
        self.api_usage_logger.write_summary()

        return reference_paths
