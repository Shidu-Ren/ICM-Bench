"""Vocal/background audio separation using Demucs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SeparationResult:
    vocals_path: Path
    background_path: Path


class AudioSeparator:
    """Separate vocals from background audio using Meta's Demucs."""

    def __init__(
        self,
        model: str = "htdemucs_ft",
        device: str = "cuda:0",
        output_dir: str | Path = "/tmp/demucs_output",
        python_executable: str | Path | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.python_executable = str(python_executable or sys.executable)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fallback_no_separation(self, audio_path: Path) -> SeparationResult:
        """Fallback when Demucs is unavailable: keep original audio as vocals and use silence as background."""
        stem_name = audio_path.stem
        fallback_dir = self.output_dir / "_fallback_no_separation" / stem_name
        fallback_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = fallback_dir / "vocals.wav"
        background_path = fallback_dir / "no_vocals.wav"
        shutil.copyfile(audio_path, vocals_path)
        duration = self._get_audio_duration(audio_path)

        silence_command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            f"{duration:.3f}",
            "-c:a",
            "pcm_s16le",
            str(background_path),
        ]
        result = subprocess.run(silence_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Fallback silent-background generation failed: {result.stderr[:300]}"
            )

        print("⚠️  Demucs unavailable or incompatible; using fallback no-separation mode.")
        print(f"   Vocals fallback: {vocals_path}")
        print(f"   Background fallback: {background_path}")
        return SeparationResult(vocals_path=vocals_path, background_path=background_path)

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Return audio duration in seconds using ffprobe."""
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to probe audio duration for fallback separation: {result.stderr[:300]}"
            )
        try:
            return float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected ffprobe duration output for {audio_path}: {result.stdout[:100]}"
            ) from exc

    def separate(self, audio_path: str | Path) -> SeparationResult:
        """
        Separate an audio file into vocals and background (accompaniment).

        Args:
            audio_path: Path to the input audio file.

        Returns:
            SeparationResult with paths to the vocals and background stems.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        stem_name = audio_path.stem

        command = [
            self.python_executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", self.model,
            "--device", self.device,
            "-o", str(self.output_dir),
            str(audio_path),
        ]

        print(f"🎵 Demucs: separating vocals from background...")
        print(f"   Model: {self.model}, Device: {self.device}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"⚠️  Demucs stderr: {result.stderr[:500]}")
            return self._fallback_no_separation(audio_path)

        # Demucs outputs to: output_dir / model_name / stem_name / {vocals,no_vocals}.wav
        model_output_dir = self.output_dir / self.model / stem_name
        vocals_path = model_output_dir / "vocals.wav"
        background_path = model_output_dir / "no_vocals.wav"

        if not vocals_path.exists():
            print(f"⚠️  Demucs did not produce vocals file at: {vocals_path}")
            return self._fallback_no_separation(audio_path)
        if not background_path.exists():
            print(f"⚠️  Demucs did not produce background file at: {background_path}")
            return self._fallback_no_separation(audio_path)

        print(f"✅ Separation complete:")
        print(f"   Vocals: {vocals_path}")
        print(f"   Background: {background_path}")
        return SeparationResult(vocals_path=vocals_path, background_path=background_path)
