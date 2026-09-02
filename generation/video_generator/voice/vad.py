"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass
class VoiceSegment:
    start: float    # seconds
    end: float      # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


class VoiceActivityDetector:
    """Detect speech segments in audio using Silero VAD."""

    def __init__(
        self,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        threshold: float = 0.5,
        speech_pad_ms: int = 50,
    ) -> None:
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.threshold = threshold
        self.speech_pad_ms = speech_pad_ms

        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.get_speech_timestamps = self.utils[0]

    def detect(self, audio_path: str | Path) -> list[VoiceSegment]:
        """
        Detect all speech segments in an audio file.

        Args:
            audio_path: Path to audio file (WAV recommended).

        Returns:
            List of VoiceSegment with start/end times in seconds.
        """
        audio_path = Path(audio_path)
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Silero VAD expects 16kHz mono
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        speech_timestamps = self.get_speech_timestamps(
            waveform.squeeze(),
            self.model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            threshold=self.threshold,
            speech_pad_ms=self.speech_pad_ms,
        )

        segments = []
        for ts in speech_timestamps:
            start = ts["start"] / sample_rate
            end = ts["end"] / sample_rate
            segments.append(VoiceSegment(start=start, end=end))

        # Merge segments that are very close together (< 300ms gap)
        merged = self._merge_close_segments(segments, max_gap=0.3)

        print(f"🎤 VAD: found {len(merged)} speech segments in {audio_path.name}")
        for i, seg in enumerate(merged):
            print(f"   [{i}] {seg.start:.2f}s - {seg.end:.2f}s ({seg.duration:.2f}s)")

        return merged

    def _merge_close_segments(
        self, segments: list[VoiceSegment], max_gap: float
    ) -> list[VoiceSegment]:
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.start - prev.end <= max_gap:
                merged[-1] = VoiceSegment(start=prev.start, end=seg.end)
            else:
                merged.append(seg)
        return merged
