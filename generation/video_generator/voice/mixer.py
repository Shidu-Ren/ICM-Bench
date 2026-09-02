"""Audio mixing, loudness normalization, and video muxing utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioMixer:
    """Mix converted vocals with original background audio and mux to video."""

    def __init__(
        self,
        vocals_loudness_lufs: float = -16.0,
        background_loudness_lufs: float = -26.0,
    ) -> None:
        self.vocals_lufs = vocals_loudness_lufs
        self.background_lufs = background_loudness_lufs

    @staticmethod
    def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
        """Extract audio track from a video file."""
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {result.stderr[:300]}")

        print(f"🔊 Audio extracted: {output_path}")
        return output_path

    @staticmethod
    def has_audio_track(video_path: str | Path) -> bool:
        """Check if a video file contains an audio track."""
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        return bool(result.stdout.strip())

    @staticmethod
    def video_duration_seconds(video_path: str | Path) -> float:
        """Return video duration in seconds."""
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Video duration probe failed: {result.stderr[:300]}")
        try:
            return float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected video duration output for {video_path}: {result.stdout[:100]}"
            ) from exc

    @staticmethod
    def create_silent_audio(
        duration_seconds: float,
        output_path: str | Path,
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> Path:
        """Create a silent PCM WAV bed for videos rendered without audio."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = max(1, int(round(duration_seconds * sample_rate)))
        data = np.zeros((frames, channels), dtype=np.float32)
        sf.write(str(output_path), data, sample_rate)
        return output_path

    @staticmethod
    def extract_segment(
        audio_path: str | Path,
        start: float,
        end: float,
        output_path: str | Path,
    ) -> Path:
        """Extract a time segment from an audio file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = end - start

        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-acodec", "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)
        return output_path

    @staticmethod
    def audio_duration_seconds(audio_path: str | Path) -> float:
        """Return audio duration in seconds."""
        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)

    @staticmethod
    def is_effectively_silent(audio_path: str | Path, threshold: float = 1e-4) -> bool:
        """Heuristic silence check for a saved audio file."""
        data, _ = sf.read(str(audio_path))
        if data.size == 0:
            return True
        if data.ndim > 1:
            data = data.mean(axis=1)
        peak = float(np.max(np.abs(data)))
        return peak < threshold

    def normalize_loudness(
        self, audio_path: str | Path, target_lufs: float, output_path: str | Path
    ) -> Path:
        """Normalize audio loudness to target LUFS using pyloudnorm."""
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        try:
            import pyloudnorm as pyln
        except ImportError:
            print("⚠️  pyloudnorm not installed, skipping loudness normalization")
            import shutil
            shutil.copyfile(audio_path, output_path)
            return output_path

        data, rate = sf.read(str(audio_path))
        meter = pyln.Meter(rate)

        # Handle mono/stereo
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        current_lufs = meter.integrated_loudness(data)

        if np.isinf(current_lufs) or np.isnan(current_lufs):
            # Silent audio – skip normalization
            sf.write(str(output_path), data, rate)
            return output_path

        normalized = pyln.normalize.loudness(data, current_lufs, target_lufs)
        normalized = np.clip(normalized, -0.999, 0.999)
        sf.write(str(output_path), normalized, rate)
        return output_path

    def duck_audio(
        self,
        audio_path: str | Path,
        segments: list[tuple[float, float]],
        output_path: str | Path,
        reduction_db: float = 18.0,
        fade_seconds: float = 0.08,
    ) -> Path:
        """Lower the original bed during dialogue windows while keeping ambience."""
        data, rate = sf.read(str(audio_path))
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        gain_floor = 10.0 ** (-abs(reduction_db) / 20.0)
        envelope = np.ones(len(data), dtype=np.float32)
        fade_samples = max(1, int(fade_seconds * rate))

        for start, end in segments:
            start_idx = max(0, int(start * rate))
            end_idx = min(len(data), int(end * rate))
            if end_idx <= start_idx:
                continue

            envelope[start_idx:end_idx] = np.minimum(
                envelope[start_idx:end_idx],
                gain_floor,
            )

            fade_in_start = max(0, start_idx - fade_samples)
            if start_idx > fade_in_start:
                fade_curve = np.linspace(1.0, gain_floor, start_idx - fade_in_start, endpoint=False)
                envelope[fade_in_start:start_idx] = np.minimum(
                    envelope[fade_in_start:start_idx],
                    fade_curve.astype(np.float32),
                )

            fade_out_end = min(len(data), end_idx + fade_samples)
            if fade_out_end > end_idx:
                fade_curve = np.linspace(gain_floor, 1.0, fade_out_end - end_idx, endpoint=False)
                envelope[end_idx:fade_out_end] = np.minimum(
                    envelope[end_idx:fade_out_end],
                    fade_curve.astype(np.float32),
                )

        ducked = data * envelope[:, None]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), ducked, rate)
        return output_path

    def assemble_dialogue_track(
        self,
        base_audio_path: str | Path,
        dialogue_segments: list[tuple[float, float, Path]],
        output_path: str | Path,
    ) -> Path:
        """Place synthesized dialogue clips on a silent timeline matching the source clip."""
        base_info = sf.info(str(base_audio_path))
        sample_rate = base_info.samplerate
        channels = max(1, base_info.channels)
        total_frames = base_info.frames
        track = np.zeros((total_frames, channels), dtype=np.float32)

        for start, end, segment_path in dialogue_segments:
            seg_data, seg_rate = sf.read(str(segment_path))
            if seg_data.ndim == 1:
                seg_data = seg_data.reshape(-1, 1)

            if seg_rate != sample_rate:
                seg_data = self._resample_audio_array(
                    seg_data,
                    orig_sr=seg_rate,
                    target_sr=sample_rate,
                )

            if seg_data.shape[1] != channels:
                if seg_data.shape[1] == 1 and channels > 1:
                    seg_data = np.tile(seg_data, (1, channels))
                else:
                    seg_data = seg_data[:, :channels]

            start_idx = max(0, int(start * sample_rate))
            end_idx = min(total_frames, int(end * sample_rate))
            if end_idx <= start_idx:
                continue

            target_len = end_idx - start_idx
            if len(seg_data) > target_len:
                seg_data = seg_data[:target_len]
            elif len(seg_data) < target_len:
                pad = np.zeros((target_len - len(seg_data), seg_data.shape[1]), dtype=np.float32)
                seg_data = np.concatenate([seg_data, pad], axis=0)

            track[start_idx:start_idx + len(seg_data)] += seg_data

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), track, sample_rate)
        return output_path

    @staticmethod
    def _resample_audio_array(
        audio_data: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Lightweight linear resample that avoids optional heavy deps."""
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

    def assemble_converted_vocals(
        self,
        original_vocals_path: str | Path,
        converted_segments: list[tuple[float, float, Path]],
        output_path: str | Path,
    ) -> Path:
        """
        Replace segments in the original vocals with converted versions.

        Args:
            original_vocals_path: Path to the original separated vocals.
            converted_segments: List of (start, end, converted_path) tuples.
            output_path: Where to save the assembled result.

        Returns:
            Path to the assembled vocals file.
        """
        original_data, rate = sf.read(str(original_vocals_path))
        if original_data.ndim == 1:
            original_data = original_data.reshape(-1, 1)

        result = original_data.copy()

        for start, end, conv_path in converted_segments:
            conv_data, conv_rate = sf.read(str(conv_path))
            if conv_data.ndim == 1:
                conv_data = conv_data.reshape(-1, 1)

            # Resample if needed
            if conv_rate != rate:
                import librosa
                conv_mono = conv_data.mean(axis=1) if conv_data.ndim > 1 else conv_data
                conv_mono = librosa.resample(conv_mono, orig_sr=conv_rate, target_sr=rate)
                conv_data = conv_mono.reshape(-1, 1)
                if result.shape[1] > 1:
                    conv_data = np.tile(conv_data, (1, result.shape[1]))

            start_sample = int(start * rate)
            end_sample = int(end * rate)
            target_len = end_sample - start_sample

            # Trim or pad to match target length exactly
            if len(conv_data) > target_len:
                conv_data = conv_data[:target_len]
            elif len(conv_data) < target_len:
                pad = np.zeros((target_len - len(conv_data), conv_data.shape[1]))
                conv_data = np.concatenate([conv_data, pad])

            # Ensure channel count matches
            if conv_data.shape[1] != result.shape[1]:
                if conv_data.shape[1] == 1 and result.shape[1] > 1:
                    conv_data = np.tile(conv_data, (1, result.shape[1]))
                else:
                    conv_data = conv_data[:, :result.shape[1]]

            result[start_sample:start_sample + len(conv_data)] = conv_data

        output_path = Path(output_path)
        sf.write(str(output_path), result, rate)
        print(f"✅ Assembled converted vocals: {output_path}")
        return output_path

    def mix_audio(
        self,
        vocals_path: str | Path,
        background_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Mix normalized vocals and background into a single audio file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize both tracks
        norm_vocals = output_path.parent / f"_norm_vocals_{output_path.stem}.wav"
        norm_bg = output_path.parent / f"_norm_bg_{output_path.stem}.wav"

        self.normalize_loudness(vocals_path, self.vocals_lufs, norm_vocals)
        self.normalize_loudness(background_path, self.background_lufs, norm_bg)

        # Mix using ffmpeg amix filter
        command = [
            "ffmpeg", "-y",
            "-i", str(norm_vocals),
            "-i", str(norm_bg),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=2,"
            "alimiter=limit=0.95[out]",
            "-map", "[out]",
            "-acodec", "pcm_s16le",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Clean up temp files
        norm_vocals.unlink(missing_ok=True)
        norm_bg.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"Audio mixing failed: {result.stderr[:300]}")

        print(f"✅ Mixed audio: {output_path}")
        return output_path

    def mux_audio_to_video(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Replace the audio track in a video with new audio."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Video muxing failed: {result.stderr[:300]}")

        print(f"✅ Video with new audio: {output_path}")
        return output_path
