"""Speaker diarization using speechbrain embeddings + spectral clustering.

No HuggingFace gated token required — uses the public ECAPA-TDNN model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio

from video_generator.voice.vad import VoiceSegment


@dataclass
class DiarizedSegment:
    start: float       # seconds
    end: float         # seconds
    speaker: str       # e.g. "speaker_0"

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerDiarizer:
    """Cluster speech segments by speaker using speechbrain embeddings."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._encoder = None

    def _load_encoder(self):
        if self._encoder is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier
        self._encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )
        print("✅ SpeakerDiarizer: ECAPA-TDNN encoder loaded")

    def _extract_embedding(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Extract a speaker embedding from a waveform chunk."""
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # speechbrain expects [batch, time]
        embedding = self._encoder.encode_batch(waveform.to(self.device))
        return embedding.squeeze().cpu().numpy()

    def diarize(
        self,
        vocals_path: str | Path,
        vad_segments: list[VoiceSegment],
        num_speakers: int | None = None,
        max_speakers: int = 8,
    ) -> list[DiarizedSegment]:
        """
        Assign speaker labels to each VAD segment.

        Args:
            vocals_path: Path to the isolated vocals audio.
            vad_segments: List of speech segments from VAD.
            num_speakers: If known, fix the number of speakers.
                          If None, auto-detect via eigen gap.
            max_speakers: Upper bound for auto-detection.

        Returns:
            List of DiarizedSegment with speaker labels.
        """
        if not vad_segments:
            return []

        self._load_encoder()

        vocals_path = Path(vocals_path)
        waveform, sample_rate = torchaudio.load(str(vocals_path))

        # Extract embeddings for each segment
        embeddings = []
        valid_segments = []
        for seg in vad_segments:
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            chunk = waveform[:, start_sample:end_sample]

            # Skip very short segments (< 0.3s)
            if chunk.shape[1] < int(0.3 * sample_rate):
                continue

            emb = self._extract_embedding(chunk, sample_rate)
            embeddings.append(emb)
            valid_segments.append(seg)

        if not embeddings:
            return []

        if len(embeddings) == 1:
            return [
                DiarizedSegment(
                    start=valid_segments[0].start,
                    end=valid_segments[0].end,
                    speaker="speaker_0",
                )
            ]

        embeddings_matrix = np.stack(embeddings)

        # Determine number of speakers
        n_clusters = num_speakers or self._estimate_num_speakers(
            embeddings_matrix, max_speakers
        )
        n_clusters = min(n_clusters, len(embeddings))

        # Cluster
        labels = self._spectral_cluster(embeddings_matrix, n_clusters)

        results = []
        for seg, label in zip(valid_segments, labels):
            results.append(
                DiarizedSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=f"speaker_{label}",
                )
            )

        # Print summary
        speaker_counts = {}
        for r in results:
            speaker_counts[r.speaker] = speaker_counts.get(r.speaker, 0) + 1
        print(f"🗣️  Diarization: {len(results)} segments, {len(speaker_counts)} speakers")
        for spk, count in sorted(speaker_counts.items()):
            total_dur = sum(r.duration for r in results if r.speaker == spk)
            print(f"   {spk}: {count} segments, {total_dur:.1f}s total")

        return results

    def _estimate_num_speakers(
        self, embeddings: np.ndarray, max_speakers: int
    ) -> int:
        """Estimate number of speakers via eigen gap analysis."""
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)
        # Ensure non-negative for spectral analysis
        sim_matrix = np.maximum(sim_matrix, 0)

        # Compute normalized Laplacian eigenvalues
        degree = np.sum(sim_matrix, axis=1)
        degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        laplacian = np.eye(len(sim_matrix)) - D_inv_sqrt @ sim_matrix @ D_inv_sqrt

        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(laplacian)))

        # Look for largest gap in first max_speakers eigenvalues
        max_k = min(max_speakers, len(eigenvalues) - 1)
        if max_k < 2:
            return 1

        gaps = np.diff(eigenvalues[1 : max_k + 1])
        if len(gaps) == 0:
            return 1

        n_speakers = int(np.argmax(gaps) + 2)
        return max(1, min(n_speakers, max_speakers))

    def _spectral_cluster(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Spectral clustering on cosine similarity matrix."""
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics.pairwise import cosine_similarity

        if n_clusters <= 1:
            return np.zeros(len(embeddings), dtype=int)

        sim_matrix = cosine_similarity(embeddings)
        sim_matrix = np.maximum(sim_matrix, 0)

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
        )
        labels = clustering.fit_predict(sim_matrix)
        return labels
