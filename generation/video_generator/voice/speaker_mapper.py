"""Automatic mapping from diarized speaker labels to character IDs.

Uses SeriesBible shot metadata (dialogue_lines, visible_characters, timing)
and optionally Gemini STT for transcript matching — fully automated, no
manual annotation required.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_generator.voice.diarizer import DiarizedSegment
from video_generator.voice.vad import VoiceSegment


@dataclass
class ShotTimeRange:
    shot_id: str
    clip_offset: float     # start time within the clip (seconds)
    duration: float        # shot duration in seconds
    visible_characters: list[str]
    dialogue_speakers: list[str]      # character IDs extracted from dialogue_lines
    dialogue_line_characters: list[str]
    dialogue_texts: list[str]         # raw dialogue text lines

    @property
    def start(self) -> float:
        return self.clip_offset

    @property
    def end(self) -> float:
        return self.clip_offset + self.duration


class SpeakerCharacterMapper:
    """Map diarized speaker labels to character IDs using metadata."""

    def __init__(self, series_bible, gemini_client=None, text_model: str = None):
        self.cast_by_id = {m.id: m for m in series_bible.cast}
        self.cast_by_name = {}
        for m in series_bible.cast:
            self.cast_by_name[m.name_en.lower()] = m.id
            if m.name_cn:
                self.cast_by_name[m.name_cn.lower()] = m.id
            # First name only
            first_name = m.name_en.split()[0].lower()
            if first_name not in self.cast_by_name:
                self.cast_by_name[first_name] = m.id

        self.gemini_client = gemini_client
        self.text_model = text_model

    def _build_shot_timeline(self, clip) -> list[ShotTimeRange]:
        """Build a timeline of shots with time offsets within the clip."""
        timeline = []
        offset = 0.0
        for shot in clip.shots:
            duration = float(shot.duration_seconds or 6)
            speakers = []
            line_characters = []
            texts = []
            for line in (shot.dialogue_lines or []):
                for speaker_name, text in self._parse_dialogue_turns(line):
                    char_id = self.cast_by_name.get(speaker_name.lower())
                    if not char_id:
                        char_id = self._extra_speaker_id(speaker_name)
                    line_characters.append(char_id)
                    if char_id not in speakers:
                        speakers.append(char_id)
                    texts.append(f"{speaker_name}: {text}")

            timeline.append(ShotTimeRange(
                shot_id=shot.id,
                clip_offset=offset,
                duration=duration,
                visible_characters=list(shot.visible_characters),
                dialogue_speakers=speakers,
                dialogue_line_characters=line_characters,
                dialogue_texts=texts,
            ))
            offset += duration
        return timeline

    def _parse_dialogue_turns(self, line: str) -> list[tuple[str, str]]:
        """Split one metadata line into one or more speaker turns."""
        raw = line.strip()
        if not raw:
            return []

        names = sorted(self.cast_by_name.keys(), key=len, reverse=True)
        if not names:
            return []

        pattern = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(name) for name in names) + r")\s*:\s*",
            flags=re.IGNORECASE,
        )
        matches = list(pattern.finditer(raw))
        if not matches:
            return []

        turns: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            text = raw[start:end].strip()
            text = re.sub(r"^[\s.?!,;:]+", "", text).strip()
            if text:
                turns.append((match.group(1).strip(), text))
        return turns

    def _extra_speaker_id(self, speaker_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", speaker_name.lower()).strip("_")
        return f"extra_{slug or 'speaker'}"

    def align_dialogue_segments(
        self,
        vad_segments: list[VoiceSegment],
        clip,
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """
        Align known dialogue lines to speech regions using shot timing.

        This is the preferred path for scripted video because the metadata
        already specifies who says each line. We only use VAD to recover the
        timing of speech inside each shot.

        Returns:
            A list of dicts with start/end/char_id/text metadata.
        """
        timeline = self._build_shot_timeline(clip)
        aligned_segments: list[dict[str, Any]] = []

        for shot_range in timeline:
            if not shot_range.dialogue_line_characters:
                continue

            speech_windows = self._speech_windows_for_shot(vad_segments, shot_range)
            if not speech_windows:
                speech_windows = self._approximate_dialogue_windows(shot_range)
            if not speech_windows:
                continue

            line_segments = self._allocate_dialogue_lines(
                shot_range=shot_range,
                speech_windows=speech_windows,
            )
            aligned_segments.extend(line_segments)

        if output_dir:
            manifest_path = output_dir / "dialogue_alignment.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "aligned_segments": aligned_segments,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if aligned_segments:
            print("🧭 Script-guided dialogue alignment:")
            for item in aligned_segments:
                member = self.cast_by_id.get(item["char_id"])
                name = member.name_en if member else item["char_id"]
                print(
                    f"   {item['shot_id']} [{item['start']:.2f}s - {item['end']:.2f}s] "
                    f"{name}: {item['text']}"
                )

        return aligned_segments

    def _speech_windows_for_shot(
        self,
        vad_segments: list[VoiceSegment],
        shot_range: ShotTimeRange,
    ) -> list[tuple[float, float]]:
        """Clip VAD speech regions to a shot's time range."""
        windows: list[tuple[float, float]] = []
        min_window = max(0.15, 0.04 * shot_range.duration)
        for seg in vad_segments:
            start = max(seg.start, shot_range.start)
            end = min(seg.end, shot_range.end)
            if end - start >= min_window:
                windows.append((start, end))

        windows.sort(key=lambda item: item[0])
        return windows

    def _approximate_dialogue_windows(
        self,
        shot_range: ShotTimeRange,
    ) -> list[tuple[float, float]]:
        """Fallback when VAD misses scripted speech: reserve the center of the shot for dialogue."""
        if not shot_range.dialogue_line_characters:
            return []

        duration = shot_range.duration
        if duration <= 0:
            return []

        lead_in = min(0.35, max(0.12 * duration, 0.18))
        lead_out = min(0.30, max(0.10 * duration, 0.15))
        start = shot_range.start + lead_in
        end = shot_range.end - lead_out

        minimum_window = min(duration, max(1.0, 1.1 * len(shot_range.dialogue_line_characters)))
        if end - start < minimum_window:
            center = (shot_range.start + shot_range.end) / 2
            half = minimum_window / 2
            start = max(shot_range.start, center - half)
            end = min(shot_range.end, center + half)

        if end <= start:
            return [(shot_range.start, shot_range.end)]
        return [(start, end)]

    def _allocate_dialogue_lines(
        self,
        shot_range: ShotTimeRange,
        speech_windows: list[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Allocate speech time in a shot across dialogue lines in order."""
        line_characters = shot_range.dialogue_line_characters
        line_texts = [self._strip_speaker_prefix(text) for text in shot_range.dialogue_texts]
        if not line_characters or not line_texts:
            return []

        total_speech = sum(end - start for start, end in speech_windows)
        if total_speech <= 0:
            return []

        weights = [self._line_weight(text) for text in line_texts]
        total_weight = sum(weights) or len(weights)

        allocated_segments: list[dict[str, Any]] = []
        window_index = 0
        cursor = speech_windows[0][0]

        for line_index, (char_id, text, weight) in enumerate(
            zip(line_characters, line_texts, weights)
        ):
            target_duration = total_speech * weight / total_weight

            # Keep the final line aligned to the remaining speech so the
            # whole shot gets covered even with rounding drift.
            if line_index == len(line_texts) - 1:
                target_duration = self._remaining_speech_duration(
                    speech_windows,
                    window_index,
                    cursor,
                )

            remaining = target_duration
            first_slice_start: float | None = None
            last_slice_end: float | None = None

            while remaining > 1e-6 and window_index < len(speech_windows):
                window_start, window_end = speech_windows[window_index]
                if cursor < window_start:
                    cursor = window_start

                available = window_end - cursor
                if available <= 1e-6:
                    window_index += 1
                    if window_index < len(speech_windows):
                        cursor = speech_windows[window_index][0]
                    continue

                take = min(remaining, available)
                slice_start = cursor
                slice_end = cursor + take
                if first_slice_start is None:
                    first_slice_start = slice_start
                last_slice_end = slice_end

                remaining -= take
                cursor = slice_end

                if cursor >= window_end - 1e-6:
                    window_index += 1
                    if window_index < len(speech_windows):
                        cursor = speech_windows[window_index][0]

            if first_slice_start is None or last_slice_end is None:
                continue

            allocated_segments.append(
                {
                    "shot_id": shot_range.shot_id,
                    "line_index": line_index,
                    "char_id": char_id,
                    "text": text,
                    "start": first_slice_start,
                    "end": last_slice_end,
                }
            )

        return allocated_segments

    def _remaining_speech_duration(
        self,
        speech_windows: list[tuple[float, float]],
        window_index: int,
        cursor: float,
    ) -> float:
        """Return remaining speech duration from the current cursor onward."""
        remaining = 0.0
        for idx in range(window_index, len(speech_windows)):
            start, end = speech_windows[idx]
            if idx == window_index:
                start = max(start, cursor)
            if end > start:
                remaining += end - start
        return remaining

    def _strip_speaker_prefix(self, line: str) -> str:
        match = re.match(r"^([^:]+):\s*(.+)", line.strip())
        if match:
            text = match.group(2).strip()
        else:
            text = line.strip()
        return text.strip("\"' \u201c\u201d")

    def _line_weight(self, text: str) -> float:
        words = re.findall(r"\w+", text)
        if words:
            return float(len(words))
        text = text.strip()
        return float(len(text) or 1)

    def _find_shot_for_time(
        self, time_point: float, timeline: list[ShotTimeRange]
    ) -> ShotTimeRange | None:
        """Find which shot a time point falls in."""
        for shot_range in timeline:
            if shot_range.start <= time_point < shot_range.end:
                return shot_range
        # If beyond last shot, assign to last shot
        if timeline and time_point >= timeline[-1].end:
            return timeline[-1]
        return None

    def map_speakers(
        self,
        diarized_segments: list[DiarizedSegment],
        clip,
        output_dir: Path | None = None,
    ) -> dict[str, str]:
        """
        Map speaker labels to character IDs.

        Strategy:
        1. Build shot timeline with character/dialogue metadata.
        2. For each diarized segment, find which shot it overlaps with.
        3. Use dialogue_lines speaker names + temporal overlap for mapping.
        4. For unambiguous shots (1 speaker, 1 character), map directly.
        5. For ambiguous cases, use voting across all segments.

        Returns:
            dict mapping speaker_label (e.g. "speaker_0") to character_id.
        """
        if not diarized_segments:
            return {}

        timeline = self._build_shot_timeline(clip)
        if not timeline:
            return {}

        # Collect evidence: for each speaker, which character IDs are likely
        speaker_evidence: dict[str, Counter] = defaultdict(Counter)

        for seg in diarized_segments:
            mid_point = (seg.start + seg.end) / 2
            shot_range = self._find_shot_for_time(mid_point, timeline)
            if shot_range is None:
                continue

            # If this shot has dialogue with known speakers
            if len(shot_range.dialogue_speakers) == 1:
                # Only one person speaks in this shot → strong evidence
                speaker_evidence[seg.speaker][shot_range.dialogue_speakers[0]] += 3
            elif shot_range.dialogue_speakers:
                # Multiple speakers in this shot → weaker evidence for each
                for char_id in shot_range.dialogue_speakers:
                    speaker_evidence[seg.speaker][char_id] += 1

            # Also use visible characters as weak evidence
            if len(shot_range.visible_characters) == 1:
                speaker_evidence[seg.speaker][shot_range.visible_characters[0]] += 1

        # Resolve: assign each speaker to the character with most evidence
        # Using greedy assignment to avoid conflicts
        mapping: dict[str, str] = {}
        assigned_characters: set[str] = set()

        # Sort speakers by strength of evidence (strongest first)
        speakers_by_confidence = sorted(
            speaker_evidence.keys(),
            key=lambda spk: (
                speaker_evidence[spk].most_common(1)[0][1]
                if speaker_evidence[spk]
                else 0
            ),
            reverse=True,
        )

        for speaker in speakers_by_confidence:
            evidence = speaker_evidence[speaker]
            for char_id, _ in evidence.most_common():
                if char_id not in assigned_characters:
                    mapping[speaker] = char_id
                    assigned_characters.add(char_id)
                    break

        # Fallback: speakers with no evidence → assign to remaining characters
        all_speakers = {seg.speaker for seg in diarized_segments}
        unmapped_speakers = all_speakers - set(mapping.keys())
        remaining_chars = []
        for shot_range in timeline:
            for char_id in shot_range.dialogue_speakers:
                if char_id not in assigned_characters:
                    remaining_chars.append(char_id)
                    assigned_characters.add(char_id)

        for speaker, char_id in zip(unmapped_speakers, remaining_chars):
            mapping[speaker] = char_id

        # Print mapping
        print(f"🔗 Speaker → Character mapping:")
        for speaker, char_id in sorted(mapping.items()):
            member = self.cast_by_id.get(char_id)
            name = member.name_en if member else char_id
            print(f"   {speaker} → {name} ({char_id})")

        unmapped = all_speakers - set(mapping.keys())
        if unmapped:
            print(f"   ⚠️  Unmapped speakers: {unmapped}")

        # Save mapping for debugging
        if output_dir:
            manifest_path = output_dir / "speaker_mapping.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mapping": mapping,
                        "evidence": {
                            spk: dict(evidence)
                            for spk, evidence in speaker_evidence.items()
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        return mapping
