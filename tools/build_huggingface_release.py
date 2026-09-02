#!/usr/bin/env python3
"""Pack and validate the public ICM-Bench dataset release.

This tool accepts finalized public-format inputs. The ``pack`` command creates
a deterministic local release folder, and the ``validate`` command checks an
existing release folder. Neither command communicates with Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


VIDEO_COUNT = 839
MEMORY_VIDEO_COUNT = 838
QUESTION_COUNT = 1217
TRANSCRIPT_COUNT = 829
RETAINED_SHOTS = 1958
TOTAL_DURATION_SECONDS = 8460.042
PUBLIC_VIDEO_IDS = tuple(f"clip_{index:03d}" for index in range(VIDEO_COUNT))
PUBLIC_VIDEO_ID_SET = set(PUBLIC_VIDEO_IDS)
PUBLIC_CHARACTER_IDS = {
    "protagonist",
    "char_001",
    "char_002",
    "char_003",
    "char_005",
    "char_008",
}
EXPECTED_CATEGORIES = {
    "Identity Recall": 400,
    "Cross-Episode Identity Retrieval": 500,
    "Long-Term Identity Profile Inference": 317,
}
QA_FIELDS = {
    "question_id",
    "question",
    "reference_answer",
    "category",
    "target_character_ids",
    "evidence_video_ids",
    "before_clip",
}
VIDEO_FIELDS = {
    "file_name",
    "video_id",
    "date",
    "is_calibration",
    "duration_seconds",
}
CHARACTER_FIELDS = {"character_id", "name", "role", "age", "gender"}
PAPER_TITLE = (
    "ICM-Bench: Person-Level Identity Reasoning in "
    "Multimodal Agents with Long-Term Memory"
)
HUGGING_FACE_REPO = "ryanren0330/ICM-Bench"
CODE_REPO_URL = "https://github.com/Shidu-Ren/ICM-Bench"
SRT_TIMECODE = re.compile(
    r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser(
        "pack",
        help="Pack finalized public-format inputs into a local release folder.",
    )
    pack.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing clip_000.mp4 through clip_838.mp4.",
    )
    pack.add_argument(
        "--video-metadata",
        type=Path,
        required=True,
        help="Final public videos/metadata.jsonl.",
    )
    pack.add_argument(
        "--qa-jsonl",
        type=Path,
        required=True,
        help="Final public annotations/qa_test.jsonl.",
    )
    pack.add_argument(
        "--characters",
        type=Path,
        required=True,
        help="Final public annotations/characters.json.",
    )
    pack.add_argument(
        "--speakerless-transcripts",
        type=Path,
        required=True,
        help="Directory containing the finalized speakerless SRT files.",
    )
    pack.add_argument(
        "--speaker-labeled-transcripts",
        type=Path,
        required=True,
        help="Directory containing matching reference SRT files with labels.",
    )
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument(
        "--dataset-license",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "LICENSE-DATASET",
        help="Dataset license text copied to RELEASE_DIR/LICENSE.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate an existing local release folder without modifying it.",
    )
    validate.add_argument("release", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(
                isinstance(value, dict),
                f"{path}:{line_number} must contain a JSON object",
            )
            rows.append(value)
    return rows


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_index(value: str) -> int:
    match = re.fullmatch(r"clip_(\d{3})", str(value))
    if not match:
        raise ValueError(f"Invalid public clip identifier: {value!r}")
    index = int(match.group(1))
    require(index < VIDEO_COUNT, f"Clip identifier is outside clip_000--clip_838: {value}")
    return index


def deterministic_query_delay(question_id: str) -> int:
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return 5 + int(digest[:8], 16) % 16


def validate_characters(characters: object) -> tuple[list[dict], list[str]]:
    require(isinstance(characters, list), "characters.json must be a JSON array")
    require(len(characters) == 6, f"Expected 6 recurring characters, found {len(characters)}")
    rows: list[dict] = []
    for index, row in enumerate(characters):
        require(isinstance(row, dict), f"Character row {index} must be an object")
        require(
            set(row) == CHARACTER_FIELDS,
            f"Character row {index} has fields {sorted(row)}; expected {sorted(CHARACTER_FIELDS)}",
        )
        require(str(row["name"]).strip(), f"Character row {index} has an empty name")
        rows.append(row)
    ids = [str(row["character_id"]) for row in rows]
    require(len(ids) == len(set(ids)), "Character identifiers must be unique")
    require(set(ids) == PUBLIC_CHARACTER_IDS, f"Unexpected character identifiers: {sorted(ids)}")
    return rows, [str(row["name"]) for row in rows]


def validate_video_metadata(rows: list[dict]) -> list[dict]:
    require(len(rows) == VIDEO_COUNT, f"Expected {VIDEO_COUNT} video rows, found {len(rows)}")
    ids: list[str] = []
    duration = 0.0
    for position, row in enumerate(rows):
        require(
            set(row) == VIDEO_FIELDS,
            f"Video row {position} has fields {sorted(row)}; expected {sorted(VIDEO_FIELDS)}",
        )
        video_id = str(row["video_id"])
        clip_index(video_id)
        require(
            row["file_name"] == f"{video_id}.mp4",
            f"Video row {video_id} has mismatched file_name {row['file_name']!r}",
        )
        require(
            isinstance(row["duration_seconds"], (int, float))
            and not isinstance(row["duration_seconds"], bool)
            and row["duration_seconds"] > 0,
            f"Video row {video_id} has invalid duration",
        )
        duration += float(row["duration_seconds"])
        if video_id == "clip_000":
            require(row["is_calibration"] is True, "clip_000 must be the calibration clip")
            require(row["date"] is None, "clip_000 must have a null date")
        else:
            require(row["is_calibration"] is False, f"{video_id} cannot be calibration")
            require(
                isinstance(row["date"], str)
                and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", row["date"]),
                f"{video_id} must have an ISO date",
            )
        ids.append(video_id)
    require(ids == list(PUBLIC_VIDEO_IDS), "Video metadata must be ordered clip_000--clip_838")
    require(
        round(duration, 3) == TOTAL_DURATION_SECONDS,
        f"Expected total duration {TOTAL_DURATION_SECONDS}, found {round(duration, 3)}",
    )
    return rows


def validate_qa(rows: list[dict], character_ids: set[str]) -> list[dict]:
    require(len(rows) == QUESTION_COUNT, f"Expected {QUESTION_COUNT} questions, found {len(rows)}")
    question_ids: list[str] = []
    for position, row in enumerate(rows):
        require(
            set(row) == QA_FIELDS,
            f"QA row {position} has fields {sorted(row)}; expected {sorted(QA_FIELDS)}",
        )
        question_id = str(row["question_id"])
        require(question_id.strip(), f"QA row {position} has an empty question_id")
        require(isinstance(row["question"], str) and row["question"].strip(), f"{question_id} has no question")
        require(
            isinstance(row["reference_answer"], str) and row["reference_answer"].strip(),
            f"{question_id} has no reference answer",
        )
        require(row["category"] in EXPECTED_CATEGORIES, f"{question_id} has an unknown category")

        targets = row["target_character_ids"]
        require(isinstance(targets, list) and targets, f"{question_id} has no target characters")
        require(len(targets) == len(set(targets)), f"{question_id} repeats a target character")
        require(set(targets) <= character_ids, f"{question_id} references an unknown character")

        evidence = row["evidence_video_ids"]
        require(isinstance(evidence, list) and evidence, f"{question_id} has no evidence clips")
        require(len(evidence) == len(set(evidence)), f"{question_id} repeats an evidence clip")
        require(set(evidence) <= PUBLIC_VIDEO_ID_SET, f"{question_id} has a non-public evidence ID")
        require("clip_000" not in evidence, f"{question_id} uses calibration as evidence")
        evidence_indices = [clip_index(value) for value in evidence]

        if row["category"] == "Long-Term Identity Profile Inference":
            require(row["before_clip"] is None, f"{question_id} Profile before_clip must be null")
            require(len(evidence) >= 2, f"{question_id} Profile item needs at least two evidence clips")
        else:
            before_clip = row["before_clip"]
            require(before_clip in PUBLIC_VIDEO_ID_SET, f"{question_id} has an invalid before_clip")
            expected = min(
                VIDEO_COUNT - 1,
                max(evidence_indices) + deterministic_query_delay(question_id),
            )
            require(
                clip_index(before_clip) == expected,
                f"{question_id} before_clip is {before_clip}; expected clip_{expected:03d}",
            )
        question_ids.append(question_id)

    require(len(question_ids) == len(set(question_ids)), "question_id values must be unique")
    category_counts = Counter(str(row["category"]) for row in rows)
    require(
        dict(category_counts) == EXPECTED_CATEGORIES,
        f"Unexpected category counts: {dict(category_counts)}",
    )
    return rows


def transcript_files(directory: Path) -> dict[str, Path]:
    require(directory.is_dir(), f"Transcript directory does not exist: {directory}")
    entries = list(directory.iterdir())
    require(all(path.is_file() for path in entries), f"Transcript directory contains a subdirectory: {directory}")
    require(all(path.suffix == ".srt" for path in entries), f"Transcript directory contains a non-SRT file: {directory}")
    files = {path.name: path for path in entries}
    require(len(files) == TRANSCRIPT_COUNT, f"Expected {TRANSCRIPT_COUNT} SRT files in {directory}, found {len(files)}")
    for name in files:
        match = re.fullmatch(r"clip_(\d{3})\.srt", name)
        require(match is not None, f"Invalid transcript filename: {name}")
        index = int(match.group(1))
        require(1 <= index < VIDEO_COUNT, f"Transcript is outside memory clip range: {name}")
    return files


def validate_srt(text: str, path: Path) -> None:
    blocks = text.strip().split("\n\n") if text.strip() else []
    require(blocks, f"Empty SRT file: {path}")
    for expected_index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        require(len(lines) >= 3, f"Malformed SRT block in {path}")
        require(lines[0] == str(expected_index), f"Non-sequential SRT index in {path}")
        require(SRT_TIMECODE.fullmatch(lines[1]) is not None, f"Malformed SRT timecode in {path}")


def validate_transcripts(
    speakerless_dir: Path,
    labeled_dir: Path,
    character_names: list[str],
) -> tuple[dict[str, Path], dict[str, Path]]:
    speakerless = transcript_files(speakerless_dir)
    labeled = transcript_files(labeled_dir)
    require(set(speakerless) == set(labeled), "Speakerless and labeled transcript filenames differ")
    speaker_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(name) for name in character_names) + r"):\s*"
    )
    for name in sorted(speakerless):
        plain_text = speakerless[name].read_text(encoding="utf-8")
        labeled_text = labeled[name].read_text(encoding="utf-8")
        validate_srt(plain_text, speakerless[name])
        validate_srt(labeled_text, labeled[name])
        require(
            speaker_pattern.search(plain_text) is None,
            f"Speaker label remains in {speakerless[name]}",
        )
        require(
            speaker_pattern.sub("", labeled_text) == plain_text,
            f"Transcript variants differ beyond speaker labels: {name}",
        )
    return speakerless, labeled


def validate_source_videos(videos_dir: Path) -> dict[str, Path]:
    require(videos_dir.is_dir(), f"Video directory does not exist: {videos_dir}")
    files = {path.name: path for path in videos_dir.glob("clip_*.mp4")}
    expected = {f"{video_id}.mp4" for video_id in PUBLIC_VIDEO_IDS}
    require(set(files) == expected, "Video directory must contain exactly clip_000.mp4--clip_838.mp4")
    require(all(path.stat().st_size > 0 for path in files.values()), "A source MP4 is empty")
    return files


def compute_statistics(metadata: list[dict], qa_rows: list[dict]) -> dict:
    delayed = [row for row in qa_rows if row["before_clip"] is not None]
    actual_delays = [
        clip_index(row["before_clip"])
        - max(clip_index(value) for value in row["evidence_video_ids"])
        for row in delayed
    ]
    target_delays = [deterministic_query_delay(row["question_id"]) for row in delayed]
    return {
        "release_version": "1.0.0",
        "dataset_id": "ICM-Bench",
        "split": "test",
        "videos": len(metadata),
        "memory_videos": MEMORY_VIDEO_COUNT,
        "calibration_videos": 1,
        "total_duration_seconds": round(sum(float(row["duration_seconds"]) for row in metadata), 3),
        "total_duration_minutes": round(sum(float(row["duration_seconds"]) for row in metadata) / 60, 3),
        "retained_shots": RETAINED_SHOTS,
        "recurring_adults": 6,
        "questions": len(qa_rows),
        "question_category_counts": dict(Counter(row["category"] for row in qa_rows)),
        "evidence_clip_references": sum(len(row["evidence_video_ids"]) for row in qa_rows),
        "query_cutoff_policy": {
            "applies_to": ["Identity Recall", "Cross-Episode Identity Retrieval"],
            "target_delay_clips": [5, 20],
            "deterministic_key": "sha256(question_id)",
            "timeline_cap": "clip_838",
            "target_capped_questions": sum(
                actual < target for actual, target in zip(actual_delays, target_delays)
            ),
            "below_five_clip_questions": sum(delay < 5 for delay in actual_delays),
        },
        "speakerless_asr_transcripts": TRANSCRIPT_COUNT,
        "speaker_labeled_transcripts": TRANSCRIPT_COUNT,
        "timeline": {"start": "2025-05-01", "end": "2026-05-01"},
    }


def build_video_archive(archive_path: Path, videos: dict[str, Path]) -> None:
    with tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for video_id in PUBLIC_VIDEO_IDS:
            source = videos[f"{video_id}.mp4"]
            info = tarfile.TarInfo(name=f"videos/{video_id}.mp4")
            info.size = source.stat().st_size
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with source.open("rb") as handle:
                archive.addfile(info, handle)


def dataset_card(statistics: dict) -> str:
    return f'''---
license: cc-by-nc-sa-4.0
language:
- en
task_categories:
- visual-question-answering
tags:
- long-video
- multimodal
- question-answering
- long-term-memory
- identity-reasoning
- synthetic
pretty_name: ICM-Bench
size_categories:
- 1K<n<10K
viewer: false
---

# {PAPER_TITLE}

ICM-Bench evaluates person-centered evidence retrieval and cross-time relation reasoning in long-term multimodal agents. It contains 839 synthetic video clips spanning approximately 141 minutes and 1,217 open-ended questions about six recurring adults in a one-year life album.

## Dataset contents

| Item | Count |
|---|---:|
| Videos | {statistics['videos']} |
| Date-stamped memory clips | {statistics['memory_videos']} |
| Calibration clips | 1 |
| Retained shots | 1,958 |
| Recurring adults | 6 |
| Open-ended questions | 1,217 |
| Identity Recall | 400 |
| Cross-Episode Identity Retrieval | 500 |
| Long-Term Identity Profile Inference | 317 |

## Download

```bash
hf download {HUGGING_FACE_REPO} \\
  --repo-type dataset \\
  --local-dir ICM-Bench
tar -xf ICM-Bench/videos.tar -C ICM-Bench
```

The archive extracts 839 files under `videos/`, named `clip_000.mp4` through `clip_838.mp4`.

## Annotations and transcripts

`annotations/qa_test.jsonl` is the recommended annotation file; `qa_test.json` contains the same records as a JSON array. Timestamped transcripts are available for 829 memory clips. `resources/asr_transcripts/` omits speaker labels and is the transcript input for evaluated ASR-based settings. `resources/transcripts_with_speakers/` retains speaker labels for reference and must not be exposed to evaluated systems.

## Evaluation protocol

For Identity Recall and Cross-Episode Identity Retrieval, systems may access the calibration clip and memory clips only up to and including the question-specific `before_clip`. Long-Term Identity Profile Inference uses the complete timeline. The evaluator-side fields `reference_answer`, `target_character_ids`, and `evidence_video_ids`, along with `annotations/characters.json`, must not be exposed to the evaluated system.

Answers are open-ended and are evaluated by semantic equivalence rather than exact string matching.

## Uses and limitations

ICM-Bench is intended for evaluating long-term multimodal memory, person-centered retrieval, cross-episode reasoning, identity-profile inference, and open-ended video question answering. It is synthetic and should not be treated as a substitute for real-world video with natural noise, occlusion, overlapping speech, diverse environments, and spontaneous social behavior. It must not be used to identify, track, profile, or make decisions about real people.

## License

ICM-Bench is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. The videos were generated with Google Gemini/Veo services. Users remain responsible for complying with applicable law, provider terms, and the dataset license.

## Code and citation

Code and evaluation instructions: {CODE_REPO_URL}

```bibtex
@misc{{ren2026icmbench,
  title={{{PAPER_TITLE}}},
  author={{Ren, Shidu and Liu, Yunze and Liu, Xing and Wu, Chi-Hao and Zhou, Enmin and Shen, Junxiao}},
  year={{2026}},
  note={{arXiv preprint}}
}}
```
'''


def citation_cff() -> str:
    return f'''cff-version: 1.2.0
message: "If you use ICM-Bench, please cite the paper."
title: "{PAPER_TITLE}"
type: dataset
version: 1.0.0
date-released: 2026-08-29
license: CC-BY-NC-SA-4.0
authors:
  - family-names: Ren
    given-names: Shidu
  - family-names: Liu
    given-names: Yunze
  - family-names: Liu
    given-names: Xing
  - family-names: Wu
    given-names: Chi-Hao
  - family-names: Zhou
    given-names: Enmin
  - family-names: Shen
    given-names: Junxiao
repository-code: "{CODE_REPO_URL}"
url: "https://huggingface.co/datasets/{HUGGING_FACE_REPO}"
'''


def release_schema() -> dict:
    return {
        "qa_test.jsonl": {
            "question_id": "Unique question identifier.",
            "question": "Open-ended benchmark question.",
            "reference_answer": "Evaluator-side target semantic content.",
            "category": "Canonical paper-facing question family.",
            "target_character_ids": "Evaluator-side recurring-character identifiers.",
            "evidence_video_ids": "Evaluator-side supporting public clip IDs.",
            "before_clip": "Latest accessible clip for Recall/Retrieval; null for Profile.",
        },
        "videos/metadata.jsonl": {
            "file_name": "MP4 filename after extracting videos.tar.",
            "video_id": "Identifier in the released evaluation timeline.",
            "date": "Visible clip date, null for calibration.",
            "is_calibration": "Whether this is the non-evidence calibration clip.",
            "duration_seconds": "Container duration in seconds.",
        },
    }


VALIDATOR_SCRIPT = r'''#!/usr/bin/env python3
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

root = Path(__file__).resolve().parents[1]
expected_ids = {f"clip_{index:03d}" for index in range(839)}
qa = [json.loads(line) for line in (root / "annotations/qa_test.jsonl").read_text().splitlines() if line]
qa_json = json.loads((root / "annotations/qa_test.json").read_text())
videos = [json.loads(line) for line in (root / "videos/metadata.jsonl").read_text().splitlines() if line]
assert qa == qa_json
assert len(qa) == 1217
assert len(videos) == 839
assert Counter(row["category"] for row in qa) == {
    "Identity Recall": 400,
    "Cross-Episode Identity Retrieval": 500,
    "Long-Term Identity Profile Inference": 317,
}
assert {row["video_id"] for row in videos} == expected_ids
assert all(set(row) == {"file_name", "video_id", "date", "is_calibration", "duration_seconds"} for row in videos)
assert all(set(row) == {"question_id", "question", "reference_answer", "category", "target_character_ids", "evidence_video_ids", "before_clip"} for row in qa)
assert len({row["question_id"] for row in qa}) == 1217
for row in qa:
    assert set(row["evidence_video_ids"]) <= expected_ids
    if row["category"] == "Long-Term Identity Profile Inference":
        assert row["before_clip"] is None
        assert len(row["evidence_video_ids"]) >= 2
    else:
        assert row["before_clip"] in expected_ids

with tarfile.open(root / "videos.tar", "r") as archive:
    members = archive.getmembers()
assert len(members) == 839
assert all(member.isfile() and not member.issym() and not member.islnk() for member in members)
assert {member.name for member in members} == {f"videos/clip_{index:03d}.mp4" for index in range(839)}
assert not list((root / "videos").glob("clip_*.mp4"))

speakerless = sorted((root / "resources/asr_transcripts").glob("*.srt"))
labeled = sorted((root / "resources/transcripts_with_speakers").glob("*.srt"))
assert len(speakerless) == len(labeled) == 829
assert [path.name for path in speakerless] == [path.name for path in labeled]
names = [row["name"] for row in json.loads((root / "annotations/characters.json").read_text())]
speaker_pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in names) + r"):\s*")
for plain, named in zip(speakerless, labeled):
    plain_text = plain.read_text()
    named_text = named.read_text()
    assert speaker_pattern.search(plain_text) is None
    assert speaker_pattern.sub("", named_text) == plain_text

checksum_file = root / "checksums/sha256.txt"
entries = {}
for line in checksum_file.read_text().splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    assert match, line
    digest, relative = match.groups()
    path = PurePosixPath(relative)
    assert not path.is_absolute() and ".." not in path.parts
    assert relative not in entries
    entries[relative] = digest
expected_paths = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != checksum_file
}
assert set(entries) == expected_paths
for relative, expected in entries.items():
    digest = hashlib.sha256()
    with (root / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == expected, relative
print("ICM-Bench release validation passed.")
'''


def safe_release_file(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    require(not posix.is_absolute() and ".." not in posix.parts, f"Unsafe checksum path: {relative}")
    target = root.joinpath(*posix.parts)
    require(target.is_file(), f"Checksum target is missing: {relative}")
    return target


def validate_checksums(root: Path) -> None:
    checksum_file = root / "checksums" / "sha256.txt"
    require(checksum_file.is_file(), "Missing checksums/sha256.txt")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed checksum line {line_number}")
        digest, relative = match.groups()
        require(relative not in entries, f"Duplicate checksum path: {relative}")
        entries[relative] = digest
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_file
    }
    require(set(entries) == expected_paths, "Checksum inventory does not match release files")
    for relative, expected in entries.items():
        actual = sha256(safe_release_file(root, relative))
        require(actual == expected, f"Checksum mismatch: {relative}")


def validate_archive(path: Path) -> None:
    require(path.is_file(), "Missing videos.tar")
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
    expected_names = {f"videos/{video_id}.mp4" for video_id in PUBLIC_VIDEO_IDS}
    require(len(members) == VIDEO_COUNT, f"Expected {VIDEO_COUNT} tar members, found {len(members)}")
    require(all(member.isfile() for member in members), "videos.tar contains a non-regular member")
    require(all(not member.issym() and not member.islnk() for member in members), "videos.tar contains a link")
    require({member.name for member in members} == expected_names, "videos.tar member names are incorrect")
    require(all(member.size > 0 for member in members), "videos.tar contains an empty MP4")


def validate_release(root: Path) -> dict:
    root = root.resolve()
    require(root.is_dir(), f"Release directory does not exist: {root}")
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "MANIFEST.json",
        "videos.tar",
        "videos/metadata.jsonl",
        "annotations/qa_test.jsonl",
        "annotations/qa_test.json",
        "annotations/characters.json",
        "annotations/dataset_statistics.json",
        "annotations/schema.json",
        "resources/asr_transcripts",
        "resources/transcripts_with_speakers",
        "scripts/validate_release.py",
        "checksums/sha256.txt",
    ]
    for relative in required:
        require((root / relative).exists(), f"Missing release path: {relative}")

    metadata = validate_video_metadata(read_jsonl(root / "videos/metadata.jsonl"))
    characters, names = validate_characters(
        json.loads((root / "annotations/characters.json").read_text(encoding="utf-8"))
    )
    qa_rows = validate_qa(
        read_jsonl(root / "annotations/qa_test.jsonl"),
        {row["character_id"] for row in characters},
    )
    qa_json = json.loads((root / "annotations/qa_test.json").read_text(encoding="utf-8"))
    require(qa_json == qa_rows, "qa_test.json and qa_test.jsonl differ")
    validate_transcripts(
        root / "resources/asr_transcripts",
        root / "resources/transcripts_with_speakers",
        names,
    )
    validate_archive(root / "videos.tar")
    require(not list((root / "videos").glob("clip_*.mp4")), "Individual MP4s must not remain beside videos.tar")

    statistics = compute_statistics(metadata, qa_rows)
    released_statistics = json.loads(
        (root / "annotations/dataset_statistics.json").read_text(encoding="utf-8")
    )
    require(released_statistics == statistics, "dataset_statistics.json is stale")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest.get("statistics") == statistics, "MANIFEST statistics are stale")
    file_count = sum(path.is_file() for path in root.rglob("*"))
    require(manifest.get("file_count") == file_count, "MANIFEST file_count is stale")
    validate_checksums(root)
    return manifest


def pack_release(args: argparse.Namespace) -> Path:
    videos_dir = args.videos_dir.resolve()
    video_metadata_path = args.video_metadata.resolve()
    qa_path = args.qa_jsonl.resolve()
    characters_path = args.characters.resolve()
    speakerless_dir = args.speakerless_transcripts.resolve()
    labeled_dir = args.speaker_labeled_transcripts.resolve()
    license_path = args.dataset_license.resolve()
    output = args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    require(license_path.is_file(), f"Dataset license does not exist: {license_path}")

    videos = validate_source_videos(videos_dir)
    metadata = validate_video_metadata(read_jsonl(video_metadata_path))
    characters, names = validate_characters(json.loads(characters_path.read_text(encoding="utf-8")))
    qa_rows = validate_qa(read_jsonl(qa_path), {row["character_id"] for row in characters})
    speakerless, labeled = validate_transcripts(speakerless_dir, labeled_dir, names)
    statistics = compute_statistics(metadata, qa_rows)

    for directory in (
        output / "videos",
        output / "annotations",
        output / "resources" / "asr_transcripts",
        output / "resources" / "transcripts_with_speakers",
        output / "scripts",
        output / "checksums",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    write_jsonl(output / "videos" / "metadata.jsonl", metadata)
    write_jsonl(output / "annotations" / "qa_test.jsonl", qa_rows)
    write_json(output / "annotations" / "qa_test.json", qa_rows)
    write_json(output / "annotations" / "characters.json", characters)
    write_json(output / "annotations" / "dataset_statistics.json", statistics)
    write_json(output / "annotations" / "schema.json", release_schema())

    for name, source in speakerless.items():
        shutil.copy2(source, output / "resources" / "asr_transcripts" / name)
    for name, source in labeled.items():
        shutil.copy2(source, output / "resources" / "transcripts_with_speakers" / name)
    build_video_archive(output / "videos.tar", videos)

    (output / "README.md").write_text(dataset_card(statistics), encoding="utf-8")
    (output / "CITATION.cff").write_text(citation_cff(), encoding="utf-8")
    shutil.copy2(license_path, output / "LICENSE")
    (output / ".gitattributes").write_text(
        "*.tar filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    validator = output / "scripts" / "validate_release.py"
    validator.write_text(VALIDATOR_SCRIPT, encoding="utf-8")
    validator.chmod(0o755)

    current_files = sum(path.is_file() for path in output.rglob("*"))
    manifest = {
        "release_version": "1.0.0",
        "dataset_id": "ICM-Bench",
        "split": "test",
        "file_count": current_files + 2,
        "statistics": statistics,
    }
    write_json(output / "MANIFEST.json", manifest)

    checksum_file = output / "checksums" / "sha256.txt"
    checksummed = [
        path
        for path in output.rglob("*")
        if path.is_file() and path != checksum_file
    ]
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(checksummed)
    ]
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    validate_release(output)
    return output


def main() -> None:
    args = parse_args()
    try:
        if args.command == "pack":
            output = pack_release(args)
            print(f"Packed and validated ICM-Bench release: {output}")
        else:
            manifest = validate_release(args.release)
            print(json.dumps(manifest, indent=2))
            print("ICM-Bench release validation passed.")
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        raise SystemExit(f"ICM-Bench release error: {exc}") from exc


if __name__ == "__main__":
    main()
