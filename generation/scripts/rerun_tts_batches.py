#!/usr/bin/env python3
"""Rerun Gemini TTS redub for clip batches with the current voice pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def clip_list(start: int, end: int) -> str:
    return ",".join(f"clip_{idx:03d}" for idx in range(start, end + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun current Gemini TTS for clip ranges.")
    parser.add_argument("--config", default="configs/video_config.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--voice-workers", type=int, default=2)
    parser.add_argument("--voice-output-dir-name", default="clips_gemini_tts")
    parser.add_argument("--voice-work-dir-name", default="voice_work")
    parser.add_argument("--semantic-shortening", action="store_true")
    parser.add_argument("--semantic-shortening-model", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    overall = 0
    for batch_start in range(args.start, args.end + 1, args.batch_size):
        batch_end = min(args.end, batch_start + args.batch_size - 1)
        clips = clip_list(batch_start, batch_end)
        print(
            f"\n===== TTS batch clip_{batch_start:03d}-clip_{batch_end:03d} START =====",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "video_generator.voice.pipeline",
            "--config",
            args.config,
            "--output-root",
            args.output_root,
            "--include-clips",
            clips,
            "--voice-output-dir-name",
            args.voice_output_dir_name,
            "--voice-work-dir-name",
            args.voice_work_dir_name,
            "--voice-workers",
            str(args.voice_workers),
        ]
        if args.semantic_shortening:
            cmd.append("--semantic-shortening")
        if args.semantic_shortening_model:
            cmd.extend(["--semantic-shortening-model", args.semantic_shortening_model])
        if args.force:
            cmd.append("--force")

        result = subprocess.run(cmd, cwd=repo_root, env=env)
        print(
            f"===== TTS batch clip_{batch_start:03d}-clip_{batch_end:03d} EXIT {result.returncode} =====",
            flush=True,
        )
        if result.returncode != 0:
            overall = result.returncode

    print(f"VOICE_EXIT_CODE={overall}", flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
