#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # Allows schema/helpers and --help to load before dependencies are installed.
    OpenAI = None

JUDGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JUDGING_ROOT))

from prompts import SEMANTIC_EQUIVALENCE_PROMPT  # noqa: E402

def load_m3_prompt() -> str:
    return SEMANTIC_EQUIVALENCE_PROMPT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def qtype(row: dict[str, Any]) -> str:
    typ = row.get("type") or row.get("types") or row.get("category") or "unknown"
    if isinstance(typ, list):
        return str(typ[0]) if typ else "unknown"
    return str(typ)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def bucket_stats(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(bucket)
        correct = sum(bool(row.get("gpt_eval")) for row in bucket)
        empty = sum(not str(row.get("judge_response", "")).strip() for row in bucket)
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
            "empty_judge_response": empty,
        }

    out = {"overall": bucket_stats(rows), "by_type": {}, "by_difficulty": {}}
    for key, fn in (
        ("by_type", qtype),
        ("by_difficulty", lambda row: str(row.get("difficulty") or "unknown")),
    ):
        values = sorted({fn(row) for row in rows})
        out[key] = {value: bucket_stats([row for row in rows if fn(row) == value]) for value in values}
    return out


class OpenAIJudgeClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        retry_sleep: float,
        reasoning_effort: str | None,
        api_key: str | None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "openai is required for the GPT cross-check; install evaluation/judging/requirements.txt"
            )
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Set it before running GPT cross-check.")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.reasoning_effort = reasoning_effort
        self._local = threading.local()
        self._api_key = api_key

    def client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(api_key=self._api_key)
            self._local.client = client
        return client

    def judge(self, prompt: str) -> tuple[str, dict[str, Any], list[str]]:
        errors: list[str] = []
        last_usage: dict[str, Any] = {}
        for attempt in range(1, self.max_retries + 1):
            try:
                is_newer_model = any(
                    name in self.model.lower() for name in ("gpt-5", "o1", "o3", "o4")
                )
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "instructions": "You are an expert in video understanding.",
                    "input": prompt,
                    "max_output_tokens": self.max_output_tokens,
                }
                if self.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                if not is_newer_model:
                    kwargs["temperature"] = self.temperature
                response = self.client().responses.create(**kwargs)
                text = str(getattr(response, "output_text", "") or "").strip()
                usage = getattr(response, "usage", None)
                if usage is not None:
                    last_usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                if text:
                    return text, last_usage, errors
                errors.append(f"attempt_{attempt}: empty_text")
            except Exception as exc:  # noqa: BLE001
                setattr(self._local, "client", None)
                errors.append(f"attempt_{attempt}: {exc!r}")
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep * attempt)
        return "", last_usage, errors


def rejudge_row(
    row: dict[str, Any],
    prompt_template: str,
    judge_client: OpenAIJudgeClient,
) -> dict[str, Any]:
    out = dict(row)
    response = str(row.get("response", "") or "").strip()
    if not response:
        out["gpt_eval"] = False
        out["judge_response"] = ""
        out["judge_model"] = judge_client.model
        out["judge_usage"] = {}
        out["judge_errors"] = ["empty_agent_response"]
        return out
    prompt = prompt_template.format(
        question=row.get("question", ""),
        ground_truth_answer=row.get("answer", ""),
        agent_answer=response,
    )
    text, usage, errors = judge_client.judge(prompt)
    verdict = text.strip().lower().rstrip(".")
    out["gpt_eval"] = verdict == "yes"
    out["judge_response"] = text
    out["judge_model"] = judge_client.model
    out["judge_usage"] = usage
    out["judge_errors"] = errors
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rejudge open-QA results with M3Agent-style OpenAI/GPT judge.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    output_path = Path(args.output)
    done_rows = read_jsonl(output_path)
    done_ids = {str(row.get("id")) for row in done_rows}
    todo = [row for row in rows if str(row.get("id")) not in done_ids]

    print(
        f"[setup] input={args.input} total={len(rows)} existing={len(done_rows)} "
        f"todo={len(todo)} judge={args.judge_model} workers={args.workers}",
        flush=True,
    )
    judge_client = OpenAIJudgeClient(
        args.judge_model,
        args.temperature,
        args.max_output_tokens,
        args.max_retries,
        args.retry_sleep,
        args.reasoning_effort,
        args.api_key,
    )
    prompt_template = load_m3_prompt()
    write_lock = threading.Lock()
    completed = 0
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(rejudge_row, row, prompt_template, judge_client) for row in todo]
            for future in concurrent.futures.as_completed(futures):
                out = future.result()
                append_jsonl(output_path, out, write_lock)
                completed += 1
                if completed <= 10 or completed % 25 == 0:
                    print(
                        f"[judge] {completed}/{len(todo)} {out.get('id')} "
                        f"eval={out.get('gpt_eval')} resp={str(out.get('judge_response', ''))[:20]!r}",
                        flush=True,
                    )

    final_rows = read_jsonl(output_path)
    summary = summarize(final_rows)
    summary.update(
        {
            "input": str(args.input),
            "output": str(output_path),
            "judge_model": args.judge_model,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "workers": args.workers,
        }
    )
    atomic_write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
