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
    from google import genai
    from google.genai import types
except ImportError:  # Allows schema/helpers and --help to load before dependencies are installed.
    genai = None
    types = None

JUDGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JUDGING_ROOT))

from prompts import SEMANTIC_EQUIVALENCE_PROMPT  # noqa: E402

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
M3_ROOT = EVALUATION_ROOT / "m3_agent"
DEFAULT_API_CONFIG = M3_ROOT / "configs" / "api_config.json"
DEFAULT_PROCESSING_CONFIG = M3_ROOT / "configs" / "processing_config.json"


def load_m3_prompt() -> str:
    return SEMANTIC_EQUIVALENCE_PROMPT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_qa_map(path: Path | None, dataset_id: str | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    obj = load_json(path, {})
    if isinstance(obj, dict) and dataset_id and dataset_id in obj:
        qas = obj[dataset_id].get("qa_list", [])
    elif isinstance(obj, dict) and obj and "qa_list" not in obj:
        first = next(iter(obj.values()))
        qas = first.get("qa_list", []) if isinstance(first, dict) else []
    elif isinstance(obj, dict):
        qas = obj.get("qa_list", [])
    elif isinstance(obj, list):
        qas = obj
    else:
        qas = []
    out = {}
    for qa in qas:
        qid = qa.get("question_id") or qa.get("id")
        if qid:
            out[str(qid)] = qa
    return out


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def api_key_for_model(model: str, config_path: Path) -> str:
    cfg = load_json(config_path, {})
    keys = [model, model.removeprefix("models/")]
    for key in keys:
        item = cfg.get(key)
        if isinstance(item, dict) and item.get("api_key"):
            return str(item["api_key"]).strip()
    for env_key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"):
        if os.getenv(env_key):
            return os.environ[env_key].strip()
    raise RuntimeError(f"Missing Gemini API key for {model}")


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    out = {}
    for key in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def candidate_debug(response: Any) -> list[dict[str, Any]]:
    out = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        part_texts = []
        for part in parts or []:
            text = getattr(part, "text", None)
            if text:
                part_texts.append(str(text))
        finish_reason = getattr(candidate, "finish_reason", None)
        out.append(
            {
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
                "text": "".join(part_texts).strip(),
            }
        )
    return out


def extract_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
    except Exception:
        pass
    texts = [item.get("text", "") for item in candidate_debug(response)]
    return "".join(texts).strip()


class JudgeClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        retry_sleep: float,
    ) -> None:
        if genai is None or types is None:
            raise RuntimeError(
                "google-genai is required for the Gemini judge; install evaluation/judging/requirements.txt"
            )
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self._local = threading.local()

    def client(self) -> genai.Client:
        client = getattr(self._local, "client", None)
        if client is None:
            client = genai.Client(api_key=self.api_key)
            self._local.client = client
        return client

    def judge(self, prompt: str) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[str]]:
        errors = []
        last_usage: dict[str, Any] = {}
        last_debug: list[dict[str, Any]] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client().models.generate_content(
                    model=self.model,
                    contents=[types.Part.from_text(text=prompt)],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        system_instruction="You are an expert in video understanding.",
                    ),
                )
                text = extract_text(response)
                last_usage = usage_dict(response)
                last_debug = candidate_debug(response)
                if text.strip():
                    return text.strip(), last_usage, last_debug, errors
                errors.append(f"attempt_{attempt}: empty_text")
            except Exception as exc:
                setattr(self._local, "client", None)
                errors.append(f"attempt_{attempt}: {exc!r}")
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep * attempt)
        return "", last_usage, last_debug, errors


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


def rejudge_row(
    row: dict[str, Any],
    prompt_template: str,
    judge_client: JudgeClient,
    qa_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out = dict(row)
    qa = qa_map.get(str(row.get("id")), {})
    for key in (
        "type",
        "types",
        "category",
        "difficulty",
        "before_clip",
        "evidence_clip_ids",
        "required_modalities",
        "reasoning_hops",
    ):
        if key not in out and key in qa:
            out[key] = qa[key]
    response = str(row.get("response", "") or "").strip()
    if not response:
        out["gpt_eval"] = False
        out["judge_response"] = ""
        out["judge_model"] = judge_client.model
        out["judge_usage"] = {}
        out["judge_candidate_debug"] = []
        out["judge_errors"] = ["empty_agent_response"]
        return out

    prompt = prompt_template.format(
        question=row.get("question", ""),
        ground_truth_answer=row.get("answer", ""),
        agent_answer=response,
    )
    text, usage, debug, errors = judge_client.judge(prompt)
    verdict = text.strip().lower().rstrip(".")
    out["gpt_eval"] = verdict == "yes"
    out["judge_response"] = text
    out["judge_model"] = judge_client.model
    out["judge_usage"] = usage
    out["judge_candidate_debug"] = debug
    out["judge_errors"] = errors
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rejudge open-QA results with M3Agent-style Gemini judge.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--judge-model", default="models/gemini-3-flash-preview")
    parser.add_argument("--api-config", default=str(DEFAULT_API_CONFIG))
    parser.add_argument("--processing-config", default=str(DEFAULT_PROCESSING_CONFIG))
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    prompt_template = load_m3_prompt()
    qa_map = load_qa_map(Path(args.qa_file) if args.qa_file else None, args.dataset_id)
    processing_config = load_json(Path(args.processing_config), {})
    temperature = float(processing_config.get("temperature", 1e-6))
    api_key = api_key_for_model(args.judge_model, Path(args.api_config))
    judge_client = JudgeClient(
        args.judge_model,
        api_key,
        temperature,
        args.max_output_tokens,
        args.max_retries,
        args.retry_sleep,
    )

    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    done_rows = read_jsonl(output_path) if output_path.exists() else []
    done_ids = {str(row.get("id")) for row in done_rows}
    todo = [row for row in rows if str(row.get("id")) not in done_ids]
    print(
        f"[setup] input={input_path} total={len(rows)} existing={len(done_rows)} "
        f"todo={len(todo)} judge={args.judge_model} temp={temperature} "
        f"max_output_tokens={args.max_output_tokens} workers={args.workers}",
        flush=True,
    )

    write_lock = threading.Lock()
    completed = 0
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(rejudge_row, row, prompt_template, judge_client, qa_map)
                for row in todo
            ]
            for future in concurrent.futures.as_completed(futures):
                out = future.result()
                append_jsonl(output_path, out, write_lock)
                completed += 1
                if completed <= 10 or completed % 25 == 0:
                    empty = not str(out.get("judge_response", "")).strip()
                    print(
                        f"[judge] {completed}/{len(todo)} {out.get('id')} "
                        f"eval={out.get('gpt_eval')} empty={empty} "
                        f"resp={str(out.get('judge_response', ''))[:20]!r}",
                        flush=True,
                    )

    final_rows = read_jsonl(output_path)
    summary = summarize(final_rows)
    summary.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "judge_model": args.judge_model,
            "temperature": temperature,
            "max_output_tokens": args.max_output_tokens,
            "workers": args.workers,
        }
    )
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
