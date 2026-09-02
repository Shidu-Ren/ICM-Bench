from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


def usage_metadata_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return dict(usage)

    result: dict[str, Any] = {}
    for key in (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
        "cached_content_token_count",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def text_model_price(model: str, prompt_tokens: int) -> tuple[float, float]:
    normalized = model.lower()
    if "tts" in normalized:
        return 1.00, 20.00
    if "flash-lite" in normalized:
        return 0.25, 1.50
    if "flash" in normalized:
        return 0.50, 3.00
    if "pro" in normalized:
        if prompt_tokens > 200_000:
            return 4.00, 18.00
        return 2.00, 12.00
    return 2.00, 12.00


def estimate_usage_cost(model: str, usage: dict[str, Any]) -> dict[str, Any]:
    normalized_model = model.lower()
    prompt_tokens = int(usage.get("prompt_token_count") or 0)
    candidate_tokens = int(usage.get("candidates_token_count") or 0)
    thought_tokens = int(usage.get("thoughts_token_count") or 0)
    output_tokens = candidate_tokens + thought_tokens
    input_price, output_price = text_model_price(model, prompt_tokens)
    input_cost = prompt_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price
    if "tts" in normalized_model:
        billing_note = (
            "Estimate uses public Gemini TTS rates: text input tokens and audio output tokens "
            "are billed separately; exact billing may differ."
        )
    else:
        billing_note = "Estimate uses public Gemini token rates; exact billing may differ."
    return {
        "billing_note": billing_note,
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
        "billable_output_tokens_est": output_tokens,
        "input_cost_usd_est": round(input_cost, 6),
        "output_cost_usd_est": round(output_cost, 6),
        "total_cost_usd_est": round(input_cost + output_cost, 6),
    }


class ApiUsageLogger:
    """Append-only usage logger shared by planning, QA, image, video, and TTS calls."""

    def __init__(self, metadata_dir: str | Path) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.metadata_dir / "00_api_usage.jsonl"
        self.summary_path = self.metadata_dir / "00_api_usage_summary.json"
        self._lock = threading.Lock()
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(self.records_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def append_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)
            with open(self.records_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_response(
        self,
        *,
        response: Any,
        operation: str,
        model: str,
        prompt: str | None = None,
        prompt_chars: int | None = None,
        attempt: int | None = None,
        schema: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        usage = usage_metadata_dict(response)
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "operation": operation,
            "schema": schema,
            "attempt": attempt,
            "model": model,
            "prompt_chars": len(prompt) if prompt is not None else prompt_chars,
            "usage_metadata": usage,
            "cost_estimate": estimate_usage_cost(model, usage),
        }
        if extra:
            record["extra"] = extra
        self.append_record(record)
        return record

    def record_failure(
        self,
        *,
        operation: str,
        model: str,
        error: Exception | str,
        prompt: str | None = None,
        prompt_chars: int | None = None,
        attempt: int | None = None,
        schema: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "operation": operation,
            "schema": schema,
            "attempt": attempt,
            "model": model,
            "status": "error",
            "prompt_chars": len(prompt) if prompt is not None else prompt_chars,
            "usage_metadata": {},
            "cost_estimate": estimate_usage_cost(model, {}),
            "error": str(error)[:1200],
        }
        if extra:
            record["extra"] = extra
        self.append_record(record)
        return record

    def write_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "record_count": len(self.records),
            "by_model": {},
            "by_operation": {},
            "totals": {
                "prompt_tokens": 0,
                "candidates_tokens": 0,
                "thoughts_tokens": 0,
                "billable_output_tokens_est": 0,
                "total_tokens": 0,
                "cost_usd_est": 0.0,
            },
            "billing_note": (
                "Estimated from response.usage_metadata when available. "
                "TTS records use text-input/audio-output token rates when the model name contains TTS. "
                "Image and video generation may have separate billing rules; "
                "records without usage_metadata are kept for auditability but contribute zero token cost here."
            ),
        }

        for record in self.records:
            usage = record.get("usage_metadata", {}) or {}
            cost = record.get("cost_estimate", {}) or {}
            model = str(record.get("model") or "unknown")
            operation = str(record.get("operation") or "unknown")
            prompt_tokens = int(usage.get("prompt_token_count") or 0)
            candidates_tokens = int(usage.get("candidates_token_count") or 0)
            thoughts_tokens = int(usage.get("thoughts_token_count") or 0)
            billable_output_tokens = int(
                cost.get("billable_output_tokens_est") or candidates_tokens + thoughts_tokens
            )
            total_tokens = int(usage.get("total_token_count") or 0)
            cost_total = float(cost.get("total_cost_usd_est") or 0.0)

            for bucket_name, bucket_key in (("by_model", model), ("by_operation", operation)):
                bucket = summary[bucket_name].setdefault(
                    bucket_key,
                    {
                        "record_count": 0,
                        "prompt_tokens": 0,
                        "candidates_tokens": 0,
                        "thoughts_tokens": 0,
                        "billable_output_tokens_est": 0,
                        "total_tokens": 0,
                        "cost_usd_est": 0.0,
                    },
                )
                bucket["record_count"] += 1
                bucket["prompt_tokens"] += prompt_tokens
                bucket["candidates_tokens"] += candidates_tokens
                bucket["thoughts_tokens"] += thoughts_tokens
                bucket["billable_output_tokens_est"] += billable_output_tokens
                bucket["total_tokens"] += total_tokens
                bucket["cost_usd_est"] += cost_total

            totals = summary["totals"]
            totals["prompt_tokens"] += prompt_tokens
            totals["candidates_tokens"] += candidates_tokens
            totals["thoughts_tokens"] += thoughts_tokens
            totals["billable_output_tokens_est"] += billable_output_tokens
            totals["total_tokens"] += total_tokens
            totals["cost_usd_est"] += cost_total

        for bucket_group in (summary["by_model"], summary["by_operation"]):
            for bucket in bucket_group.values():
                bucket["cost_usd_est"] = round(bucket["cost_usd_est"], 6)
        summary["totals"]["cost_usd_est"] = round(summary["totals"]["cost_usd_est"], 6)

        with open(self.summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        return summary
