#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


CATEGORY_TO_LEGACY_TYPE = {
    "Identity Recall": "memory_recall",
    "Cross-Episode Identity Retrieval": "memory_rag_qa",
    "Long-Term Identity Profile Inference": "user_profile",
}


def normalize_qa(item):
    normalized = dict(item)
    if "answer" not in normalized and "reference_answer" in normalized:
        normalized["answer"] = normalized["reference_answer"]
    if "type" not in normalized and normalized.get("category") in CATEGORY_TO_LEGACY_TYPE:
        normalized["type"] = [CATEGORY_TO_LEGACY_TYPE[normalized["category"]]]
    before_clip = normalized.get("before_clip")
    if isinstance(before_clip, str):
        match = re.search(r"(\d+)$", before_clip)
        if not match:
            raise ValueError(f"Cannot parse before_clip: {before_clip!r}")
        normalized["before_clip"] = int(match.group(1))
    if not normalized.get("question_id"):
        raise ValueError("Every QA row must contain question_id")
    if "answer" not in normalized:
        raise ValueError(f"QA row {normalized['question_id']} has no reference answer")
    return normalized


def normalize_container(container, mem_path):
    if isinstance(container, list):
        return {"icm_bench": {"qa_list": [normalize_qa(row) for row in container], "mem_path": mem_path}}
    if isinstance(container, dict) and "qa_list" in container:
        normalized = dict(container)
        normalized["qa_list"] = [normalize_qa(row) for row in normalized["qa_list"]]
        normalized["mem_path"] = mem_path
        return {"icm_bench": normalized}
    if isinstance(container, dict):
        normalized = {}
        for key, value in container.items():
            if not isinstance(value, dict) or "qa_list" not in value:
                continue
            entry = dict(value)
            entry["qa_list"] = [normalize_qa(row) for row in entry["qa_list"]]
            entry["mem_path"] = mem_path
            normalized[key] = entry
        if normalized:
            return normalized
    raise ValueError(f"Unsupported annotation format: {type(container).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mem-path", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = normalize_container(data, args.mem_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total = sum(len(item.get("qa_list", [])) for item in data.values())
    print(json.dumps({"output": str(output), "mem_path": args.mem_path, "qa": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
