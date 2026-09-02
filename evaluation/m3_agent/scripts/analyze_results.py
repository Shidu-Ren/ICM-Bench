#!/usr/bin/env python
import argparse
import json
import os
import statistics
from collections import defaultdict


def safe_load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_video_id(item_id):
    if not item_id:
        return "(unknown)"
    if "_Q" in item_id:
        return item_id.split("_Q")[0]
    return item_id


def count_actions(conversations):
    search_count = 0
    answer_count = 0
    for msg in conversations or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "Action: [Search]" in content:
            search_count += 1
        if "Action: [Answer]" in content:
            answer_count += 1
    return search_count, answer_count


def count_empty_searches(conversations):
    empty = 0
    for msg in conversations or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if "(The search result is empty" in content:
            empty += 1
        elif content.strip() == "Searched knowledge: {}":
            empty += 1
        elif "Searched knowledge: {}" in content:
            empty += 1
    return empty


def last_action(conversations):
    for msg in reversed(conversations or []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "Action: [Answer]" in content:
            return "Answer"
        if "Action: [Search]" in content:
            return "Search"
    return "(unknown)"


def response_len_stats(values):
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0}
    values_sorted = sorted(values)

    def pct(p):
        idx = int(round((p / 100.0) * (len(values_sorted) - 1)))
        return values_sorted[idx]

    return {
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "avg": round(statistics.mean(values_sorted), 2),
        "p50": pct(50),
        "p90": pct(90),
    }


def bucket(value, buckets):
    for label, (lo, hi) in buckets:
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to control output jsonl")
    parser.add_argument("--output", required=True, help="Path to write markdown summary")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Missing results file: {args.input}")
        return 1

    rows = list(safe_load_lines(args.input))
    total = len(rows)
    correct = sum(1 for r in rows if r.get("gpt_eval"))
    acc = (correct / total) if total else 0.0

    by_video = defaultdict(lambda: {"total": 0, "correct": 0})
    by_before_clip = defaultdict(lambda: {"total": 0, "correct": 0})
    by_last_action = defaultdict(lambda: {"total": 0, "correct": 0})
    by_search_bucket = defaultdict(lambda: {"total": 0, "correct": 0})
    by_empty_search_bucket = defaultdict(lambda: {"total": 0, "correct": 0})
    by_rounds_bucket = defaultdict(lambda: {"total": 0, "correct": 0})

    search_counts = []
    empty_search_counts = []
    rounds_counts = []
    response_lengths = []
    response_word_counts = []
    missing_response = 0
    failures = []

    search_buckets = [
        ("0", (0, 0)),
        ("1", (1, 1)),
        ("2-3", (2, 3)),
        ("4+", (4, None)),
    ]
    empty_search_buckets = [
        ("0", (0, 0)),
        ("1", (1, 1)),
        ("2+", (2, None)),
    ]
    rounds_buckets = [
        ("1", (1, 1)),
        ("2", (2, 2)),
        ("3", (3, 3)),
        ("4+", (4, None)),
    ]

    for r in rows:
        vid = get_video_id(r.get("id"))
        by_video[vid]["total"] += 1
        if r.get("gpt_eval"):
            by_video[vid]["correct"] += 1

        bc = r.get("before_clip", "(none)")
        by_before_clip[bc]["total"] += 1
        if r.get("gpt_eval"):
            by_before_clip[bc]["correct"] += 1

        conversations = r.get("conversations", [])
        search_count, answer_count = count_actions(conversations)
        empty_search_count = count_empty_searches(conversations)
        last = last_action(conversations)

        by_last_action[last]["total"] += 1
        if r.get("gpt_eval"):
            by_last_action[last]["correct"] += 1

        search_counts.append(search_count)
        empty_search_counts.append(empty_search_count)
        rounds_counts.append(answer_count + search_count)

        by_search_bucket[bucket(search_count, search_buckets)]["total"] += 1
        by_empty_search_bucket[bucket(empty_search_count, empty_search_buckets)]["total"] += 1
        by_rounds_bucket[bucket(answer_count + search_count, rounds_buckets)]["total"] += 1

        if r.get("gpt_eval"):
            by_search_bucket[bucket(search_count, search_buckets)]["correct"] += 1
            by_empty_search_bucket[bucket(empty_search_count, empty_search_buckets)]["correct"] += 1
            by_rounds_bucket[bucket(answer_count + search_count, rounds_buckets)]["correct"] += 1

        response = r.get("response", "")
        if not response:
            missing_response += 1
            failures.append(r)
        else:
            response_lengths.append(len(response))
            response_word_counts.append(len(response.split()))

    md = []
    md.append("# Control Results Summary\n")
    md.append(f"- total: {total}")
    md.append(f"- correct: {correct}")
    md.append(f"- acc: {round(acc, 4)}")
    md.append(f"- missing_response: {missing_response}")
    md.append("")

    md.append("## Response Lengths")
    md.append(json.dumps(response_len_stats(response_lengths), indent=2))
    md.append("")
    md.append("## Response Word Counts")
    md.append(json.dumps(response_len_stats(response_word_counts), indent=2))
    md.append("")

    def table(title, data):
        md.append(f"## {title}")
        md.append("| key | total | correct | acc |")
        md.append("| --- | --- | --- | --- |")
        for k, s in sorted(data.items(), key=lambda x: str(x[0])):
            v_acc = s["correct"] / s["total"] if s["total"] else 0
            md.append(f"| {k} | {s['total']} | {s['correct']} | {round(v_acc, 4)} |")
        md.append("")

    table("By Video", by_video)
    table("By before_clip", by_before_clip)
    table("By last_action", by_last_action)
    table("By search_count", by_search_bucket)
    table("By empty_search_count", by_empty_search_bucket)
    table("By rounds", by_rounds_bucket)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
