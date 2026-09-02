#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from hipporag import HippoRAG
from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig


DEFAULT_DATASET_ID = "icm_bench"

QA_SYSTEM_NO_UNKNOWN = (
    "You are a memory QA assistant. Use the provided evidence to answer.\n"
    "Give the most likely concise answer. Do not answer 'Unknown'. Respond with only the answer.\n"
    "Do not output reasoning or analysis. /no_think"
)
QA_SYSTEM_REASONABLE_GUESS = (
    "You are a memory QA assistant. Use the provided evidence to answer.\n"
    "Give the most likely concise answer. Do not answer 'Unknown'. Respond with only the answer.\n"
    "If there is insufficient information, you can make reasonable guesses.\n"
    "Do not output reasoning or analysis. /no_think"
)
QA_SYSTEM_BEST_SUPPORTED = (
    "You are answering an open-ended question about a long personal video album. "
    "Relevant clips were retrieved from a graph-based memory index. "
    "Use the memory snippets below. "
    "Answer directly and concisely. If the evidence is insufficient, give the best supported answer and avoid inventing details. "
    "Do not output reasoning or analysis. /no_think"
)
QA_USER_NO_UNKNOWN = """Question: {question}

Evidence:
{evidence}

Provide the most likely concise answer using the evidence. /no_think"""
QA_USER_REASONABLE_GUESS = """Question: {question}

Evidence:
{evidence}

Provide the most likely concise answer using the evidence. If there is insufficient information, you can make reasonable guesses. /no_think"""
QA_USER_BEST_SUPPORTED = """Question: {question}

Memory snippets:
{evidence}

Answer: /no_think"""


def get_answer_prompt(mode: str) -> tuple[str, str]:
    if mode == "reasonable_guess":
        return QA_SYSTEM_REASONABLE_GUESS, QA_USER_REASONABLE_GUESS
    if mode == "best_supported":
        return QA_SYSTEM_BEST_SUPPORTED, QA_USER_BEST_SUPPORTED
    return QA_SYSTEM_NO_UNKNOWN, QA_USER_NO_UNKNOWN


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_qa(path: Path, dataset_id: str) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and dataset_id in obj:
        return list(obj[dataset_id].get("qa_list", []))
    if isinstance(obj, dict) and "qa_list" in obj:
        return list(obj.get("qa_list", []))
    if isinstance(obj, dict) and obj:
        first = next(iter(obj.values()))
        if isinstance(first, dict):
            return list(first.get("qa_list", []))
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Cannot load QA list from {path}")


def clip_number_from_id(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    if text == "clip_000":
        return 0
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else None


def clip_number_from_doc(text: str) -> int | None:
    match = re.search(r"Clip id:\s*(clip_\d+)", text)
    if not match:
        match = re.search(r"\b(clip_\d+)\b", text)
    return clip_number_from_id(match.group(1)) if match else None


def question_type(qa: dict[str, Any]) -> str:
    typ = qa.get("type") or qa.get("types") or qa.get("category") or "unknown"
    if isinstance(typ, list):
        return str(typ[0]) if typ else "unknown"
    return str(typ)


def should_use_before_clip(qa: dict[str, Any], mode: str, user_profile_mode: str) -> bool:
    if mode == "all":
        return False
    if question_type(qa) in {"user_profile", "Long-Term Identity Profile Inference"}:
        return user_profile_mode == "before_clip"
    return True


def strip_model_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<agent-think>.*?</agent-think>", "", text, flags=re.S).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"<\|.*?\|>", "", text).strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1].strip()
    if "Direct Answer:" in text:
        text = text.split("Direct Answer:")[-1].strip()
    return text.strip()


def build_rag(args: argparse.Namespace, corpus_size: int) -> HippoRAG:
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "sk-local"

    config = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url or None,
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        qa_top_k=args.qa_top_k,
        graph_type=args.graph_type,
        embedding_batch_size=args.embedding_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        force_index_from_scratch=args.force_rebuild,
        force_openie_from_scratch=args.force_openie,
        corpus_len=corpus_size,
    )
    return HippoRAG(global_config=config)


def configure_answerer(args: argparse.Namespace, rag: HippoRAG) -> None:
    if not args.answer_llm_name:
        return
    config = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.answer_llm_name,
        llm_base_url=args.answer_llm_base_url or None,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url or None,
        max_new_tokens=args.answer_max_new_tokens or args.max_new_tokens,
        temperature=args.temperature,
    )
    rag.llm_model = _get_llm_class(config)
    if hasattr(rag, "rerank_filter"):
        rag.rerank_filter.llm_infer_fn = rag.llm_model.infer
        rag.rerank_filter.model_name = config.llm_name


def answer_one(
    rag: HippoRAG,
    qa: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    qid = str(qa.get("question_id") or qa.get("id"))
    question = str(qa.get("question", "")).strip()
    before_clip = qa.get("before_clip")
    query_solutions = rag.retrieve(queries=[question], num_to_retrieve=args.retrieval_top_k)
    query_solution = query_solutions[0]

    docs = [str(doc).strip() for doc in query_solution.docs]
    scores = (
        [float(score) for score in query_solution.doc_scores.tolist()]
        if getattr(query_solution, "doc_scores", None) is not None
        else [None] * len(docs)
    )
    pairs = []
    for doc, score in zip(docs, scores):
        doc_clip = clip_number_from_doc(doc)
        pairs.append((doc, score, doc_clip))

    use_before = should_use_before_clip(qa, args.timeline_mode, args.user_profile_timeline_mode)
    if use_before and before_clip is not None:
        cutoff = clip_number_from_id(before_clip)
        if cutoff is None:
            raise ValueError(f"Cannot parse before_clip: {before_clip!r}")
        filtered = [(doc, score, clip) for doc, score, clip in pairs if clip is None or clip <= cutoff]
    else:
        filtered = pairs

    selected = filtered[: args.qa_top_k]
    evidence = "\n\n".join(doc for doc, _score, _clip in selected)
    if not evidence:
        evidence = "No retrieved evidence was selected."
    qa_system, qa_user = get_answer_prompt(args.answer_prompt_mode)
    messages = [
        {"role": "system", "content": qa_system},
        {"role": "user", "content": qa_user.format(question=question, evidence=evidence)},
    ]
    answer_max_tokens = args.answer_max_new_tokens or args.max_new_tokens
    raw_response, usage, _cache_hit = rag.llm_model.infer(
        messages=messages,
        max_tokens=answer_max_tokens,
    )
    response = strip_model_answer(raw_response)

    return {
        "id": qid,
        "question": question,
        "answer": qa.get("answer") or qa.get("reference_answer", ""),
        "response": response,
        "raw_response": raw_response,
        "type": qa.get("type") or qa.get("category"),
        "difficulty": qa.get("difficulty"),
        "before_clip": before_clip,
        "timeline_mode": args.timeline_mode,
        "user_profile_timeline_mode": args.user_profile_timeline_mode,
        "source_query_clip_id": qa.get("source_query_clip_id"),
        "evidence_clip_ids": qa.get("evidence_clip_ids") or qa.get("evidence_video_ids"),
        "retrieved_clip_nums": [clip for _doc, _score, clip in pairs],
        "selected_clip_nums": [clip for _doc, _score, clip in selected],
        "retrieved_scores": scores,
        "usage": usage,
        "llm_name": args.llm_name,
        "answer_llm_name": args.answer_llm_name or args.llm_name,
        "embedding_name": args.embedding_name,
        "answer_prompt_mode": args.answer_prompt_mode,
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_type: dict[str, int] = {}
    unknown = 0
    for row in rows:
        typ = question_type(row)
        by_type[typ] = by_type.get(typ, 0) + 1
        if str(row.get("response", "")).strip().lower() == "unknown":
            unknown += 1
    return {
        "rows": len(rows),
        "by_type": by_type,
        "unknown_responses": unknown,
        "output": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HippoRAG2 open-ended evaluation on ICM-Bench.")
    parser.add_argument("--stage", choices=["build", "answer", "all"], default="all")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--qa-file", required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--llm-name", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--answer-llm-name", default="")
    parser.add_argument("--answer-llm-base-url", default="")
    parser.add_argument("--answer-max-new-tokens", type=int, default=0)
    parser.add_argument("--embedding-name", default="Transformers/BAAI/bge-large-en-v1.5")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--openai-api-key", default="sk-local")
    parser.add_argument("--retrieval-top-k", type=int, default=80)
    parser.add_argument("--linking-top-k", type=int, default=5)
    parser.add_argument("--qa-top-k", type=int, default=10)
    parser.add_argument("--graph-type", default="facts_and_sim_passage_node_unidirectional")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-openie", action="store_true")
    parser.add_argument("--timeline-mode", choices=["before_clip", "all"], default="before_clip")
    parser.add_argument("--user-profile-timeline-mode", choices=["before_clip", "all"], default="all")
    parser.add_argument(
        "--answer-prompt-mode",
        choices=["no_unknown", "reasonable_guess", "best_supported"],
        default="no_unknown",
        help="Answer prompt variant. Default preserves the prior no_unknown behavior.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    docs_obj = json.loads(corpus_path.read_text(encoding="utf-8"))
    docs = [str(row["text"]) for row in docs_obj]
    qa_items = load_qa(Path(args.qa_file), args.dataset_id)
    if args.limit:
        qa_items = qa_items[: args.limit]

    print(
        f"[setup] stage={args.stage} docs={len(docs)} qa={len(qa_items)} "
        f"llm={args.llm_name} embed={args.embedding_name} save={args.save_dir}",
        flush=True,
    )
    rag = build_rag(args, len(docs))

    if args.stage in {"build", "all"}:
        start = time.time()
        rag.index(docs)
        print(f"[build] done in {time.time() - start:.1f}s", flush=True)
        if args.stage == "build":
            return 0

    configure_answerer(args, rag)

    output_path = Path(args.output)
    done_rows = read_jsonl(output_path)
    done_ids = {str(row.get("id")) for row in done_rows}
    todo = [qa for qa in qa_items if str(qa.get("question_id") or qa.get("id")) not in done_ids]
    print(f"[answer] existing={len(done_rows)} todo={len(todo)} output={output_path}", flush=True)
    for idx, qa in enumerate(todo, start=1):
        try:
            row = answer_one(rag, qa, args)
        except Exception as exc:
            row = {
                "id": str(qa.get("question_id") or qa.get("id")),
                "question": qa.get("question", ""),
                "answer": qa.get("answer") or qa.get("reference_answer", ""),
                "response": "",
                "error": repr(exc),
                "type": qa.get("type") or qa.get("category"),
                "difficulty": qa.get("difficulty"),
                "before_clip": qa.get("before_clip"),
                "llm_name": args.llm_name,
                "embedding_name": args.embedding_name,
            }
        append_jsonl(output_path, row)
        if idx <= 10 or idx % 25 == 0:
            print(
                f"[answer] {len(done_rows) + idx}/{len(qa_items)} {row['id']} "
                f"resp={str(row.get('response', ''))[:80]!r}",
                flush=True,
            )

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    write_json(summary_path, summarize(output_path))
    print(f"[summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
