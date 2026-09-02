#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icm_runtime import activate_upstream, load_vgent_class  # noqa: E402

activate_upstream()
from models.utils import fetch_video, resize_video  # noqa: E402
from utils.retrieval import compute_text_similarity  # noqa: E402

Vgent = load_vgent_class()


def load_qa(qa_path: Path) -> list[dict[str, Any]]:
    if qa_path.suffix == ".jsonl":
        return [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.load(qa_path.open("r", encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "qa_list" in payload:
            return payload["qa_list"]
        first = next(iter(payload.values()))
        if isinstance(first, dict) and "qa_list" in first:
            return first["qa_list"]
    raise ValueError(f"Unsupported QA format: {qa_path}")


def clip_num_from_name(name: str) -> int:
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        raise ValueError(f"Cannot parse clip number from {name}")
    return int(m.group(1))


def graph_node_text(node_data: dict[str, Any]) -> str:
    parts = []
    for key in ("entities", "actions", "scenes", "subtitles"):
        values = node_data.get(key)
        if not values:
            continue
        if isinstance(values, str):
            values = [values]
        values = [str(v).strip() for v in values if str(v).strip()]
        if values:
            parts.append(f"{key}: " + "; ".join(values))
    return "\n".join(parts)


def build_docs(graph_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    video_by_name: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            video_by_name[item["video_name"]] = item["video_path"]

    docs = []
    for pkl_path in sorted(graph_dir.glob("*.pkl"), key=lambda p: clip_num_from_name(p.name)):
        clip_num = clip_num_from_name(pkl_path.name)
        video_name = f"clip_{clip_num:03d}.mp4"
        if video_name not in video_by_name:
            continue
        saved = pickle.load(pkl_path.open("rb"))
        graph = saved.get("video_graph")
        if graph is None:
            continue
        for node_id, node_data in graph.nodes(data=True):
            text = graph_node_text(node_data)
            if not text:
                continue
            docs.append(
                {
                    "clip_num": clip_num,
                    "video_name": video_name,
                    "video_path": video_by_name[video_name],
                    "node_id": int(node_id),
                    "text": text,
                }
            )
    return docs


def qtype_of(qa: dict[str, Any]) -> str:
    qtype = qa.get("type") or qa.get("category") or "unknown"
    if isinstance(qtype, list):
        return qtype[0] if qtype else "unknown"
    return str(qtype)


def allowed_docs(docs: list[dict[str, Any]], qa: dict[str, Any]) -> list[dict[str, Any]]:
    if qtype_of(qa) in {"user_profile", "Long-Term Identity Profile Inference"}:
        return docs
    before_clip = qa.get("before_clip")
    if before_clip is None:
        return docs
    cutoff = clip_num_from_name(str(before_clip))
    return [doc for doc in docs if doc["clip_num"] <= cutoff]


def retrieve(vgent: Vgent, qa: dict[str, Any], docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    pool = allowed_docs(docs, qa)
    if not pool:
        return []
    question = qa["question"]
    query_list = [question]
    # Keep Vgent's query decomposition when it parses cleanly, but do not let
    # that step block album QA. Our questions are open-ended and have no options.
    try:
        extracted, _ = vgent.extract_keywords(question, [], [torch.empty(0)])
        query_list = list(dict.fromkeys([q for q in extracted + [question] if q]))
    except Exception:
        pass

    key_list = [doc["text"] for doc in pool]
    sims = compute_text_similarity(
        query_list,
        key_list,
        vgent.embedding_model,
        vgent.embedding_tokenizer,
        return_all=True,
    )
    scores = torch.mean(sims, dim=0)
    order = torch.argsort(scores, descending=True).tolist()[:top_k]
    selected = []
    for idx in order:
        doc = dict(pool[idx])
        doc["score"] = float(scores[idx].detach().cpu())
        selected.append(doc)
    return selected


def load_selected_video(selected: list[dict[str, Any]], frames_per_clip: int, total_pixels: int):
    videos = []
    fps_values = []
    for doc in selected:
        request = {"video": doc["video_path"], "nframes": frames_per_clip}
        raw_video, _, fps = fetch_video(request, resize=False)
        videos.append(raw_video)
        fps_values.append(fps)
    if not videos:
        return None, None
    video = torch.cat(videos, dim=0)
    video, fps = resize_video(video, fps_values[0] if fps_values else 1.0, total_pixels=total_pixels * 28 * 28)
    return video, fps


def answer_question(vgent: Vgent, qa: dict[str, Any], selected: list[dict[str, Any]], args) -> str:
    evidence = "\n\n".join(
        f"[clip_{doc['clip_num']:03d}, score={doc['score']:.3f}]\n{doc['text']}"
        for doc in selected
    )
    prompt = (
        "You are answering an open-ended question about a long personal video album. "
        "Relevant clips were retrieved from a graph-based video memory index. "
        "Use the retrieved video frames and the memory snippets below. "
        "Answer directly and concisely. If the evidence is insufficient, give the best supported answer and avoid inventing details.\n\n"
        f"Memory snippets:\n{evidence}\n\n"
        f"Question: {qa['question']}\n"
        "Answer:"
    )
    video, fps = load_selected_video(selected, args.frames_per_clip, args.answer_total_pixels)
    if video is None:
        video = None
    return vgent.mllm_response(
        vgent.video_llm,
        vgent.processor,
        vgent.image_processor,
        prompt,
        None,
        video,
        max_new_tokens=args.max_new_tokens,
        fps=fps,
    ).strip()


def save_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--qa_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--frames_per_clip", type=int, default=12)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--total_pixels", type=int, default=16384)
    parser.add_argument("--answer_total_pixels", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    args = parser.parse_args()
    args.task = "album"
    args.n_retrieval = args.top_k
    args.n_refine = args.top_k
    args.uniform_frame = args.top_k * args.frames_per_clip

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line).get("id"))

    qa_list = load_qa(Path(args.qa_path))
    if args.limit:
        qa_list = qa_list[: args.limit]
    docs = build_docs(Path(args.graph_dir), Path(args.manifest_path))
    if not docs:
        raise RuntimeError(f"No graph docs found under {args.graph_dir}")
    print(f"Loaded {len(docs)} graph docs from {args.graph_dir}", flush=True)

    vgent = Vgent(args)
    vgent.embedding_model.eval()

    for qa in tqdm(qa_list, desc=f"Vgent album eval {args.model_name}"):
        qid = qa["question_id"]
        if qid in done_ids:
            continue
        try:
            selected = retrieve(vgent, qa, docs, args.top_k)
            response = answer_question(vgent, qa, selected, args)
            row = {
                "id": qid,
                "question": qa["question"],
                "answer": qa.get("answer") or qa.get("reference_answer", ""),
                "type": qa.get("type") or qa.get("category"),
                "difficulty": qa.get("difficulty"),
                "before_clip": None if qtype_of(qa) in {"user_profile", "Long-Term Identity Profile Inference"} else qa.get("before_clip"),
                "response": response,
                "retrieved": [
                    {
                        "clip_num": doc["clip_num"],
                        "video_name": doc["video_name"],
                        "node_id": doc["node_id"],
                        "score": doc["score"],
                        "text": doc["text"],
                    }
                    for doc in selected
                ],
                "model_name": args.model_name,
            }
        except Exception as exc:
            row = {
                "id": qid,
                "question": qa.get("question"),
                "answer": qa.get("answer") or qa.get("reference_answer", ""),
                "type": qa.get("type") or qa.get("category"),
                "difficulty": qa.get("difficulty"),
                "before_clip": None if qtype_of(qa) in {"user_profile", "Long-Term Identity Profile Inference"} else qa.get("before_clip"),
                "response": "",
                "error": repr(exc),
                "model_name": args.model_name,
            }
        save_jsonl_row(output_path, row)
        done_ids.add(qid)


if __name__ == "__main__":
    main()
