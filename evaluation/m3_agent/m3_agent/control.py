# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by the ICM-Bench authors in 2026 for benchmark integration.
import re
import os
import sys
import json
import time
import argparse
import multiprocessing
import concurrent.futures
import mmagent.videograph
from mmagent.retrieve import search
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages, get_response_with_retry
from mmagent.prompts import prompt_agent_verify_answer_referencing

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))
if os.environ.get("M3_CONTROL_BATCH_SIZE"):
    processing_config["batch_size"] = int(os.environ["M3_CONTROL_BATCH_SIZE"])
if os.environ.get("M3_CONTROL_TOTAL_ROUND"):
    processing_config["total_round"] = int(os.environ["M3_CONTROL_TOTAL_ROUND"])
model_name = "models/M3-Agent-Control"
eval_model = os.environ.get("M3_EVAL_MODEL", "models/gemini-2.5-pro")
control_backend = os.environ.get("M3_CONTROL_GENERATION_BACKEND", "vllm").strip().lower()
control_gemini_model = os.environ.get("M3_CONTROL_GEMINI_MODEL", "models/gemini-3.5-flash")

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=ground_truth,
                    agent_answer=predict,
                ),
            }   
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(eval_model, messages, timeout=60)
        result = response[0].lower() if response else ""
    except Exception as e:
        print(f"Error verifying qa: {question} | {str(e)}")
        return False
    return True if "yes" in result else False

system_prompt = "You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>."""

tokenizer = None if control_backend == "gemini" else AutoTokenizer.from_pretrained(model_name)
sampling_params = None if control_backend == "gemini" else SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=int(os.environ.get("M3_CONTROL_MAX_TOKENS", "1024")),
)
pattern = r"Action: \[(.*)\].*Content: (.*)"


def format_gemini_control_prompt(conversations):
    """Render M3Agent chat turns as a single text prompt for API control models."""
    chunks = []
    for turn in conversations:
        role = str(turn.get("role", "user")).upper()
        content = str(turn.get("content", ""))
        chunks.append(f"{role}:\n{content}")
    chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def gemini_control_one(conversations):
    prompt = format_gemini_control_prompt(conversations)
    messages = generate_messages([{"type": "text", "content": prompt}])
    response = get_response_with_retry(
        control_gemini_model,
        messages,
        timeout=int(os.environ.get("M3_CONTROL_GEMINI_TIMEOUT", "120")),
    )
    return response[0] if response else ""


def gemini_control_batch(conversations_list):
    workers = int(os.environ.get("M3_CONTROL_GEMINI_WORKERS", "4"))
    workers = max(1, min(workers, len(conversations_list) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(gemini_control_one, conversations_list))


def consumer(data):
    if not data["finish"]:
        before_clip = data.get("before_clip", None)
        response = data["conversations"][-1]["content"]
        match_result = re.search(pattern, response.split("</think>")[-1], re.DOTALL)
        if match_result:
            action = match_result.group(1)
            content = match_result.group(2)
        else:
            action = "Search"
            content = None
        if action == "Answer":
            data["response"] = content
            data["finish"] = True
        else:
            new_memories = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if mem_node is None:
                    search_result = (
                        "Searched knowledge: {}"
                        "\n(The memory graph is missing for this sample.)"
                    )
                    data["conversations"].append({"role": "user", "content": search_result})
                    return data
                if before_clip is not None:
                    mem_node.truncate_memory_by_clip(before_clip, False)
                mem_node.refresh_equivalences()
                if "character id" in content:
                    memories, _, _ = search(
                        mem_node,
                        content,
                        [],
                        mem_wise=True,
                        topk=20,
                        before_clip=before_clip,
                    )
                    new_memories.update(memories)
                else:
                    memories, currenr_clips, _ = search(
                        mem_node,
                        content,
                        data["currenr_clips"],
                        threshold=0.5,
                        topk=processing_config["topk"],
                        before_clip=before_clip,
                    )
                    data["currenr_clips"] = currenr_clips
                    new_memories.update(memories)
            search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            if len(new_memories) == 0:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/robot.json")
    parser.add_argument("--list_file", type=str, default=None, help="Optional file with video IDs (one per line) to evaluate")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs for vLLM (default: 2)")
    parser.add_argument("--output_name", type=str, default=None, help="Optional output filename (without extension) for results")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Override clip retrieval top-k for normal search.",
    )
    args = parser.parse_args()
    if args.topk is not None:
        processing_config["topk"] = args.topk
    dataset_name = args.data_file.split("/")[-1].split(".")[0]
    if args.output_name:
        dataset_name = args.output_name
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
    llm_kwargs = {}
    if os.environ.get("M3_VLLM_GPU_MEMORY_UTILIZATION"):
        llm_kwargs["gpu_memory_utilization"] = float(
            os.environ["M3_VLLM_GPU_MEMORY_UTILIZATION"]
        )
    if os.environ.get("M3_VLLM_MAX_MODEL_LEN"):
        llm_kwargs["max_model_len"] = int(os.environ["M3_VLLM_MAX_MODEL_LEN"])
    if os.environ.get("M3_VLLM_MAX_NUM_SEQS"):
        llm_kwargs["max_num_seqs"] = int(os.environ["M3_VLLM_MAX_NUM_SEQS"])
    if os.environ.get("M3_VLLM_MAX_NUM_BATCHED_TOKENS"):
        llm_kwargs["max_num_batched_tokens"] = int(
            os.environ["M3_VLLM_MAX_NUM_BATCHED_TOKENS"]
        )
    if control_backend == "gemini":
        model = None
        print(
            f"[M3 eval] using Gemini control backend: {control_gemini_model}",
            flush=True,
        )
    else:
        model = LLM(
            model=model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            **llm_kwargs,
        )

    batched_datas, data = [], []
    datas = json.load(open(args.data_file))
    id_list = None
    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            id_list = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        missing = [vid for vid in id_list if vid not in datas]
        if missing:
            print(f"[WARN] {len(missing)} IDs not found in {args.data_file}: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
    items = ((vid, datas[vid]) for vid in id_list if vid in datas) if id_list else datas.items()
    for _, v in items:
        for qa in v["qa_list"]:
            data.append({
                "id": qa["question_id"],
                "mem_path": v["mem_path"],
                "question": qa["question"],
                "answer": qa["answer"],
            })
            if "before_clip" in qa:
                data[-1]["before_clip"] = qa["before_clip"]
            if len(data) == processing_config["batch_size"]:
                batched_datas.append(data)
                data = []
    if len(data) > 0:
        batched_datas.append(data)

    with open(output_path, "w") as f:
        pass

    result = []
    for batch_idx, batched_data in enumerate(batched_datas, start=1):
        print(
            f"[M3 eval] batch {batch_idx}/{len(batched_datas)} "
            f"size={len(batched_data)}",
            flush=True,
        )
        for i in range(len(batched_data)):
            batched_data[i]["conversations"] = [{"role": "system", "content": system_prompt.format(question=batched_data[i]["question"])}, {"role": "user", "content": "Searched knowledge: {}"}]
            batched_data[i]["finish"] = False
            batched_data[i]["currenr_clips"] = []

        for idx in range(processing_config["total_round"]):
            print(
                f"[M3 eval] batch {batch_idx} round "
                f"{idx + 1}/{processing_config['total_round']}",
                flush=True,
            )
            vllm_inputs = []
            gemini_inputs = []
            for data in batched_data:
                if data["finish"]:
                    continue
                data["conversations"][-1]["content"] += instruction
                if idx == processing_config["total_round"] - 1:
                    data["conversations"][-1]["content"] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                if control_backend == "gemini":
                    gemini_inputs.append(data["conversations"])
                else:
                    text = tokenizer.apply_chat_template(
                        data["conversations"],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=os.environ.get("M3_CONTROL_ENABLE_THINKING", "1") != "0"
                    )
                    vllm_inputs.append({"prompt_token_ids": text})

            if control_backend == "gemini":
                outputs = gemini_control_batch(gemini_inputs)
            else:
                outputs = model.generate(
                    prompts=vllm_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            print(
                f"[M3 eval] batch {batch_idx} round {idx + 1} "
                f"generated={len(outputs)}",
                flush=True,
            )

            i = 0
            for data in batched_data:
                if data["finish"]:
                    continue
                if control_backend == "gemini":
                    assistant_text = outputs[i]
                else:
                    assistant_text = outputs[i].outputs[0].text
                data["conversations"].append({"role": "assistant", "content": assistant_text})
                i += 1
            assert i == (len(gemini_inputs) if control_backend == "gemini" else len(vllm_inputs))
            
            with multiprocessing.Pool() as pool:
                batched_data = pool.map(consumer, batched_data)
            print(
                f"[M3 eval] batch {batch_idx} round {idx + 1} "
                "retrieval done",
                flush=True,
            )

        batch_lines = []
        for data in batched_data:
            if "response" in data:
                data["gpt_eval"] = eval_answer(data["question"], data["response"], data["answer"])
                time.sleep(0.5)
            else:
                data["gpt_eval"] = False
            line = json.dumps(data, ensure_ascii=False) + '\n'
            result.append(line)
            batch_lines.append(line)
        with open(output_path, "a") as f:
            for line in batch_lines:
                f.write(line)
        print(f"[M3 eval] batch {batch_idx} written", flush=True)
