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
import base64
import json
import logging
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .utils.chat_api import (
    generate_messages as generate_api_messages,
    get_response_with_retry as get_api_response_with_retry,
    parallel_get_embedding,
)
from .utils.chat_qwen import generate_messages, get_response
from .utils.general import validate_and_fix_json
from .prompts import prompt_generate_memory_with_ids_sft
from .memory_processing import parse_video_caption

processing_config = json.load(open("configs/processing_config.json"))
logging_level = processing_config["logging"]

MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

def _build_voices_input(voices_list, include_speaker_meta=False):
    voices_input = {}
    for id, voices in voices_list.items():
        if len(voices) == 0:
            continue
        packed = []
        for voice in voices:
            row = {
                "start_time": voice["start_time"],
                "end_time": voice["end_time"],
                "asr": voice["asr"],
            }
            if include_speaker_meta:
                row["speaker_id"] = voice.get("speaker_id", f"<voice_{id}>")
                if "speaker_confidence" in voice:
                    row["speaker_confidence"] = voice["speaker_confidence"]
            packed.append(row)
        voices_input[f"<voice_{id}>"] = packed
    return voices_input

def generate_video_context(
    base64_frames, faces_list, voices_list, video_path=None, faces_input="face_only"
):
    face_frames = []
    face_only = []

    # Iterate through faces directly
    for char_id, faces in faces_list.items():
        if len(faces) == 0:
            continue
        face = faces[0]
        frame_id = face["frame_id"]
        if 0 <= frame_id < len(base64_frames):
            frame_base64 = base64_frames[frame_id]

            # Convert base64 to PIL Image
            frame_bytes = base64.b64decode(frame_base64)
            frame_img = Image.open(BytesIO(frame_bytes))
            draw = ImageDraw.Draw(frame_img)

            # Draw current face
            bbox = face["bounding_box"]
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=(0, 255, 0), width=4
            )

            # Convert back to base64
            buffered = BytesIO()
            frame_img.save(buffered, format="JPEG")
            frame_base64 = base64.b64encode(buffered.getvalue()).decode()
            face_frames.append((f"<face_{char_id}>:", frame_base64))
        face_only.append((f"<face_{char_id}>:", face["extra_data"]["face_base64"]))
    
    if faces_input == "face_only":
        faces_input = face_only
    elif faces_input == "face_frames":
        faces_input = face_frames
    else:
        raise ValueError(f"Invalid face input: {faces_input}")
    
    num_faces = len(faces_input)
    if num_faces == 0:
        logger.warning("No qualified faces detected")
    
    # Visualize face frames with IDs
    if logging_level == "DETAIL" and num_faces > 0:
        num_rows = (num_faces + 2) // 3  # Round up division to get number of rows needed

        _, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.ravel()  # Flatten axes array for easier indexing

        for i, face_pic in enumerate(faces_input):
            # Convert base64 to image array
            img_bytes = base64.b64decode(face_pic[1])
            img_array = np.array(Image.open(BytesIO(img_bytes)))

            axes[i].imshow(img_array)
            axes[i].set_title(face_pic[0])
            axes[i].axis("off")

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    voices_input = _build_voices_input(voices_list)
    
    num_voices = len(voices_input)
    if num_voices == 0:
        logger.warning("No qualified voices detected")

    if logging_level == "DETAIL" and num_voices > 0:
        logger.debug(f"Diarized dialogues: {voices_input}")

    video_mode = os.environ.get("M3_QWEN_VIDEO_MODE", "video").strip().lower()
    if video_mode == "frames":
        frame_limit = max(1, int(os.environ.get("M3_QWEN_FRAME_LIMIT", "4")))
        frame_max_side = max(128, int(os.environ.get("M3_QWEN_FRAME_MAX_SIDE", "448")))

        def _resize_frame(frame_b64):
            try:
                frame_bytes = base64.b64decode(frame_b64)
                frame_img = Image.open(BytesIO(frame_bytes)).convert("RGB")
                width, height = frame_img.size
                scale = min(frame_max_side / max(width, height), 1.0)
                if scale < 1.0:
                    frame_img = frame_img.resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                buffered = BytesIO()
                frame_img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode()
            except Exception:
                return frame_b64

        if len(base64_frames) <= frame_limit:
            selected_frames = list(base64_frames)
        else:
            indexes = np.linspace(0, len(base64_frames) - 1, frame_limit, dtype=int)
            selected_frames = [base64_frames[int(index)] for index in indexes]
        selected_frames = [_resize_frame(frame) for frame in selected_frames]
        video_context = [
            {
                "type": "text",
                "content": "Selected scene frames sampled from the video clip:",
            },
            {
                "type": "images/jpeg",
                "content": selected_frames,
            },
        ]
    else:
        video_context = [
            {
                "type": "video_base64/mp4",
                "content": video_path,
            },
        ]

    video_context.extend(
        [
            {
                "type": "text",
                "content": "Face features:",
            },
            {
                "type": "images/jpeg",
                "content": faces_input,
            },
            {
                "type": "text",
                "content": "Voice features:",
            },
            {
                "type": "text",
                "content": json.dumps(voices_input),
            },
        ]
    )

    return video_context

def generate_audio_context(voices_list):
    voices_input = _build_voices_input(voices_list, include_speaker_meta=False)
    num_voices = len(voices_input)
    if num_voices == 0:
        logger.warning("No qualified voices detected")
    if logging_level == "DETAIL" and num_voices > 0:
        logger.debug(f"Diarized dialogues: {voices_input}")
    audio_context = [
        {
            "type": "text",
            "content": "Voice features:"
        },
        {
            "type": "text",
            "content": json.dumps(voices_input),
        }
    ]
    return audio_context


def generate_audio_context_speaker_aware(voices_list):
    voices_input = _build_voices_input(voices_list, include_speaker_meta=True)
    num_voices = len(voices_input)
    if num_voices == 0:
        logger.warning("No qualified voices detected")
    if logging_level == "DETAIL" and num_voices > 0:
        logger.debug(f"Speaker-aware diarized dialogues: {voices_input}")
    audio_context = [
        {
            "type": "text",
            "content": "Voice features with speaker tracking metadata:",
        },
        {
            "type": "text",
            "content": json.dumps(voices_input),
        },
    ]
    return audio_context

def generate_all_memories(
    video_context,
    model_type="sft",
    video_path=None,
    prompt_text=prompt_generate_memory_with_ids_sft,
):
    input = [
        {
            "type": "text",
            "content": prompt_text,
        },
    ] + video_context
    
    backend = os.environ.get("M3_MEMORY_GENERATION_BACKEND", "qwen").strip().lower()
    if backend == "gemini":
        messages = generate_api_messages(input)
        gemini_model = os.environ.get("M3_MEMORY_GEMINI_MODEL", "models/gemini-2.5-pro")
    else:
        messages = generate_messages(input)
    # prompt 里是 video_description（单数），代码兼容两种
    epi_key_alt = "video_description"
    epi_key = "video_descriptions"
    sem_key = "high_level_conclusions"
    
    memories = None
    correction_added = False
    last_raw_response = None
    for i in range(MAX_RETRIES):
        if backend == "gemini":
            memories_string = get_api_response_with_retry(gemini_model, messages, timeout=120)[0]
        else:
            memories_string = get_response(messages)[0]
        if not memories_string:
            logger.warning(f"Empty memory response, retry {i + 1}/{MAX_RETRIES}")
            continue
        last_raw_response = memories_string
        memories = validate_and_fix_json(memories_string)
        if not isinstance(memories, dict):
            logger.warning(f"Invalid memory format, retry {i + 1}/{MAX_RETRIES}")
            memories = None
            continue
        has_epi = epi_key in memories or epi_key_alt in memories
        if not has_epi or sem_key not in memories:
            logger.warning(f"Missing memory keys, retry {i + 1}/{MAX_RETRIES}")
            if not correction_added:
                correction_text = (
                    "Your previous response was missing required keys. "
                    "Return ONLY valid JSON with keys "
                    "\"video_descriptions\" and \"high_level_conclusions\". "
                    "If you cannot infer any high-level conclusions, return "
                    "an empty list for \"high_level_conclusions\"."
                )
                messages = messages + [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": correction_text,
                            }
                        ],
                    }
                ]
                correction_added = True
            memories = None
            continue
        break
    if memories is None:
        if last_raw_response is not None:
            if video_path:
                logger.error(
                    "Failed to generate memories with required keys for %s. "
                    "Last raw response (full): %s",
                    video_path,
                    last_raw_response,
                )
            else:
                logger.error(
                    "Failed to generate memories with required keys. "
                    "Last raw response (full): %s",
                    last_raw_response,
                )
        failure_log_path = "logs/memory_failures.jsonl"
        try:
            os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
            with open(failure_log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "video_path": video_path,
                            "reason": "missing_required_keys",
                            "last_raw_response": last_raw_response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as write_error:
            logger.error("Failed to write memory failure log: %s", write_error)
        # Fallback: return empty lists instead of raising to keep the pipeline running.
        logger.warning("Fallback to empty memories due to missing required keys.")
        memories = {
            epi_key: [],
            sem_key: [],
        }

    episodic_memories = memories.get(epi_key_alt) or memories.get(epi_key)
    semantic_memories = memories[sem_key]
    
    return episodic_memories, semantic_memories

def generate_memories(
    base64_frames, faces_list, voices_list, video_path, model_type="sft"
):
    video_context = generate_video_context(base64_frames, faces_list, voices_list, video_path)
    episodic_memories, semantic_memories = generate_all_memories(
        video_context,
        model_type,
        video_path=video_path,
    )
    return episodic_memories, semantic_memories

def generate_memories_audio_only(voices_list, video_path, model_type="sft"):
    audio_context = generate_audio_context(voices_list)
    episodic_memories, semantic_memories = generate_all_memories(
        audio_context,
        model_type,
        video_path=video_path,
    )
    return episodic_memories, semantic_memories


def generate_memories_audio_only_speaker_aware(voices_list, video_path, model_type="sft"):
    audio_context = generate_audio_context_speaker_aware(voices_list)
    speaker_hint = (
        "Additional instruction for speaker-aware audio memory generation:\n"
        "- Each utterance may include `speaker_id` and `speaker_confidence`.\n"
        "- Prefer high-confidence speaker evidence when inferring identity, relationships, and attributions.\n"
        "- For uncertain speaker attribution, explicitly avoid over-claiming.\n"
        "- Keep all entity tags in the existing format (e.g., <voice_x>).\n"
    )
    episodic_memories, semantic_memories = generate_all_memories(
        audio_context,
        model_type,
        video_path=video_path,
        prompt_text=prompt_generate_memory_with_ids_sft + "\n\n" + speaker_hint,
    )
    return episodic_memories, semantic_memories

def process_memories(video_graph, memory_contents, clip_id, type='episodic'):
    def normalize_memory_text(memory):
        if isinstance(memory, str):
            return memory
        if isinstance(memory, dict):
            contents = memory.get("contents")
            if isinstance(contents, list) and contents:
                return str(contents[0])
            if "text" in memory:
                return str(memory["text"])
        return str(memory)

    def get_memory_embeddings(memory_contents):
        # calculate the embedding for each memory
        model = "gemini-embedding-001"
        memory_texts = [normalize_memory_text(memory) for memory in memory_contents]
        embeddings = parallel_get_embedding(model, memory_texts)[0]
        return embeddings

    def insert_memory(video_graph, memory, type='episodic'):
        # create a new text node for each memory
        new_node_id = video_graph.add_text_node(memory, clip_id, type)
        entities = parse_video_caption(video_graph, memory['contents'][0])
        for entity in entities:
            video_graph.add_edge(new_node_id, entity[1])

    def update_video_graph(video_graph, memories, type='episodic'):
        # append all episodic memories to the graph
        if type == 'episodic':
            # create a new text node for each memory
            for memory in memories:
                insert_memory(video_graph, memory, type)
        # semantic memories can be used to update the existing text nodes, or create new text nodes
        elif type == 'semantic':
            for memory in memories:
                entities = parse_video_caption(video_graph, memory['contents'][0])

                if len(entities) == 0:
                    insert_memory(video_graph, memory, type)
                    continue
                
                # update the existing text node for each memory, if needed
                positive_threshold = 0.85
                negative_threshold = 0
                
                # get all (possible) related nodes            
                node_id = entities[0][1]
                related_nodes = video_graph.get_connected_nodes(node_id, type=['semantic'])
                
                # if there is a node with similarity > positive_threshold, then update the edge weight by +1
                # if there is a node with similarity < negative_threshold, then update the edge weight by -1, and add a new text node and connect it to the existing node
                # otherwise, add a new text node and connect it to the existing node
                create_new_node = True
                
                for node_id in related_nodes:
                    # related nodes to be updated should satisfy two condtions:
                    # 1. the caption entities are a subset of the existing node entities
                    # 2. the semantic similarity between the memory and the existing node shows a positive correlation or a negative correlation
                    
                    # see if the memory entities are a subset of the existing node entities
                    related_node_entities = parse_video_caption(video_graph, video_graph.nodes[node_id].metadata['contents'][0])
                    embedding = video_graph.nodes[node_id].embeddings[0]
                    if all(entity in related_node_entities for entity in entities):
                        similarity = np.dot(memory['embeddings'][0], embedding) / (np.linalg.norm(memory['embeddings'][0]) * np.linalg.norm(embedding))
                        if similarity > positive_threshold:
                            video_graph.reinforce_node(node_id)
                            create_new_node = False
                        elif similarity < negative_threshold:
                            video_graph.weaken_node(node_id)
                            create_new_node = False
                
                if create_new_node:
                    insert_memory(video_graph, memory, type)
    
    memories_embeddings = get_memory_embeddings(memory_contents)

    memories = []
    for memory, embedding in zip(memory_contents, memories_embeddings):
        memories.append({
            'contents': [memory],
            'embeddings': [embedding]
        })

    update_video_graph(video_graph, memories, type)
    
