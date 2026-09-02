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
import json
import re
import logging
import random
from difflib import SequenceMatcher
from collections import defaultdict
from .utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
    get_embedding_with_retry,
)
from .utils.general import validate_and_fix_python_list
from .prompts import *
from .memory_processing import parse_video_caption
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

_VOICE_NAME_PATTERNS = [
    re.compile(r"<voice_(\d+)>'s\s+name\s+is\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
    re.compile(r"<voice_(\d+)>\s+is\s+named\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
    re.compile(r"<voice_(\d+)>\s+introduces\s+(?:themselves|herself|himself)\s+as\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
]
_CHARACTER_NAME_PATTERNS = [
    re.compile(r"<character_(\d+)>'s\s+name\s+is\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
    re.compile(r"<character_(\d+)>\s+is\s+named\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
]
_CHARACTER_ID_QUERY = re.compile(
    r"what\s+is\s+the\s+character\s+id\s+of\s+(.+?)[\?\.!\s]*$",
    re.IGNORECASE,
)
_CHARACTER_NAME_QUERY = re.compile(
    r"what\s+is\s+the\s+name\s+of\s+(<character_\d+>)[\?\.!\s]*$",
    re.IGNORECASE,
)


def _normalize_person_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9\-\' ]+", "", (name or "").strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _name_to_voice_nodes(video_graph):
    """Build weak supervision map: spoken names -> related voice node IDs."""
    mapping = defaultdict(set)
    name_patterns = [
        ("voice", re.compile(r"<voice_(\d+)>[^\n]*?name is\s+([A-Za-z][A-Za-z\-\' ]{0,40})", re.IGNORECASE)),
        ("character", re.compile(r"<character_(\d+)>[^\n]*?name is\s+([A-Za-z][A-Za-z\-\' ]{0,40})", re.IGNORECASE)),
        ("face", re.compile(r"<face_(\d+)>[^\n]*?name is\s+([A-Za-z][A-Za-z\-\' ]{0,40})", re.IGNORECASE)),
    ]
    for node in video_graph.nodes.values():
        if node.type not in {"semantic", "episodic"}:
            continue
        contents = node.metadata.get("contents", [])
        for text in contents:
            if not isinstance(text, str):
                continue
            for ent_type, pat in name_patterns:
                for m in pat.finditer(text):
                    ent_id = int(m.group(1))
                    name = _normalize_person_name(m.group(2))
                    if not name:
                        continue
                    if ent_type == "voice":
                        if ent_id in video_graph.nodes and video_graph.nodes[ent_id].type == "voice":
                            mapping[name].add(ent_id)
                    elif ent_type == "character":
                        key = f"character_{ent_id}"
                        for tag in video_graph.character_mappings.get(key, []):
                            if tag.startswith("voice_"):
                                voice_id = int(tag.split("_", 1)[1])
                                if voice_id in video_graph.nodes and video_graph.nodes[voice_id].type == "voice":
                                    mapping[name].add(voice_id)
                    else:
                        # Face name evidence: map through current character mapping when possible,
                        # then fallback to direct graph connectivity.
                        face_tag = f"face_{ent_id}"
                        character_id = video_graph.reverse_character_mappings.get(face_tag)
                        if character_id:
                            for tag in video_graph.character_mappings.get(character_id, []):
                                if tag.startswith("voice_"):
                                    voice_id = int(tag.split("_", 1)[1])
                                    if voice_id in video_graph.nodes and video_graph.nodes[voice_id].type == "voice":
                                        mapping[name].add(voice_id)
                        if ent_id in video_graph.nodes and video_graph.nodes[ent_id].type == "img":
                            for voice_id in video_graph.get_connected_nodes(ent_id, type=["voice"]):
                                if voice_id in video_graph.nodes and video_graph.nodes[voice_id].type == "voice":
                                    mapping[name].add(voice_id)
    return mapping


def _build_identity_hint_cache(video_graph):
    cached = getattr(video_graph, "_identity_hint_cache", None)
    if cached is not None:
        return cached

    name_to_characters = defaultdict(set)
    character_to_names = defaultdict(set)

    def add_character_name(character_id, raw_name):
        normalized = _normalize_person_name(raw_name)
        if not normalized:
            return
        name_to_characters[normalized].add(character_id)
        character_to_names[character_id].add(raw_name.strip())

    for character_id, raw_name in getattr(video_graph, "character_names", {}).items():
        add_character_name(character_id, raw_name)

    for voice_id, raw_name in getattr(video_graph, "voice_names", {}).items():
        character_id = video_graph.reverse_character_mappings.get(f"voice_{voice_id}")
        if character_id:
            add_character_name(character_id, raw_name)

    for node in video_graph.nodes.values():
        if node.type not in {"semantic", "episodic"}:
            continue
        contents = node.metadata.get("contents", [])
        for text in contents:
            if not isinstance(text, str):
                continue
            for pattern in _VOICE_NAME_PATTERNS:
                for match in pattern.finditer(text):
                    voice_id = int(match.group(1))
                    raw_name = match.group(2).strip()
                    character_id = video_graph.reverse_character_mappings.get(f"voice_{voice_id}")
                    if character_id:
                        add_character_name(character_id, raw_name)
            for pattern in _CHARACTER_NAME_PATTERNS:
                for match in pattern.finditer(text):
                    character_id = f"character_{int(match.group(1))}"
                    raw_name = match.group(2).strip()
                    add_character_name(character_id, raw_name)

    cached = {
        "name_to_characters": {k: sorted(v) for k, v in name_to_characters.items()},
        "character_to_names": {k: sorted(v) for k, v in character_to_names.items()},
    }
    setattr(video_graph, "_identity_hint_cache", cached)
    return cached


def _score_name_match(query_name, candidate_name):
    query_norm = _normalize_person_name(query_name)
    candidate_norm = _normalize_person_name(candidate_name)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    if len(query_norm) >= 3 and query_norm in candidate_norm:
        return 0.95
    if len(candidate_norm) >= 3 and candidate_norm in query_norm:
        return 0.95
    prefix_len = 0
    for a, b in zip(query_norm, candidate_norm):
        if a != b:
            break
        prefix_len += 1
    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    if prefix_len >= 2 and ratio >= 0.45:
        return ratio
    return 0.0


def get_identity_hints(video_graph, query):
    cache = _build_identity_hint_cache(video_graph)

    character_id_match = _CHARACTER_ID_QUERY.match(query.strip())
    if character_id_match:
        query_name = character_id_match.group(1).strip()
        candidates = []
        for candidate_name, character_ids in cache["name_to_characters"].items():
            score = _score_name_match(query_name, candidate_name)
            if score <= 0:
                continue
            for character_id in character_ids:
                candidates.append((score, candidate_name, character_id))
        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        hints = []
        seen = set()
        for _, candidate_name, character_id in candidates:
            key = (candidate_name, character_id)
            if key in seen:
                continue
            seen.add(key)
            hints.append(f"{query_name} may refer to {character_id}, who may have been named {candidate_name.title()}.")
        return hints

    character_name_match = _CHARACTER_NAME_QUERY.match(query.strip())
    if character_name_match:
        character_token = character_name_match.group(1)
        character_id = character_token.strip("<>")
        names = cache["character_to_names"].get(character_id, [])
        return [f"{character_token} may have been named {name}." for name in names]

    return []


def infer_speaker_nodes_from_query(video_graph, query):
    name_map = getattr(video_graph, "_speaker_name_map_cache", None)
    if name_map is None:
        name_map = _name_to_voice_nodes(video_graph)
        setattr(video_graph, "_speaker_name_map_cache", name_map)
    query_l = f" {_normalize_person_name(query)} "
    if not query_l.strip():
        return set()
    selected = set()
    for name, voice_ids in name_map.items():
        if not name:
            continue
        if f" {name} " in query_l:
            selected.update(voice_ids)
    return selected


def _apply_speaker_bias(video_graph, nodes, speaker_nodes, speaker_bias=0.0, speaker_hard_filter=False):
    if not speaker_nodes:
        return nodes
    rescored = []
    bias = max(0.0, float(speaker_bias))
    for node_id, node_score in nodes:
        connected_voices = set(video_graph.get_connected_nodes(node_id, type=["voice"]))
        hit = bool(connected_voices & speaker_nodes)
        if speaker_hard_filter and not hit:
            continue
        if hit and bias > 0:
            node_score = float(node_score) * (1.0 + bias)
        rescored.append((node_id, float(node_score)))
    if not rescored:
        return nodes
    return sorted(rescored, key=lambda x: x[1], reverse=True)

def translate(video_graph, memories):
    new_memories = []
    for memory in memories:
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        entities = list(set(entities))
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.reverse_character_mappings.keys():
                new_memory = new_memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories

def back_translate(video_graph, queries):
    translated_queries = []
    for query in queries:
        entities = parse_video_caption(video_graph, query)
        entities = list(set(entities))
        to_be_translated = [query]
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.character_mappings.keys():
                mappings = video_graph.character_mappings[entity_str]
                
                # Create new queries for each mapping
                new_queries = []
                for mapping in mappings:
                    for partially_translated in to_be_translated:
                        new_query = partially_translated.replace(entity_str, mapping)
                        new_queries.append(new_query)
                
                # Update translated_query with all variants
                to_be_translated = new_queries
                
        # Add all variants of the translated query
        translated_queries.extend(to_be_translated)
    return translated_queries

# retrieve by clip
def retrieve_from_videograph(
    video_graph,
    query,
    topk=5,
    mode='max',
    threshold=0,
    before_clip=None,
    speaker_nodes=None,
    speaker_bias=0.0,
    speaker_hard_filter=False,
    scene_nodes=None,
    scene_rerank_weight=0.14,
):
    top_clips = []
    # find all CLIP_x in query
    pattern = r"CLIP_(\d+)"
    matches = re.finditer(pattern, query)
    top_clips = []
    for match in matches:
        try:
            clip_id = int(match.group(1))
            top_clips.append(clip_id)
        except ValueError:
            continue
    
    queries = back_translate(video_graph, [query])
    if len(queries) > 100:
        logger.error(f"Anomaly detected from query: {query}, randomly sample 100 translatedqueries")
        queries = random.sample(queries, 100)
    
    related_nodes = get_related_nodes(video_graph, query)

    model = "gemini-embedding-001"
    query_embeddings = parallel_get_embedding(model, queries)[0]

    full_clip_scores = {}
    clip_scores = {}

    if mode not in ['sum', 'max', 'mean']:
        raise ValueError(f"Unknown mode: {mode}")

    # calculate scores for each node
    nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode='max')
    nodes = _apply_speaker_bias(
        video_graph,
        nodes,
        speaker_nodes=speaker_nodes or set(),
        speaker_bias=speaker_bias,
        speaker_hard_filter=speaker_hard_filter,
    )
    
    
    # collect node scores for each clip
    for node_id, node_score in nodes:
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        if clip_id not in full_clip_scores:
            full_clip_scores[clip_id] = []
        full_clip_scores[clip_id].append(node_score)

    # calculate scores for each clip
    for clip_id, scores in full_clip_scores.items():
        if mode == 'sum':
            clip_score = sum(scores)
        elif mode == 'max':
            clip_score = max(scores)
        elif mode == 'mean':
            clip_score = np.mean(scores)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        clip_scores[clip_id] = clip_score

    # Scene Node reranking: boost clips near relevant scene nodes
    if scene_nodes and clip_scores:
        scene_embeddings = np.array([emb for emb, _ in scene_nodes])
        scene_clips = [cid for _, cid in scene_nodes]
        query_emb_mean = np.mean(np.array(query_embeddings), axis=0, keepdims=True)
        scene_sims = cosine_similarity(query_emb_mean, scene_embeddings)[0]
        for sim, scene_clip in zip(scene_sims, scene_clips):
            if sim > 0.3:
                for offset in range(-5, 1):  # scene covers ~6 clips
                    c = scene_clip + offset
                    if c in clip_scores:
                        clip_scores[c] += scene_rerank_weight * sim

    # sort clips by score
    sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
    # filter out clips that have 0 score and get top k clips
    if before_clip is not None:
        top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold and clip_id <= before_clip][:topk]
    else:
        top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold][:topk]
    return top_clips, clip_scores, nodes

def get_related_nodes(video_graph, query):
    related_nodes = []
    entities = parse_video_caption(video_graph, query)
    for entity in entities:
        type = entity[0]
        node_id = entity[1]
        if not (f"{type}_{node_id}" in video_graph.character_mappings.keys() or f"{type}_{node_id}" in video_graph.reverse_character_mappings.keys()):
            continue
        if type == "character":
            related_nodes.extend([int(node.split("_")[1]) for node in video_graph.character_mappings[f"{type}_{node_id}"]])
        else:
            related_nodes.append(node_id)
    return list(set(related_nodes))

def generate_action(question, knowledge, retrieval_plan=None, multiple_queries=False, responses=[], switch=False, model="models/gemini-2.5-pro"):
    # select prompt
    if not switch:
        if multiple_queries:
            prompt = prompt_generate_action_with_plan_multiple_queries
        else:
            prompt = prompt_generate_action_with_plan
            # prompt = prompt_generate_action_with_plan_multiple_queries
    else:
        logger.info(f"Route switch triggered.")
        if multiple_queries:
            prompt = prompt_generate_action_with_plan_multiple_queries_new_direction
        else:
            prompt = prompt_generate_action_with_plan_new_direction
            # prompt = prompt_generate_action_with_plan_multiple_queries_new_direction
    
    input = [
        {
            "type": "text",
            "content": prompt.format(
                question=question,
                knowledge=knowledge,
                retrieval_plan=retrieval_plan,
            )
        }
    ]
    messages = generate_messages(input)
    action_type = None
    action_content = None
    for i in range(MAX_RETRIES):
        action = get_response_with_retry(model, messages)[0]
        if "[ANSWER]" in action:
            action_type = "answer"
            reasoning = action.split("[ANSWER]")[0].strip()
            action_content = action.split("[ANSWER]")[1].strip()
        elif "[SEARCH]" in action:
            if not multiple_queries:
                action_type = "search"
                reasoning = action.split("[SEARCH]")[0].strip()
                action_content = action.split("[SEARCH]")[1].strip() 
            else:
                action_type = "search"
                reasoning = action.split("[SEARCH]")[0].strip()
                action_content = select_queries(validate_and_fix_python_list(action.split("[SEARCH]")[1].strip()), responses)
        else:
            raise ValueError(f"Unknown action type: {action}")
        if action_content is not None:
            break
    if action_content is None:
        raise Exception("Failed to generate action")
    return reasoning, action_type, action_content

def select_queries(action_content, responses):
    if not action_content:
        return None
    
    history_queries = [response["action_content"] for response in responses]
    history_embeddings = parallel_get_embedding("gemini-embedding-001", history_queries)[0]
    
    queries = action_content
    embeddings = parallel_get_embedding("gemini-embedding-001", queries)[0]
    
    # If there are no history queries, return the first query
    if not history_queries:
        return queries[0]
    
    # Calculate cosine similarity between each query and all history queries
    avg_similarities = []
    for query_embedding in embeddings:
        similarities = []
        for history_embedding in history_embeddings:
            # Compute cosine similarity
            dot_product = sum(a*b for a,b in zip(query_embedding, history_embedding))
            query_norm = sum(a*a for a in query_embedding) ** 0.5
            history_norm = sum(b*b for b in history_embedding) ** 0.5
            cos_sim = dot_product / (query_norm * history_norm)
            similarities.append(cos_sim)
        # Calculate average similarity for this query
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarities.append(avg_similarity)
    
    # Return query with lowest average similarity
    min_similarity_idx = avg_similarities.index(min(avg_similarities))
    return queries[min_similarity_idx]

def search(
    video_graph,
    query,
    current_clips,
    topk=5,
    mode='max',
    threshold=0,
    mem_wise=False,
    before_clip=None,
    episodic_only=False,
    speaker_aware=False,
    speaker_bias=0.0,
    speaker_hard_filter=False,
    scene_nodes=None,
    scene_rerank_weight=0.3,
):
    speaker_nodes = set()
    if speaker_aware:
        speaker_nodes = infer_speaker_nodes_from_query(video_graph, query)
    top_clips, clip_scores, nodes = retrieve_from_videograph(
        video_graph,
        query,
        topk,
        mode,
        threshold,
        before_clip,
        speaker_nodes=speaker_nodes,
        speaker_bias=speaker_bias,
        speaker_hard_filter=speaker_hard_filter,
        scene_nodes=scene_nodes,
        scene_rerank_weight=scene_rerank_weight,
    )
    
    if mem_wise:
        new_memories = {}
        top_nodes_num = 0
        # fetch top nodes
        for top_node, _ in nodes:
            clip_id = video_graph.nodes[top_node].metadata['timestamp']
            if before_clip is not None and clip_id > before_clip:
                continue
            if clip_id not in new_memories:
                new_memories[clip_id] = []
            new_ = translate(video_graph, video_graph.nodes[top_node].metadata['contents'])
            new_memories[clip_id].extend(new_)
            top_nodes_num += len(new_)
            if top_nodes_num >= topk:
                break
        # sort related_memories by timestamp
        new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
        new_memories = {f"CLIP_{k}": v for k, v in new_memories.items() if len(v) > 0}
        return new_memories, current_clips, clip_scores
    
    new_clips = [top_clip for top_clip in top_clips if top_clip not in current_clips]
    new_memories = {}
    current_clips.extend(new_clips)
    
    for new_clip in new_clips:
        if new_clip not in video_graph.text_nodes_by_clip:
            new_memories[new_clip] = [f"CLIP_{new_clip} not found in memory bank, please search for other information"]
        else:
            related_nodes = video_graph.text_nodes_by_clip[new_clip]
            new_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes if (not episodic_only or video_graph.nodes[node_id].type != "semantic")])
                        
    # sort related_memories by timestamp
    new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
    new_memories = {f"CLIP_{k}": v for k, v in new_memories.items()}
    
    return new_memories, current_clips, clip_scores

def answer_with_retrieval(video_graph, question, video_clip_base64=None, topk=5, auto_refresh=False, mode='max', multiple_queries=False, max_retrieval_steps=10, route_switch=True, threshold=0, model="models/gemini-2.5-pro", before_clip=None):
    if before_clip is not None:
        video_graph.truncate_memory_by_clip(before_clip)
    
    if auto_refresh:
        video_graph.refresh_equivalences()
        
    related_clips = []
    context = []

    final_answer = None
    
    memories = [[]]
    responses = []
    
    if video_clip_base64 is not None:
        input = [
            {
                "type": "video_base64/mp4",
                "content": video_clip_base64,
            },
            {
                "type": "text",
                "content": prompt_generate_plan.format(question=question),
            }
        ]

        messages = generate_messages(input)
        plan_model = "gemini-1.5-pro-002"
        retrieval_plan = get_response_with_retry(plan_model, messages)[0]
        logger.info(f"Retrieval plan: {retrieval_plan}")
    else:
        retrieval_plan = None
        
    switch = False
    for i in range(max_retrieval_steps):
        # reasoning, action_type, action_content = generate_action(question, context, retrieval_plan)
        reasoning, action_type, action_content = generate_action(question, context, retrieval_plan, multiple_queries=multiple_queries, responses=responses, switch=switch, model=model)
        reasoning = reasoning.strip("### Reasoning:").strip("### Answer or Search:").strip("Reasoning:").strip()
        if action_type == "answer":
            final_answer = action_content
            responses.append({
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            })
            logger.info(f"Answer: {final_answer}")
            break
        elif action_type == "search":
            if i == max_retrieval_steps - 1:
                input = [
                    {
                        "type": "text",
                        "content": prompt_answer_with_retrieval_final.format(
                            question=question,
                            information=context,
                        ),
                    }
                ]
                messages = generate_messages(input)
                resp = get_response_with_retry(model, messages)[0]
                reasoning = resp.split("[ANSWER]")[0].strip()
                final_answer = resp.split("[ANSWER]")[1].strip()
                responses.append({
                    "reasoning": reasoning,
                    "action_type": "answer",
                    "action_content": final_answer
                })
                logger.info(f"Forced answer: {final_answer}")
                break
            
            new_memories, related_clips, _ = search(video_graph, action_content, related_clips, topk, mode, threshold=threshold, before_clip=before_clip)
            
            if len(new_memories.items()) == 0 and route_switch:
                switch = True
            else:
                switch = False
            
            context.append({
                "reasoning": reasoning,
                "query": action_content,
                "retrieved memories": new_memories
            })
            
            new_response_item = {
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            }
            responses.append(new_response_item)
            
            new_memory_items = [{
                "clip_id": k,
                "memory": v
            } for k, v in new_memories.items()]
            memories.append(new_memory_items)
            
            if processing_config["logging"] == "DETAIL":
                logger.debug("=" * 10 + "Retrieval Step " + str(i+1) + "=" * 10)
                logger.debug(new_response_item)
                logger.debug(new_memory_items)
            
    return final_answer, (memories, responses)

def verify_qa(question, gt, pred, model="models/gemini-2.5-pro"):
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=gt,
                    agent_answer=pred,
                ),
            }   
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(model, messages)
        result = response[0]
    except Exception as e:
        logger.error(f"Error verifying qa: {question}")
        logger.error(str(e))
        return None
    return result

def calculate_similarity(mem, query, related_nodes):
    related_nodes_embeddings = np.array([mem.nodes[node_id].embeddings[0] for node_id in related_nodes])
    query_embedding = np.array(get_embedding_with_retry("gemini-embedding-001", query)[0]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, related_nodes_embeddings)[0]
    return similarities.tolist()

def retrieve_all_episodic_memories(video_graph):
    episodic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "episodic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in episodic_memories:
                episodic_memories[clips_id] = []
            episodic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return episodic_memories

def retrieve_all_semantic_memories(video_graph):
    semantic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "semantic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in semantic_memories:
                semantic_memories[clips_id] = []
            semantic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return semantic_memories
