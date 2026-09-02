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
import mimetypes
import os
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.CRITICAL)
# Disable urllib3 logging (which httpx uses)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
# Disable httpcore logging (which httpx uses)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

# api utils

processing_config = json.load(open("configs/processing_config.json"))
temp = processing_config["temperature"]

try:
    config = json.load(open("configs/api_config.json"))
except Exception:
    config = {}

MAX_RETRIES = 5


def _safe_positive_int(value, default):
    try:
        value = int(value)
        if value > 0:
            return value
    except Exception:
        pass
    return default

def _get_model_config(model):
    if model in config:
        return config[model]
    if model.startswith("models/"):
        return config.get(model[len("models/"):], {})
    return {}


_CLIENT = None
_CLIENT_KEY = None


def _configure_gemini(model):
    model_config = _get_model_config(model)
    api_key = (
        model_config.get("api_key")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise ValueError(f"Missing api_key for model: {model}")
    global _CLIENT, _CLIENT_KEY
    if _CLIENT is None or _CLIENT_KEY != api_key:
        _CLIENT = genai.Client(api_key=api_key)
        _CLIENT_KEY = api_key
    return _CLIENT


def _model_for_embedding(model):
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _decode_base64(data):
    try:
        return base64.b64decode(data)
    except Exception:
        return None


def _build_parts(inputs):
    parts = []
    for input in inputs:
        if not input["content"]:
            logger.warning("empty content, skip")
            continue
        if input["type"] == "text":
            parts.append(types.Part.from_text(text=input["content"]))
        elif input["type"] in ["images/jpeg", "images/png"]:
            img_format = input["type"].split("/")[1]
            if isinstance(input["content"][0], str):
                for img in input["content"]:
                    img_bytes = _decode_base64(img)
                    if img_bytes:
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=f"image/{img_format}"))
            else:
                for img in input["content"]:
                    parts.append(types.Part.from_text(text=img[0]))
                    img_bytes = _decode_base64(img[1])
                    if img_bytes:
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=f"image/{img_format}"))
        elif input["type"] in ["video_base64/mp4", "video_base64/webm"]:
            video_format = input["type"].split("/")[1]
            video_bytes = _decode_base64(input["content"])
            if video_bytes:
                parts.append(types.Part.from_bytes(data=video_bytes, mime_type=f"video/{video_format}"))
        elif input["type"] == "video_url":
            parts.append(types.Part.from_text(text=f"Video URL: {input['content']}"))
        elif input["type"] in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = input["type"].split("/")[1]
            audio_bytes = _decode_base64(input["content"])
            if audio_bytes:
                parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=f"audio/{audio_format}"))
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    return parts


def get_response(model, messages, timeout=30):
    """Get chat completion response from specified model.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
    """
    client = _configure_gemini(model)
    response = client.models.generate_content(
        model=model,
        contents=messages,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=8192,
            system_instruction="You are an expert in video understanding.",
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    total_tokens = getattr(usage, "total_token_count", 0) if usage else 0
    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None):
        content = getattr(response.candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            text = "".join([getattr(p, "text", "") for p in parts])
    return text or "", total_tokens

def get_response_with_retry(model, messages, timeout=30):
    """Retry get_response up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_response(model, messages, timeout)
        except Exception as e:
            global _CLIENT, _CLIENT_KEY
            _CLIENT = None
            _CLIENT_KEY = None
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from message")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_response(model, messages, timeout=30):
    """Process multiple messages in parallel using ThreadPoolExecutor.
    Messages are processed in batches, with each batch completing before starting the next.

    Args:
        model (str): Model identifier
        messages (list): List of message lists to process

    Returns:
        tuple: (list of responses, total tokens used)
    """
    batch_size = _get_model_config(model).get("qpm", len(messages)) or len(messages)
    responses = []
    total_tokens = 0

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            batch_responses = list(executor.map(lambda msg: get_response_with_retry(model, msg, timeout), batch))
            
        # Extract answers and tokens from batch responses
        batch_answers = [response[0] for response in batch_responses]
        batch_tokens = [response[1] for response in batch_responses]
        
        responses.extend(batch_answers)
        total_tokens += sum(batch_tokens)

    return responses, total_tokens


def get_embedding(model, text, timeout=15):
    """Get embedding for text using specified model.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
    """
    client = _configure_gemini(model)
    if not isinstance(text, str):
        if isinstance(text, dict):
            contents = text.get("contents")
            if isinstance(contents, list) and contents:
                text = str(contents[0])
            else:
                text = json.dumps(text, ensure_ascii=False)
        else:
            text = str(text)
    response = client.models.embed_content(
        model=_model_for_embedding(model),
        contents=[text],
    )
    embedding = None
    if hasattr(response, "embeddings") and response.embeddings:
        emb = response.embeddings[0]
        embedding = getattr(emb, "values", None) or getattr(emb, "embedding", None)
    if embedding is None and isinstance(response, dict):
        if "embedding" in response:
            embedding = response["embedding"]
        elif "embeddings" in response and response["embeddings"]:
            embedding = response["embeddings"][0].get("values") or response["embeddings"][0].get("embedding")
    usage = getattr(response, "usage_metadata", None)
    total_tokens = getattr(usage, "total_token_count", 0) if usage else 0
    return embedding, total_tokens


def get_embedding_with_retry(model, text, timeout=15):
    """Retry get_embedding up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_embedding(model, text, timeout)
        except Exception as e:
            global _CLIENT, _CLIENT_KEY
            _CLIENT = None
            _CLIENT_KEY = None
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from get embedding")
            continue
    raise Exception(f"Failed to get embedding after {MAX_RETRIES} retries")

def parallel_get_embedding(model, texts, timeout=15):
    """Process multiple texts in parallel to get embeddings.

    Args:
        model (str): Model identifier
        texts (list): List of texts to embed

    Returns:
        tuple: (list of embeddings, total tokens used)
    """
    model_config = _get_model_config(model)
    batch_size = model_config.get("qpm", len(texts)) or len(texts)
    batch_size = _safe_positive_int(os.getenv("EMBEDDING_BATCH_SIZE"), batch_size)
    max_workers_cap = model_config.get("embedding_max_workers")
    if max_workers_cap is None:
        max_workers_cap = os.getenv("EMBEDDING_MAX_WORKERS", 4)
    max_workers_cap = _safe_positive_int(max_workers_cap, 4)
    embeddings = []
    total_tokens = 0
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_workers = min(len(batch), max_workers_cap)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda x: get_embedding_with_retry(model, x, timeout), batch))
            
        # Split batch results into embeddings and tokens
        batch_embeddings = [result[0] for result in results]
        batch_tokens = [result[1] for result in results]
        
        embeddings.extend(batch_embeddings)
        total_tokens += sum(batch_tokens)
        
    return embeddings, total_tokens

def get_whisper(model, file_path):
    """Transcribe audio file using Whisper model.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
    """
    client = _configure_gemini(model)
    with open(file_path, "rb") as file:
        audio_bytes = file.read()
    mime_type = mimetypes.guess_type(file_path)[0] or "audio/mpeg"
    prompt = "Transcribe the audio to text. Return only the transcription."
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0),
    )
    return getattr(response, "text", None) or ""

def get_whisper_with_retry(model, file_path):
    """Retry Whisper transcription with error handling.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_whisper(model, file_path)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e}")
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_whisper(model, file_paths):
    """Process multiple audio files in parallel using Whisper model.

    Args:
        model (str): Model identifier
        file_paths (list): List of audio file paths

    Returns:
        list: List of transcription results
    """
    batch_size = _get_model_config(model).get("qpm", len(file_paths)) or len(file_paths)
    responses = []
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        max_workers = len(batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_responses = list(executor.map(lambda x: get_whisper_with_retry(model, x), batch))
            
        responses.extend(batch_responses)
        
    return responses

def generate_messages(inputs):
    """Generate message list for chat completion from mixed inputs.

    Args:
        inputs (list): List of input dictionaries with 'type' and 'content' keys
        type can be:
            "text" - text content
            "image/jpeg", "image/png" - base64 encoded images
            "video/mp4", "video/webm" - base64 encoded videos
            "video_url" - video URL
            "audio/mp3", "audio/wav" - base64 encoded audio
        content should be a string for text,
        a list of base64 encoded media for images/video/audio,
        or a string (url) for video_url
        inputs are like: 
        [
            {
                "type": "video_base64/mp4",
                "content": <base64>
            },
            {
                "type": "text",
                "content": "Describe the video content."
            },
            ...
        ]

    Returns:
        list: Formatted messages for chat completion
    """
    return _build_parts(inputs)

def print_messages(messages):
    for item in messages:
        if hasattr(item, "text"):
            logger.debug(item.text)
