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
import struct
import json
import os
import logging
import re
import torchaudio
import torch
from io import BytesIO
from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import

from pydub import AudioSegment
from .prompts import prompt_audio_segmentation
from .utils.chat_api import generate_messages, get_response_with_retry
from .utils.general import validate_and_fix_json, normalize_embedding
import io

processing_config = json.load(open("configs/processing_config.json"))

MAX_RETRIES = processing_config["max_retries"]

pretrained_state = torch.load("models/pretrained_eres2netv2.ckpt", map_location='cpu')
model = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}
embedding_model = dynamic_import(model['obj'])(**model['args'])
embedding_model.load_state_dict(pretrained_state)
embedding_model.to(torch.device('cuda'))
embedding_model.eval()
feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

def get_embedding(wav):

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav
    
    def compute_embedding(wav_file, save=True):
        wav = load_wav(wav_file)
        feat = feature_extractor(wav).unsqueeze(0).to(torch.device('cuda'))
        with torch.no_grad():
            embedding = embedding_model(feat).detach().squeeze(0).cpu().numpy()
        return embedding

    return compute_embedding(wav)

@torch.no_grad()
def generate(wav):
    wav = base64.b64decode(wav)
    wav_file = BytesIO(wav)
    emb = get_embedding(wav_file)
    return emb

@torch.no_grad()
def get_audio_embeddings(audio_segments):
    res = []
    for wav in audio_segments:
        completion = generate(wav.decode("utf-8"))
        bytes_data = struct.pack('f' * len(completion), *completion)
        res.append(bytes_data)
        
    return res


# Configure logging
logger = logging.getLogger(__name__)

_METADATA_VOICE_CACHE = None


def _load_metadata_voice_map():
    global _METADATA_VOICE_CACHE
    if _METADATA_VOICE_CACHE is not None:
        return _METADATA_VOICE_CACHE
    metadata_path = os.environ.get("M3_VOICE_METADATA_PATH")
    if not metadata_path:
        _METADATA_VOICE_CACHE = {}
        return _METADATA_VOICE_CACHE
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        _METADATA_VOICE_CACHE = {str(k): v for k, v in raw_map.items()}
    except Exception as error:
        logger.warning("Failed to load M3 voice metadata map %s: %s", metadata_path, error)
        _METADATA_VOICE_CACHE = {}
    return _METADATA_VOICE_CACHE


def _metadata_segments_for_save_path(save_path):
    metadata_map = _load_metadata_voice_map()
    if not metadata_map:
        return None
    match = re.search(r"clip_(\d+)_voices\.json$", save_path)
    if not match:
        return None
    return metadata_map.get(str(int(match.group(1))))


def process_voices(video_graph, base64_audio, base64_video, save_path, preprocessing=[]):
    def _b64_to_str(value):
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return value
    def get_audio_segments(base64_audio, dialogs):
        # Decode base64 audio into bytes
        audio_data = base64.b64decode(base64_audio)
        
        # Create BytesIO object to hold audio data
        audio_io = io.BytesIO(audio_data)
        audio = AudioSegment.from_wav(audio_io)
        
        audio_segments = []
        for start_time, end_time in dialogs: 
            try:
                start_min, start_sec = map(int, start_time.split(':'))
                end_min, end_sec = map(int, end_time.split(':'))
            except ValueError:
                audio_segments.append(None)
                continue

            if (start_min < 0 or start_sec < 0 or start_sec >= 60) or (end_min < 0 or end_sec < 0 or end_sec >= 60):
                audio_segments.append(None)
                continue

            start_time_msec = (start_min * 60 + start_sec) * 1000
            end_time_msec = (end_min * 60 + end_sec) * 1000

            if start_time_msec >= end_time_msec:
                audio_segments.append(None)
                continue

            # Extract segment
            if end_time_msec > len(audio):  # AudioSegment uses milliseconds
                audio_segments.append(None)
                continue
            
            segment = audio[start_time_msec:end_time_msec]
        
            # Export segment to bytes buffer
            with io.BytesIO() as segment_buffer:
                segment.export(segment_buffer, format='wav')
                segment_buffer.seek(0)
                audio_segments.append(base64.b64encode(segment_buffer.getvalue()))
        
        return audio_segments

    def diarize_audio(base64_video, base64_audio, filter=None):
        if base64_video:
            input = [
                {
                    "type": "video_base64/mp4",
                    "content": _b64_to_str(base64_video),
                },
                {
                    "type": "text",
                    "content": prompt_audio_segmentation,
                },
            ]
        else:
            input = [
                {
                    "type": "audio_base64/wav",
                    "content": _b64_to_str(base64_audio),
                },
                {
                    "type": "text",
                    "content": prompt_audio_segmentation,
                },
            ]
        messages = generate_messages(input)
        model = "models/gemini-2.5-pro"
        asrs = None
        for i in range(MAX_RETRIES):
            response, _ = get_response_with_retry(model, messages, timeout=30)
            asrs = validate_and_fix_json(response)
            if asrs is not None:
                break
        if asrs is None:
            raise Exception("Failed to diarize audio")

        for asr in asrs:
            start_min, start_sec = map(int, asr["start_time"].split(':'))
            end_min, end_sec = map(int, asr["end_time"].split(':'))
            asr["duration"] = (end_min * 60 + end_sec) - (start_min * 60 + start_sec)
            
        asrs = [asr for asr in asrs if filter(asr)]

        return asrs

    def get_normed_audio_embeddings(audios):
        """
        Get normalized audio embeddings for a list of base64 audio strings
        
        Args:
            base64_audios (list): List of base64 encoded audio strings
            
        Returns:
            list: List of normalized audio embeddings
        """
        audio_segments = [audio["audio_segment"] for audio in audios]
        embeddings = get_audio_embeddings(audio_segments)
        normed_embeddings = [normalize_embedding(embedding) for embedding in embeddings]
        for audio, embedding in zip(audios, normed_embeddings):
            audio["embedding"] = embedding
        return audios

    def create_audio_segments(base64_audio, asrs):
        dialogs = [(asr["start_time"], asr["end_time"]) for asr in asrs]
        audio_segments = get_audio_segments(base64_audio, dialogs)
        for asr, audio_segment in zip(asrs, audio_segments):
            asr["audio_segment"] = audio_segment

        return asrs

    def filter_duration_based(audio):
        min_duration = processing_config["min_duration_for_audio"]
        return audio["duration"] >= min_duration
    
    def update_videograph(video_graph, audios):
        id2audios = {}
        
        for audio in audios:
            audio_info = {
                "embeddings": [audio["embedding"]],
                "contents": [audio["asr"]]
            }
            matched_nodes = video_graph.search_voice_nodes(audio_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, audio_info)
                audio["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_voice_node(audio_info)
                audio["matched_node"] = matched_node
                
            if matched_node not in id2audios:
                id2audios[matched_node] = []
            id2audios[matched_node].append(audio)

        return id2audios

    # Check if intermediate results exist
    try:
        with open(save_path, "r") as f:
            audios = json.load(f)
        for audio in audios:
            audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
    except Exception as e:
        if not base64_audio:
            logger.warning("Missing audio and no cached voice processing at %s.", save_path)
            return {}
        metadata_segments = _metadata_segments_for_save_path(save_path)
        if metadata_segments is not None:
            asrs = [
                segment
                for segment in metadata_segments
                if segment.get("asr")
            ]
            logger.info(
                "Use metadata dialogue segmentation for %s (%d segments)",
                save_path,
                len(asrs),
            )
        else:
            try:
                asrs = diarize_audio(base64_video, base64_audio, filter=filter_duration_based)
            except Exception as diarize_error:
                logger.warning(f"Skip diarization for {save_path}: {diarize_error}")
                return {}
        audios = create_audio_segments(base64_audio, asrs)
        audios = [audio for audio in audios if audio["audio_segment"] is not None]

        if len(audios) > 0:
            audios = get_normed_audio_embeddings(audios)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            for audio in audios:
                audio["audio_segment"] = audio["audio_segment"].decode("utf-8")
            json.dump(audios, f)
            for audio in audios:
                audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
        
        logger.info(f"Write voice detection results to {save_path}")
    
    if "voice" in preprocessing:
        return
    
    if len(audios) == 0:
        return {}

    id2audios = update_videograph(video_graph, audios)

    return id2audios
if __name__ == "__main__":
    pass
