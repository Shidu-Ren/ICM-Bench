"""Qwen video-language backend used by the ICM-Bench Vgent runs."""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from models.utils import fetch_video, resize_video


def load_video(video_path, args):
    raw_video, frame_indices, fps = fetch_video(
        {"video": video_path, "nframes": args.chunk_size},
        resize=False,
    )
    chunks = max(1, math.ceil(len(raw_video) / args.chunk_size))
    video, fps = resize_video(
        raw_video,
        fps,
        total_pixels=args.total_pixels * chunks * 28 * 28,
    )
    return [raw_video], None, None, frame_indices, fps, [video], None


def load_model(model_name):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.to("cuda").eval()
    return None, model, processor, None


def mllm_response(
    video_llm,
    tokenizer,
    processor,
    text,
    image_inputs,
    video,
    max_new_tokens=512,
    size_list=None,
    fps=None,
):
    content = []
    if video is not None:
        content.append(
            {
                "type": "video",
                "video": "provided_frames",
                "max_pixels": 360 * 420,
                "fps": fps or 1.0,
            }
        )
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        outputs = video_llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
    generated = outputs.sequences[0][inputs.input_ids.shape[1] :]
    return processor.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
