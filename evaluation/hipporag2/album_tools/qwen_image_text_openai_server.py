#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class Qwen35Server:
    def __init__(self, model_path: str, model_id: str, max_input_tokens: int) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.generation_lock = threading.Lock()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 256)
        temperature = float(payload.get("temperature") or 0.0)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.generation_lock, torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_tokens = int(new_tokens.shape[-1])
        return {
            "id": f"chatcmpl-local-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length" if completion_tokens >= max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def make_handler(server_state: Qwen35Server):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in {"/v1/models", "/models"}:
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": server_state.model_id,
                                "object": "model",
                                "created": 0,
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions"}:
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send(200, server_state.chat(payload))
            except Exception as exc:
                self._send(500, {"error": {"message": repr(exc)}})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="Qwen3.5-9B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-input-tokens", type=int, default=65536)
    args = parser.parse_args()

    print(f"[server] loading model={args.model_path}", flush=True)
    state = Qwen35Server(args.model_path, args.served_model_name, args.max_input_tokens)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[server] ready http://{args.host}:{args.port}/v1 model={args.served_model_name}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
