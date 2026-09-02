#!/usr/bin/env python3
"""Google Gemini image-generation helper used by the video anchor pipeline."""

from google import genai
from google.genai import types
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO
import sys
import time
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import get_google_api_key, get_image_model


class VertexNanobanana:
    """Generate and edit still images through the Google GenAI SDK."""
    
    def __init__(self, output_dir="output", api_key=None, model=None, usage_logger=None):
        """
        初始化 Nanobanana 客户端
        
        Args:
            output_dir: 输出目录路径
            api_key: Google API Key (可选，如果不传则尝试读取环境变量 GOOGLE_API_KEY)
            model: 模型名称 (可选)
        """
        # 初始化 Gemini 客户端
        if not api_key:
            api_key = get_google_api_key()
        
        if not api_key:
            print("⚠️  未找到 API Key。请设置环境变量 GOOGLE_API_KEY。")
            
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self._client_local = threading.local()
        self._client_local.client = self.client
        
        # 默认模型，如果传入了model则使用传入的，否则使用默认
        self.model = model or get_image_model() or "gemini-3.1-flash-image-preview"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.usage_logger = usage_logger
        
        print("✅ Gemini 图像客户端初始化成功")
        print(f"📁 输出目录: {self.output_dir.absolute()}")
        print(f"🤖 使用模型: {self.model}")

    def _thread_client(self):
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = genai.Client(api_key=self.api_key)
            self._client_local.client = client
        return client

    def _record_usage(
        self,
        response,
        *,
        operation: str,
        prompt: str,
        attempt: int | None = None,
        extra: dict | None = None,
    ) -> None:
        if self.usage_logger is None:
            return
        self.usage_logger.record_response(
            response=response,
            operation=operation,
            model=self.model,
            prompt=prompt,
            attempt=attempt,
            extra=extra,
        )

    def _image_generation_config(
        self,
        *,
        aspect_ratio: str,
        response_modalities: list | None = None,
    ):
        """Build an image-generation config across google-genai SDK versions."""
        if hasattr(types, "ImageConfig"):
            config = types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                )
            )
            if response_modalities:
                config.response_modalities = response_modalities
            return config

        config_dict = {}
        if response_modalities:
            config_dict["response_modalities"] = response_modalities
        return config_dict

    def _extract_response_parts(self, response, operation_name: str) -> list:
        """Return a best-effort list of response parts or raise a diagnostic error."""
        parts = getattr(response, "parts", None)
        if parts:
            return list(parts)

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None) if content is not None else None
            if candidate_parts:
                return list(candidate_parts)

        diagnostics: list[str] = []
        response_text = getattr(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            diagnostics.append(f"text={response_text.strip()[:500]}")

        for index, candidate in enumerate(candidates):
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is not None:
                diagnostics.append(f"candidate[{index}].finish_reason={finish_reason}")

            finish_message = getattr(candidate, "finish_message", None)
            if finish_message:
                diagnostics.append(f"candidate[{index}].finish_message={str(finish_message)[:300]}")

            safety_ratings = getattr(candidate, "safety_ratings", None)
            if safety_ratings:
                diagnostics.append(f"candidate[{index}].safety_ratings={safety_ratings}")

        if not diagnostics:
            diagnostics.append("response contained no parts and no candidate diagnostics")

        raise RuntimeError(
            f"{operation_name} 返回了空响应，没有任何可用的图片 parts。"
            f" 可能是模型未产出图片、响应被过滤、或当前 mixed content 过于复杂。"
            f" 诊断信息: {' | '.join(diagnostics)}"
        )

    def _part_to_image(self, part):
        """Decode image output across google-genai SDK versions."""
        if hasattr(part, "as_image"):
            return part.as_image()

        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            raise RuntimeError("响应 part 不包含 inline image data。")

        data = getattr(inline_data, "data", None)
        if data is None:
            raise RuntimeError("响应 inline image data 为空。")

        if isinstance(data, str):
            data = base64.b64decode(data)

        return Image.open(BytesIO(data)).copy()
    
    def generate_image(
        self, 
        prompt: str,
        aspect_ratio: str = "1:1",
        response_modalities: list = None,
        save_path: str = None,
        max_attempts: int = 3,
        retry_wait_seconds: int = 4,
    ):
        """
        文本生成图片 (Text-to-Image)
        
        Args:
            prompt: 图像描述文本
            aspect_ratio: 宽高比 (1:1, 16:9, 9:16, 4:3, 3:4, 4:5, 5:4, 21:9, 2:3, 3:2)
            response_modalities: 响应模态 ['Image'] 或 ['Text', 'Image']
            save_path: 保存路径（可选）
            
        Returns:
            生成的图片列表
        """
        print(f"\n🎨 生成图片...")
        print(f"📝 提示词: {prompt}")
        print(f"📐 宽高比: {aspect_ratio}")
        
        # 配置参数
        config = self._image_generation_config(
            aspect_ratio=aspect_ratio,
            response_modalities=response_modalities,
        )
        
        response_parts = None
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._thread_client().models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=config
                )
                self._record_usage(
                    response,
                    operation="image_generate",
                    prompt=prompt,
                    attempt=attempt,
                    extra={
                        "aspect_ratio": aspect_ratio,
                        "save_path": str(save_path) if save_path else None,
                    },
                )
                response_parts = self._extract_response_parts(response, "generate_image")
                break
            except Exception as exc:
                last_error = exc
                print(f"⚠️  图片生成失败 ({attempt}/{max_attempts}): {exc}")
                if attempt < max_attempts:
                    print(f"⏳ 等待 {retry_wait_seconds} 秒后重试...")
                    time.sleep(retry_wait_seconds)
                else:
                    raise

        if response_parts is None:
            raise RuntimeError(f"generate_image 未获得有效响应: {last_error}")
        
        # 处理响应
        generated_images = []
        for i, part in enumerate(response_parts):
            if part.text is not None:
                print(f"📄 模型说明: {part.text}")
            elif part.inline_data is not None:
                image = self._part_to_image(part)
                
                # 保存图片
                if save_path:
                    filename = save_path
                else:
                    filename = self.output_dir / f"generated_{i}_{aspect_ratio.replace(':', 'x')}.png"
                
                image.save(filename)
                generated_images.append(image)
                print(f"✅ 图片已保存: {filename}")
        
        return generated_images
    
    def edit_image(
        self,
        image_path: str,
        edit_prompt: str,
        save_path: str = None
    ):
        """
        图片编辑 (Image + Text-to-Image)
        
        Args:
            image_path: 要编辑的图片路径
            edit_prompt: 编辑指令
            save_path: 保存路径（可选）
            
        Returns:
            编辑后的图片
        """
        print(f"\n✏️  编辑图片...")
        print(f"🖼️  原图: {image_path}")
        print(f"📝 编辑指令: {edit_prompt}")
        
        # 加载图片
        image = Image.open(image_path)
        
        # 调用 API
        response = self._thread_client().models.generate_content(
            model=self.model,
            contents=[edit_prompt, image],
        )
        self._record_usage(
            response,
            operation="image_edit",
            prompt=edit_prompt,
            extra={
                "image_path": str(image_path),
                "save_path": str(save_path) if save_path else None,
            },
        )
        
        # 处理响应
        edited_images = []
        for i, part in enumerate(self._extract_response_parts(response, "edit_image")):
            if part.text is not None:
                print(f"📄 模型说明: {part.text}")
            elif part.inline_data is not None:
                edited_image = self._part_to_image(part)
                
                # 保存图片
                if save_path:
                    filename = save_path
                else:
                    filename = self.output_dir / f"edited_{i}.png"
                
                edited_image.save(filename)
                edited_images.append(edited_image)
                print(f"✅ 编辑后图片已保存: {filename}")
        
        return edited_images
    
    def blend_images(
        self,
        image_paths: list,
        blend_prompt: str,
        aspect_ratio: str = "1:1",
        save_path: str = None
    ):
        """
        多图合成 (Multi-Image to Image)
        
        Args:
            image_paths: 要合成的图片路径列表 (2-3张)
            blend_prompt: 合成指令
            aspect_ratio: 宽高比 (1:1, 16:9, 9:16, 4:3, 3:4, 4:5, 5:4, 21:9, 2:3, 3:2)
            save_path: 保存路径（可选）
            
        Returns:
            合成后的图片
        """
        print(f"\n🎭 合成图片...")
        print(f"🖼️  输入图片: {len(image_paths)} 张")
        print(f"📝 合成指令: {blend_prompt}")
        print(f"📐 宽高比: {aspect_ratio}")
        
        # 加载所有图片
        images = [Image.open(path) for path in image_paths]
        
        # 构建内容: 提示词 + 多张图片
        contents = [blend_prompt] + images
        
        # 配置参数
        config = self._image_generation_config(aspect_ratio=aspect_ratio)
        
        # 调用 API
        response = self._thread_client().models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        self._record_usage(
            response,
            operation="image_blend",
            prompt=blend_prompt,
            extra={
                "image_count": len(image_paths),
                "aspect_ratio": aspect_ratio,
                "save_path": str(save_path) if save_path else None,
            },
        )
        
        # 处理响应
        blended_images = []
        for i, part in enumerate(self._extract_response_parts(response, "blend_images")):
            if part.text is not None:
                print(f"📄 模型说明: {part.text}")
            elif part.inline_data is not None:
                blended_image = self._part_to_image(part)
                
                # 保存图片
                if save_path:
                    filename = save_path
                else:
                    filename = self.output_dir / f"blended_{i}.png"
                
                blended_image.save(filename)
                blended_images.append(blended_image)
                print(f"✅ 合成图片已保存: {filename}")
        
        return blended_images
    
    def iterative_editing(self, initial_prompt: str, edits: list):
        """
        迭代式编辑 (Multi-turn Conversational Editing)
        
        Args:
            initial_prompt: 初始生成提示词
            edits: 编辑指令列表 [(edit_prompt, save_name), ...]
            
        Returns:
            最终图片
        """
        print(f"\n🔄 开始迭代式编辑...")
        
        # 第一步：生成初始图片
        print(f"\n步骤 1: 生成初始图片")
        images = self.generate_image(initial_prompt, save_path=self.output_dir / "iter_step_0.png")
        current_image = images[0]
        current_path = self.output_dir / "iter_step_0.png"
        
        # 后续步骤：逐步编辑
        for i, (edit_prompt, save_name) in enumerate(edits, 1):
            print(f"\n步骤 {i+1}: {edit_prompt}")
            edited_images = self.edit_image(
                str(current_path),
                edit_prompt,
                save_path=self.output_dir / save_name
            )
            current_image = edited_images[0]
            current_path = self.output_dir / save_name
        
        print(f"\n✅ 迭代编辑完成!")
        return current_image
    
    def generate_with_mixed_content(
        self,
        contents: list,
        aspect_ratio: str = "1:1",
        save_path: str = None,
        max_attempts: int = 1,
        retry_wait_seconds: int = 4,
    ):
        """
        混合内容生成 (支持文本和图片交替)
        
        Args:
            contents: 内容列表，可以包含字符串(提示词)和PIL.Image对象
            aspect_ratio: 宽高比 (1:1, 16:9, 9:16, 4:3, 3:4, 4:5, 5:4, 21:9, 2:3, 3:2)
            save_path: 保存路径（可选）
            
        Returns:
            生成的图片列表
        """
        print(f"\n🎨 混合内容生成...")
        print(f"📝 内容项数量: {len(contents)}")
        text_items = sum(1 for item in contents if isinstance(item, str))
        image_items = sum(1 for item in contents if isinstance(item, Image.Image))
        print(f"🧾 文本项: {text_items}")
        print(f"🖼️  图片项: {image_items}")
        print(f"📐 宽高比: {aspect_ratio}")
        
        # 配置参数
        config = self._image_generation_config(aspect_ratio=aspect_ratio)
        
        # 调用 API
        response = None
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._thread_client().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                self._record_usage(
                    response,
                    operation="image_mixed_content",
                    prompt="\n".join(item for item in contents if isinstance(item, str)),
                    attempt=attempt,
                    extra={
                        "text_items": text_items,
                        "image_items": image_items,
                        "aspect_ratio": aspect_ratio,
                        "save_path": str(save_path) if save_path else None,
                    },
                )
                break
            except Exception as exc:
                last_error = exc
                print(f"⚠️  混合内容生成失败 ({attempt}/{max_attempts}): {exc}")
                if attempt < max_attempts:
                    print(f"⏳ 等待 {retry_wait_seconds} 秒后重试...")
                    time.sleep(retry_wait_seconds)
                else:
                    raise

        if response is None:
            raise RuntimeError(f"generate_with_mixed_content 未获得有效响应: {last_error}")
        
        # 处理响应
        generated_images = []
        saved_primary_image = False
        skipped_extra_images = 0
        for i, part in enumerate(self._extract_response_parts(response, "generate_with_mixed_content")):
            if part.text is not None:
                print(f"📄 模型说明: {part.text}")
            elif part.inline_data is not None:
                image = self._part_to_image(part)
                generated_images.append(image)

                if save_path:
                    if saved_primary_image:
                        skipped_extra_images += 1
                        continue
                    filename = save_path
                    saved_primary_image = True
                else:
                    filename = self.output_dir / f"generated_mixed_{i}.png"

                image.save(filename)
                print(f"✅ 图片已保存: {filename}")

        if save_path and skipped_extra_images:
            print(f"ℹ️  额外返回了 {skipped_extra_images} 张图片，已忽略，只保留首张作为目标输出。")

        return generated_images
