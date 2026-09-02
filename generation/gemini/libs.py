"""
Gemini Video Processor Library
A utility class for working with Google's Gemini AI models with video, image, and audio support.
"""

import re
import time
import asyncio
import datetime
from pathlib import Path
from functools import wraps
import base64
import subprocess
from google.genai.types import Part, Blob, VideoMetadata
from google import genai


def get_video_duration_cn(path: str) -> str:
    """
    使用 ffprobe 获取视频总时长（秒），并格式化为 "xx时xx分xx秒"。

    :param path: 视频文件路径
    :return: 时长字符串，格式为 "xx时xx分xx秒"
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nologger.info_wrappers=1:nokey=1",
        path,
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    total_seconds = float(output)

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}时{minutes:02d}分{seconds:02d}秒"


def retry(max_attempts=None, delay=0):
    """
    通用重试装饰器，自动检测同步/异步函数

    Args:
        max_attempts: 最大重试次数，None表示无限重试
        delay: 重试间隔（秒）
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            # 异步函数
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                attempt = 0
                while True:
                    try:
                        attempt += 1
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if max_attempts and attempt >= max_attempts:
                            print(f"⛔ {func.__name__} 失败: 已达到最大重试次数 ({max_attempts}次)")
                            raise
                        print(f"🔄 {func.__name__} 重试 {attempt}/{max_attempts}: {e}")
                        await asyncio.sleep(delay)

            return async_wrapper
        else:
            # 同步函数
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                attempt = 0
                while True:
                    try:
                        attempt += 1
                        return func(*args, **kwargs)
                    except Exception as e:
                        if max_attempts and attempt >= max_attempts:
                            print(f"⛔ {func.__name__} 失败: 已达到最大重试次数 ({max_attempts}次)")
                            raise
                        print(f"🔄 {func.__name__} 重试 {attempt}/{max_attempts}: {e}")
                        time.sleep(delay)

            return sync_wrapper

    return decorator


class GeminiVideoProcessor:
    """
    Gemini Video Processor - A helper class for working with Gemini AI models
    
    Features:
    - Text, video, audio, and image input support
    - Automatic model fallback (pro -> flash -> 2.0)
    - Token usage tracking and cost estimation
    - Retry mechanism for robustness
    
    Usage:
        client = genai.Client()
        processor = GeminiVideoProcessor(client, output_dir="output")
        
        # Simple text call
        response = processor.call_gemini(
            prompt_text="What is AI?",
            primary_model="gemini-2.5-flash"
        )
        
        # Video analysis
        with open("video.mp4", "rb") as f:
            video_data = f.read()
        response = processor.call_gemini(
            prompt_text="Describe this video",
            video_bytes=video_data,
            primary_model="gemini-2.5-pro"
        )
    """
    
    def __init__(
        self,
        llm,
        input_dir: str = "",
        output_dir: str = "",
        fps: int = 60,
        merge_episode_number: int = 10,
        semaphore: int = 200,
        additional_prompt: str = "",
        usage_logger=None,
    ):
        self.llm = llm
        self.fps = fps
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0
        self.semaphore = semaphore
        self.last_video_generation_diagnostics = {}
        self.usage_logger = usage_logger

        if output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{self.output_dir}/token_count.txt", "w") as f:
                f.write(f"Started at {datetime.datetime.now()}\n")

    def _record_api_usage(
        self,
        response,
        *,
        operation: str,
        model: str,
        prompt: str | None = None,
        attempt: int | None = None,
        extra: dict | None = None,
    ) -> None:
        if self.usage_logger is None:
            return
        self.usage_logger.record_response(
            response=response,
            operation=operation,
            model=model,
            prompt=prompt,
            attempt=attempt,
            extra=extra,
        )

    def _print_video_filter_diagnostics(self, operation) -> None:
        """尽量输出 Veo 过滤原因，方便定位空视频响应。"""
        response = getattr(operation, "response", None)
        if not response:
            return

        filtered_count = getattr(response, "rai_media_filtered_count", None)
        if filtered_count is None:
            filtered_count = getattr(response, "raiMediaFilteredCount", None)

        filtered_reasons = getattr(response, "rai_media_filtered_reasons", None)
        if filtered_reasons is None:
            filtered_reasons = getattr(response, "raiMediaFilteredReasons", None)

        if filtered_count is not None:
            print(f"   RAI filtered count: {filtered_count}")

        if filtered_reasons:
            print(f"   RAI filtered reasons: {filtered_reasons}")

        if hasattr(response, "model_dump"):
            try:
                print(f"   Response dump: {response.model_dump(exclude_none=True)}")
            except Exception:
                pass
        elif hasattr(response, "__dict__"):
            print(f"   Response attrs: {response.__dict__}")

    def _extract_video_generation_diagnostics(self, operation) -> dict:
        """Collect lightweight diagnostics about the latest Veo response."""
        diagnostics: dict = {}
        response = getattr(operation, "response", None)
        if not response:
            return diagnostics

        filtered_count = getattr(response, "rai_media_filtered_count", None)
        if filtered_count is None:
            filtered_count = getattr(response, "raiMediaFilteredCount", None)

        filtered_reasons = getattr(response, "rai_media_filtered_reasons", None)
        if filtered_reasons is None:
            filtered_reasons = getattr(response, "raiMediaFilteredReasons", None)

        if filtered_count is not None:
            diagnostics["rai_media_filtered_count"] = filtered_count
        if filtered_reasons:
            diagnostics["rai_media_filtered_reasons"] = list(filtered_reasons)
            diagnostics["audio_filtered"] = any(
                "audio" in str(reason).lower() for reason in filtered_reasons
            )

        return diagnostics

    def _set_video_generation_diagnostics(
        self,
        *,
        status: str,
        operation=None,
        error=None,
        model: str | None = None,
    ) -> None:
        diagnostics = {"status": status}
        if model:
            diagnostics["model"] = model
        if error is not None:
            diagnostics["error"] = str(error)
        if operation is not None:
            diagnostics.update(self._extract_video_generation_diagnostics(operation))
        self.last_video_generation_diagnostics = diagnostics

    def _get_generated_video_items(self, operation):
        """兼容不同 SDK 字段名，提取视频结果列表。"""
        response = getattr(operation, "response", None)
        if not response:
            return []

        generated_videos = getattr(response, "generated_videos", None)
        if generated_videos is None:
            generated_videos = getattr(response, "videos", None)

        return generated_videos or []

    def _wait_for_file_active(self, file_ref, timeout_seconds: int = 180, poll_seconds: int = 5):
        """等待 Gemini Files 中的视频文件进入 ACTIVE，便于后续 extend。"""
        file_name = getattr(file_ref, "name", None)
        if not file_name:
            return file_ref

        deadline = time.time() + timeout_seconds
        current_file = file_ref

        while time.time() < deadline:
            try:
                current_file = self.llm.files.get(name=file_name)
            except Exception as e:
                print(f"⚠️  查询视频文件状态失败，稍后重试: {e}")
                time.sleep(poll_seconds)
                continue

            state = getattr(current_file, "state", None)
            state_name = getattr(state, "name", str(state)) if state is not None else None

            if state_name == "ACTIVE":
                return current_file

            if state_name == "FAILED":
                error = getattr(current_file, "error", None)
                raise RuntimeError(f"视频文件处理失败: {error or current_file}")

            print(f"⏳ 等待视频文件进入 ACTIVE 状态... 当前状态: {state_name or 'UNKNOWN'}")
            time.sleep(poll_seconds)

        raise TimeoutError(f"等待视频文件 ACTIVE 超时: {file_name}")

    def _generate_videos_with_compatibility_retry(
        self,
        *,
        model: str,
        prompt: str = None,
        image=None,
        video=None,
        config=None,
    ):
        """在 Gemini API 不支持某些字段时自动降级重试。"""
        active_config = config
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                return self.llm.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=image,
                    video=video,
                    config=active_config,
                )
            except ValueError as e:
                error_text = str(e)
                if (
                    active_config is not None
                    and "generate_audio parameter is not supported in Gemini API" in error_text
                ):
                    print("⚠️  当前 Gemini API 不支持 generate_audio，自动移除该参数后重试。")
                    if hasattr(active_config, "model_dump"):
                        config_dict = active_config.model_dump(exclude_none=True)
                    elif isinstance(active_config, dict):
                        config_dict = dict(active_config)
                    else:
                        raise
                    config_dict.pop("generate_audio", None)
                    active_config = self._build_generate_videos_config(config_dict)
                    continue
                raise
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                wait_seconds = min(20, 4 * attempt)
                print(
                    f"⚠️  generate_videos 调用失败 ({attempt}/{max_attempts}): {e}，"
                    f"{wait_seconds}s 后重试。"
                )
                time.sleep(wait_seconds)

        raise RuntimeError("generate_videos 未获得有效 operation。")

    def _build_generate_videos_config(self, config_dict: dict | None):
        """Build Veo config across google-genai SDK versions."""
        if not config_dict:
            return None
        from google.genai import types

        if hasattr(types, "GenerateVideosConfig"):
            return types.GenerateVideosConfig(**config_dict)
        return dict(config_dict)

    def _generate_video_extension_with_processed_retry(
        self,
        *,
        model: str,
        prompt: str,
        video_ref,
        config,
        max_attempts: int = 4,
        wait_seconds: int = 15,
    ):
        """针对 Gemini API 中 extend 的 processed 时序问题做等待重试。"""
        last_error = None
        current_video_ref = video_ref

        for attempt in range(1, max_attempts + 1):
            current_video_ref = self._wait_for_file_active(current_video_ref)
            if attempt > 1:
                print(
                    f"🔁 重新尝试视频续写 ({attempt}/{max_attempts})，等待输入视频完全 processed 后再提交..."
                )
            try:
                return self._generate_videos_with_compatibility_retry(
                    model=model,
                    prompt=prompt,
                    video=current_video_ref,
                    config=config,
                )
            except Exception as exc:
                error_text = str(exc)
                last_error = exc
                if "Input video must be a video that was generated by VEO that has been processed" not in error_text:
                    raise
                if attempt >= max_attempts:
                    raise
                print(
                    f"⚠️  输入视频还未完全 processed ({attempt}/{max_attempts})，"
                    f"等待 {wait_seconds} 秒后重试..."
                )
                time.sleep(wait_seconds)

        if last_error:
            raise last_error
        raise RuntimeError("视频续写重试失败，未获得有效操作对象。")
    def call_gemini(
        self,
        prompt_text,
        system_instruction=None,
        video_bytes=None,
        primary_model="gemini-2.5-pro",
        image_list=None,
        video_list=None,
        temperature=1.0,
        response_config=None,
        enable_fallback=True,
        calling_from="",
        audio_bytes=None,
    ):
        """
        统一的Gemini调用方法，支持文本、视频、图片、音频输入

        Args:
            prompt_text: 提示文本
            system_instruction: 系统指令
            video_bytes: 视频字节数据（可选）
            primary_model: 主要使用的模型
            image_list: 图片列表 [(image_bytes, description), ...]
            video_list: 视频列表 [(prefix_text, video_bytes, suffix_text), ...]
            temperature: 温度参数
            response_config: 额外的响应配置（如response_mime_type等）
            enable_fallback: 是否启用模型降级
            calling_from: 调用来源标识
            audio_bytes: 音频字节数据

        Returns:
            Gemini响应对象
        """
        # 模型降级顺序
        models_fallback_order = [primary_model]
        if enable_fallback and video_bytes:
            models_fallback_order.extend(
                [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                ]
            )

        # 构建内容
        contents = []
        if system_instruction:
            contents.append(f"system_instruction:{system_instruction}\n\n")

        # 添加视频
        if video_bytes:
            contents.append(
                Part(
                    inline_data=Blob(data=video_bytes, mime_type="video/mp4"),
                    video_metadata=VideoMetadata(fps=1),
                )
            )

        # 添加音频
        if audio_bytes:
            contents.append(Part(inline_data=Blob(data=audio_bytes, mime_type="audio/mpeg")))

        # 添加视频列表
        if video_list:
            for video in video_list:
                contents.append(video[0])
                contents.append(
                    Part(
                        inline_data=Blob(data=video[1], mime_type="video/mp4"),
                        video_metadata=VideoMetadata(fps=1),
                    )
                )
                contents.append(video[2])

        # 添加图片列表
        if image_list:
            for image in image_list:
                contents.append(Part.from_bytes(data=image[0], mime_type="image/jpeg"))
                contents.append(f"上面的图片为该集短剧时间戳为{image[1]}的截图")

        contents.append(prompt_text)

        # 构建配置
        config = {"temperature": temperature, "safety_settings": None}
        if response_config:
            config.update(response_config)

        # 尝试不同的模型
        last_error = None
        for model in models_fallback_order:
            try:
                response = self.llm.models.generate_content(model=model, contents=contents, config=config)
                self._update_tokens(response=response, model_type=model, calling_from=calling_from)
                self._record_api_usage(
                    response,
                    operation="gemini_generate_content",
                    model=model,
                    prompt=prompt_text,
                    extra={
                        "calling_from": calling_from,
                        "has_video": bool(video_bytes or video_list),
                        "has_audio": bool(audio_bytes),
                        "image_count": len(image_list or []),
                    },
                )
                if response.text or not enable_fallback:
                    return response
            except Exception as e:
                last_error = e
                if not enable_fallback:
                    raise
                continue

        raise Exception(f"All Gemini models failed. Last error: {last_error}")

    def _update_tokens(self, response, model_type, calling_from="unknown"):
        """Update token counters and track cost info for Gemini models"""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        this_token_input = getattr(usage, "prompt_token_count", None) or 0
        this_token_output = getattr(usage, "candidates_token_count", None) or 0
        self.input_tokens += this_token_input
        self.output_tokens += this_token_output

        # 根据 model_type 选取单价（单位：美元 / 1M token）
        if "2.5-pro" in model_type:
            rate_in = 1.25
            rate_out = 10.00
        elif "2.5-flash" in model_type:
            rate_in = 0.15
            rate_out = 3.50
        elif "2.0-flash" in model_type:
            rate_in = 0.15
            rate_out = 0.60
        else:
            rate_in = 0.0
            rate_out = 0.0

        cost_this = (this_token_input * rate_in + this_token_output * rate_out) / 1000000
        self.total_cost += cost_this

        token_info = (
            f"Calling time: {datetime.datetime.now()}\n"
            f"Calling from {calling_from}, \n"
            f"Tokens - Input: {this_token_input}, Output: {this_token_output}, \n"
            f"Total Tokens - Input: {self.input_tokens}, Output: {self.output_tokens}, \n"
            f"Total: {self.input_tokens + self.output_tokens}; \n"
            f"Cost This Call: ${cost_this:.6f}, Total Cost: ${self.total_cost:.6f}\n\n"
        )

        print(token_info)
        if self.output_dir:
            with open(f"{self.output_dir}/token_count.txt", "a") as f:
                f.write(f"{token_info}\n")

    def get_cost_summary(self) -> dict:
        """
        获取费用统计摘要
        
        Returns:
            dict: 包含费用统计信息的字典
        """
        return {
            "total_cost_usd": self.total_cost,
            "total_cost_cny": self.total_cost * 7.2,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens
        }

    @retry(delay=0)
    def file_to_base64(self, file_path: str) -> str:
        """
        读取本地文件并返回其 Base64 编码字符串。
        """
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")

    def generate_timestamps(self, line, step=0.2):
        """
        从形如 "MM:SS 到 MM:SS" 的行中提取时间区间，并以 step 秒为间隔生成所有时间点。
        返回格式示例：["00:00", "00:00.2", "00:00.4", ...]
        """
        pattern = r"(\d{2}):(\d{2})\s*到\s*(\d{2}):(\d{2})"
        matches = re.findall(pattern, line)
        timestamps = []

        for sh, ss, eh, es in matches:
            start = int(sh) * 60 + int(ss)
            end = int(eh) * 60 + int(es)
            t = start
            while t <= end + 1e-9:
                mm = int(t // 60)
                sec = t % 60
                if abs(sec - int(sec)) < 1e-9:
                    timestamps.append(f"{mm:02d}:{int(sec):02d}")
                else:
                    timestamps.append(f"{mm:02d}:{sec:04.1f}")
                t += step

        return timestamps

    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        duration: int = 8,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        save_path: str = None,
        model: str = "veo-3.1-fast-generate-preview",
        negative_prompt: str = None,
        calling_from: str = "generate_video",
        person_generation: str = None,
        generate_audio: bool = None,
        return_video_ref: bool = False,
    ):
        """
        图片生成视频 (Image-to-Video)
        
        Args:
            image_path: 输入图片路径
            prompt: 视频描述文本（支持对话提示，用引号包裹对话）
            duration: 视频时长（4, 6, 8秒）
            aspect_ratio: 宽高比 ("16:9" 或 "9:16")
            resolution: 分辨率 ("720p" 或 "1080p"，1080p仅支持8秒）
            save_path: 视频保存路径
            model: Veo模型 ("veo-3.1-fast-generate-preview", "veo-3.1-generate-preview")
            negative_prompt: 负向提示词
            calling_from: 调用来源标识
            
        Returns:
            视频文件路径
        """
        self.last_video_generation_diagnostics = {}
        print(f"\n🎬 生成视频...")
        print(f"📝 提示词: {prompt[:100]}...")
        print(f"🖼️  输入图片: {image_path}")
        print(f"⏱️  时长: {duration}秒")
        print(f"📐 宽高比: {aspect_ratio}")
        
        # 读取图片并转换为 Image 对象
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        from google.genai import types
        image = types.Image(
            image_bytes=image_bytes,
            mime_type="image/png"
        )
        
        # 配置参数
        config_dict = {
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }

        if negative_prompt:
            config_dict["negative_prompt"] = negative_prompt
        if person_generation:
            config_dict["person_generation"] = person_generation
        if generate_audio is not None:
            config_dict["generate_audio"] = generate_audio
        
        config = self._build_generate_videos_config(config_dict)
        
        # 调用 Veo API
        print(f"🚀 调用 {model}...")
        operation = self._generate_videos_with_compatibility_retry(
            model=model,
            prompt=prompt,
            image=image,
            config=config
        )
        
        # 轮询等待完成（增加轮询间隔）
        print("⏳ 等待视频生成完成...")
        poll_count = 0
        poll_error_count = 0
        max_poll_errors = 40
        while not operation.done:
            poll_count += 1
            time.sleep(15)  # 15秒轮询间隔
            try:
                operation = self.llm.operations.get(operation)
                poll_error_count = 0
            except Exception as e:
                poll_error_count += 1
                if poll_error_count > max_poll_errors:
                    raise
                wait_seconds = min(30, 5 * poll_error_count)
                print(
                    f"⚠️  查询视频生成状态失败 ({poll_error_count}/{max_poll_errors}): {e}，"
                    f"{wait_seconds}s 后继续轮询。"
                )
                time.sleep(wait_seconds)
                continue
            if poll_count % 6 == 0:  # 每90秒打印一次
                print(f"   仍在生成中... ({poll_count * 15}秒)")

        # 检查是否有错误
        if operation.error:
            error_msg = str(operation.error)
            print(f"❌ 视频生成失败: {error_msg}")
            self._set_video_generation_diagnostics(
                status="operation_error",
                operation=operation,
                error=error_msg,
                model=model,
            )
            # 检查是否是内容审核问题
            if 'sensitive words' in error_msg or 'Responsible AI' in error_msg:
                print("   原因: 内容违反了Google的AI政策，尝试简化prompt")
            self._record_api_usage(
                operation,
                operation="veo_image_to_video",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "operation_error",
                    "duration_seconds": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                },
            )
            return (None, None) if return_video_ref else None

        if not operation.response:
            print(f"❌ 没有生成视频响应 (operation.response is None)")
            print(f"   Operation状态: done={operation.done}, error={operation.error}")
            self._set_video_generation_diagnostics(
                status="missing_response",
                operation=operation,
                model=model,
            )
            self._record_api_usage(
                operation,
                operation="veo_image_to_video",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "missing_response",
                    "duration_seconds": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                },
            )
            return (None, None) if return_video_ref else None
            
        generated_videos = self._get_generated_video_items(operation)
        if not generated_videos:
            print(f"❌ generated_videos列表为空")
            self._print_video_filter_diagnostics(operation)
            self._set_video_generation_diagnostics(
                status="empty_generated_videos",
                operation=operation,
                model=model,
            )
            self._record_api_usage(
                operation,
                operation="veo_image_to_video",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "empty_generated_videos",
                    "duration_seconds": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                },
            )
            return (None, None) if return_video_ref else None
        
        # 下载视频
        generated_video = generated_videos[0]
        video_file = getattr(generated_video, "video", None) or generated_video
        video_bytes = self.llm.files.download(file=video_file)
        ready_video_file = self._wait_for_file_active(video_file)
        
        # 保存视频
        if save_path is None:
            if self.output_dir:
                save_path = f"{self.output_dir}/video_{int(time.time())}.mp4"
            else:
                save_path = f"video_{int(time.time())}.mp4"
        
        # 使用Video对象的save方法保存
        with open(save_path, "wb") as f:
            f.write(video_bytes)
        
        print(f"✅ 视频已保存: {save_path}")
        self._set_video_generation_diagnostics(
            status="success",
            operation=operation,
            model=model,
        )
        
        # 记录token使用（如果operation有usage信息）
        if hasattr(operation, 'usage_metadata'):
            self._update_tokens(response=operation, model_type=model, calling_from=calling_from)
        self._record_api_usage(
            operation,
            operation="veo_image_to_video",
            model=model,
            prompt=prompt,
            extra={
                "calling_from": calling_from,
                "status": "success",
                "duration_seconds": duration,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "save_path": str(save_path) if save_path else None,
            },
        )
        
        if return_video_ref:
            return save_path, ready_video_file
        return save_path

    def generate_video_with_references(
        self,
        prompt: str,
        reference_images: list,
        duration: int = 8,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        save_path: str = None,
        model: str = "veo-3.1-fast-generate-preview",
        negative_prompt: str = None,
        calling_from: str = "generate_video_refs",
        person_generation: str = None,
        generate_audio: bool = None,
    ):
        """
        使用参考图生成视频 (Reference Images to Video)
        支持最多3张参考图保持风格一致性
        
        Args:
            prompt: 视频描述文本
            reference_images: 参考图路径列表（最多3张）[(image_path, reference_type), ...]
                             reference_type: "asset" (人物/物品) 或 "style" (风格)
            duration: 视频时长（8秒，使用参考图时必须为8）
            aspect_ratio: 宽高比（仅支持"16:9"）
            resolution: 分辨率
            save_path: 视频保存路径
            model: Veo模型（仅支持3.1版本）
            negative_prompt: 负向提示词
            calling_from: 调用来源标识
            
        Returns:
            视频文件路径
        """
        print(f"\n🎬 使用参考图生成视频...")
        print(f"📝 提示词: {prompt[:100]}...")
        print(f"🖼️  参考图数量: {len(reference_images)}")
        print(f"⏱️  时长: {duration}秒")
        
        if len(reference_images) > 3:
            raise ValueError("最多支持3张参考图")
        
        if duration != 8:
            print("⚠️  使用参考图时时长必须为8秒，已自动调整")
            duration = 8
        
        if aspect_ratio != "16:9":
            print("⚠️  使用参考图时仅支持16:9宽高比，已自动调整")
            aspect_ratio = "16:9"
        
        # 加载参考图
        from PIL import Image
        from google.genai import types
        
        ref_images = []
        for img_path, ref_type in reference_images:
            image = Image.open(img_path)
            ref = types.VideoGenerationReferenceImage(
                image=image,
                reference_type=ref_type  # "asset" 或 "style"
            )
            ref_images.append(ref)
            print(f"  📎 {img_path} ({ref_type})")
        
        # 配置参数
        config_dict = {
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "reference_images": ref_images
        }
        
        if negative_prompt:
            config_dict["negative_prompt"] = negative_prompt
        if person_generation:
            config_dict["person_generation"] = person_generation
        if generate_audio is not None:
            config_dict["generate_audio"] = generate_audio
        
        config = self._build_generate_videos_config(config_dict)
        
        # 调用 Veo API
        print(f"🚀 调用 {model}...")
        operation = self._generate_videos_with_compatibility_retry(
            model=model,
            prompt=prompt,
            config=config
        )
        
        # 轮询等待完成
        print("⏳ 等待视频生成完成...")
        poll_count = 0
        while not operation.done:
            poll_count += 1
            time.sleep(10)
            operation = self.llm.operations.get(operation)
            if poll_count % 6 == 0:
                print(f"   仍在生成中... ({poll_count * 10}秒)")
        
        generated_videos = self._get_generated_video_items(operation)
        if not generated_videos:
            print(f"❌ generated_videos列表为空")
            self._print_video_filter_diagnostics(operation)
            self._record_api_usage(
                operation,
                operation="veo_reference_images_to_video",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "empty_generated_videos",
                    "duration_seconds": duration,
                    "reference_image_count": len(reference_images),
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                },
            )
            return None

        # 下载视频
        generated_video = generated_videos[0]
        video_file = getattr(generated_video, "video", None) or generated_video
        video_bytes = self.llm.files.download(file=video_file)
        
        # 保存视频
        if save_path is None:
            if self.output_dir:
                save_path = f"{self.output_dir}/video_refs_{int(time.time())}.mp4"
            else:
                save_path = f"video_refs_{int(time.time())}.mp4"
        
        with open(save_path, "wb") as f:
            f.write(video_bytes)
        print(f"✅ 视频已保存: {save_path}")
        
        # 记录token使用
        if hasattr(operation, 'usage_metadata'):
            self._update_tokens(response=operation, model_type=model, calling_from=calling_from)
        self._record_api_usage(
            operation,
            operation="veo_reference_images_to_video",
            model=model,
            prompt=prompt,
            extra={
                "calling_from": calling_from,
                "status": "success",
                "duration_seconds": duration,
                "reference_image_count": len(reference_images),
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "save_path": str(save_path) if save_path else None,
            },
        )
        
        return save_path

    def generate_video_interpolation(
        self,
        first_frame_path: str,
        last_frame_path: str,
        prompt: str,
        duration: int = 8,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        save_path: str = None,
        model: str = "veo-3.1-fast-generate-preview",
        negative_prompt: str = None,
        calling_from: str = "generate_video_interp",
        person_generation: str = None,
        generate_audio: bool = None,
    ):
        """
        首帧+末帧插值生成视频 (Frame Interpolation)
        
        Args:
            first_frame_path: 首帧图片路径
            last_frame_path: 末帧图片路径
            prompt: 视频描述文本
            duration: 视频时长（8秒，插值时必须为8）
            aspect_ratio: 宽高比
            resolution: 分辨率
            save_path: 视频保存路径
            model: Veo模型（仅支持3.1版本）
            negative_prompt: 负向提示词
            calling_from: 调用来源标识
            
        Returns:
            视频文件路径
        """
        print(f"\n🎬 首末帧插值生成视频...")
        print(f"📝 提示词: {prompt[:100]}...")
        print(f"🖼️  首帧: {first_frame_path}")
        print(f"🖼️  末帧: {last_frame_path}")
        print(f"⏱️  时长: {duration}秒")
        
        if duration != 8:
            print("⚠️  插值时时长必须为8秒，已自动调整")
            duration = 8
        
        # 加载图片
        from PIL import Image
        from google.genai import types
        
        first_image = Image.open(first_frame_path)
        last_image = Image.open(last_frame_path)
        
        # 配置参数
        config_dict = {
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "last_frame": last_image
        }
        
        if negative_prompt:
            config_dict["negative_prompt"] = negative_prompt
        if person_generation:
            config_dict["person_generation"] = person_generation
        if generate_audio is not None:
            config_dict["generate_audio"] = generate_audio
        
        config = self._build_generate_videos_config(config_dict)
        
        # 调用 Veo API
        print(f"🚀 调用 {model}...")
        operation = self._generate_videos_with_compatibility_retry(
            model=model,
            prompt=prompt,
            image=first_image,
            config=config
        )
        
        # 轮询等待完成
        print("⏳ 等待视频生成完成...")
        poll_count = 0
        while not operation.done:
            poll_count += 1
            time.sleep(10)
            operation = self.llm.operations.get(operation)
            if poll_count % 6 == 0:
                print(f"   仍在生成中... ({poll_count * 10}秒)")
        
        generated_videos = self._get_generated_video_items(operation)
        if not generated_videos:
            print(f"❌ generated_videos列表为空")
            self._print_video_filter_diagnostics(operation)
            self._record_api_usage(
                operation,
                operation="veo_frame_interpolation",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "empty_generated_videos",
                    "duration_seconds": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                },
            )
            return None

        # 下载视频
        generated_video = generated_videos[0]
        video_file = getattr(generated_video, "video", None) or generated_video
        video_bytes = self.llm.files.download(file=video_file)
        
        # 保存视频
        if save_path is None:
            if self.output_dir:
                save_path = f"{self.output_dir}/video_interp_{int(time.time())}.mp4"
            else:
                save_path = f"video_interp_{int(time.time())}.mp4"
        
        with open(save_path, "wb") as f:
            f.write(video_bytes)
        print(f"✅ 视频已保存: {save_path}")
        
        # 记录token使用
        if hasattr(operation, 'usage_metadata'):
            self._update_tokens(response=operation, model_type=model, calling_from=calling_from)
        self._record_api_usage(
            operation,
            operation="veo_frame_interpolation",
            model=model,
            prompt=prompt,
            extra={
                "calling_from": calling_from,
                "status": "success",
                "duration_seconds": duration,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "save_path": str(save_path) if save_path else None,
            },
        )
        
        return save_path

    def generate_video_extension(
        self,
        video_path: str = None,
        video_ref=None,
        prompt: str = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        save_path: str = None,
        model: str = "veo-3.1-fast-generate-preview",
        negative_prompt: str = None,
        calling_from: str = "generate_video_extend",
        person_generation: str = None,
        generate_audio: bool = None,
    ):
        """
        对已有 Veo 视频做续写。

        Args:
            video_path: 输入视频路径（应为 Veo 生成视频）
            prompt: 续写提示词
            aspect_ratio: 输出宽高比
            resolution: 输出分辨率
            save_path: 保存路径
            model: Veo 模型
            negative_prompt: 负向提示词
            calling_from: 调用来源
            person_generation: 人像安全设置
            generate_audio: 是否生成音频
        """
        from google.genai import types
        self.last_video_generation_diagnostics = {}

        print(f"\n🎬 续写视频...")
        print(f"📝 提示词: {(prompt or '')[:100]}...")
        print(f"📼 输入视频: {video_path or '<video_ref>'}")
        print(f"📐 宽高比: {aspect_ratio}")

        if video_ref is not None:
            video = self._wait_for_file_active(video_ref)
        else:
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            video = types.Video(
                video_bytes=video_bytes,
                mime_type="video/mp4",
            )

        config_dict = {}
        if aspect_ratio:
            config_dict["aspect_ratio"] = aspect_ratio
        if resolution:
            config_dict["resolution"] = resolution
        if negative_prompt:
            config_dict["negative_prompt"] = negative_prompt
        if person_generation:
            config_dict["person_generation"] = person_generation
        if generate_audio is not None:
            config_dict["generate_audio"] = generate_audio

        config = self._build_generate_videos_config(config_dict)

        print(f"🚀 调用 {model}...")
        if video_ref is not None:
            operation = self._generate_video_extension_with_processed_retry(
                model=model,
                prompt=prompt,
                video_ref=video,
                config=config,
            )
        else:
            operation = self._generate_videos_with_compatibility_retry(
                model=model,
                prompt=prompt,
                video=video,
                config=config,
            )

        print("⏳ 等待视频续写完成...")
        poll_count = 0
        poll_error_count = 0
        max_poll_errors = 40
        while not operation.done:
            poll_count += 1
            time.sleep(15)
            try:
                operation = self.llm.operations.get(operation)
                poll_error_count = 0
            except Exception as e:
                poll_error_count += 1
                if poll_error_count > max_poll_errors:
                    raise
                wait_seconds = min(30, 5 * poll_error_count)
                print(
                    f"⚠️  查询视频续写状态失败 ({poll_error_count}/{max_poll_errors}): {e}，"
                    f"{wait_seconds}s 后继续轮询。"
                )
                time.sleep(wait_seconds)
                continue
            if poll_count % 6 == 0:
                print(f"   仍在生成中... ({poll_count * 15}秒)")

        if operation.error:
            error_msg = str(operation.error)
            print(f"❌ 视频续写失败: {error_msg}")
            self._set_video_generation_diagnostics(
                status="operation_error",
                operation=operation,
                error=error_msg,
                model=model,
            )
            self._record_api_usage(
                operation,
                operation="veo_video_extension",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "operation_error",
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                    "used_video_ref": video_ref is not None,
                },
            )
            return None, None

        generated_videos = self._get_generated_video_items(operation)
        if not generated_videos:
            print(f"❌ generated_videos列表为空")
            self._print_video_filter_diagnostics(operation)
            self._set_video_generation_diagnostics(
                status="empty_generated_videos",
                operation=operation,
                model=model,
            )
            self._record_api_usage(
                operation,
                operation="veo_video_extension",
                model=model,
                prompt=prompt,
                extra={
                    "calling_from": calling_from,
                    "status": "empty_generated_videos",
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "save_path": str(save_path) if save_path else None,
                    "used_video_ref": video_ref is not None,
                },
            )
            return None, None

        generated_video = generated_videos[0]
        video_file = getattr(generated_video, "video", None) or generated_video
        video_bytes = self.llm.files.download(file=video_file)
        ready_video_file = self._wait_for_file_active(video_file)

        if save_path is None:
            if self.output_dir:
                save_path = f"{self.output_dir}/video_extend_{int(time.time())}.mp4"
            else:
                save_path = f"video_extend_{int(time.time())}.mp4"

        with open(save_path, "wb") as f:
            f.write(video_bytes)

        print(f"✅ 续写视频已保存: {save_path}")
        self._set_video_generation_diagnostics(
            status="success",
            operation=operation,
            model=model,
        )

        if hasattr(operation, 'usage_metadata'):
            self._update_tokens(response=operation, model_type=model, calling_from=calling_from)
        self._record_api_usage(
            operation,
            operation="veo_video_extension",
            model=model,
            prompt=prompt,
            extra={
                "calling_from": calling_from,
                "status": "success",
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "save_path": str(save_path) if save_path else None,
                "used_video_ref": video_ref is not None,
            },
        )

        return save_path, ready_video_file
