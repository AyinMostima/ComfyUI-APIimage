"""
Qwen Image Generation node for ComfyUI.

Uses the DashScope SDK (dashscope.MultiModalConversation) for Qwen image
generation and editing models.

Reference: https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-wanxiang
- Text-to-image: qwen-image-plus
- Image editing:  qwen-image-edit (supports multiple reference images)
- SDK: dashscope.MultiModalConversation.call()
- Parameters: model, messages, result_format, watermark, negative_prompt, prompt_extend, size
"""
import logging
import io
import base64
import requests

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import tensor_to_pil, mask_to_pil, bytes_to_tensor, sanitize_url, validate_ref_images

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-wanxiang
# qwen-image-plus: text-to-image only, no reference images (0)
# qwen-image-edit: editing model, requires 1-3 reference images per request
MODEL_REF_IMAGE_LIMITS = {
    "qwen-image-plus": (0, 0),
    "qwen-image-edit": (1, 3),
}


class QwenImageGenerate:
    """
    Generate or edit images using Alibaba Qwen (Tongyi Wanxiang) via DashScope SDK.

    Supports:
    - Text-to-image generation (qwen-image-plus)
    - Image editing with multiple reference images (qwen-image-edit)
    - Watermark control, negative prompt, prompt extension
    """

    CATEGORY = "APIImage/Qwen"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "text_response",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("Qwen Image")
        if not models:
            models = BUILTIN_MODELS.get("Qwen Image", ["qwen-image-plus"])

        saved_config = get_api_config("Qwen Image")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://dashscope.aliyuncs.com/api/v1")

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "sk-... (DashScope API Key)"
                }),
                "model_name": (models, {
                    "default": models[0] if models else "qwen-image-plus"
                }),
                "size": ([
                    "1664*928",   # 16:9 (default)
                    "1472*1104",  # 4:3
                    "1328*1328",  # 1:1
                    "1104*1472",  # 3:4
                    "928*1664",   # 9:16
                ], {
                    "default": "1664*928"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": "CN: dashscope.aliyuncs.com | Intl: dashscope-intl.aliyuncs.com"
                }),
                "ref_images": ("IMAGE",),
                "mask": ("MASK",),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Negative prompt (what to avoid)"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                }),
                "prompt_extend": ("BOOLEAN", {
                    "default": True,
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty to use dropdown; fill to override"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0x7FFFFFFF,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _upload_image_and_get_url(self, pil_img):
        """
        Convert PIL image to a temporary base64 data URI for DashScope.
        DashScope accepts both URLs and local file paths.
        We save to a temp file and return the path.
        """
        import tempfile
        import os

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        # Save to temp file (DashScope SDK can read local file paths)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(buf.getvalue())
        tmp.close()
        return tmp.name

    def generate(self, prompt, api_key, model_name, size, num_images=1,
                 base_url="", ref_images=None,
                 mask=None,
                 negative_prompt="", watermark=False, prompt_extend=True,
                 custom_model="", seed=0):
        """
        Main execution function for Qwen image generation.

        Reference: https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-wanxiang
        - SDK: dashscope.MultiModalConversation.call(
              api_key=..., model=..., messages=[{role, content}],
              result_format='message', watermark=..., negative_prompt=..., size=...)
        - Messages content format: [{"image": url_or_path}, {"text": prompt}]
        - Response: response.output.choices[0].message.content[0] contains result URL
        - Inpainting: mask image passed as additional reference with editing instruction
        """
        import torch
        import os

        # --- Input Validation ---
        if not prompt or not prompt.strip():
            raise ValueError(
                "[APIImage Qwen] Prompt is empty. "
                "Please enter a text prompt describing the image you want to generate."
            )

        if not api_key or not api_key.strip():
            raise ValueError(
                "[APIImage Qwen] API Key is not set. "
                "Please provide a valid DashScope API key (format: sk-...). "
                "Get one at: https://dashscope.console.aliyun.com/"
            )

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name

        # --- Validate ref_images compatibility ---
        validate_ref_images("Qwen", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS)

        logger.info(
            f"[Qwen] Starting generation | Model: {effective_model} | "
            f"Size: {size} | HasRefImages: {ref_images is not None} | "
            f"HasMask: {mask is not None} | Seed: {seed}"
        )

        # --- Import SDK ---
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as e:
            raise ImportError(
                f"[APIImage Qwen] Missing required package: {e}. "
                f"Please run: pip install dashscope"
            )

        # Set base URL if provided
        effective_url = sanitize_url(base_url)
        if effective_url:
            dashscope.base_http_api_url = effective_url

        # --- Build Message Content ---
        content = []
        temp_files = []

        # Process reference images from ref_images (no limit applied)
        ref_count = 0

        if ref_images is not None:
            try:
                batch_pils = tensor_to_pil(ref_images)
                for bidx, bimg in enumerate(batch_pils):
                    tmp_path = self._upload_image_and_get_url(bimg)
                    temp_files.append(tmp_path)
                    content.append({"image": f"file://{tmp_path}"})
                    ref_count += 1
                    logger.info(f"[Qwen] Added ref image {bidx+1}/{len(batch_pils)}: {tmp_path}")
            except Exception as e:
                logger.warning(f"[Qwen] Failed to process ref_images: {e}")

        if ref_count > 0:
            logger.info(f"[Qwen] Total reference images: {ref_count}")

        # Add text prompt
        content.append({"text": prompt})

        # Add mask as additional image if provided (for inpainting)
        # Mask is sent as a separate image; the editing model uses it
        # to identify which regions to modify
        if mask is not None:
            try:
                mask_pil = mask_to_pil(mask)
                tmp_path = self._upload_image_and_get_url(mask_pil)
                temp_files.append(tmp_path)
                content.append({"image": f"file://{tmp_path}"})
                logger.info(f"[Qwen] Added mask image: {tmp_path}")
            except Exception as e:
                logger.warning(f"[Qwen] Failed to process mask: {e}")

        messages = [{"role": "user", "content": content}]

        # --- Call API (loop for multi-image) ---
        all_images_data = []
        text_messages = []

        try:
            for gen_idx in range(num_images):
                try:
                    call_kwargs = {
                        "api_key": api_key.strip(),
                        "model": effective_model,
                        "messages": messages,
                        "result_format": "message",
                        "stream": False,
                        "watermark": watermark,
                    }
                    if negative_prompt and negative_prompt.strip():
                        call_kwargs["negative_prompt"] = negative_prompt.strip()
                    if size:
                        call_kwargs["size"] = size.replace("x", "*")
                    if "edit" not in effective_model.lower():
                        call_kwargs["prompt_extend"] = prompt_extend

                    response = MultiModalConversation.call(**call_kwargs)

                except Exception as e:
                    error_str = str(e)
                    if "InvalidApiKey" in error_str or "Invalid API-key" in error_str:
                        raise RuntimeError(
                            f"[APIImage Qwen] API Key invalid. Please check:\n"
                            f"1. Your DashScope API key is correct (format: sk-...)\n"
                            f"2. Base URL matches your account region:\n"
                            f"   - China: https://dashscope.aliyuncs.com/api/v1\n"
                            f"   - International: https://dashscope-intl.aliyuncs.com/api/v1\n"
                            f"Get a key at: https://dashscope.console.aliyun.com/"
                        )
                    elif "UploadFileException" in error_str or "upload" in error_str.lower():
                        raise RuntimeError(
                            f"[APIImage Qwen] Image upload failed: {error_str}\n"
                            f"This usually means your API key is invalid or expired."
                        )
                    else:
                        raise RuntimeError(f"[APIImage Qwen] API Error: {error_str}")

                # Parse response for this iteration
                if hasattr(response, 'status_code') and response.status_code != 200:
                    error_msg = getattr(response, 'message', str(response))
                    error_code = getattr(response, 'code', 'unknown')
                    raise RuntimeError(
                        f"[APIImage Qwen] API Error ({response.status_code}): "
                        f"Code={error_code}, Message={error_msg}"
                    )

                try:
                    choices = response.output.get("choices", []) if isinstance(response.output, dict) else []
                    if not choices and isinstance(response.output, dict):
                        result_url = response.output.get("result_url", "")
                        if result_url:
                            img_resp = requests.get(result_url, timeout=60)
                            img_resp.raise_for_status()
                            all_images_data.append(img_resp.content)

                    for choice in choices:
                        msg = choice.get("message", {})
                        msg_content = msg.get("content", [])
                        if isinstance(msg_content, list):
                            for item in msg_content:
                                if isinstance(item, dict):
                                    if "image" in item:
                                        img_url = item["image"]
                                        logger.info(f"[Qwen] Downloading: {img_url[:100]}...")
                                        img_resp = requests.get(img_url, timeout=60)
                                        img_resp.raise_for_status()
                                        all_images_data.append(img_resp.content)
                                    elif "text" in item:
                                        text_messages.append(item["text"])
                                elif isinstance(item, str):
                                    text_messages.append(item)
                        elif isinstance(msg_content, str):
                            text_messages.append(msg_content)
                except Exception as e:
                    logger.error(f"[Qwen] Failed to parse response: {e}")
                    raw = str(response) if response else "No response"
                    raise RuntimeError(
                        f"[APIImage Qwen] Failed to parse response: {e}. Raw: {raw[:500]}"
                    )

                if gen_idx < num_images - 1:
                    logger.info(f"[Qwen] Generated image {gen_idx+1}/{num_images}")

        finally:
            # Clean up temp files
            for tf in temp_files:
                try:
                    os.remove(tf)
                except Exception:
                    pass

        text_response = "\n".join(text_messages) if text_messages else ""

        if not all_images_data:
            raise RuntimeError(
                f"[APIImage Qwen] No image data returned. "
                f"Model response: {text_response[:500] if text_response else 'None'}"
            )

        result_tensor = bytes_to_tensor(all_images_data)
        logger.info(
            f"[Qwen] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]}"
        )

        return (result_tensor, text_response,)
