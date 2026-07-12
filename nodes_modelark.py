"""
BytePlus ModelArk image generation node for ComfyUI.

Reference: https://docs.byteplus.com/en/docs/ModelArk/1541523
- POST /api/v3/images/generations
- Authentication: Authorization: Bearer ARK_API_KEY
- Response: data[].b64_json or data[].url
"""
import base64
import io
import logging
import re

import requests

from .config import BUILTIN_MODELS, get_api_config, get_model_list
from .utils import bytes_to_tensor, sanitize_url, tensor_to_pil


logger = logging.getLogger("ComfyUI-APIImage")

DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
MAX_INPUT_PIXELS = 36_000_000
MAX_INPUT_BYTES = 30 * 1024 * 1024
MIN_OUTPUT_PIXELS = 921_600
MAX_OUTPUT_PIXELS = 4_624_220
MODEL_FAMILY_HINTS = {
    "auto": None,
    "seedream-5-0-pro": "pro",
    "seedream-5-0-lite": "lite",
    "seedream-4-5": "4.5",
    "seedream-4-0": "4.0",
    "seedream-3-0-t2i": "3.0-t2i",
}


def _model_family(model_name):
    """Map a model ID or endpoint ID to its documented capability family."""
    normalized = (model_name or "").strip().lower()
    if "seedream-5-0-pro" in normalized:
        return "pro"
    if "seedream-5-0-lite" in normalized or normalized == "seedream-5-0-260128":
        return "lite"
    if "seedream-4-5" in normalized:
        return "4.5"
    if "seedream-4-0" in normalized:
        return "4.0"
    if "seedream-3-0-t2i" in normalized:
        return "3.0-t2i"
    return "unknown"


def _collect_reference_images(ref_images, image1, image2, image3):
    """Collect batch and individual ComfyUI image inputs in connection order."""
    images = []
    for tensor in (ref_images, image1, image2, image3):
        if tensor is not None:
            images.extend(tensor_to_pil(tensor))
    return images


def _validate_reference_image(image, index):
    """Validate ModelArk reference-image dimensions before encoding."""
    width, height = image.size
    if width <= 14 or height <= 14:
        raise ValueError(
            f"[APIImage ModelArk] Reference image {index} must be larger than 14x14 px."
        )
    if max(width, height) / min(width, height) > 16:
        raise ValueError(
            f"[APIImage ModelArk] Reference image {index} aspect ratio must be within 1:16-16:1."
        )
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            f"[APIImage ModelArk] Reference image {index} exceeds 36000000 pixels."
        )


def _image_to_data_url(image, index):
    """Encode a reference image as a documented lowercase JPEG data URL."""
    _validate_reference_image(image, index)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    if len(image_bytes) > MAX_INPUT_BYTES:
        raise ValueError(
            f"[APIImage ModelArk] Reference image {index} exceeds the 30 MB limit."
        )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _validate_output_size(size, family):
    """Validate Seedream 4.x and 5.x resolution levels or exact dimensions."""
    if family not in {"pro", "lite", "4.5", "4.0"}:
        return
    if size in {"1K", "2K"}:
        return

    match = re.fullmatch(r"(\d+)x(\d+)", size or "")
    if not match:
        raise ValueError(
            "[APIImage ModelArk] ModelArk size must be 1K, 2K, or WIDTHxHEIGHT."
        )

    width, height = (int(value) for value in match.groups())
    pixels = width * height
    if (
        width % 16 != 0
        or height % 16 != 0
        or max(width, height) / min(width, height) > 16
        or pixels < MIN_OUTPUT_PIXELS
        or pixels > MAX_OUTPUT_PIXELS
    ):
        raise ValueError(
            "[APIImage ModelArk] ModelArk size must use edges divisible by 16, "
            "an aspect ratio within 1:16-16:1, and 921600-4624220 total pixels."
        )


def _generation_url(base_url):
    """Accept either a regional API base URL or the complete generation URL."""
    if base_url.endswith("/images/generations"):
        return base_url
    return f"{base_url}/images/generations"


def _error_text(error):
    """Format ModelArk top-level and per-image error objects consistently."""
    if isinstance(error, dict):
        code = error.get("code", "UnknownError")
        message = error.get("message", "No error message")
        return f"{code}: {message}"
    return str(error or "Unknown error")


class ModelArkImageGenerate:
    """Generate or edit images through the BytePlus ModelArk REST API."""

    CATEGORY = "APIImage/ModelArk"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("BytePlus ModelArk")
        if not models:
            models = BUILTIN_MODELS.get(
                "BytePlus ModelArk",
                ["dola-seedream-5-0-pro-260628"],
            )

        saved_config = get_api_config("BytePlus ModelArk")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", DEFAULT_BASE_URL)
        saved_model = saved_config.get("model_name", "")
        default_model = saved_model if saved_model in models else models[0]

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter a Seedream image prompt...",
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "BytePlus ModelArk API Key",
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": DEFAULT_BASE_URL,
                }),
                "model_name": (models, {
                    "default": default_model,
                }),
                "size": ([
                    "2K",
                    "1K",
                    "1024x1024",
                    "1152x864",
                    "864x1152",
                    "1424x800",
                    "800x1424",
                    "1248x832",
                    "832x1248",
                    "1568x672",
                    "2048x2048",
                    "2368x1776",
                    "1776x2368",
                    "2816x1584",
                    "1584x2816",
                    "2496x1664",
                    "1664x2496",
                    "3136x1344",
                ], {
                    "default": "2K",
                }),
            },
            "optional": {
                "ref_images": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "custom_model": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty to use the model dropdown",
                }),
                "model_family": (list(MODEL_FAMILY_HINTS), {
                    "default": "auto",
                }),
                "custom_size": ("STRING", {
                    "default": "",
                    "placeholder": "Optional WIDTHxHEIGHT override",
                }),
                "sequential_image_generation": (["disabled", "auto"], {
                    "default": "disabled",
                }),
                "max_images": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 15,
                }),
                "output_format": (["png", "jpeg"], {
                    "default": "png",
                }),
                "response_format": (["b64_json", "url"], {
                    "default": "b64_json",
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                }),
                "optimize_prompt_mode": (["standard", "fast", "disabled"], {
                    "default": "standard",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate(
        self,
        prompt,
        api_key,
        base_url,
        model_name,
        size,
        ref_images=None,
        image1=None,
        image2=None,
        image3=None,
        custom_model="",
        custom_size="",
        sequential_image_generation="disabled",
        max_images=15,
        output_format="png",
        response_format="b64_json",
        watermark=False,
        optimize_prompt_mode="standard",
        seed=-1,
        guidance_scale=2.5,
        model_family="auto",
    ):
        """Execute a non-streaming ModelArk image generation request."""
        if not prompt or not prompt.strip():
            raise ValueError("[APIImage ModelArk] Prompt is empty.")
        if not api_key or not api_key.strip():
            raise ValueError("[APIImage ModelArk] API Key is not set.")
        if not base_url or not base_url.strip():
            raise ValueError("[APIImage ModelArk] Base URL is not set.")

        effective_model = (
            custom_model.strip()
            if custom_model and custom_model.strip()
            else model_name
        )
        effective_size = (
            custom_size.strip()
            if custom_size and custom_size.strip()
            else size
        )
        if not effective_model or not effective_model.strip():
            raise ValueError("[APIImage ModelArk] Model name is empty.")
        if model_family not in MODEL_FAMILY_HINTS:
            raise ValueError("[APIImage ModelArk] Invalid model family hint.")

        clean_url = sanitize_url(base_url) or DEFAULT_BASE_URL
        family = MODEL_FAMILY_HINTS[model_family] or _model_family(effective_model)
        _validate_output_size(effective_size, family)

        if sequential_image_generation not in {"disabled", "auto"}:
            raise ValueError("[APIImage ModelArk] Invalid sequential generation mode.")
        if not 1 <= max_images <= 15:
            raise ValueError("[APIImage ModelArk] max_images must be between 1 and 15.")
        if output_format not in {"png", "jpeg"}:
            raise ValueError("[APIImage ModelArk] Invalid output_format.")
        if response_format not in {"url", "b64_json"}:
            raise ValueError("[APIImage ModelArk] Invalid response_format.")
        if optimize_prompt_mode not in {"standard", "fast", "disabled"}:
            raise ValueError("[APIImage ModelArk] Invalid prompt optimization mode.")
        if not -1 <= seed <= 0x7FFFFFFF:
            raise ValueError("[APIImage ModelArk] seed must be between -1 and 2147483647.")
        if not 1.0 <= guidance_scale <= 10.0:
            raise ValueError("[APIImage ModelArk] guidance_scale must be between 1 and 10.")

        reference_images = _collect_reference_images(
            ref_images,
            image1,
            image2,
            image3,
        )
        max_references = {
            "pro": 10,
            "lite": 14,
            "4.5": 14,
            "4.0": 14,
            "3.0-t2i": 0,
            "unknown": 14,
        }[family]
        if reference_images and max_references == 0:
            raise ValueError(
                f"[APIImage ModelArk] Model '{effective_model}' does not support reference images."
            )
        if len(reference_images) > max_references:
            raise ValueError(
                f"[APIImage ModelArk] Model '{effective_model}' supports at most "
                f"{max_references} reference images."
            )

        payload = {
            "model": effective_model,
            "prompt": prompt,
            "size": effective_size,
            "response_format": response_format,
            "watermark": bool(watermark),
        }

        if reference_images:
            encoded_images = [
                _image_to_data_url(image, index)
                for index, image in enumerate(reference_images, start=1)
            ]
            payload["image"] = (
                encoded_images[0] if len(encoded_images) == 1 else encoded_images
            )

        if family in {"pro", "lite"}:
            payload["output_format"] = output_format

        if family in {"lite", "4.5", "4.0"}:
            payload["sequential_image_generation"] = sequential_image_generation
            if sequential_image_generation == "auto":
                if len(reference_images) + max_images > 15:
                    raise ValueError(
                        "[APIImage ModelArk] The input and output image count must not exceed 15."
                    )
                payload["sequential_image_generation_options"] = {
                    "max_images": max_images,
                }

        if family in {"pro", "lite", "4.5", "4.0"} and optimize_prompt_mode != "disabled":
            if optimize_prompt_mode == "fast" and family in {"pro", "lite", "4.5"}:
                raise ValueError(
                    f"[APIImage ModelArk] Model family {family} does not support fast prompt optimization."
                )
            payload["optimize_prompt_options"] = {
                "mode": optimize_prompt_mode,
            }

        if family == "3.0-t2i":
            payload["seed"] = seed
            payload["guidance_scale"] = guidance_scale

        url = _generation_url(clean_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }
        logger.info(
            f"[ModelArk] Starting generation | URL: {url} | Model: {effective_model} | "
            f"Size: {effective_size} | References: {len(reference_images)}"
        )

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=300,
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "[APIImage ModelArk] Request timed out after 300 seconds."
            )
        except requests.exceptions.ConnectionError as error:
            raise RuntimeError(
                f"[APIImage ModelArk] Connection failed to {clean_url}. Error: {error}"
            )
        except Exception as error:
            raise RuntimeError(f"[APIImage ModelArk] Request failed: {error}")

        if response.status_code != 200:
            try:
                response_error = response.json().get("error")
                error_message = _error_text(response_error or response.text[:500])
            except Exception:
                error_message = response.text[:500]

            if response.status_code == 401:
                raise RuntimeError(
                    "[APIImage ModelArk] Authentication failed (401). Check the API key."
                )
            if response.status_code == 403:
                raise RuntimeError(
                    f"[APIImage ModelArk] Permission denied (403): {error_message}"
                )
            if response.status_code == 429:
                raise RuntimeError(
                    f"[APIImage ModelArk] Rate limit exceeded (429): {error_message}"
                )
            raise RuntimeError(
                f"[APIImage ModelArk] API Error ({response.status_code}): {error_message}"
            )

        try:
            json_response = response.json()
        except Exception:
            raise RuntimeError("[APIImage ModelArk] Invalid JSON response from server.")

        if json_response.get("error"):
            raise RuntimeError(
                f"[APIImage ModelArk] API Error: {_error_text(json_response['error'])}"
            )

        image_bytes = []
        item_errors = []
        for item in json_response.get("data") or []:
            if item.get("error"):
                item_errors.append(_error_text(item["error"]))
                continue
            if item.get("b64_json"):
                try:
                    image_bytes.append(base64.b64decode(item["b64_json"], validate=True))
                except Exception as error:
                    item_errors.append(f"Base64DecodeError: {error}")
                continue
            if item.get("url"):
                try:
                    download = requests.get(item["url"], timeout=60)
                    download.raise_for_status()
                    image_bytes.append(download.content)
                except Exception as error:
                    item_errors.append(f"DownloadError: {error}")
                continue
            item_errors.append("InvalidData: image item has no data")

        if not image_bytes:
            details = "; ".join(item_errors) if item_errors else "No data items"
            raise RuntimeError(
                f"[APIImage ModelArk] No image data returned. Errors: {details}"
            )

        result_tensor = bytes_to_tensor(image_bytes)
        usage = json_response.get("usage") or {}
        logger.info(
            f"[ModelArk] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | "
            f"Size: {result_tensor.shape[1]}x{result_tensor.shape[2]} | "
            f"Generated: {usage.get('generated_images', result_tensor.shape[0])} | "
            f"InputImages: {usage.get('input_images', len(reference_images))} | "
            f"OutputTokens: {usage.get('output_tokens', 'N/A')} | "
            f"TotalTokens: {usage.get('total_tokens', 'N/A')}"
        )
        if item_errors:
            logger.warning(
                f"[ModelArk] Some images failed: {'; '.join(item_errors)}"
            )
        return (result_tensor,)
