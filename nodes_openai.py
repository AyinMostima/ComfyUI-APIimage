"""
OpenAI Compatible Image Generation node for ComfyUI.

Uses HTTP REST API to generate images via GPT Image, DALL-E, or any
OpenAI-compatible endpoint.
Supports custom base_url for third-party providers.

Reference: https://developers.openai.com/api/reference/resources/images
- POST /v1/images/generations
- POST /v1/images/edits
- Response: data[].b64_json or data[].url
"""
import base64
import io
import logging
import re

import requests

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import (
    bytes_to_tensor,
    mask_to_pil,
    sanitize_url,
    tensor_to_pil,
    validate_ref_images,
)

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://developers.openai.com/api/reference/resources/images/methods/edit
# dall-e-3: text-to-image only, no /images/edits support (0)
# GPT Image models: support up to 16 reference images via /images/edits
# dall-e-2: supports 1 reference image via /images/edits
MODEL_REF_IMAGE_LIMITS = {
    "dall-e-3": (0, 0),
    "gpt-image-2": (0, 16),
    "gpt-image-1.5": (0, 16),
    "gpt-image-1": (0, 16),
    "gpt-image-1-mini": (0, 16),
    "chatgpt-image-latest": (0, 16),
    "dall-e-2": (0, 1),
}

GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400
GPT_IMAGE_2_MAX_EDGE = 3_840


def _is_gpt_image_model(model_name):
    """Return whether a model uses the GPT Image request contract."""
    normalized = (model_name or "").strip().lower()
    return normalized.startswith("gpt-image-") or normalized == "chatgpt-image-latest"


def _is_gpt_image_2(model_name):
    """Return whether a model is GPT Image 2 or one of its snapshots."""
    normalized = (model_name or "").strip().lower()
    return normalized == "gpt-image-2" or normalized.startswith("gpt-image-2-")


def _validate_gpt_image_2_size(size):
    """Validate the flexible GPT Image 2 WIDTHxHEIGHT size contract."""
    if size == "auto":
        return

    match = re.fullmatch(r"(\d+)x(\d+)", size or "")
    if not match:
        raise ValueError(
            "[APIImage OpenAI] GPT-image-2 size must be 'auto' or WIDTHxHEIGHT."
        )

    width, height = (int(value) for value in match.groups())
    pixels = width * height
    edge_ratio = max(width, height) / min(width, height)
    if (
        width % 16 != 0
        or height % 16 != 0
        or max(width, height) > GPT_IMAGE_2_MAX_EDGE
        or edge_ratio > 3
        or pixels < GPT_IMAGE_2_MIN_PIXELS
        or pixels > GPT_IMAGE_2_MAX_PIXELS
    ):
        raise ValueError(
            "[APIImage OpenAI] GPT-image-2 size must use edges divisible by 16, "
            "an aspect ratio no wider than 3:1, each edge at most 3840 px, and "
            "655360-8294400 total pixels."
        )


def _collect_reference_images(ref_images, image1, image2, image3):
    """Collect all ComfyUI image inputs while preserving their original sizes."""
    images = []
    for tensor in (ref_images, image1, image2, image3):
        if tensor is not None:
            images.extend(tensor_to_pil(tensor))
    return images


def _model_ref_limit(model_name):
    """Resolve a known reference-image limit using the shared matching rules."""
    normalized = (model_name or "").lower()
    for known_model, bounds in MODEL_REF_IMAGE_LIMITS.items():
        if known_model == normalized or known_model in normalized:
            return bounds
    return None


def _add_request_options(
    payload,
    model_name,
    quality,
    output_format,
    output_compression,
    background,
    moderation,
):
    """Add only parameters supported by the selected OpenAI model family."""
    if _is_gpt_image_model(model_name):
        payload["output_format"] = output_format
        if quality != "auto":
            payload["quality"] = quality
        if output_format in {"jpeg", "webp"} and output_compression != 100:
            payload["output_compression"] = output_compression
        if background != "auto":
            payload["background"] = background
        if moderation != "auto":
            payload["moderation"] = moderation
        return

    payload["response_format"] = "b64_json"
    if quality != "auto":
        payload["quality"] = quality


class OpenAIImageGenerate:
    """
    Generate or edit images using GPT Image, legacy DALL-E, or compatible APIs.

    Supports custom base_url for third-party providers (e.g., Azure, proxy APIs).
    Uses /v1/images/generations for text-only requests and /v1/images/edits
    when one or more reference images are connected.
    """

    CATEGORY = "APIImage/OpenAI"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("OpenAI Compatible")
        if not models:
            models = BUILTIN_MODELS.get("OpenAI Compatible", ["gpt-image-2"])

        saved_config = get_api_config("OpenAI Compatible")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://api.openai.com")
        saved_model = saved_config.get("model_name", "")
        default_model = saved_model if saved_model in models else models[0]

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "sk-... (OpenAI API Key)"
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": "https://api.openai.com"
                }),
                "model_name": (models, {
                    "default": default_model if models else "gpt-image-2"
                }),
                "size": ([
                    "auto",       # GPT Image automatic sizing
                    "1024x1024",  # Square (all models)
                    "1024x1536",  # Portrait (GPT Image)
                    "1536x1024",  # Landscape (GPT Image)
                    "2048x2048",  # 2K square (GPT Image 2)
                    "2048x1152",  # 2K landscape (GPT Image 2)
                    "3840x2160",  # 4K landscape (GPT Image 2)
                    "2160x3840",  # 4K portrait (GPT Image 2)
                    "1024x1792",  # Portrait (DALL-E 3)
                    "1792x1024",  # Landscape (DALL-E 3)
                    "512x512",    # DALL-E 2
                    "256x256",    # DALL-E 2
                ], {
                    "default": "1024x1024"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }),
                "quality": (["auto", "low", "medium", "high", "standard", "hd"], {
                    "default": "auto"
                }),
                "ref_images": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "mask": ("MASK",),
                "custom_model": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty to use dropdown; fill to override"
                }),
                "custom_size": ("STRING", {
                    "default": "",
                    "placeholder": "GPT-image-2 override, for example 1536x864"
                }),
                "output_format": (["png", "jpeg", "webp"], {
                    "default": "png"
                }),
                "output_compression": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100,
                }),
                "background": (["auto", "opaque", "transparent"], {
                    "default": "auto"
                }),
                "moderation": (["auto", "low"], {
                    "default": "auto"
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

    def generate(self, prompt, api_key, base_url, model_name, size, num_images=1,
                 quality="auto", ref_images=None,
                 image1=None, image2=None, image3=None,
                 mask=None, custom_model="", seed=0,
                 custom_size="", output_format="png", output_compression=100,
                 background="auto", moderation="auto"):
        """
        Main execution function for OpenAI-compatible image generation.

        Reference: https://developers.openai.com/api/reference/resources/images
        - Generation: POST {base_url}/v1/images/generations
        - Editing: POST {base_url}/v1/images/edits (when images are provided)
          - Request: multipart form data with image(s), prompt, model, size, n
          - Mask: PNG with transparent (alpha=0) areas to edit
        - Response: { data: [{ b64_json: "..." }, ...] }
        """
        # --- Input Validation ---
        if not prompt or not prompt.strip():
            raise ValueError(
                "[APIImage OpenAI] Prompt is empty. "
                "Please enter a text prompt describing the image you want to generate."
            )

        if not api_key or not api_key.strip():
            raise ValueError(
                "[APIImage OpenAI] API Key is not set. "
                "Please provide a valid API key."
            )

        if not base_url or not base_url.strip():
            raise ValueError(
                "[APIImage OpenAI] Base URL is not set. "
                "Please provide a valid API endpoint URL (e.g., https://api.openai.com)."
            )

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        effective_size = custom_size.strip() if custom_size and custom_size.strip() else size
        clean_url = sanitize_url(base_url) or "https://api.openai.com"

        if not effective_model or not effective_model.strip():
            raise ValueError("[APIImage OpenAI] Model name is empty.")
        if num_images < 1 or num_images > 10:
            raise ValueError("[APIImage OpenAI] num_images must be between 1 and 10.")
        if effective_model.lower() == "dall-e-3" and num_images != 1:
            raise ValueError("[APIImage OpenAI] DALL-E 3 supports only num_images=1.")
        if output_format not in {"png", "jpeg", "webp"}:
            raise ValueError("[APIImage OpenAI] Invalid output_format.")
        if not 0 <= output_compression <= 100:
            raise ValueError("[APIImage OpenAI] output_compression must be 0-100.")
        if background not in {"auto", "opaque", "transparent"}:
            raise ValueError("[APIImage OpenAI] Invalid background option.")
        if moderation not in {"auto", "low"}:
            raise ValueError("[APIImage OpenAI] Invalid moderation option.")

        if _is_gpt_image_model(effective_model):
            if quality not in {"auto", "low", "medium", "high"}:
                raise ValueError(
                    "[APIImage OpenAI] GPT Image quality must be auto, low, medium, or high."
                )
        elif effective_model.lower() == "dall-e-3":
            if quality not in {"auto", "standard", "hd"}:
                raise ValueError(
                    "[APIImage OpenAI] DALL-E 3 quality must be auto, standard, or hd."
                )
        elif effective_model.lower() == "dall-e-2" and quality not in {"auto", "standard"}:
            raise ValueError(
                "[APIImage OpenAI] DALL-E 2 quality must be auto or standard."
            )

        if _is_gpt_image_2(effective_model):
            _validate_gpt_image_2_size(effective_size)
            if background == "transparent":
                raise ValueError(
                    "[APIImage OpenAI] GPT-image-2 does not support transparent backgrounds."
                )

        # --- Validate ref_images compatibility ---
        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images(
            "OpenAI",
            effective_model,
            ref_images,
            MODEL_REF_IMAGE_LIMITS,
            extra_count=extra_img_count,
        )
        reference_images = _collect_reference_images(
            ref_images,
            image1,
            image2,
            image3,
        )
        ref_limit = _model_ref_limit(effective_model)
        if ref_limit and ref_limit[1] > 0 and len(reference_images) > ref_limit[1]:
            raise ValueError(
                f"[APIImage OpenAI] Model '{effective_model}' supports at most "
                f"{ref_limit[1]} reference images, but {len(reference_images)} were provided."
            )
        if mask is not None and not reference_images:
            raise ValueError(
                "[APIImage OpenAI] A mask requires at least one reference image."
            )

        logger.info(
            f"[OpenAI] Starting generation | URL: {clean_url} | "
            f"Model: {effective_model} | Size: {effective_size} | NumImages: {num_images}"
        )

        if reference_images:
            # === EDITING MODE: /v1/images/edits with multipart form data ===

            url = f"{clean_url}/v1/images/edits"
            logger.info(
                f"[OpenAI] EDIT mode | URL: {url} | References: {len(reference_images)}"
            )

            headers = {
                "Authorization": f"Bearer {api_key.strip()}"
            }

            buffers = []
            files = []
            image_field = "image[]" if _is_gpt_image_model(effective_model) else "image"
            for index, image in enumerate(reference_images):
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_buffer.seek(0)
                buffers.append(image_buffer)
                files.append(
                    (
                        image_field,
                        (f"image_{index}.png", image_buffer, "image/png"),
                    )
                )

            if mask is not None:
                mask_pil = mask_to_pil(mask)
                if mask_pil.size != reference_images[0].size:
                    raise ValueError(
                        "[APIImage OpenAI] Mask dimensions must match the first reference image."
                    )

                import numpy as np
                from PIL import Image as PILImage

                mask_array = np.array(mask_pil.convert("RGBA"))
                mask_array[:, :, 3] = 255 - np.array(mask_pil)
                mask_rgba = PILImage.fromarray(mask_array, "RGBA")
                mask_buffer = io.BytesIO()
                mask_rgba.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)
                buffers.append(mask_buffer)
                files.append(("mask", ("mask.png", mask_buffer, "image/png")))

            data = {
                "model": effective_model,
                "prompt": prompt,
                "n": str(num_images),
                "size": effective_size,
            }
            _add_request_options(
                data,
                effective_model,
                quality,
                output_format,
                output_compression,
                background,
                moderation,
            )

            try:
                response = requests.post(url, headers=headers, files=files,
                                         data=data, timeout=300)
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    "[APIImage OpenAI] Editing request timed out after 300 seconds."
                )
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"[APIImage OpenAI] Connection failed to {clean_url}. Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"[APIImage OpenAI] Editing request failed: {e}")

        else:
            # === GENERATION MODE: /v1/images/generations with JSON ===
            url = f"{clean_url}/v1/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
            payload = {
                "model": effective_model,
                "prompt": prompt,
                "n": num_images,
                "size": effective_size,
            }
            _add_request_options(
                payload,
                effective_model,
                quality,
                output_format,
                output_compression,
                background,
                moderation,
            )

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300)
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    f"[APIImage OpenAI] Request timed out after 300 seconds. "
                    f"Check your network connection or try a simpler prompt."
                )
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"[APIImage OpenAI] Connection failed to {clean_url}. "
                    f"Please verify the base URL is correct and accessible. Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"[APIImage OpenAI] Request failed: {e}")

        # --- Handle HTTP Errors ---
        if response.status_code != 200:
            try:
                error_body = response.json()
                error_msg = error_body.get("error", {}).get("message", response.text[:500])
            except Exception:
                error_msg = response.text[:500]

            if response.status_code == 401:
                raise RuntimeError(
                    f"[APIImage OpenAI] Authentication failed (401). "
                    f"Please check your API key."
                )
            elif response.status_code == 403:
                raise RuntimeError(
                    f"[APIImage OpenAI] Permission denied (403). "
                    f"Your API key may not have access to this model."
                )
            elif response.status_code == 404:
                raise RuntimeError(
                    f"[APIImage OpenAI] Endpoint not found (404). "
                    f"Please verify the base URL: {clean_url}"
                )
            elif response.status_code == 429:
                raise RuntimeError(
                    f"[APIImage OpenAI] Rate limit exceeded (429). "
                    f"Please wait and retry. Detail: {error_msg}"
                )
            else:
                raise RuntimeError(
                    f"[APIImage OpenAI] API Error ({response.status_code}): {error_msg}"
                )

        # --- Parse Response ---
        try:
            json_response = response.json()
        except Exception:
            raise RuntimeError(
                f"[APIImage OpenAI] Invalid JSON response from server."
            )

        images_data = []
        data_list = json_response.get("data") or []

        for item in data_list:
            b64_data = item.get("b64_json")
            if b64_data:
                try:
                    images_data.append(base64.b64decode(b64_data))
                except Exception as e:
                    logger.error(f"[OpenAI] Failed to decode base64: {e}")
            elif item.get("url"):
                # Download image from URL
                try:
                    img_resp = requests.get(item["url"], timeout=60)
                    img_resp.raise_for_status()
                    images_data.append(img_resp.content)
                except Exception as e:
                    logger.error(f"[OpenAI] Failed to download image: {e}")

        if not images_data:
            raise RuntimeError(
                f"[APIImage OpenAI] No image data returned. "
                f"Raw response: {str(json_response)[:500]}"
            )

        # Convert bytes to tensor
        result_tensor = bytes_to_tensor(images_data)

        # Extract current Image API usage with legacy-compatible fallbacks.
        usage_str = "N/A"
        try:
            usage = json_response.get("usage")
            if usage:
                input_t = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_t = usage.get("output_tokens", usage.get("completion_tokens", 0))
                total_t = usage.get("total_tokens", 0)
                usage_str = f"Input: {input_t} | Output: {output_t} | Total: {total_t}"
        except Exception:
            pass

        logger.info(
            f"[OpenAI] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]} | "
            f"Tokens: {usage_str}"
        )

        return (result_tensor,)
