"""
OpenAI Compatible Image Generation node for ComfyUI.

Uses HTTP REST API to generate images via DALL-E or any OpenAI-compatible endpoint.
Supports custom base_url for third-party providers.

Reference: https://platform.openai.com/docs/api-reference/images/create
- POST /v1/images/generations
- Request: model, prompt, n, size, response_format
- Response: data[].b64_json or data[].url
"""
import base64
import logging

import requests

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import tensor_to_pil, pil_to_tensor, mask_to_pil, bytes_to_tensor, sanitize_url, validate_ref_images

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://platform.openai.com/docs/api-reference/images
# dall-e-3: text-to-image only, no /images/edits support (0)
# gpt-image-1: supports up to 16 reference images via /images/edits
# dall-e-2: supports 1 reference image via /images/edits
MODEL_REF_IMAGE_LIMITS = {
    "dall-e-3": (0, 0),
    "gpt-image-1": (0, 16),
    "dall-e-2": (0, 1),
}


class OpenAIImageGenerate:
    """
    Generate images using OpenAI DALL-E or any OpenAI-compatible API.

    Supports custom base_url for third-party providers (e.g., Azure, proxy APIs).
    Uses the standard /v1/images/generations endpoint.
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
            models = BUILTIN_MODELS.get("OpenAI Compatible", ["dall-e-3"])

        saved_config = get_api_config("OpenAI Compatible")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://api.openai.com")

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
                    "default": models[0] if models else "dall-e-3"
                }),
                "size": ([
                    "1024x1024",  # Square (all models)
                    "1024x1536",  # Portrait (gpt-image-1)
                    "1536x1024",  # Landscape (gpt-image-1)
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
                    "max": 4,
                }),
                "quality": (["auto", "high", "standard"], {
                    "default": "auto"
                }),
                "ref_images": ("IMAGE",),
                "mask": ("MASK",),
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

    def generate(self, prompt, api_key, base_url, model_name, size, num_images=1,
                 quality="auto", ref_images=None, mask=None, custom_model="", seed=0):
        """
        Main execution function for OpenAI-compatible image generation.

        Reference: https://platform.openai.com/docs/api-reference/images/create
        - Generation: POST {base_url}/v1/images/generations
        - Inpainting: POST {base_url}/v1/images/edits (when image+mask provided)
          - Request: multipart form data with image, mask, prompt, model, size, n
          - Mask: PNG with transparent (alpha=0) areas to edit
        - Response: { data: [{ b64_json: "..." }, ...] }
        """
        import torch

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
        clean_url = sanitize_url(base_url) or "https://api.openai.com"

        # --- Validate ref_images compatibility ---
        validate_ref_images("OpenAI", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS)

        logger.info(
            f"[OpenAI] Starting generation | URL: {clean_url} | "
            f"Model: {effective_model} | Size: {size} | NumImages: {num_images}"
        )

        # --- Handle ref_images: use first image for inpainting ---
        image = None
        if ref_images is not None:
            ref_pils = tensor_to_pil(ref_images)
            if ref_pils:
                if mask is None:
                    logger.warning(
                        f"[OpenAI] ref_images provided but no mask connected. "
                        f"OpenAI /images/edits requires both image AND mask. "
                        f"ref_images will be ignored for text-to-image generation."
                    )
                else:
                    image = pil_to_tensor([ref_pils[0]])

        # --- Determine mode: generation vs inpainting ---
        is_inpaint = image is not None and mask is not None

        if is_inpaint:
            # === INPAINTING MODE: /v1/images/edits with multipart form data ===
            import io

            url = f"{clean_url}/v1/images/edits"
            logger.info(f"[OpenAI] INPAINT mode | URL: {url}")

            headers = {
                "Authorization": f"Bearer {api_key.strip()}"
            }

            # Convert image tensor to PNG bytes
            pil_images = tensor_to_pil(image)
            img_buf = io.BytesIO()
            pil_images[0].save(img_buf, format="PNG")
            img_buf.seek(0)

            # Convert mask tensor to RGBA PNG (transparent areas = edit region)
            mask_pil = mask_to_pil(mask)
            # OpenAI expects RGBA mask where alpha=0 means edit area
            mask_rgba = mask_pil.convert("RGBA")
            # Invert: white (255) in mask -> transparent (alpha=0) in RGBA
            import numpy as np
            mask_arr = np.array(mask_rgba)
            # Original mask: white=edit region, so alpha should be 0 where white
            gray = np.array(mask_pil)
            mask_arr[:, :, 3] = 255 - gray  # white->transparent, black->opaque
            from PIL import Image as PILImage
            mask_rgba = PILImage.fromarray(mask_arr, "RGBA")
            mask_buf = io.BytesIO()
            mask_rgba.save(mask_buf, format="PNG")
            mask_buf.seek(0)

            files = {
                "image": ("image.png", img_buf, "image/png"),
                "mask": ("mask.png", mask_buf, "image/png"),
            }
            data = {
                "model": effective_model,
                "prompt": prompt,
                "n": str(num_images),
                "size": size,
                "response_format": "b64_json",
            }

            try:
                response = requests.post(url, headers=headers, files=files,
                                         data=data, timeout=120)
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    f"[APIImage OpenAI] Inpainting request timed out after 120s."
                )
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"[APIImage OpenAI] Connection failed to {clean_url}. Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"[APIImage OpenAI] Inpainting request failed: {e}")

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
                "size": size,
                "response_format": "b64_json"
            }
            if quality and quality != "auto":
                payload["quality"] = quality

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    f"[APIImage OpenAI] Request timed out after 120 seconds. "
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
        logger.info(
            f"[OpenAI] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]}"
        )

        return (result_tensor,)
