"""
GLM Image Generation node for ComfyUI.

Uses the ZhipuAI REST API for image generation via GLM-Image / CogView models.

Reference: https://open.bigmodel.cn/dev/api/image/cogview
- Endpoint: POST /paas/v4/images/generations
- Auth: Bearer token in Authorization header
- Request: { model, prompt, quality, size, user_id }
- Response: { data: [{ url: "..." }] }
- Models: glm-image, cogview-4-250304
- Size (glm-image): 1280x1280, 1568x1056, 1056x1568, 1472x1088, 1088x1472, 1728x960, 960x1728
- Size (cogview-4): 1024x1024, 768x1344, 864x1152, 1344x768, 1152x864, 1440x720, 720x1440
"""
import logging
import base64

import requests as http_requests

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import bytes_to_tensor, sanitize_url, validate_ref_images

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://open.bigmodel.cn/dev/api/image/cogview
# GLM API (glm-image, cogview-4) is text-to-image only.
# No models support reference images or image editing.
MODEL_REF_IMAGE_LIMITS = {
    "glm-image": (0, 0),
    "cogview-4-250304": (0, 0),
}


class GLMImageGenerate:
    """
    Generate images using ZhipuAI GLM-Image / CogView API.

    Supports text-to-image generation with quality and size control.
    Uses REST API with Bearer token authentication.
    """

    CATEGORY = "APIImage/GLM"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("GLM Image")
        if not models:
            models = BUILTIN_MODELS.get("GLM Image", ["glm-image"])

        saved_config = get_api_config("GLM Image")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://open.bigmodel.cn/api")

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "API Key (ZhipuAI)"
                }),
                "model_name": (models, {
                    "default": models[0] if models else "glm-image"
                }),
                "quality": (["hd", "standard"], {
                    "default": "hd"
                }),
                "size": ([
                    "1280x1280", "1568x1056", "1056x1568",
                    "1472x1088", "1088x1472", "1728x960", "960x1728",
                    "1024x1024", "768x1344", "864x1152",
                    "1344x768", "1152x864", "1440x720", "720x1440",
                ], {
                    "default": "1280x1280"
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
                    "placeholder": "https://open.bigmodel.cn/api"
                }),
                "ref_images": ("IMAGE",),
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

    def generate(self, prompt, api_key, model_name, quality, size, num_images=1,
                 base_url="", ref_images=None, custom_model="", seed=0):
        """
        Main execution function for GLM image generation.

        Reference: https://open.bigmodel.cn/dev/api/image/cogview
        - Endpoint: POST {base_url}/paas/v4/images/generations
        - Request body: { model, prompt, quality, size }
        - Authorization: Bearer {api_key}
        - Response: { created, data: [{ url: "..." }] }
        - Image URL is temporary (30 days), downloaded immediately.
        """
        import torch

        # --- Input Validation ---
        if not prompt or not prompt.strip():
            raise ValueError(
                "[APIImage GLM] Prompt is empty. "
                "Please enter a text prompt describing the image you want to generate."
            )

        if not api_key or not api_key.strip():
            raise ValueError(
                "[APIImage GLM] API Key is not set. "
                "Please provide a valid ZhipuAI API key. "
                "Get one at: https://open.bigmodel.cn/"
            )

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name

        # --- Validate ref_images compatibility ---
        validate_ref_images("GLM", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS)

        effective_url = sanitize_url(base_url) or "https://open.bigmodel.cn/api"

        logger.info(
            f"[GLM] Starting generation | URL: {effective_url} | "
            f"Model: {effective_model} | Quality: {quality} | Size: {size} | "
            f"NumImages: {num_images}"
        )

        # --- Generate (loop for multi-image) ---
        all_images_data = []
        for gen_idx in range(num_images):
            images_data = self._call_api(
                effective_url, api_key, effective_model, prompt, quality, size
            )
            all_images_data.extend(images_data)
            if gen_idx < num_images - 1:
                logger.info(f"[GLM] Generated image {gen_idx+1}/{num_images}")

        if not all_images_data:
            raise RuntimeError(
                "[APIImage GLM] No image data returned from any request."
            )

        result_tensor = bytes_to_tensor(all_images_data)
        logger.info(
            f"[GLM] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]}"
        )

        return (result_tensor,)

    def _call_api(self, base_url, api_key, model, prompt, quality, size):
        """Make a single API call and return list of image bytes."""
        url = f"{base_url}/paas/v4/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "quality": quality,
            "size": size,
        }

        # --- Call API ---
        try:
            response = http_requests.post(url, headers=headers, json=payload, timeout=120)
        except http_requests.exceptions.Timeout:
            raise RuntimeError(
                f"[APIImage GLM] Request timed out after 120 seconds. "
                f"GLM hd quality can take up to 20s. Check your network."
            )
        except http_requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"[APIImage GLM] Connection failed to {base_url}. "
                f"Please verify the base URL is correct. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"[APIImage GLM] Request failed: {e}")
        # --- Handle HTTP Errors ---
        if response.status_code != 200:
            try:
                error_body = response.json()
                error_msg = error_body.get("error", {}).get("message", response.text[:500])
            except Exception:
                error_msg = response.text[:500]

            if response.status_code == 401:
                raise RuntimeError(
                    f"[APIImage GLM] Authentication failed (401). "
                    f"Please check your ZhipuAI API key."
                )
            elif response.status_code == 429:
                raise RuntimeError(
                    f"[APIImage GLM] Rate limit exceeded (429). "
                    f"Please wait and retry."
                )
            else:
                raise RuntimeError(
                    f"[APIImage GLM] API Error ({response.status_code}): {error_msg}"
                )

        # --- Parse Response ---
        try:
            json_response = response.json()
        except Exception:
            raise RuntimeError(
                f"[APIImage GLM] Invalid JSON response from server."
            )

        images_data = []
        data_list = json_response.get("data") or []

        for item in data_list:
            img_url = item.get("url")
            if img_url:
                try:
                    logger.info(f"[GLM] Downloading result image: {img_url[:100]}...")
                    img_resp = http_requests.get(img_url, timeout=60)
                    img_resp.raise_for_status()
                    images_data.append(img_resp.content)
                except Exception as e:
                    logger.error(f"[GLM] Failed to download image: {e}")

            # Also check for base64 data in case API returns it
            b64_data = item.get("b64_json")
            if b64_data:
                try:
                    images_data.append(base64.b64decode(b64_data))
                except Exception as e:
                    logger.error(f"[GLM] Failed to decode base64: {e}")

        if not images_data:
            # Check content_filter for blocked content
            content_filter = json_response.get("content_filter", [])
            if content_filter:
                filter_info = ", ".join(
                    f"role={cf.get('role')} level={cf.get('level')}"
                    for cf in content_filter
                )
                raise RuntimeError(
                    f"[APIImage GLM] Content filtered: {filter_info}. "
                    f"Your prompt may have been blocked by safety filters."
                )
            raise RuntimeError(
                f"[APIImage GLM] No image data returned. "
                f"Response: {str(json_response)[:500]}"
            )

        return images_data
