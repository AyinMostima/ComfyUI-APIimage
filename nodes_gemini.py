"""
Gemini Image Generation node for ComfyUI.

Uses the official google-genai SDK for text-to-image and image editing.
Supports reference images, masks, aspect ratio, and multi-turn conversations.

Reference: https://ai.google.dev/gemini-api/docs/image-generation
- Text-to-image:  contents = [prompt_string]
- Image editing:  contents = [prompt_string, PIL.Image, ...]
- Config: GenerateContentConfig(response_modalities=['TEXT','IMAGE'],
          image_config=ImageConfig(aspect_ratio=..., image_size=...))
"""
import logging
import io
import base64

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import tensor_to_pil, pil_to_tensor, bytes_to_tensor, mask_to_pil, sanitize_url, validate_ref_images

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://ai.google.dev/gemini-api/docs/image-generation
# gemini-2.5-flash-image: up to 3 reference images per request
# gemini-3-pro-image-preview: up to 14 reference images per request
MODEL_REF_IMAGE_LIMITS = {
    "gemini-2.5-flash-image": (0, 3),
    "gemini-3-pro-image-preview": (0, 14),
}


class GeminiImageGenerate:
    """
    Generate or edit images using Google Gemini API.

    Supports text-to-image generation, image editing with reference images,
    and mask-based inpainting. Uses the official google-genai SDK.
    """

    CATEGORY = "APIImage/Gemini"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "text_response",)

    @classmethod
    def INPUT_TYPES(cls):
        # Get available models (built-in + custom)
        models = get_model_list("Gemini Native")
        if not models:
            models = BUILTIN_MODELS.get("Gemini Native", ["gemini-2.5-flash-image"])

        # Load saved api_key as default
        saved_config = get_api_config("Gemini Native")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "")

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "AIzaSy... (Google API Key)"
                }),
                "model_name": (models, {
                    "default": models[0] if models else "gemini-2.5-flash-image"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "aspect_ratio": (["Default", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "4:5"], {
                    "default": "Default"
                }),
                "resolution": (["Default", "1K", "2K", "4K"], {
                    "default": "Default"
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": "Leave empty for default, or set proxy URL"
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
        # Always re-execute when seed changes or inputs change
        return float("NaN")

    def generate(self, prompt, api_key, model_name, num_images=1,
                 aspect_ratio="Default", resolution="Default",
                 base_url="", ref_images=None,
                 mask=None, custom_model="", seed=0):
        """
        Main execution function for Gemini image generation.

        Mathematical formulation: Not applicable (API-based generation).
        Reference: https://ai.google.dev/gemini-api/docs/image-generation
        """
        import torch

        # --- Input Validation ---
        if not prompt or not prompt.strip():
            raise ValueError(
                "[APIImage Gemini] Prompt is empty. "
                "Please enter a text prompt describing the image you want to generate."
            )

        if not api_key or not api_key.strip():
            raise ValueError(
                "[APIImage Gemini] API Key is not set. "
                "Please provide a valid Google API key (format: AIzaSy...). "
                "Get one at: https://aistudio.google.com/apikey"
            )

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        logger.info(
            f"[Gemini] Starting generation | Model: {effective_model} | "
            f"AspectRatio: {aspect_ratio} | "
            f"HasMask: {mask is not None} | Seed: {seed}"
        )

        # --- Import SDK ---
        try:
            from google import genai
            from google.genai import types
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                f"[APIImage Gemini] Missing required package: {e}. "
                f"Please run: pip install google-genai Pillow"
            )

        # --- Create SDK Client ---
        try:
            client_kwargs = {"api_key": api_key.strip()}
            effective_url = sanitize_url(base_url)
            if effective_url:
                client_kwargs["http_options"] = {"base_url": effective_url}
                logger.info(f"[Gemini] Using custom endpoint: {effective_url}")
            client = genai.Client(**client_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"[APIImage Gemini] Failed to create Gemini client: {e}. "
                f"Please verify your API key is valid."
            )

        # --- Build Image Config ---
        # Reference: core/api_client.py _generate_gemini_native_single
        # Resolution: "1K", "2K", "4K" (must be uppercase K)
        # Aspect ratio: "1:1", "16:9", "9:16", "4:3", "3:4"
        image_config_kwargs = {}
        if resolution and resolution != "Default":
            image_config_kwargs["image_size"] = resolution
        if aspect_ratio and aspect_ratio != "Default":
            image_config_kwargs["aspect_ratio"] = aspect_ratio

        gen_config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        )
        if image_config_kwargs:
            gen_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(**image_config_kwargs),
            )

        # --- Validate reference images ---
        validate_ref_images("Gemini", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS)

        # --- Prepare Contents ---
        contents = [prompt]

        # Process batch reference images
        if ref_images is not None:
            try:
                batch_pil = tensor_to_pil(ref_images)
                contents.extend(batch_pil)
                logger.info(f"[Gemini] Added {len(batch_pil)} reference image(s)")
            except Exception as e:
                logger.warning(f"[Gemini] Failed to process ref_images: {e}")

        # Convert mask tensor to PIL and add context
        if mask is not None:
            try:
                mask_pil = mask_to_pil(mask)
                if mask_pil:
                    contents.append(mask_pil)
                    logger.info("[Gemini] Added mask image")
            except Exception as e:
                logger.warning(f"[Gemini] Failed to process mask: {e}")

        # --- Call API (loop for multi-image) ---
        all_images_data = []
        all_text_messages = []

        for gen_idx in range(num_images):
            try:
                response = client.models.generate_content(
                    model=effective_model,
                    contents=contents,
                    config=gen_config,
                )
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "UNAUTHENTICATED" in error_str:
                    raise RuntimeError(
                        f"[APIImage Gemini] Authentication failed (401). "
                        f"Please check your Google API key."
                    )
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    raise RuntimeError(
                        f"[APIImage Gemini] Model '{effective_model}' not found (404)."
                    )
                elif "403" in error_str or "PERMISSION_DENIED" in error_str:
                    raise RuntimeError(
                        f"[APIImage Gemini] Permission denied (403). "
                        f"Check billing at: https://aistudio.google.com"
                    )
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    raise RuntimeError(
                        f"[APIImage Gemini] API quota exceeded (429). "
                        f"Wait for quota reset or switch to a different model."
                    )
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    raise RuntimeError(
                        f"[APIImage Gemini] Model overloaded (503). Retry in 1-2 minutes."
                    )
                elif "timed out" in error_str.lower() or "timeout" in error_str.lower():
                    raise RuntimeError(
                        f"[APIImage Gemini] Network timeout. Check your connection."
                    )
                else:
                    raise RuntimeError(f"[APIImage Gemini] API Error: {error_str}")

            # Parse response for this iteration
            if response.parts:
                for part in response.parts:
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if part.text is not None:
                        all_text_messages.append(part.text)
                    elif part.inline_data is not None:
                        try:
                            img = part.as_image()
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            all_images_data.append(buf.getvalue())
                        except Exception:
                            try:
                                raw = part.inline_data.data
                                if isinstance(raw, bytes):
                                    all_images_data.append(raw)
                                elif isinstance(raw, str):
                                    all_images_data.append(base64.b64decode(raw))
                            except Exception as e2:
                                logger.error(f"[Gemini] Failed to extract image: {e2}")

            if gen_idx < num_images - 1:
                logger.info(f"[Gemini] Generated image {gen_idx+1}/{num_images}")

        text_response = "\n".join(all_text_messages) if all_text_messages else ""

        if not all_images_data:
            # Build detailed error
            error_parts = []
            if hasattr(response, 'candidates') and response.candidates:
                for cand in response.candidates:
                    if hasattr(cand, 'finish_reason') and cand.finish_reason:
                        reason_str = str(cand.finish_reason)
                        if reason_str not in ('STOP', 'FinishReason.STOP'):
                            error_parts.append(f"Blocked: {reason_str}")
                    if hasattr(cand, 'safety_ratings') and cand.safety_ratings:
                        blocked_cats = [
                            f"{r.category}={r.probability}"
                            for r in cand.safety_ratings
                            if hasattr(r, 'blocked') and r.blocked
                        ]
                        if blocked_cats:
                            error_parts.append(f"Safety: {', '.join(blocked_cats)}")

            if all_text_messages:
                error_parts.append(f"Model response: {text_response[:500]}")
            if not error_parts:
                error_parts.append("No image data in response")

            raise RuntimeError(
                f"[APIImage Gemini] Generation failed: {' | '.join(error_parts)}"
            )

        # Convert bytes to ComfyUI IMAGE tensor
        result_tensor = bytes_to_tensor(all_images_data)
        logger.info(
            f"[Gemini] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]}"
        )

        return (result_tensor, text_response,)
