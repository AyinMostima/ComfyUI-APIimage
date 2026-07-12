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
from typing import List, Tuple

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
    # Allow returning a list of images with different resolutions.
    # Each image is an independent [1,H,W,C] tensor - no resize, no distortion.
    # Reference: https://docs.comfy.org/custom-nodes/backend/lists
    OUTPUT_IS_LIST = (True, True,)

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
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
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
                 image1=None, image2=None, image3=None,
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

        effective_url = sanitize_url(base_url)

        # --- Build GenerateContentConfig ---
        # Reference: https://ai.google.dev/gemini-api/docs/image-generation
        # Both aspect_ratio and image_size (resolution) are supported in
        # text-to-image AND image editing modes.
        image_config_kwargs = {}
        if resolution and resolution != "Default":
            image_config_kwargs["image_size"] = resolution
        if aspect_ratio and aspect_ratio != "Default":
            image_config_kwargs["aspect_ratio"] = aspect_ratio

        # Keep safety settings enabled explicitly for predictable moderation policy.
        # Reference: https://ai.google.dev/gemini-api/docs/safety-settings
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        ]

        gen_config_kwargs = {
            "response_modalities": ["TEXT", "IMAGE"],
            "safety_settings": safety_settings,
        }
        if image_config_kwargs:
            gen_config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

        gen_config_primary = types.GenerateContentConfig(**gen_config_kwargs)
        # Fallback config: strict parity with the simplest core path.
        # When moderation false-positives happen in multi-image editing,
        # dropping optional image_config can improve pass rate.
        gen_config_fallback = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=safety_settings,
        )

        # --- Validate reference images ---
        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images("Gemini", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS, extra_count=extra_img_count)

        # Look up max allowed ref images for this model
        max_ref = None
        for known_model, (_, mx) in MODEL_REF_IMAGE_LIMITS.items():
            if known_model == effective_model or known_model in effective_model.lower():
                max_ref = mx
                break

        def _build_contents() -> List:
            """Build contents list with prompt + ref images (JPEG) + mask.

            Reference images are always JPEG-encoded because:
            - Image.fromarray() PIL objects have no 'format' attribute
            - Gemini SDK pil_to_blob defaults to PNG for such PIL objects
            - PNG encoding produces ~7MB per large image (vs ~500KB JPEG q95)
            - JPEG roundtrip sets format='JPEG' so SDK preserves it
            """
            contents_local = [prompt]

            # Collect all reference images: batch ref_images + individual image1-3
            all_ref_pils = []
            if ref_images is not None:
                try:
                    all_ref_pils.extend(tensor_to_pil(ref_images))
                except Exception as e:
                    logger.warning(f"[Gemini] Failed to process ref_images: {e}")
            for slot_name, slot_val in [("image1", image1), ("image2", image2), ("image3", image3)]:
                if slot_val is not None:
                    try:
                        all_ref_pils.extend(tensor_to_pil(slot_val))
                    except Exception as e:
                        logger.warning(f"[Gemini] Failed to process {slot_name}: {e}")

            if all_ref_pils:
                if max_ref is not None and max_ref > 0 and len(all_ref_pils) > max_ref:
                    logger.warning(
                        f"[Gemini] Model '{effective_model}' supports max {max_ref} "
                        f"reference image(s), but {len(all_ref_pils)} provided. "
                        f"Truncating to first {max_ref} image(s)."
                    )
                    all_ref_pils = all_ref_pils[:max_ref]

                for idx, pil_img in enumerate(all_ref_pils):
                    rgb_img = pil_img.convert("RGB")
                    # Always JPEG-encode to prevent SDK PNG bloat
                    buf = io.BytesIO()
                    rgb_img.save(buf, format="JPEG", quality=95)
                    buf.seek(0)
                    final_img = Image.open(buf)
                    final_img.load()
                    contents_local.append(final_img)
                    logger.info(
                        f"[Gemini] Added ref image {idx+1}/{len(all_ref_pils)} | "
                        f"Size: {final_img.size} | Mode: {final_img.mode} | "
                        f"Format: {final_img.format}"
                    )

            if mask is not None:
                try:
                    mask_pil = mask_to_pil(mask)
                    if mask_pil:
                        contents_local.append(mask_pil)
                        logger.info(f"[Gemini] Added mask | Size: {mask_pil.size}")
                except Exception as e:
                    logger.warning(f"[Gemini] Failed to process mask: {e}")
            return contents_local

        contents = _build_contents()

        def _build_client(use_custom_endpoint: bool):
            try:
                client_kwargs = {"api_key": api_key.strip()}
                if use_custom_endpoint and effective_url:
                    client_kwargs["http_options"] = {"base_url": effective_url}
                    logger.info(f"[Gemini] Using custom endpoint: {effective_url}")
                return genai.Client(**client_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIImage Gemini] Failed to create Gemini client: {e}. "
                    f"Please verify your API key is valid."
                )

        def _extract_parts(response_obj):
            parts_obj = None
            try:
                parts_obj = response_obj.parts
            except Exception:
                pass
            if not parts_obj:
                try:
                    if (
                        hasattr(response_obj, 'candidates')
                        and response_obj.candidates
                        and hasattr(response_obj.candidates[0], 'content')
                        and response_obj.candidates[0].content
                    ):
                        parts_obj = response_obj.candidates[0].content.parts
                        logger.info("[Gemini] Using candidates fallback for response parts")
                except Exception:
                    pass
            return parts_obj

        def _extract_block_reason(response_obj) -> str:
            reasons = []
            if hasattr(response_obj, "prompt_feedback") and response_obj.prompt_feedback:
                pf = response_obj.prompt_feedback
                if hasattr(pf, "block_reason") and pf.block_reason:
                    reasons.append(str(pf.block_reason))
            if hasattr(response_obj, "candidates") and response_obj.candidates:
                for cand in response_obj.candidates:
                    if hasattr(cand, "finish_reason") and cand.finish_reason:
                        reasons.append(str(cand.finish_reason))
            return " | ".join(reasons)

        def _parse_response(response_obj) -> Tuple[List[bytes], List[str], str]:
            images_data_local = []
            text_messages_local = []
            parts_obj = _extract_parts(response_obj)
            if parts_obj:
                for part in parts_obj:
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if part.text is not None:
                        text_messages_local.append(part.text)
                    elif part.inline_data is not None:
                        try:
                            img = part.as_image()
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            images_data_local.append(buf.getvalue())
                        except Exception as e1:
                            logger.warning(
                                f"[Gemini] part.as_image() failed: {e1}, trying raw data"
                            )
                            try:
                                raw = part.inline_data.data
                                if isinstance(raw, bytes):
                                    images_data_local.append(raw)
                                elif isinstance(raw, str):
                                    images_data_local.append(base64.b64decode(raw))
                            except Exception as e2:
                                logger.error(f"[Gemini] Failed to extract image data: {e2}")
            return images_data_local, text_messages_local, _extract_block_reason(response_obj)

        # --- Call API (loop for multi-image) ---
        all_images_data = []
        all_text_messages = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_all_tokens = 0
        response = None

        for gen_idx in range(num_images):
            use_custom_endpoint = bool(effective_url)
            has_any_ref = ref_images is not None or image1 is not None or image2 is not None or image3 is not None
            attempt_plan = [
                ("official-primary", False, contents, gen_config_primary),
            ]
            # Add a conservative fallback for multi-reference/image-edit scenarios.
            # Drops optional image_config to improve pass rate on false-positive moderation.
            if has_any_ref:
                attempt_plan.append(
                    ("official-fallback", False, contents, gen_config_fallback)
                )
            if use_custom_endpoint:
                attempt_plan.append(
                    ("custom-primary", True, contents, gen_config_primary)
                )
                if has_any_ref:
                    attempt_plan.append(
                        ("custom-fallback", True, contents, gen_config_fallback)
                    )

            last_exception = None
            got_image_this_round = False
            block_reason_last = ""
            for attempt_idx, (attempt_name, use_custom, attempt_contents, attempt_config) in enumerate(attempt_plan):
                try:
                    client = _build_client(use_custom_endpoint=use_custom)
                    response = client.models.generate_content(
                        model=effective_model,
                        contents=attempt_contents,
                        config=attempt_config,
                    )
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"[Gemini] Attempt {attempt_idx + 1}/{len(attempt_plan)} "
                        f"({attempt_name}) failed: {e}"
                    )
                    if attempt_idx < len(attempt_plan) - 1:
                        continue
                    break

                images_data, text_messages, block_reason = _parse_response(response)
                block_reason_last = block_reason
                all_text_messages.extend(text_messages)

                # Log token usage from response metadata
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    um = response.usage_metadata
                    prompt_tokens = getattr(um, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(um, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(um, 'total_token_count', 0) or 0
                    total_prompt_tokens += prompt_tokens
                    total_output_tokens += output_tokens
                    total_all_tokens += total_tokens
                    logger.info(
                        f"[Gemini] Token usage | Attempt: {attempt_name} | "
                        f"Prompt: {prompt_tokens} | Output: {output_tokens} | "
                        f"Total: {total_tokens}"
                    )

                if images_data:
                    all_images_data.extend(images_data)
                    got_image_this_round = True
                    break

                logger.warning(
                    f"[Gemini] Attempt {attempt_idx + 1}/{len(attempt_plan)} "
                    f"({attempt_name}) returned no image. "
                    f"BlockReason: {block_reason or 'N/A'}"
                )
                if attempt_idx < len(attempt_plan) - 1:
                    continue

            if gen_idx < num_images - 1:
                logger.info(f"[Gemini] Generated image {gen_idx+1}/{num_images}")

            if not got_image_this_round and last_exception is not None:
                error_str = str(last_exception)
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
                raise RuntimeError(f"[APIImage Gemini] API Error: {error_str}")

            if not got_image_this_round and block_reason_last:
                logger.warning(
                    f"[Gemini] No image generated after retries. "
                    f"Last block reason: {block_reason_last}"
                )

        text_response = "\n".join(all_text_messages) if all_text_messages else ""

        if not all_images_data:
            # Build detailed error with as much diagnostic info as possible
            error_parts = []

            # Check prompt_feedback for blocked prompts
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                pf = response.prompt_feedback
                if hasattr(pf, 'block_reason') and pf.block_reason:
                    error_parts.append(f"Prompt blocked: {pf.block_reason}")
                if hasattr(pf, 'safety_ratings') and pf.safety_ratings:
                    flagged = [
                        f"{r.category}={r.probability}"
                        for r in pf.safety_ratings
                    ]
                    if flagged:
                        error_parts.append(f"Prompt safety: {', '.join(flagged)}")

            # Check candidates for finish_reason and safety
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
            else:
                error_parts.append("No candidates in response (model may be overloaded)")

            # Check if model returned text instead of image
            if all_text_messages:
                error_parts.append(f"Model returned text instead of image: {text_response[:300]}")

            # Check parts info
            if hasattr(response, 'parts') and response.parts:
                part_types = []
                for p in response.parts:
                    if hasattr(p, 'thought') and p.thought:
                        part_types.append("thought")
                    elif p.text is not None:
                        part_types.append("text")
                    elif p.inline_data is not None:
                        part_types.append(f"inline_data({p.inline_data.mime_type})")
                    else:
                        part_types.append("unknown")
                error_parts.append(f"Response parts: [{', '.join(part_types)}]")
            elif not hasattr(response, 'parts') or not response.parts:
                error_parts.append("Response has no parts at all")

            if not error_parts:
                error_parts.append("No image data in response (unknown reason)")

            error_parts.append(
                "Tips: 1) Make prompt more descriptive for image generation "
                "2) Check if model supports image output "
                "3) Try a different model or simplify the prompt"
            )

            raise RuntimeError(
                f"[APIImage Gemini] Generation failed: {' | '.join(error_parts)}"
            )

        # Convert bytes to list of individual IMAGE tensors.
        # Each image keeps its original resolution (no resize / no distortion).
        # OUTPUT_IS_LIST = (True, True) tells ComfyUI to treat each element
        # as a separate execution for downstream nodes.
        image_tensors = []
        for img_idx, img_bytes in enumerate(all_images_data):
            try:
                single_tensor = bytes_to_tensor([img_bytes])  # [1, H, W, C]
                image_tensors.append(single_tensor)
                logger.info(
                    f"[Gemini] Image {img_idx+1}/{len(all_images_data)} | "
                    f"Size: {single_tensor.shape[1]}x{single_tensor.shape[2]}"
                )
            except Exception as e:
                logger.error(f"[Gemini] Failed to decode image {img_idx+1}: {e}")

        if not image_tensors:
            raise RuntimeError(
                "[APIImage Gemini] All generated images failed to decode."
            )

        logger.info(
            f"[Gemini] Success | Model: {effective_model} | "
            f"Images: {len(image_tensors)} | "
            f"Tokens(prompt/output/total): {total_prompt_tokens}/{total_output_tokens}/{total_all_tokens}"
        )

        # OUTPUT_IS_LIST requires each output to be a list.
        # text_response is duplicated for each image so downstream nodes
        # always receive matched (image, text) pairs.
        text_list = [text_response] * len(image_tensors)
        return (image_tensors, text_list,)
