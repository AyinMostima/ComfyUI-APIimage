"""
Grok Image Generation node for ComfyUI.

Uses the xAI xai_sdk for image generation and editing via Grok Imagine models.

Reference: https://docs.x.ai/developers/model-capabilities/images/generation
- Generation: client.image.sample() or sample_batch()
- Editing: via protobuf ImageUrlContent with base64 data URI
- Models: grok-imagine-image, grok-imagine-image-pro
"""
import os
import base64
import logging
import io

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import tensor_to_pil, mask_to_pil, bytes_to_tensor, detect_mime, sanitize_url, validate_ref_images

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://docs.x.ai/developers/model-capabilities/images/generation
# Grok Imagine models support 1 reference image per request for editing.
MODEL_REF_IMAGE_LIMITS = {
    "grok-imagine-image": (0, 1),
    "grok-imagine-image-pro": (0, 1),
}


class GrokImageGenerate:
    """
    Generate or edit images using xAI Grok Imagine API.

    Supports text-to-image and image editing with reference images.
    Uses the xai_sdk Python SDK with protobuf for editing mode.
    """

    CATEGORY = "APIImage/Grok"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("Grok API")
        if not models:
            models = BUILTIN_MODELS.get("Grok API", ["grok-imagine-image-pro"])

        saved_config = get_api_config("Grok API")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://api.x.ai")

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "xai-... (xAI API Key)"
                }),
                "model_name": (models, {
                    "default": models[0] if models else "grok-imagine-image-pro"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "1:1",
                }),
                "resolution": (["1k", "2k"], {
                    "default": "1k",
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": "https://api.x.ai (default)"
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

    def generate(self, prompt, api_key, model_name, num_images=1,
                 aspect_ratio="1:1", resolution="1k",
                 base_url="", ref_images=None,
                 mask=None, custom_model="", seed=0):
        """
        Main execution function for Grok image generation.

        Reference: https://docs.x.ai/developers/model-capabilities/images/generation
        - Generation mode: uses xai_sdk client.image.sample()
        - Edit mode: uses protobuf GenerateImageRequest with ImageUrlContent
        - image_url accepts base64 data URI: data:image/jpeg;base64,...
        - Mask: encoded as base64 data URI and passed via mask field in protobuf
        """
        import torch

        # --- Input Validation ---
        if not prompt or not prompt.strip():
            raise ValueError(
                "[APIImage Grok] Prompt is empty. "
                "Please enter a text prompt describing the image you want to generate."
            )

        if not api_key or not api_key.strip():
            raise ValueError(
                "[APIImage Grok] API Key is not set. "
                "Please provide a valid xAI API key (format: xai-...). "
                "Get one at: https://console.x.ai"
            )

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        logger.info(
            f"[Grok] Starting generation | Model: {effective_model} | "
            f"AspectRatio: {aspect_ratio} | Resolution: {resolution} | "
            f"NumImages: {num_images} | HasRefImages: {ref_images is not None} | "
            f"HasMask: {mask is not None} | Seed: {seed}"
        )

        # --- Import SDK ---
        try:
            import xai_sdk
            from xai_sdk.proto import image_pb2
            from xai_sdk.image import convert_image_format_to_pb
        except ImportError as e:
            raise ImportError(
                f"[APIImage Grok] Missing required package: {e}. "
                f"Please run: pip install xai_sdk"
            )

        # Set XAI_API_KEY env var for xai_sdk.Client()
        os.environ["XAI_API_KEY"] = api_key.strip()

        # Set custom base URL if provided
        effective_url = sanitize_url(base_url)
        if effective_url:
            os.environ["XAI_API_BASE"] = effective_url
            logger.info(f"[Grok] Using custom endpoint: {effective_url}")
        elif "XAI_API_BASE" in os.environ:
            del os.environ["XAI_API_BASE"]

        # --- Validate reference images ---
        validate_ref_images("Grok", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS)

        # --- Process Reference Image ---
        # Grok API only supports 1 reference image per request
        has_reference = False
        reference_uri = None

        # Collect reference images from ref_images
        all_ref_pils = []
        if ref_images is not None:
            try:
                all_ref_pils.extend(tensor_to_pil(ref_images))
            except Exception as e:
                logger.warning(f"[Grok] Failed to process ref_images: {e}")

        if len(all_ref_pils) > 1:
            logger.warning(
                f"[Grok] Grok API only supports 1 reference image, but {len(all_ref_pils)} provided. "
                f"Only the first image will be used."
            )

        if all_ref_pils:
            try:
                ref_img = all_ref_pils[0]
                buf = io.BytesIO()
                ref_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                mime = detect_mime(img_bytes)
                reference_uri = f"data:{mime};base64,{b64_str}"
                has_reference = True
                logger.info(
                    f"[Grok] Reference image prepared | "
                    f"Size: {ref_img.size} | Base64Length: {len(b64_str)}"
                )
            except Exception as e:
                logger.warning(f"[Grok] Failed to process reference image: {e}")

        # --- Process Mask ---
        mask_uri = None
        if mask is not None and has_reference:
            try:
                mask_pil = mask_to_pil(mask)
                buf = io.BytesIO()
                mask_pil.save(buf, format="PNG")
                mask_bytes = buf.getvalue()
                b64_mask = base64.b64encode(mask_bytes).decode("utf-8")
                mask_uri = f"data:image/png;base64,{b64_mask}"
                logger.info(f"[Grok] Mask prepared | Size: {mask_pil.size}")
            except Exception as e:
                logger.warning(f"[Grok] Failed to process mask: {e}")

        # --- Call API ---
        try:
            client = xai_sdk.Client()

            if has_reference and reference_uri:
                # === EDIT MODE via protobuf ===
                image_content = image_pb2.ImageUrlContent(image_url=reference_uri)
                # Edit mode: protobuf does NOT support resolution/aspect_ratio
                request_kwargs = {
                    "prompt": prompt,
                    "model": effective_model,
                    "image": image_content,
                    "n": num_images,
                    "format": convert_image_format_to_pb("base64"),
                }
                # Add mask if available
                if mask_uri:
                    mask_content = image_pb2.ImageUrlContent(image_url=mask_uri)
                    request_kwargs["mask"] = mask_content
                    logger.info(f"[Grok] INPAINT mode | model={effective_model}")
                else:
                    logger.info(f"[Grok] EDIT mode | model={effective_model}")

                request = image_pb2.GenerateImageRequest(**request_kwargs)
                response_pb = client.image._stub.GenerateImage(request)

                from xai_sdk.sync.image import ImageResponse as SdkImageResponse
                images_data = []
                for i in range(num_images):
                    try:
                        img_resp = SdkImageResponse(response_pb, i)
                        images_data.append(img_resp.image)
                    except Exception as e:
                        logger.error(f"[Grok] Failed to extract image {i}: {e}")

            else:
                # === GENERATE MODE via SDK ===
                logger.info(f"[Grok] GENERATE mode | model={effective_model}")
                # Generation mode: both aspect_ratio and resolution supported
                gen_kwargs = {
                    "prompt": prompt,
                    "model": effective_model,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "image_format": "base64",
                }
                if num_images == 1:
                    responses = [client.image.sample(**gen_kwargs)]
                else:
                    gen_kwargs["n"] = num_images
                    responses = client.image.sample_batch(**gen_kwargs)

                images_data = []
                for i, resp in enumerate(responses):
                    try:
                        images_data.append(resp.image)
                    except Exception as e:
                        logger.error(f"[Grok] Failed to extract image {i}: {e}")

        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "UNAUTHENTICATED" in error_str:
                raise RuntimeError(
                    f"[APIImage Grok] Authentication failed. "
                    f"Please check your xAI API key."
                )
            elif "429" in error_str or "rate" in error_str.lower():
                raise RuntimeError(
                    f"[APIImage Grok] Rate limit exceeded. "
                    f"Please wait a moment and retry."
                )
            else:
                raise RuntimeError(f"[APIImage Grok] API Error: {error_str}")

        # --- Process Results ---
        if not images_data:
            raise RuntimeError(
                "[APIImage Grok] No images returned from the API. "
                "The model may have rejected the prompt or encountered an internal error."
            )

        # Convert bytes to tensor
        result_tensor = bytes_to_tensor(images_data)
        logger.info(
            f"[Grok] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]}"
        )

        return (result_tensor,)
