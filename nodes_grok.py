"""
Grok Image Generation node for ComfyUI.

Uses the xAI xai_sdk protobuf API for image generation and editing.

CRITICAL IMPLEMENTATION NOTE (2026-05-13):
  The xai_sdk high-level Python methods (client.image.sample / sample_batch) do NOT
  pass through image_url, image_urls, aspect_ratio, or resolution parameters.
  They only accept (prompt, model, user, image_format) and silently ignore all other kwargs.
  Therefore ALL calls MUST go through the low-level protobuf path:
    request = image_pb2.GenerateImageRequest(...)
    response = client.image._stub.GenerateImage(request)

  Protobuf GenerateImageRequest fields (xai_sdk >= 1.12.2):
    - prompt: string
    - model: string
    - image: ImageUrlContent (single reference image for editing)
    - images: repeated ImageUrlContent (multiple reference images, up to 3)
    - n: int32 (number of output images)
    - format: ImageFormat enum (IMG_FORMAT_BASE64 / IMG_FORMAT_URL)
    - aspect_ratio: ImageAspectRatio enum (IMG_ASPECT_RATIO_16_9, etc.)
    - resolution: ImageResolution enum (IMG_RESOLUTION_1K / IMG_RESOLUTION_2K)
    - user: string

Reference: https://docs.x.ai/developers/model-capabilities/images/generation
"""
import os
import base64
import logging
import io

from .config import get_api_config, get_model_list, BUILTIN_MODELS
from .utils import tensor_to_pil, mask_to_pil, bytes_to_tensor, detect_mime, sanitize_url, validate_ref_images

logger = logging.getLogger("ComfyUI-APIImage")


# Reference: https://docs.x.ai/developers/model-capabilities/images/generation
# grok-imagine-image-quality supports up to 3 reference images for multi-image editing.
# grok-imagine-image and grok-imagine-image-pro support 1 reference image.
MODEL_REF_IMAGE_LIMITS = {
    "grok-imagine-image-quality": (0, 3),
    "grok-imagine-image": (0, 1),
    "grok-imagine-image-pro": (0, 1),
}

# Supported aspect ratios as user-facing strings
# Reference: https://docs.x.ai/developers/model-capabilities/images/generation#aspect-ratio
SUPPORTED_ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "2:1", "1:2",
    "19.5:9", "9:19.5", "20:9", "9:20",
    "auto",
]

# Mapping from user-facing aspect ratio strings to protobuf enum names
# Reference: image_pb2.ImageAspectRatio enum values
ASPECT_RATIO_TO_PB = {
    "1:1":    "IMG_ASPECT_RATIO_1_1",
    "16:9":   "IMG_ASPECT_RATIO_16_9",
    "9:16":   "IMG_ASPECT_RATIO_9_16",
    "4:3":    "IMG_ASPECT_RATIO_4_3",
    "3:4":    "IMG_ASPECT_RATIO_3_4",
    "3:2":    "IMG_ASPECT_RATIO_3_2",
    "2:3":    "IMG_ASPECT_RATIO_2_3",
    "2:1":    "IMG_ASPECT_RATIO_2_1",
    "1:2":    "IMG_ASPECT_RATIO_1_2",
    "19.5:9": "IMG_ASPECT_RATIO_19_5_9",
    "9:19.5": "IMG_ASPECT_RATIO_9_19_5",
    "20:9":   "IMG_ASPECT_RATIO_20_9",
    "9:20":   "IMG_ASPECT_RATIO_9_20",
    "auto":   "IMG_ASPECT_RATIO_AUTO",
}

# Mapping from user-facing resolution strings to protobuf enum names
# Reference: image_pb2.ImageResolution enum values
RESOLUTION_TO_PB = {
    "1k": "IMG_RESOLUTION_1K",
    "2k": "IMG_RESOLUTION_2K",
}

# Maximum dimension (px) for reference images before base64 encoding.
# The xAI gRPC channel has a server-side message size limit of ~4MB (4194304 bytes).
MAX_REF_IMAGE_DIMENSION = 1536

# gRPC message size limit in bytes
GRPC_MESSAGE_LIMIT = 4194304  # 4MB


class GrokImageGenerate:
    """
    Generate or edit images using xAI Grok Imagine API.

    All API calls go through protobuf GenerateImageRequest directly,
    because the SDK's sample() method does not support image/images/aspect_ratio/resolution fields.

    grok-imagine-image-quality is the recommended model, supporting:
    - Up to 10 images per generation request
    - Multi-image editing (up to 3 source images via 'images' repeated field)
    - Single-image editing (via 'image' field)
    - Extended aspect ratios including auto (via 'aspect_ratio' enum field)
    - 1K (1024x1024) and 2K (2048x2048) resolutions (via 'resolution' enum field)
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
            models = BUILTIN_MODELS.get("Grok API", ["grok-imagine-image-quality"])

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
                    "default": models[0] if models else "grok-imagine-image-quality"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {
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
        return float("NaN")

    # ------------------------------------------------------------------ #
    #  Helper: resize a PIL image so longest side <= max_dim              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_if_needed(pil_img, max_dim, idx=0):
        """
        Resize a PIL image so its longest side does not exceed max_dim.
        Returns (possibly resized image, was_resized).

        Reference: gRPC server limit = 4194304 bytes.
        Multiple large base64-encoded images easily exceed this limit.
        """
        w, h = pil_img.size
        longest = max(w, h)
        if longest <= max_dim:
            return pil_img, False

        from PIL import Image as PILImage
        scale = max_dim / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        logger.info(
            f"[Grok] Reference image {idx + 1} auto-resized: "
            f"({w}x{h}) -> ({new_w}x{new_h}) "
            f"[max dimension limit: {max_dim}px]"
        )
        return resized, True

    # ------------------------------------------------------------------ #
    #  Helper: encode PIL image to base64 data URI                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_to_data_uri(pil_img, jpeg_quality=85):
        """
        Encode a PIL image to a base64 JPEG data URI string.
        Returns (data_uri_string, base64_length).
        """
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality)
        img_bytes = buf.getvalue()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        uri = f"data:image/jpeg;base64,{b64_str}"
        return uri, len(b64_str)

    # ------------------------------------------------------------------ #
    #  Helper: convert aspect_ratio string to protobuf enum value        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_aspect_ratio_pb(aspect_ratio_str):
        """
        Convert user-facing aspect ratio string (e.g. '16:9') to protobuf enum int.
        Returns None if not found or not applicable.

        Reference: image_pb2.GenerateImageRequest.aspect_ratio field
        Enum: IMG_ASPECT_RATIO_1_1=1, ..., IMG_ASPECT_RATIO_AUTO=8, etc.
        """
        from xai_sdk.proto import image_pb2
        pb_name = ASPECT_RATIO_TO_PB.get(aspect_ratio_str)
        if pb_name is None:
            return None
        # Look up the enum value by name
        try:
            ar_field = image_pb2.GenerateImageRequest.DESCRIPTOR.fields_by_name['aspect_ratio']
            enum_type = ar_field.enum_type
            value_desc = enum_type.values_by_name.get(pb_name)
            return value_desc.number if value_desc else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Helper: convert resolution string to protobuf enum value          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_resolution_pb(resolution_str):
        """
        Convert user-facing resolution string (e.g. '2k') to protobuf enum int.
        Returns None if not found.

        Reference: image_pb2.GenerateImageRequest.resolution field
        Enum: IMG_RESOLUTION_1K=1, IMG_RESOLUTION_2K=2
        """
        from xai_sdk.proto import image_pb2
        pb_name = RESOLUTION_TO_PB.get(resolution_str)
        if pb_name is None:
            return None
        try:
            res_field = image_pb2.GenerateImageRequest.DESCRIPTOR.fields_by_name['resolution']
            enum_type = res_field.enum_type
            value_desc = enum_type.values_by_name.get(pb_name)
            return value_desc.number if value_desc else None
        except Exception:
            return None

    def generate(self, prompt, api_key, model_name, num_images=1,
                 aspect_ratio="1:1", resolution="1k",
                 base_url="", ref_images=None,
                 image1=None, image2=None, image3=None,
                 mask=None, custom_model="", seed=0):
        """
        Main execution function for Grok image generation.

        IMPORTANT: ALL API calls go through protobuf directly:
          request = image_pb2.GenerateImageRequest(
              prompt=..., model=...,
              image=ImageUrlContent(...),       # single-image edit
              images=[ImageUrlContent(...),...], # multi-image edit (up to 3)
              aspect_ratio=<enum>,              # aspect ratio
              resolution=<enum>,                # resolution (1k/2k)
              n=..., format=<enum>,
          )
          response = client.image._stub.GenerateImage(request)

        The SDK's sample()/sample_batch() methods are NOT used because they
        silently ignore image/images/aspect_ratio/resolution parameters.
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
            f"Image1: {image1 is not None} | Image2: {image2 is not None} | "
            f"Image3: {image3 is not None} | HasMask: {mask is not None} | Seed: {seed}"
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
        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images("Grok", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS, extra_count=extra_img_count)

        # --- Collect Reference Images ---
        all_ref_pils = []
        if ref_images is not None:
            try:
                all_ref_pils.extend(tensor_to_pil(ref_images))
            except Exception as e:
                logger.warning(f"[Grok] Failed to process ref_images: {e}")
        for slot_name, slot_val in [("image1", image1), ("image2", image2), ("image3", image3)]:
            if slot_val is not None:
                try:
                    all_ref_pils.extend(tensor_to_pil(slot_val))
                except Exception as e:
                    logger.warning(f"[Grok] Failed to process {slot_name}: {e}")

        # Enforce model-specific reference image limits
        model_limits = MODEL_REF_IMAGE_LIMITS.get(effective_model, (0, 3))
        max_refs = model_limits[1]
        if len(all_ref_pils) > max_refs:
            logger.warning(
                f"[Grok] Model '{effective_model}' supports max {max_refs} reference image(s), "
                f"but {len(all_ref_pils)} provided. Extra images will be ignored."
            )
            all_ref_pils = all_ref_pils[:max_refs]

        logger.info(f"[Grok] Total reference images collected: {len(all_ref_pils)}")

        # --- Encode Reference Images to Base64 Data URIs ---
        # Auto-resize large images to prevent gRPC 4MB payload overflow
        reference_uris = []
        jpeg_quality = 85 if len(all_ref_pils) > 1 else 90

        for idx, ref_img in enumerate(all_ref_pils):
            try:
                ref_img, was_resized = self._resize_if_needed(ref_img, MAX_REF_IMAGE_DIMENSION, idx)
                uri, b64_len = self._encode_to_data_uri(ref_img, jpeg_quality)
                reference_uris.append(uri)
                logger.info(
                    f"[Grok] Reference image {idx + 1} prepared | "
                    f"Size: {ref_img.size} | JPEG quality: {jpeg_quality} | "
                    f"Resized: {was_resized} | Base64Length: {b64_len}"
                )
            except Exception as e:
                logger.warning(f"[Grok] Failed to process reference image {idx + 1}: {e}")

        # Estimate total payload and warn if close to gRPC limit
        if reference_uris:
            total_b64_len = sum(len(u) for u in reference_uris)
            estimated_payload = int(total_b64_len * 1.1)
            if estimated_payload > GRPC_MESSAGE_LIMIT * 0.85:
                logger.warning(
                    f"[Grok] WARNING: Estimated payload ~{estimated_payload // 1024}KB is close to "
                    f"gRPC limit ({GRPC_MESSAGE_LIMIT // 1024}KB). May fail with large images."
                )

        has_reference = len(reference_uris) > 0
        is_multi_image = len(reference_uris) > 1

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

        # --- Build Protobuf Request ---
        # CRITICAL: We build GenerateImageRequest directly because the SDK's
        # sample()/sample_batch() methods do NOT support image/images/aspect_ratio/resolution.
        # See xai_sdk/sync/image.py lines 24-53: sample() only passes prompt, model, user, format.
        try:
            client = xai_sdk.Client()

            # Common request fields
            request_kwargs = {
                "prompt": prompt,
                "model": effective_model,
                "n": num_images,
                "format": convert_image_format_to_pb("base64"),
            }

            # --- Aspect Ratio ---
            # Protobuf field: aspect_ratio (ImageAspectRatio enum)
            # Supported in generation mode and multi-image edit mode.
            # In single-image edit, the output follows the input image's aspect ratio.
            ar_pb = self._get_aspect_ratio_pb(aspect_ratio)
            if ar_pb is not None:
                request_kwargs["aspect_ratio"] = ar_pb
                logger.info(f"[Grok] Aspect ratio set: {aspect_ratio} -> pb_enum={ar_pb}")

            # --- Resolution ---
            # Protobuf field: resolution (ImageResolution enum)
            # Only effective in generation mode (no reference images).
            res_pb = self._get_resolution_pb(resolution)
            if res_pb is not None:
                request_kwargs["resolution"] = res_pb
                logger.info(f"[Grok] Resolution set: {resolution} -> pb_enum={res_pb}")

            # --- Reference Images ---
            if has_reference:
                if is_multi_image:
                    # MULTI-IMAGE EDIT: use 'images' repeated field
                    # Protobuf: images = [ImageUrlContent(image_url=uri1), ...]
                    image_contents = [
                        image_pb2.ImageUrlContent(image_url=uri) for uri in reference_uris
                    ]
                    request_kwargs["images"] = image_contents
                    logger.info(
                        f"[Grok] MULTI-IMAGE EDIT mode (protobuf) | model={effective_model} | "
                        f"RefImages: {len(image_contents)} | AspectRatio: {aspect_ratio} | "
                        f"Resolution: {resolution}"
                    )
                else:
                    # SINGLE-IMAGE EDIT: use 'image' singular field
                    # Protobuf: image = ImageUrlContent(image_url=uri)
                    image_content = image_pb2.ImageUrlContent(image_url=reference_uris[0])
                    request_kwargs["image"] = image_content

                    # Add mask if available (only for single-image edit)
                    if mask_uri:
                        mask_content = image_pb2.ImageUrlContent(image_url=mask_uri)
                        request_kwargs["mask"] = mask_content
                        logger.info(
                            f"[Grok] INPAINT mode (protobuf) | model={effective_model} | "
                            f"AspectRatio: {aspect_ratio}"
                        )
                    else:
                        logger.info(
                            f"[Grok] SINGLE-IMAGE EDIT mode (protobuf) | model={effective_model} | "
                            f"AspectRatio: {aspect_ratio}"
                        )
            else:
                # GENERATE MODE: text-to-image, no reference images
                logger.info(
                    f"[Grok] GENERATE mode (protobuf) | model={effective_model} | "
                    f"AspectRatio: {aspect_ratio} | Resolution: {resolution}"
                )

            # --- Execute Request ---
            # Build and send protobuf request directly via gRPC stub
            request = image_pb2.GenerateImageRequest(**request_kwargs)

            # Log the actual protobuf fields for debugging
            logger.info(
                f"[Grok] Protobuf request built | "
                f"HasImage: {request.HasField('image') if 'image' not in ['images'] else False} | "
                f"ImagesCount: {len(request.images)} | "
                f"AspectRatio: {request.aspect_ratio} | "
                f"Resolution: {request.resolution} | "
                f"N: {request.n}"
            )

            response_pb = client.image._stub.GenerateImage(request)

            # --- Extract Images ---
            from xai_sdk.sync.image import ImageResponse as SdkImageResponse
            images_data = []
            actual_count = len(response_pb.images)
            logger.info(f"[Grok] Response received | Images in response: {actual_count}")

            for i in range(min(num_images, actual_count)):
                try:
                    img_resp = SdkImageResponse(response_pb, i)
                    images_data.append(img_resp.image)
                except Exception as e:
                    logger.error(f"[Grok] Failed to extract image {i}: {e}")

        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()

            # --- Error Classification ---
            # 1. Authentication error
            if "401" in error_str or "UNAUTHENTICATED" in error_str:
                raise RuntimeError(
                    f"[APIImage Grok] Authentication failed. "
                    f"Please check your xAI API key."
                )
            # 2. gRPC payload too large (message size exceeded 4MB)
            #    Error: "decoded message length too large: found 4662429 bytes, limit is 4194304"
            #    MUST check BEFORE rate-limit because "4662429" contains substring "429"!
            elif "message length too large" in error_lower or "message_length" in error_lower:
                raise RuntimeError(
                    f"[APIImage Grok] Request payload too large (gRPC 4MB limit). "
                    f"Your reference images are too large even after auto-resize. "
                    f"Please reduce the number of reference images or use smaller source images. "
                    f"Detail: {error_str[-300:]}"
                )
            # 3. Rate limit (exclude false positives from byte counts containing "429")
            elif "rate_limit" in error_lower or "resource_exhausted" in error_lower or (
                "429" in error_str and "bytes" not in error_lower
            ):
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

        # Log success with details
        logger.info(
            f"[Grok] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | "
            f"Size(HxW): {result_tensor.shape[1]}x{result_tensor.shape[2]} | "
            f"Mode: {'MultiEdit' if is_multi_image else 'SingleEdit' if has_reference else 'Generate'}"
        )

        return (result_tensor,)
