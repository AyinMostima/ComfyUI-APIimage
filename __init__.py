"""
ComfyUI Custom Nodes: API Image Generator
==========================================

Provides ComfyUI nodes for AI image generation via:
- Google Gemini (google-genai SDK)
- xAI Grok (xai_sdk)
- Alibaba Qwen / Tongyi Wanxiang (dashscope SDK)
- ZhipuAI GLM / CogView (REST API)
- OpenAI Compatible (REST API, GPT Image 2 plus legacy image models)
- BytePlus ModelArk / Seedream (REST API)

Features:
- Persistent API configuration (api_config.json)
- Custom model management (add/remove models dynamically)
- Model-specific reference image input for editing mode
- Mask input for inpainting (Gemini/Grok/OpenAI/Qwen)
- Comprehensive error handling with user-friendly messages
- Output node for saving generated images

Installation:
    Copy or symlink this directory into ComfyUI/custom_nodes/
    Dependencies are auto-installed on first load.
"""

import subprocess
import importlib
import os
import sys
import logging

logger = logging.getLogger("ComfyUI-APIImage")

# Support direct module loading by test runners when the plugin directory name
# cannot be used as a regular Python package identifier.
if not __package__:
    _DIRECT_PACKAGE_NAME = "_comfyui_apiimage_direct"
    __package__ = _DIRECT_PACKAGE_NAME
    __path__ = [os.path.dirname(os.path.abspath(__file__))]
    __spec__ = importlib.util.spec_from_file_location(
        _DIRECT_PACKAGE_NAME,
        __file__,
        submodule_search_locations=__path__,
    )
    sys.modules.setdefault(_DIRECT_PACKAGE_NAME, sys.modules[__name__])

# ============================================================
# Auto-install missing dependencies
# ============================================================
# Required packages: (import_name, pip_name)
_REQUIRED_PACKAGES = [
    ("PIL", "Pillow"),
    ("requests", "requests"),
    ("google.genai", "google-genai"),
    ("xai_sdk", "xai_sdk"),
    ("dashscope", "dashscope"),
]


def _ensure_packages():
    """Check and auto-install missing packages at startup."""
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            logger.info(f"[APIImage] Installing missing package: {pip_name}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                    stdout=subprocess.DEVNULL,
                )
                logger.info(f"[APIImage] Successfully installed {pip_name}")
            except Exception as e:
                logger.warning(
                    f"[APIImage] Failed to install {pip_name}: {e}. "
                    f"You can install it manually: pip install {pip_name}"
                )


_ensure_packages()

# ============================================================
# Node Imports
# ============================================================
from .nodes_gemini import GeminiImageGenerate
from .nodes_grok import GrokImageGenerate
from .nodes_openai import OpenAIImageGenerate
from .nodes_modelark import ModelArkImageGenerate
from .nodes_qwen import QwenImageGenerate
from .nodes_glm import GLMImageGenerate
from .nodes_config import APIImageConfigLoader, APIImageConfigSaver
from .nodes_save import APIImageSave

# ============================================================
# Node Registration
# ============================================================
# NODE_CLASS_MAPPINGS maps internal node IDs to Python classes.
# NODE_DISPLAY_NAME_MAPPINGS maps internal node IDs to display names
# shown in the ComfyUI node picker.

NODE_CLASS_MAPPINGS = {
    "APIImage_GeminiGenerate": GeminiImageGenerate,
    "APIImage_GrokGenerate": GrokImageGenerate,
    "APIImage_OpenAIGenerate": OpenAIImageGenerate,
    "APIImage_ModelArkGenerate": ModelArkImageGenerate,
    "APIImage_QwenGenerate": QwenImageGenerate,
    "APIImage_GLMGenerate": GLMImageGenerate,
    "APIImage_ConfigLoader": APIImageConfigLoader,
    "APIImage_ConfigSaver": APIImageConfigSaver,
    "APIImage_SaveImage": APIImageSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIImage_GeminiGenerate": "Gemini Image Generate",
    "APIImage_GrokGenerate": "Grok Image Generate",
    "APIImage_OpenAIGenerate": "OpenAI Image Generate",
    "APIImage_ModelArkGenerate": "ModelArk Image Generate",
    "APIImage_QwenGenerate": "Qwen Image Generate",
    "APIImage_GLMGenerate": "GLM Image Generate",
    "APIImage_ConfigLoader": "API Config Loader",
    "APIImage_ConfigSaver": "API Config Saver",
    "APIImage_SaveImage": "API Image Save",
}

# Optional: web directory for JavaScript extensions
# WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
