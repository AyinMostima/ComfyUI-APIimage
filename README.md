# ComfyUI-APIImage

**ComfyUI custom nodes for image generation and editing through multiple API providers.**

Generate and edit images directly in ComfyUI using Google Gemini, xAI Grok, OpenAI GPT Image, BytePlus ModelArk Seedream, Alibaba Qwen, and ZhipuAI GLM through a unified node interface.

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue)
![Python](https://img.shields.io/badge/Python-%3E%3D3.10-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Current defaults:** OpenAI uses `gpt-image-2`; BytePlus ModelArk uses
> `dola-seedream-5-0-pro-260628`. Older OpenAI image models remain selectable
> for compatibility but are deprecated upstream.

## Features

- **6 AI Providers** — Gemini, Grok, OpenAI-compatible, BytePlus ModelArk, Qwen, and GLM in one plugin
- **Text-to-Image** — Generate images from text prompts with all providers
- **Image Editing** — Edit existing images with reference inputs (Gemini, Grok, OpenAI, ModelArk, Qwen)
- **Flexible Image Input** — Reference-capable nodes expose batch `ref_images` plus 3 individual `image1`/`image2`/`image3` slots
- **Inpainting** — Mask-based inpainting support (Gemini, Grok, OpenAI, Qwen)
- **Provider-Aware Image Encoding** — Reference and mask files use formats appropriate to each API
- **Token Usage Tracking** — Usage is logged when the provider returns usage metadata
- **Persistent Configuration** — API keys and settings saved across sessions in `api_config.json`
- **Custom Models** — Dynamically add/remove models through Config nodes
- **Batch Generation** — Provider-specific output counts, including ModelArk sequential generation where supported
- **Auto-Install** — Missing Python packages are installed automatically on first load

---

## Supported Nodes

| Node                      | Provider            | Capabilities                                                                                  |
| ------------------------- | ------------------- | --------------------------------------------------------------------------------------------- |
| **Gemini Image Generate** | Google Gemini       | Text-to-image, editing (model-dependent, up to 14 refs), inpainting, aspect ratio and resolution |
| **Grok Image Generate**   | xAI Grok            | Text-to-image, editing (up to 3 refs on the quality model), inpainting and size controls       |
| **OpenAI Image Generate** | OpenAI / Compatible | GPT Image 2 generation/editing (plugin limit: 16 refs), flexible size, quality and output controls |
| **ModelArk Image Generate** | BytePlus ModelArk  | Seedream Pro/Lite/4.x/3.0 with model-family validation, multi-reference editing and supported batch modes |
| **Qwen Image Generate**   | Alibaba Qwen        | Text-to-image, editing (1-3 ref images), inpainting, negative prompt, watermark control       |
| **GLM Image Generate**    | ZhipuAI GLM         | Text-to-image, quality (HD/standard), multiple size presets                                   |
| **API Config Loader**     | —                   | Load saved API configurations with optional overrides                                         |
| **API Config Saver**      | —                   | Save API keys, URLs, and manage custom model lists                                            |
| **API Image Save**        | —                   | Save generated images to output directory with metadata                                       |

---

## Installation

### Method 1: ComfyUI Manager (Recommended)

Search for **"API Image Generator"** in ComfyUI Manager and install.

### Method 2: Manual Install

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/AyinMostima/ComfyUI-APIimage.git
```

Dependencies are auto-installed on first load. To install manually:

```powershell
python -m pip install -r .\ComfyUI-APIimage\requirements.txt
```

### Dependencies

| Package        | Version   | Purpose                     |
| -------------- | --------- | --------------------------- |
| `google-genai` | >= 1.0.0  | Google Gemini SDK           |
| `xai_sdk`      | >= 1.5.0  | xAI Grok SDK                |
| `dashscope`    | >= 1.17.0 | Alibaba Qwen SDK            |
| `Pillow`       | >= 9.0.0  | Image processing            |
| `requests`     | >= 2.28.0 | HTTP requests (OpenAI, ModelArk, GLM) |

---

## Quick Start

### 1. Get API Keys

| Provider | Get Key                                                    |
| -------- | ---------------------------------------------------------- |
| Gemini   | [Google AI Studio](https://aistudio.google.com/apikey)     |
| Grok     | [xAI Console](https://console.x.ai)                        |
| OpenAI   | [OpenAI Platform](https://platform.openai.com/api-keys)    |
| ModelArk | [BytePlus Console](https://console.byteplus.com/ark)        |
| Qwen     | [DashScope Console](https://dashscope.console.aliyun.com/) |
| GLM      | [ZhipuAI Platform](https://open.bigmodel.cn/)              |

### 2. Add a Generation Node

1. Right-click canvas > **Add Node** > **APIImage** > Choose your provider
2. Enter your **API Key** and **prompt**
3. Connect the output `images` to a **Preview Image** or **API Image Save** node
4. Click **Queue Prompt**

### 3. Basic Workflow

```
[Provider Image Generate] --> [Preview Image]
         |
         +--> [API Image Save]
```

---

## Usage Guide

### Text-to-Image Generation

Simply enter a prompt and your API key. Select model and optional parameters:

- **Gemini**: Aspect ratio (1:1, 16:9, 9:16, etc.), Resolution (1K, 2K, 4K)
- **Grok**: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4), Resolution (1k, 2k)
- **OpenAI**: GPT Image 2 flexible sizes up to a 3840 px edge, quality (low/medium/high), PNG/JPEG/WebP output; transparent background is not supported by GPT Image 2
- **ModelArk**: Seedream 5.x/4.x supports 1K/2K or validated exact sizes; PNG/JPEG output and sequential generation are enabled only for compatible model families
- **Qwen**: Size presets (928x1664 to 1664x928), Negative prompt, Watermark toggle
- **GLM**: Quality (HD, standard), Multiple size presets per model

### Image Editing with References

Generation nodes expose the following reference inputs. Whether they can be used
depends on the selected model; GLM and text-only models reject reference images.

| Input        | Type          | Description                                                                                    |
| ------------ | ------------- | ---------------------------------------------------------------------------------------------- |
| `ref_images` | IMAGE (batch) | Standard ComfyUI batch — connect a **Batch Images** node to pass multiple images as one tensor |
| `image1`     | IMAGE         | Individual image slot #1 — connect a **Load Image** node directly                              |
| `image2`     | IMAGE         | Individual image slot #2                                                                       |
| `image3`     | IMAGE         | Individual image slot #3                                                                       |

All inputs are merged: `ref_images` batch + `image1` + `image2` + `image3`, in that order.

#### Why use individual slots instead of batch?

> **IMPORTANT:** ComfyUI’s **Batch Images** node requires all images to have the **same resolution**. If your images have different sizes (e.g., 1390×1800 and 1024×1024), one will be **forcibly resized** to match the other. This distortion can trigger API content moderation errors (e.g., `PROHIBITED_CONTENT` on Gemini).
>
> Using `image1`/`image2`/`image3` keeps each image at its **original resolution** — no resizing, no distortion.

#### Example: Individual image inputs

```
[Load Image A] --> image1 --+
[Load Image B] --> image2 --+--> [Gemini Image Generate] --> [Preview Image]
                  prompt ---+
```

#### Example: Batch input (same-size images only)

```
[Load Image A] --+
                 +--> [Batch Images] --> ref_images --> [Gemini Image Generate]
[Load Image B] --+
```

**Reference image limits per model:**

| Model                                           | Max Ref Images |
| ----------------------------------------------- | -------------- |
| `gemini-2.5-flash-image`                        | 3              |
| `gemini-3-pro-image-preview`                    | 14             |
| `grok-imagine-image-quality`                    | 3              |
| `grok-imagine-image` / `grok-imagine-image-pro` | 1              |
| `gpt-image-2` / `gpt-image-1`                   | 16             |
| `dola-seedream-5-0-pro-260628`                  | 10             |
| `seedream-5-0-260128` / `seedream-5-0-lite`     | 14             |
| `seedream-4-5` / `seedream-4-0`                 | 14             |
| `dall-e-2`                                      | 1              |
| `qwen-image-edit`                               | 3 (min 1)      |

> All limits apply to the **total combined count** from `ref_images` +
> `image1-3`. For GPT Image models, providing any reference image selects
> `/v1/images/edits`; a mask is optional. OpenAI and ModelArk reject known
> over-limit requests locally before making an API call.

### Inpainting

Connect both a reference image and `mask` input:

```
[Load Image] ----> image1 ------+
                                 +--> [Generate Node] --> [Preview Image]
[Mask Editor] ---> mask ---------+
```

Mask input is supported by the Gemini, Grok, OpenAI, and Qwen nodes. OpenAI
requires the mask dimensions to match the first reference image and sends the
mask as RGBA PNG. ModelArk does not expose a mask parameter.

### Image Encoding

Encoding is provider-specific:

- **ModelArk references** are sent as lowercase `data:image/jpeg;base64,...`
  values using JPEG quality 95 after local dimension and size validation.
- **OpenAI edit inputs** are sent as PNG multipart files. Masks are sent as
  RGBA PNG files so their alpha channel is preserved.
- **Gemini, Grok, and Qwen** use the encoding required by their current SDK or
  endpoint implementation.

Actual upload size depends on image dimensions and content; the plugin does not
promise a fixed compression ratio.

### Token Usage Tracking

Nodes log usage to the ComfyUI console when the API response includes usage
metadata. Examples:

```
[Gemini] Token usage | Attempt: official-primary | Prompt: 1856 | Output: 4231 | Total: 6087
[OpenAI] Success | Model: gpt-image-2 | Images: 1 | Tokens: Input: ... | Output: ... | Total: ...
[ModelArk] Success | Model: dola-seedream-5-0-pro-260628 | Generated: 1 | OutputTokens: ... | TotalTokens: ...
```

> Note: Some image generation APIs may not return token usage data (displayed as `N/A`).

### Persistent Configuration

Use **API Config Saver** to persist your settings:

```
[API Config Saver] -- save api_key, base_url, model
        |
[API Config Loader] -- load saved config --> [Generate Node]
```

### Custom Models

Add new models via the **API Config Saver** node:

1. Set `api_type` to your provider
2. Enter the model name in `add_custom_model_name`
3. Queue the node — the model will appear in dropdowns

Or override any model by filling the `custom_model` field on generation nodes.

For an opaque ModelArk Endpoint ID such as `ep-...`, enter it in `custom_model`
and select the matching `model_family` so model-specific parameters are enabled.

---

## Configuration

API configurations are stored in `api_config.json` within the plugin directory. You can edit this file directly or use the Config nodes.

```json
{
  "api_configs": {
    "Gemini Native": {
      "api_key": "YOUR_KEY",
      "base_url": "(SDK - Automatic)",
      "model_name": "gemini-2.5-flash-image",
      "custom_models": []
    },
    "Grok API": {
      "api_key": "YOUR_KEY",
      "base_url": "https://api.x.ai",
      "model_name": "grok-imagine-image-pro",
      "custom_models": []
    },
    "OpenAI Compatible": {
      "api_key": "YOUR_KEY",
      "base_url": "https://api.openai.com",
      "model_name": "gpt-image-2",
      "custom_models": []
    },
    "BytePlus ModelArk": {
      "api_key": "YOUR_KEY",
      "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
      "model_name": "dola-seedream-5-0-pro-260628",
      "custom_models": []
    }
  }
}
```

### ModelArk Regions and Model Families

| Region | Base URL |
| ------ | -------- |
| `ap-southeast-1` | `https://ark.ap-southeast.bytepluses.com/api/v3` |
| `eu-west-1` | `https://ark.eu-west.bytepluses.com/api/v3` |

| Family | Reference images | Sequential output | Custom output format | Family-only controls |
| ------ | ---------------- | ----------------- | -------------------- | -------------------- |
| Seedream 5.0 Pro | Up to 10 | No | PNG/JPEG | Prompt optimization |
| Seedream 5.0 Lite | Up to 14 | Yes; input + output <= 15 | PNG/JPEG | Prompt optimization |
| Seedream 4.5 / 4.0 | Up to 14 | Yes; input + output <= 15 | No; JPEG response | Prompt optimization |
| Seedream 3.0 T2I | None | No | No; JPEG response | `seed`, `guidance_scale` |

For an opaque inference Endpoint ID such as `ep-...`, set `custom_model` to the
Endpoint ID and select the matching `model_family`. This prevents unsupported
parameters from being sent to that endpoint.

### Custom Base URL / Proxy

All nodes support custom `base_url` for proxy or self-hosted endpoints. Enter a valid `https://` URL to override the default endpoint.

---

## Built-in Models

| Provider | Models                                                 |
| -------- | ------------------------------------------------------ |
| Gemini   | `gemini-2.5-flash-image`, `gemini-3-pro-image-preview` |
| Grok     | `grok-imagine-image-quality`, `grok-imagine-image`, `grok-imagine-image-pro` |
| OpenAI   | `gpt-image-2`, `gpt-image-2-2026-04-21`, `gpt-image-1`, `dall-e-3`, `dall-e-2` |
| ModelArk | `dola-seedream-5-0-pro-260628`, `seedream-5-0-260128`, `seedream-5-0-lite`, `seedream-4-5`, `seedream-4-0`, `seedream-3-0-t2i` |
| Qwen     | `qwen-image-plus`, `qwen-image-edit`                   |
| GLM      | `glm-image`, `cogview-4-250304`                        |

`gpt-image-2` is the current OpenAI default. `gpt-image-1`, `dall-e-3`, and
`dall-e-2` remain in the dropdown for compatibility but are deprecated by
OpenAI. Third-party OpenAI-compatible endpoints may support a different subset
of models and parameters.

---

## File Structure

```
comfyui-apiimage\
  __init__.py          # Node registration & auto-install
  config.py            # Persistent config management
  utils.py             # Image tensor/PIL/bytes conversion utilities
  nodes_gemini.py      # Google Gemini node
  nodes_grok.py        # xAI Grok node
  nodes_openai.py      # OpenAI-compatible node
  nodes_modelark.py    # BytePlus ModelArk Seedream node
  nodes_qwen.py        # Alibaba Qwen node
  nodes_glm.py         # ZhipuAI GLM node
  nodes_config.py      # Config Loader & Saver nodes
  nodes_save.py        # Image save node
  api_config.json      # Saved API configurations
  requirements.txt     # Python dependencies
  pyproject.toml       # Package metadata
```

---

## Troubleshooting

| Error                         | Solution                                                  |
| ----------------------------- | --------------------------------------------------------- |
| `Authentication failed (401)` | Check your API key is correct and active                  |
| `Rate limit exceeded (429)`   | Wait a moment and retry, or switch models                 |
| `Model not found (404)`       | Verify model name; use `custom_model` for new models      |
| `Request timed out`           | Check network; try a simpler prompt; high-resolution output may be slow |
| `Content filtered`            | Modify your prompt — it may have triggered safety filters |
| `GPT Image permission denied` | Confirm model access; OpenAI may require organization verification |
| `ModelArk invalid parameter`  | Select the correct `model_family`; Pro and 3.0 reject sequential controls |
| `Missing package`             | Run `python -m pip install -r .\requirements.txt` manually |

### Official API References

- [OpenAI GPT Image 2 model](https://developers.openai.com/api/docs/models/gpt-image-2)
- [OpenAI image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
- [BytePlus ModelArk image generation API](https://docs.byteplus.com/en/docs/ModelArk/1541523)

---

## License

[MIT License](LICENSE)

---

## Contributing

Contributions are welcome! Please open an issue or pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-provider`)
3. Commit your changes
4. Push to your branch and open a Pull Request
