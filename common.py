"""Helpers shared by the annotation tabs (text boxes and structured document)."""

import re
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from providers import get_provider

LOC_RE = re.compile(r"<\|LOC_(\d+)\|>")


def make_cfg(prov, url, key, mdl, mt) -> dict:
    return {"provider": prov, "base_url": url, "api_key": key,
            "model": mdl, "max_tokens": mt}


def ask_model(img: Image.Image, prompt: str, cfg: dict) -> str:
    if not cfg["model"]:
        raise gr.Error("Pick a model in Settings first.")
    provider = get_provider(cfg["provider"], cfg["base_url"], cfg["api_key"])
    try:
        return provider.chat(img, prompt, cfg["model"], cfg["max_tokens"])
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{cfg['provider']} request failed: {e}")


def parse_loc_lines(text: str, width: int, height: int) -> list[tuple]:
    """Parse PaddleOCR-VL `text<|LOC_x1|><|LOC_y1|>...<|LOC_x4|><|LOC_y4|>` lines
    into (text, xmin, ymin, xmax, ymax) in pixels.

    Coordinates are normalised to 0..1000; the quad is reduced to its
    axis-aligned bounding rectangle.
    """
    lines = []
    for line in text.splitlines():
        locs = [int(v) for v in LOC_RE.findall(line)]
        label = LOC_RE.sub("", line).strip()
        if len(locs) < 4 or not label:
            continue
        xs = [locs[i] * width / 1000 for i in range(0, len(locs) - 1, 2)]
        ys = [locs[i] * height / 1000 for i in range(1, len(locs), 2)]
        lines.append((label, min(xs), min(ys), max(xs), max(ys)))
    return lines


def crop_box(img: Image.Image, box: dict, pad: int) -> Image.Image:
    w, h = img.size
    crop = img.crop((
        max(0, box["xmin"] - pad), max(0, box["ymin"] - pad),
        min(w, box["xmax"] + pad), min(h, box["ymax"] + pad),
    ))
    # The model does poorly on tiny crops; upscale short lines a bit.
    if crop.height < 48:
        scale = 48 / max(crop.height, 1)
        crop = crop.resize((max(1, int(crop.width * scale)), 48), Image.BICUBIC)
    return crop


def open_image(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def status_text(paths, idx, extra=""):
    if not paths:
        return "No images loaded."
    msg = f"Image {idx + 1}/{len(paths)} — `{Path(paths[idx]).name}`"
    return f"{msg}  \n{extra}" if extra else msg
