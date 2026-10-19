"""Structured-document tab: parse a page into structured JSON.

"Parse page" runs the whole pipeline (`parse_page`):
1. detect the page's layout sections with a vision model, in reading order;
2. read the sections that have no text yet;
3. pick the output schema from the page's content (document, Q&A, MCQ or
   QCM; see schemas.py), by rules or by asking the model;
4. relabel the sections to fit it (e.g. "1. ..." -> question, "A. ..." -> choice);
5. fill the schema and validate it.
Each step can also be run on its own, and the schema can be chosen by hand.

The model family is picked with a preset (see `PROFILES`):

- PaddleOCR-VL-1.5 has no layout task, so sections are built from the
  `Spotting:` text lines, merged into blocks. Draw or retype tables, formulas
  and figures by hand; "Read sections" then uses the matching task prompt
  (`Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`).
- Qwen3-VL (e.g. qwen3-vl-4b-thinking) and the Generic VLM preset use the
  prompts in prompt.md: one prompt returns every section with its type, box
  (0-1000 page coordinates) and content, and the model can also pick and
  fill an output schema.
- dots.ocr is a document parser: its own layout prompt returns every section
  with its category, box (pixels of the image it was sent) and content.

Each image is autosaved to <out>/structured/<image>.json holding the flat
`sections` list (reading order), the nested `document` built from it, and the
latest schema `extraction`.
"""

import json
import math
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pandas as pd
from gradio_image_annotation import image_annotator
from PIL import Image

import schemas
from prompts import PROMPTS
from formulas import bare_latex, segments, whole_formula
from common import ask_model, crop_box, make_cfg, open_image, parse_loc_lines, status_text
from schemas import SCHEMA_CHOICES, SCHEMAS, line_role

SECTION_TYPES = ["title", "heading", "paragraph", "list", "question", "choice", "answer",
                 "explanation", "table", "formula", "figure", "caption", "footnote",
                 "header", "footer", "page_number"]
TYPE_COLORS = {
    "title": (220, 38, 38), "heading": (234, 88, 12), "paragraph": (37, 99, 235),
    "list": (8, 145, 178), "question": (190, 18, 60), "choice": (13, 148, 136),
    "answer": (21, 128, 61), "explanation": (161, 98, 7), "table": (22, 163, 74),
    "formula": (147, 51, 234), "figure": (219, 39, 119), "caption": (202, 138, 4),
    "footnote": (120, 113, 108), "header": (100, 116, 139), "footer": (100, 116, 139),
    "page_number": (100, 116, 139),
}
OTHER_COLOR = (64, 64, 64)
TYPE_ALIASES = {
    "text": "paragraph", "plain_text": "paragraph", "section_header": "heading",
    "section_title": "heading", "subtitle": "heading", "doc_title": "title",
    "list_item": "list", "equation": "formula", "isolate_formula": "formula",
    "image": "figure", "picture": "figure", "chart": "figure",
    "figure_caption": "caption", "table_caption": "caption",
    "page_header": "header", "page_footer": "footer",
    "question_stem": "question", "stem": "question", "option": "choice",
    "answer_key": "answer", "solution": "explanation",
}
# Types whose read text is collapsed onto one line.
INLINE_TYPES = {"title", "heading", "paragraph", "caption", "header", "footer", "page_number"}
COORDS = ("xmin", "ymin", "xmax", "ymax")
TABLE_HEADERS = ["order", "type", "level", "content", *COORDS]
LABEL_RE = re.compile(r"\s*(\d+)\s*:\s*(.*)")
LIST_RE = re.compile(r"^\s*([•●▪◦·\-–*]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.S)

# dots.ocr's own prompts (prompt_layout_all_en and prompt_ocr); the model is
# trained on this exact wording, so keep it as is.
DOTS_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
DOTS_OCR_PROMPT = "Extract the text content from this image."

# detect: "spotting" (PaddleOCR-VL text lines) or "layout" (JSON list of elements).
# bbox: "norm1000" (0-1000 page coordinates) or "pixels" (pixels of the image sent).
# max_pixels: resize the page like the model's processor (Qwen2-VL smart_resize)
#   before sending, so pixel boxes can be scaled back to the original image.
# can_fill: the model follows free-form instructions well enough to pick and fill a schema.
# describe_figures: the parse pipeline asks the model to describe empty figures.
# General instruction-following VLMs share the prompts in prompt.md.
GENERAL_VLM = {
    "detect": "layout",
    "bbox": "norm1000",
    "detect_prompt": PROMPTS["layout"],
    "read_prompts": {
        "table": PROMPTS["read-table"],
        "formula": PROMPTS["read-formula"],
        "figure": PROMPTS["read-figure"],
        "_default": PROMPTS["read-text"],
    },
    "can_fill": True,
    "describe_figures": True,
}

PROFILES = {
    "PaddleOCR-VL-1.5": {
        "match": ["paddleocr-vl-1.5", "paddleocr-vl", "paddle"],
        "model": "paddleocr-vl-1.5",
        "detect": "spotting",
        "detect_prompt": "Spotting:",
        "read_prompts": {
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "figure": "Chart Recognition:",
            "_default": "OCR:",
        },
        "can_fill": False,
    },
    "Qwen3-VL-4B-Thinking": {
        "match": ["qwen3-vl-4b-thinking", "qwen3-vl-4b", "qwen3-vl"],
        "model": "qwen/qwen3-vl-4b-thinking",
        **GENERAL_VLM,
    },
    "dots.ocr": {
        "match": ["dots.ocr", "dots_ocr", "dots-ocr", "dotsocr", "dots"],
        "model": "dots.ocr",
        "detect": "layout",
        "bbox": "pixels",
        "max_pixels": 11289600,
        "detect_prompt": DOTS_LAYOUT_PROMPT,
        "read_prompts": {
            # The layout prompt formats tables as HTML and formulas as LaTeX;
            # clean_read() keeps just the text of its JSON answer.
            "table": DOTS_LAYOUT_PROMPT,
            "formula": DOTS_LAYOUT_PROMPT,
            "_default": DOTS_OCR_PROMPT,
        },
        "can_fill": False,
    },
    # Any other VLM (InternVL, Gemma 3, MiniCPM-V, an API model...): choose
    # this preset, then the model. It has no name patterns, so it never
    # switches the model.
    "Generic VLM": {
        "match": [],
        "model": "",
        **GENERAL_VLM,
    },
}


def profile_for(model: str) -> str | None:
    """The preset whose model this looks like, or None."""
    name = (model or "").lower()
    for preset, p in PROFILES.items():
        if any(kw in name for kw in p["match"]):
            return preset
    return None


def pick_model(models: list[str], preset: str) -> str | None:
    """The server model that best matches a preset, or None."""
    for kw in PROFILES[preset]["match"]:
        hits = [m for m in models if kw in m.lower()]
        if hits:
            return hits[0]
    return None


def smart_resize(width: int, height: int, factor=28, min_pixels=3136,
                 max_pixels=11289600) -> tuple[int, int]:
    """The size Qwen2-VL-style processors (used by dots.ocr) resize an image to:
    both sides multiples of `factor`, area within [min_pixels, max_pixels]."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return w, h


def page_for_model(img: Image.Image, profile: dict) -> Image.Image:
    """The page as sent for detection. Pixel-box models get it pre-resized the
    way their processor would, so the server leaves it alone and the boxes are
    in a known size."""
    if not profile.get("max_pixels"):
        return img
    size = smart_resize(*img.size, max_pixels=profile["max_pixels"])
    return img if size == img.size else img.resize(size, Image.BICUBIC)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def normalise_type(t) -> str:
    t = re.sub(r"[\s-]+", "_", str(t or "").strip().lower())
    return TYPE_ALIASES.get(t, t) or "paragraph"


def clamp_rect(x0, y0, x1, y1, width, height) -> dict:
    clamp = lambda v, hi: int(round(max(0, min(float(v), hi))))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return {"xmin": clamp(x0, width), "ymin": clamp(y0, height),
            "xmax": clamp(x1, width), "ymax": clamp(y1, height)}


def new_section(type_, rect, content="", level=None) -> dict:
    return {"type": type_, "level": level if type_ == "heading" else None,
            "content": content, **{k: rect[k] for k in COORDS}}


def group_lines(lines: list[tuple], width: int, height: int) -> list[dict]:
    """Merge PaddleOCR-VL `Spotting:` lines into blocks and guess each block's type
    from its line height (headings) and leading bullets (lists).

    On a question page (two or more lines start with an option letter, or one
    is an answer line) each question, option, answer and explanation line
    starts its own block, typed by its role."""
    if not lines:
        return []
    med = statistics.median(y1 - y0 for _, _, y0, _, y1 in lines) or 1
    roles = [line_role(t) for t, *_ in lines]
    exam = roles.count("choice") >= 2 or "answer" in roles
    blocks = []
    for text, x0, y0, x1, y1 in sorted(lines, key=lambda l: (l[2], l[1])):
        lh = y1 - y0
        big = lh > 1.3 * med
        bullet = bool(LIST_RE.match(text))
        role = line_role(text) if exam else None
        # A role line always starts a block; other lines join one when close below it.
        for b in [] if role else reversed(blocks):
            gap = y0 - b["ymax"]
            overlap = min(x1, b["xmax"]) - max(x0, b["xmin"])
            if (b["big"] == big and -0.5 * med < gap < 0.8 * med and overlap > 0
                    and (not bullet or b["list"])):
                b["lines"].append(text)
                b["xmin"], b["ymin"] = min(b["xmin"], x0), min(b["ymin"], y0)
                b["xmax"], b["ymax"] = max(b["xmax"], x1), max(b["ymax"], y1)
                b["lh"] = max(b["lh"], lh)
                break
        else:
            blocks.append({"lines": [text], "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                           "big": big, "list": bullet and not role, "role": role, "lh": lh})

    tallest = max(b["lh"] for b in blocks)
    sections, has_title = [], False
    for b in reading_order(blocks, width):
        rect = clamp_rect(b["xmin"], b["ymin"], b["xmax"], b["ymax"], width, height)
        if b["role"]:
            sections.append(new_section(b["role"], rect, " ".join(b["lines"])))
        elif b["big"] and not has_title and b["lh"] == tallest and b["ymin"] < height / 3:
            sections.append(new_section("title", rect, " ".join(b["lines"])))
            has_title = True
        elif b["big"]:
            level = 1 if b["lh"] > 1.6 * med else 2
            sections.append(new_section("heading", rect, " ".join(b["lines"]), level))
        elif b["list"]:
            sections.append(new_section("list", rect, "\n".join(b["lines"])))
        else:
            sections.append(new_section("paragraph", rect, " ".join(b["lines"])))
    return sections


def reading_order(blocks: list[dict], width: int) -> list[dict]:
    """Top-to-bottom, reading a two-column stretch left column first. Blocks
    that cross the page middle (titles, wide tables) separate the stretches."""
    mid = width / 2
    ordered, band = [], []
    for b in sorted(blocks, key=lambda b: (b["ymin"], b["xmin"])):
        if b["xmin"] < mid < b["xmax"]:
            ordered += sorted(band, key=lambda b: (b["xmin"] >= mid, b["ymin"])) + [b]
            band = []
        else:
            band.append(b)
    return ordered + sorted(band, key=lambda b: (b["xmin"] >= mid, b["ymin"]))


# Backslashes that aren't valid JSON escapes, e.g. LaTeX "\pm" written unescaped.
BAD_ESCAPE_RE = re.compile(r'\\(?![\\/"bfnrt]|u[0-9a-fA-F]{4})')


def extract_json(text: str):
    """The first JSON value in `text`; if the output was cut off, every complete
    object found in it. Unescaped LaTeX backslashes are repaired."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    dec = json.JSONDecoder()
    for text in (text, BAD_ESCAPE_RE.sub(r"\\\\", text)):
        try:
            return dec.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            pass
    items, i = [], start + 1
    while (i := text.find("{", i)) >= 0:
        try:
            obj, i = dec.raw_decode(text, i)
            items.append(obj)
        except json.JSONDecodeError:
            i += 1
    return items or None


# LaTeX commands whose unescaped backslash JSON reads as a control character:
# "\frac" -> form feed + "rac", "\times" -> tab + "imes", "\nabla" -> newline...
LATEX_CONTROL_FIXES = [
    (re.compile("\x0c"), r"\\f"),
    (re.compile("\x08"), r"\\b"),
    (re.compile(r"\t(?=[a-zA-Z])"), r"\\t"),
    (re.compile(r"\r(?=[a-zA-Z])"), r"\\r"),
    (re.compile(r"\n(?=(?:abla|eq|eg|ot|u|ewline|exists|mid)\b)"), r"\\n"),
]


def fix_latex(value):
    """Undo LATEX_CONTROL_FIXES damage in a string, or in every string of a JSON value."""
    if isinstance(value, dict):
        return {k: fix_latex(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fix_latex(v) for v in value]
    if not isinstance(value, str):
        return value
    for regex, repl in LATEX_CONTROL_FIXES:
        value = regex.sub(repl, value)
    return value


def parse_layout(text: str, width: int, height: int, bbox: str = "norm1000",
                 sent_size: tuple[int, int] | None = None) -> list[dict]:
    """Parse a JSON list of layout elements: Qwen-VL style {type, bbox_2d,
    content, level} or dots.ocr style {category, bbox, text}.

    `bbox` says how boxes are given: "norm1000" (0-1000 page coordinates) or
    "pixels" (pixels of the image sent, of size `sent_size`)."""
    data = extract_json(text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    items = [d for d in data or [] if isinstance(d, dict)]
    boxes = []
    for d in items:
        b = d.get("bbox_2d") or d.get("bbox") or d.get("box")
        try:
            b = [float(v) for v in b]
        except (TypeError, ValueError):
            b = None
        boxes.append(b if b and len(b) == 4 else None)
    sw, sh = sent_size or (width, height)
    # Qwen3-VL answers in 0-1000 page coordinates; Qwen2.5-VL in pixels.
    if bbox == "norm1000" and max((v for b in boxes if b for v in b), default=0) <= 1000:
        sx, sy = width / 1000, height / 1000
    else:
        sx, sy = width / sw, height / sh

    sections = []
    for d, b in zip(items, boxes):
        if not b:
            continue
        rect = clamp_rect(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy, width, height)
        if rect["xmax"] <= rect["xmin"] or rect["ymax"] <= rect["ymin"]:
            continue
        type_ = normalise_type(d.get("type") or d.get("category") or d.get("label"))
        content = d.get("content", d.get("text", ""))
        if isinstance(content, list):
            content = "\n".join(map(str, content))
        content = fix_latex(str(content or "").strip())
        level = d.get("level")
        # dots.ocr writes headings as Markdown: "## 1.2 Results".
        if (m := MD_HEADING_RE.match(content)) and type_ in ("title", "heading"):
            content, level = m[2].strip(), level or len(m[1])
        try:
            level = int(level or 1)
        except (TypeError, ValueError):
            level = 1
        sections.append(new_section(type_, rect, content, level))
    return sections


def clean_read(text: str) -> str:
    """A section's read text. dots.ocr answers its layout prompt with JSON even
    for a crop; keep only the text of its elements."""
    if text.lstrip()[:1] in "[{":
        data = extract_json(text)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            parts = [str(d.get("text", d.get("content", ""))).strip() for d in data]
            return fix_latex("\n".join(p for p in parts if p))
    return fix_latex(text)


# Section types that become a formula when their text is only a formula.
FORMULA_FROM = {"paragraph", "list", "text", "caption", "footnote"}


def mark_formulas(sections: list[dict]) -> int:
    """In place: a text section that is only a formula (`\\[ E = mc^2 \\]`) becomes
    a formula section, and formula sections lose their delimiters. Returns how
    many sections were retyped."""
    retyped = 0
    for s in sections:
        if s["type"] in FORMULA_FROM and (latex := whole_formula(s["content"])):
            s.update(type="formula", level=None, content=latex)
            retyped += 1
        elif s["type"] == "formula":
            s["content"] = bare_latex(s["content"])
    return retyped


def element(s: dict, bbox: list[int]) -> dict:
    """A document block. Text that contains formulas also gets `segments`,
    marking which parts are text and which are formulas."""
    elem = {"type": s["type"], "bbox": bbox, "content": s["content"]}
    if s["type"] not in ("formula", "table", "figure") and (parts := segments(s["content"])):
        elem["segments"] = parts
    return elem


def build_document(sections: list[dict]) -> dict:
    """Nest the flat, reading-ordered sections under their headings."""
    doc = {"title": None, "page_header": [], "page_footer": [], "blocks": [], "sections": []}
    stack = [(0, doc)]
    for s in sections:
        bbox = [s[k] for k in COORDS]
        elem = element(s, bbox)
        if s["type"] == "title" and doc["title"] is None:
            doc["title"] = s["content"]
        elif s["type"] == "header":
            doc["page_header"].append(elem)
        elif s["type"] in ("footer", "page_number"):
            doc["page_footer"].append(elem)
        elif s["type"] == "heading":
            level = s.get("level") or 1
            while stack[-1][0] >= level:
                stack.pop()
            node = {"heading": s["content"], "level": level, "bbox": bbox,
                    "blocks": [], "sections": []}
            stack[-1][1]["sections"].append(node)
            stack.append((level, node))
        else:
            stack[-1][1]["blocks"].append(elem)
    return doc


def sections_to_markdown(sections: list[dict]) -> str:
    """The page as Markdown, used as reference text when a model fills a schema."""
    parts = []
    for s in sections:
        t, c = s["type"], s["content"].strip()
        if not c or t in ("header", "footer", "page_number"):
            continue
        if t == "title":
            parts.append(f"# {c}")
        elif t == "heading":
            parts.append(f"{'#' * min((s.get('level') or 1) + 1, 6)} {c}")
        elif t == "formula":
            parts.append(f"\\[ {bare_latex(c)} \\]")
        elif t == "figure":
            parts.append(f"[Figure: {c}]")
        else:
            parts.append(c)
    return "\n\n".join(parts)


def merge_annot(boxes: list[dict], base: list[dict]) -> list[dict]:
    """Fold annotator edits (move, resize, delete, draw, relabel) back into the
    sections. Boxes are labelled "<n>:<type>", n indexing `base` (the sections
    the annotator was last drawn from); a new box's label is just its type."""
    out = []
    for b in boxes:
        rect = {k: b[k] for k in COORDS}
        m = LABEL_RE.match(b.get("label", ""))
        idx, type_ = (int(m[1]), m[2]) if m else (None, b.get("label", ""))
        if idx is None or idx >= len(base):
            # A box relabelled without its "n:" keeps its content if it didn't move.
            idx = next((i for i, s in enumerate(base)
                        if all(abs(s[k] - rect[k]) <= 2 for k in COORDS)), None)
        prev = base[idx] if idx is not None else {}
        type_ = normalise_type(type_ or prev.get("type"))
        out.append(new_section(type_, rect, prev.get("content", ""), prev.get("level") or 1))
    return out


def sections_to_table(sections: list[dict]) -> pd.DataFrame:
    rows = [[i, s["type"], s["level"], s["content"], *(s[k] for k in COORDS)]
            for i, s in enumerate(sections)]
    return pd.DataFrame(rows, columns=TABLE_HEADERS)


def table_to_sections(df: pd.DataFrame, width: int, height: int) -> list[dict]:
    rows = []
    for n, r in df.iterrows():
        try:
            rect = clamp_rect(*(float(r[k]) for k in COORDS), width, height)
        except (TypeError, ValueError):
            continue
        try:
            order = float(r["order"])
        except (TypeError, ValueError):
            order = float("inf")
        try:
            level = int(float(r["level"]))
        except (TypeError, ValueError):
            level = 1
        type_ = normalise_type("" if pd.isna(r["type"]) else r["type"])
        content = "" if pd.isna(r["content"]) else str(r["content"])
        rows.append((order, n, new_section(type_, rect, content, level)))
    return [s for *_, s in sorted(rows, key=lambda t: t[:2])]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def struct_path(out_dir: str, image_path: str) -> Path:
    return Path(out_dir) / "structured" / (Path(image_path).name + ".json")


def load_record(out_dir: str, image_path: str, width: int, height: int) -> dict:
    p = struct_path(out_dir, image_path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: data.get(k) for k in ("image", "width", "height", "model", "extraction")} | {
            "sections": data.get("sections", [])}
    return {"image": str(image_path), "width": width, "height": height,
            "model": None, "sections": [], "extraction": None}


def save_record(out_dir: str, record: dict) -> dict:
    full = {**record, "document": build_document(record["sections"])}
    p = struct_path(out_dir, record["image"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    return full


def dump(record: dict | None) -> str:
    if not record:
        return ""
    return json.dumps({**record, "document": build_document(record["sections"])},
                      ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Pipeline: detect -> read -> pick schema -> label sections -> convert
# --------------------------------------------------------------------------- #
def detect_sections(img: Image.Image, profile: dict, prompt: str,
                    cfg: dict) -> tuple[list[dict], str]:
    """The page's sections in reading order, and the raw model output."""
    w, h = img.size
    if profile["detect"] == "spotting":
        raw = ask_model(img, prompt, cfg)
        sections = group_lines(parse_loc_lines(raw, w, h), w, h)
    else:
        page = page_for_model(img, profile)
        raw = ask_model(page, prompt, cfg)
        sections = parse_layout(raw, w, h, profile["bbox"], page.size)
    mark_formulas(sections)
    return sections, raw


def no_sections_message(preset_name: str, raw: str) -> str:
    hint = ("The model returned nothing — it may have used up Max tokens while thinking."
            if not raw else f"Model output starts: {raw[:200]!r}")
    return f"No sections found for preset {preset_name}. {hint}"


def read_sections(img: Image.Image, sections: list[dict], targets: list[int], profile: dict,
                  cfg: dict, padding: int, progress=None) -> None:
    """Read the sections at `targets` (indexes) in place."""
    prompts = profile["read_prompts"]
    for i in progress.tqdm(targets, desc="Reading sections") if progress else targets:
        s = sections[i]
        text = clean_read(ask_model(crop_box(img, s, int(padding)),
                                    prompts.get(s["type"], prompts["_default"]), cfg))
        s["content"] = " ".join(text.split()) if s["type"] in INLINE_TYPES else text
    mark_formulas(sections)


def pick_schema(sections: list[dict], img: Image.Image, cfg: dict,
                use_model: bool) -> tuple[str, list[str]]:
    """The schema that fits the page, and why. Asks the model when `use_model`,
    falling back to the rules if its answer is unusable."""
    if not use_model:
        return schemas.classify(sections)
    raw = ask_model(img, schemas.classify_prompt(sections_to_markdown(sections)), cfg)
    answer = extract_json(raw)
    if isinstance(answer, dict) and answer.get("schema") in SCHEMAS:
        return answer["schema"], [f"Model: {answer.get('reason') or 'no reason given'}"]
    name, reasons = schemas.classify(sections)
    return name, ["The model's answer wasn't usable, so the rules decided.", *reasons]


def convert_record(record: dict, img: Image.Image, label: str, method: str, preset_name: str,
                   cfg: dict) -> dict:
    """Pick the schema (when `label` is Auto), relabel the sections to match it,
    fill it, and validate. Returns the record with new sections and extraction."""
    notes = []
    sections = [dict(s) for s in record["sections"]]
    if retyped := mark_formulas(sections):
        notes.append(f"Marked {retyped} section(s) as formula.")
    record = {**record, "sections": sections}
    use_model = method == "Model" and PROFILES[preset_name]["can_fill"]
    if method == "Model" and not use_model:
        notes.append(f"{preset_name} only parses pages and can't follow a schema, so the "
                     "rules were used.")
    chosen = SCHEMA_CHOICES[label]
    name, reasons = chosen, []
    if chosen == "auto":
        name, reasons = pick_schema(record["sections"], img, cfg, use_model)
    sections, changed = schemas.label_sections(record["sections"], name)
    if changed:
        notes.append(f"Relabelled {changed} section(s) as question parts.")
    if name == "document":
        data = None  # built from the sections whenever it's shown or saved
    elif use_model:
        raw = ask_model(img, schemas.fill_prompt(name, sections_to_markdown(sections)), cfg)
        data = fix_latex(extract_json(raw))
        if not isinstance(data, dict):
            raise gr.Error(f"The model didn't return a JSON object. Output starts: {raw[:200]!r}")
        data["schema"] = name
    else:
        data, more = schemas.extract(name, sections)
        notes += more
    errors = schemas.validate(name, build_document(sections) if data is None else data)
    return {**record, "sections": sections, "extraction": {
        "schema": name, "auto": chosen == "auto", "reasons": reasons,
        "method": "model" if use_model else "rules", "data": data, "errors": errors,
        "notes": notes}}


def parse_page(path: str, img: Image.Image, preset_name: str, cfg: dict, prompt: str,
               padding: int, label: str, method: str) -> dict:
    """The whole pipeline for one page; returns its record (not saved)."""
    profile = PROFILES[preset_name]
    sections, raw = detect_sections(img, profile, prompt, cfg)
    if not sections:
        raise gr.Error(no_sections_message(preset_name, raw))
    empty = [i for i, s in enumerate(sections) if not s["content"].strip()
             and (s["type"] != "figure" or profile.get("describe_figures"))]
    read_sections(img, sections, empty, profile, cfg, padding)
    w, h = img.size
    record = {"image": str(path), "width": w, "height": h, "model": cfg["model"],
              "sections": sections, "extraction": None}
    return convert_record(record, img, label, method, preset_name, cfg)


def schema_data(record: dict) -> dict | None:
    """The record's schema output; a document is built from its sections."""
    ext = record.get("extraction")
    if not ext:
        return None
    return build_document(record["sections"]) if ext["data"] is None else ext["data"]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def schema_label(name: str) -> str:
    return next(k for k, v in SCHEMA_CHOICES.items() if v == name)


def extraction_status(ext: dict) -> str:
    name, data, errors = ext["schema"], ext["data"], ext["errors"]
    msg = (f"{'Auto-detected' if ext.get('auto') else 'Schema'}: **{schema_label(name)}**"
           f" · {ext['method']}")
    if isinstance(data, dict) and name != "document":
        msg += f" · {len(data.get('items') or [])} questions"
    msg += f" · ⚠️ {len(errors)} schema problem(s)" if errors else " · ✅ valid"
    lines = ([f"- {r}" for r in ext.get("reasons", [])] + [f"- `{e}`" for e in errors[:8]]
             + [f"- {n}" for n in ext.get("notes", [])])
    return msg + ("\n\n" + "\n".join(lines) if lines else "")


def extraction_view(record) -> tuple[str, str]:
    if not record or not record.get("extraction"):
        return "", "No schema output yet: click **⚡ Parse page**, or **🧩 Convert**."
    return (json.dumps(schema_data(record), ensure_ascii=False, indent=2),
            extraction_status(record["extraction"]))


def annot_value(image_path, sections, img=None):
    if not image_path:
        return None
    boxes = [{"label": f"{i}:{s['type']}", "color": TYPE_COLORS.get(s["type"], OTHER_COLOR),
              **{k: s[k] for k in COORDS}} for i, s in enumerate(sections)]
    return {"image": img or open_image(image_path), "boxes": boxes}


def view(record, img=None):
    """Values for StructureView.outputs:
    (annot, base, table, json, record, schema output, schema status)."""
    if not record:
        return None, [], sections_to_table([]), "", None, *extraction_view(None)
    sections = record["sections"]
    return (annot_value(record["image"], sections, img), sections,
            sections_to_table(sections), dump(record), record, *extraction_view(record))


def show(image_path: str | None, out_dir: str):
    if not image_path:
        return view(None)
    img = open_image(image_path)
    return view(load_record(out_dir, image_path, *img.size), img)


def schema_text(label: str, record=None) -> str:
    name = SCHEMA_CHOICES[label]
    if name == "auto":
        detected = ((record or {}).get("extraction") or {}).get("schema")
        shown = {detected: SCHEMAS[detected]} if detected else SCHEMAS
    else:
        shown = SCHEMAS[name]
    return json.dumps(shown, ensure_ascii=False, indent=2)


@dataclass
class StructureView:
    annot: gr.components.Component
    base: gr.State
    table: gr.Dataframe
    json: gr.Code
    record: gr.State
    extract_code: gr.Code
    extract_status: gr.Markdown
    parse_btn: gr.Button
    parse_all_btn: gr.Button
    skip_parsed: gr.Checkbox
    detect_btn: gr.Button
    read_mode: gr.Radio
    read_btn: gr.Button
    clear_btn: gr.Button
    apply_btn: gr.Button
    export_btn: gr.Button
    export_files: gr.File
    schema: gr.Dropdown
    method: gr.Radio
    convert_btn: gr.Button
    save_extract_btn: gr.Button
    schema_view: gr.Code
    detect_prompt: gr.Textbox
    max_tokens: gr.Number

    @property
    def outputs(self):
        return [self.annot, self.base, self.table, self.json, self.record,
                self.extract_code, self.extract_status]


def build_tab(default_preset: str) -> StructureView:
    """Lay out the tab's components; call inside a `gr.Tab`."""
    base = gr.State([])
    record = gr.State(None)
    with gr.Row(equal_height=True):
        parse_btn = gr.Button("⚡ Parse page", variant="primary")
        parse_all_btn = gr.Button("⚡ Parse all pages")
        skip_parsed = gr.Checkbox(True, label="Skip pages already parsed", container=False)
    with gr.Row():
        detect_btn = gr.Button("🧭 Detect sections", size="sm")
        read_mode = gr.Radio(["Empty sections", "All sections"], value="Empty sections",
                             show_label=False, container=False)
        read_btn = gr.Button("📖 Read sections", size="sm")
        clear_btn = gr.Button("🗑 Clear sections", variant="stop", size="sm")
    with gr.Row():
        with gr.Column(scale=3):
            annot = image_annotator(
                label="Sections — labelled \"<order>:<type>\". Draw a box and name it by "
                      f"type ({', '.join(SECTION_TYPES)}); double-click to retype; "
                      "select + Delete to remove.",
                image_type="filepath",
                sources=[],
                boxes_alpha=0.12,
                box_min_size=4,
                handle_size=6,
                show_clear_button=False,
                show_remove_button=False,
                height=720,
            )
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Schema output"):
                    with gr.Row():
                        schema = gr.Dropdown(list(SCHEMA_CHOICES), value=next(iter(SCHEMA_CHOICES)),
                                             label="Output schema", scale=3)
                        method = gr.Radio(["Rules", "Model"], value="Rules",
                                          label="Decide + fill with", scale=2)
                    with gr.Row():
                        convert_btn = gr.Button("🧩 Convert")
                        save_extract_btn = gr.Button("💾 Validate + save edits")
                    extract_status = gr.Markdown()
                    extract_code = gr.Code(language="json", label="Schema output (editable)",
                                           interactive=True, lines=22, max_lines=22,
                                           buttons=["copy", "download"])
                with gr.Tab("Document JSON"):
                    json_view = gr.Code(language="json", label="Structured JSON (autosaved)",
                                        interactive=False, lines=30, max_lines=30,
                                        buttons=["copy", "download"])
                with gr.Tab("JSON Schema"):
                    schema_view = gr.Code(schema_text(next(iter(SCHEMA_CHOICES))),
                                          language="json", label="Output schema",
                                          interactive=False, lines=30, max_lines=30,
                                          buttons=["copy"])
    with gr.Row():
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["number", "str", "number", "str", "number", "number", "number", "number"],
            interactive=True,
            label="Sections in reading order — edit type/level/content/order here, "
                  "then click Apply",
            wrap=True,
            scale=4,
        )
        with gr.Column(scale=1, min_width=160):
            apply_btn = gr.Button("✅ Apply table edits")
            export_btn = gr.Button("📦 Export JSONL", variant="primary")
            export_files = gr.File(label="Export", file_count="multiple")
    with gr.Accordion("Structure settings", open=False):
        detect_prompt = gr.Textbox(PROFILES[default_preset]["detect_prompt"], lines=6,
                                   max_lines=30, label="Detect prompt (set by the model preset)")
        max_tokens = gr.Number(16384, precision=0,
                               label="Max tokens (thinking models need room to think)")
    return StructureView(annot, base, table, json_view, record, extract_code, extract_status,
                         parse_btn, parse_all_btn, skip_parsed, detect_btn, read_mode,
                         read_btn, clear_btn, apply_btn, export_btn, export_files, schema,
                         method, convert_btn, save_extract_btn, schema_view, detect_prompt,
                         max_tokens)


def wire(v: StructureView, *, paths_state, idx_state, conn_inputs, preset, pad, out_dir,
         status):
    """Attach the tab's event handlers. `conn_inputs` = [provider, base_url, api_key, model]."""
    skip_view = [gr.skip()] * len(v.outputs)

    def parse(paths, idx, prov, url, key, mdl, preset_name, mt, prompt, label, method, padding,
              out, progress=gr.Progress()):
        if not paths:
            raise gr.Error("Load an image first.")
        path = paths[idx]
        img = open_image(path)
        progress(0.05, desc=f"Parsing with {mdl}…")
        record = parse_page(path, img, preset_name, make_cfg(prov, url, key, mdl, mt), prompt,
                            padding, label, method)
        save_record(out, record)
        ext = record["extraction"]
        msg = f"Parsed {len(record['sections'])} sections → {schema_label(ext['schema'])}."
        return *view(record, img), status_text(paths, idx, msg)

    def parse_all(paths, idx, skip, prov, url, key, mdl, preset_name, mt, prompt, label, method,
                  padding, out, progress=gr.Progress()):
        if not paths:
            raise gr.Error("Load images first.")
        cfg = make_cfg(prov, url, key, mdl, mt)
        counts, failed, skipped = {}, [], 0
        for path in progress.tqdm(paths, desc="Parsing pages"):
            p = struct_path(out, path)
            if skip and p.exists() and json.loads(p.read_text(encoding="utf-8")).get("sections"):
                skipped += 1
                continue
            try:
                record = parse_page(path, open_image(path), preset_name, cfg, prompt, padding,
                                    label, method)
            except gr.Error as e:
                failed.append(f"{Path(path).name}: {e.message}")
                continue
            save_record(out, record)
            name = schema_label(record["extraction"]["schema"])
            counts[name] = counts.get(name, 0) + 1
        msg = f"Parsed {sum(counts.values())} pages"
        if counts:
            msg += " (" + ", ".join(f"{n} {k}" for k, n in counts.items()) + ")"
        msg += f", skipped {skipped}, failed {len(failed)}."
        if failed:
            gr.Warning("Some pages failed: " + "; ".join(failed[:3]))
        return *show(paths[idx], out), status_text(paths, idx, msg)

    def detect(paths, idx, prov, url, key, mdl, preset_name, mt, prompt, out,
               progress=gr.Progress()):
        if not paths:
            raise gr.Error("Load an image first.")
        path = paths[idx]
        img = open_image(path)
        w, h = img.size
        progress(0.1, desc=f"Detecting sections with {mdl}…")
        sections, raw = detect_sections(img, PROFILES[preset_name], prompt,
                                        make_cfg(prov, url, key, mdl, mt))
        if not sections:
            gr.Warning(no_sections_message(preset_name, raw))
            return *skip_view, status_text(paths, idx, "Detected 0 sections.")
        record = {"image": str(path), "width": w, "height": h, "model": mdl,
                  "sections": sections, "extraction": None}
        save_record(out, record)
        return *view(record, img), status_text(paths, idx, f"Detected {len(sections)} sections.")

    def read(paths, idx, annot_val, base, record, mode, prov, url, key, mdl, preset_name, mt,
             padding, out, progress=gr.Progress()):
        if not paths or not record:
            raise gr.Error("Load an image first.")
        sections = (merge_annot(annot_val["boxes"], base) if annot_val
                    else [dict(s) for s in record["sections"]])
        targets = [i for i, s in enumerate(sections)
                   if mode == "All sections" or not s["content"].strip()]
        if not targets:
            gr.Info("No sections to read.")
            return *skip_view, gr.skip()
        img = open_image(paths[idx])
        read_sections(img, sections, targets, PROFILES[preset_name],
                      make_cfg(prov, url, key, mdl, mt), padding, progress)
        record = {**record, "model": mdl, "sections": sections}
        save_record(out, record)
        return *view(record, img), status_text(paths, idx, f"Read {len(targets)} sections.")

    def convert(paths, idx, record, label, method, prov, url, key, mdl, preset_name, mt, out,
                progress=gr.Progress()):
        if not paths or not record:
            raise gr.Error("Load an image first.")
        if not record["sections"]:
            raise gr.Error("No sections yet — parse the page first.")
        img = open_image(paths[idx])
        progress(0.1, desc=f"Converting to {label}…")
        record = convert_record(record, img, label, method, preset_name,
                                make_cfg(prov, url, key, mdl, mt))
        save_record(out, record)
        return view(record, img)

    def save_extraction(record, label, text, out):
        if not record:
            raise gr.Error("Load an image first.")
        name = SCHEMA_CHOICES[label]
        if name == "auto":
            name = (record.get("extraction") or {}).get("schema")
        if not name or name == "document":
            raise gr.Error("Pick a Q&A, MCQ or QCM schema to save edits; the document JSON "
                           "is built from the sections, so edit those instead.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise gr.Error(f"Not valid JSON: {e}")
        prev = record.get("extraction") or {}
        record = {**record, "extraction": {
            "schema": name, "auto": prev.get("auto", False) and prev.get("schema") == name,
            "reasons": prev.get("reasons", []) if prev.get("schema") == name else [],
            "method": "edited", "data": data, "errors": schemas.validate(name, data),
            "notes": []}}
        full = save_record(out, record)
        return (*extraction_view(record), record,
                json.dumps(full, ensure_ascii=False, indent=2))

    def on_annot_change(annot_val, base, record, out):
        if not record or not annot_val:
            return gr.skip(), gr.skip(), gr.skip()
        sections = merge_annot(annot_val.get("boxes", []), base)
        if sections == record["sections"]:
            return gr.skip(), gr.skip(), gr.skip()
        record = {**record, "sections": sections}
        return sections_to_table(sections), json.dumps(save_record(out, record),
                                                       ensure_ascii=False, indent=2), record

    def apply_table(df, record, out):
        if not record:
            raise gr.Error("Load an image first.")
        record = {**record, "sections": table_to_sections(df, record["width"], record["height"])}
        save_record(out, record)
        return view(record)

    def clear(record, out):
        if not record:
            return skip_view
        record = {**record, "sections": [], "extraction": None}
        save_record(out, record)
        return view(record)

    def export(paths, out):
        if not paths:
            raise gr.Error("Load images first.")
        records = []
        for path in paths:
            p = struct_path(out, path)
            if p.exists():
                records.append(json.loads(p.read_text(encoding="utf-8")))
        if not records:
            raise gr.Error("No structured annotations yet — parse a page first.")
        files = {"structured.jsonl": records}
        # One file per output schema: {"image": ..., <schema data>} per page.
        for r in records:
            if ext := r.get("extraction"):
                files.setdefault(f"{ext['schema']}.jsonl", []).append(
                    {"image": r["image"], **schema_data(r)})
        dl_dir = Path(tempfile.mkdtemp(prefix="ppocr_struct_"))
        served = []
        for fname, rows in files.items():
            dest = Path(out) / fname
            dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                            encoding="utf-8")
            # Serve copies from temp so any output folder works with Gradio's file sandbox.
            served.append(shutil.copy(dest, dl_dir))
        gr.Info(f"Exported {len(records)} pages to {Path(out).resolve()}: "
                f"{', '.join(files)}")
        return served

    model_inputs = [*conn_inputs, preset, v.max_tokens]
    pipeline_inputs = [*model_inputs, v.detect_prompt, v.schema, v.method, pad, out_dir]
    v.parse_btn.click(parse, [paths_state, idx_state, *pipeline_inputs],
                      [*v.outputs, status], api_name="parse_page")
    v.parse_all_btn.click(parse_all, [paths_state, idx_state, v.skip_parsed, *pipeline_inputs],
                          [*v.outputs, status], api_name="parse_all_pages")
    v.detect_btn.click(detect, [paths_state, idx_state, *model_inputs, v.detect_prompt, out_dir],
                       [*v.outputs, status], api_name="detect_sections")
    v.read_btn.click(read, [paths_state, idx_state, v.annot, v.base, v.record, v.read_mode,
                            *model_inputs, pad, out_dir],
                     [*v.outputs, status], api_name="read_sections")
    v.convert_btn.click(convert, [paths_state, idx_state, v.record, v.schema, v.method,
                                  *model_inputs, out_dir],
                        v.outputs, api_name="convert_schema")
    v.save_extract_btn.click(save_extraction, [v.record, v.schema, v.extract_code, out_dir],
                             [v.extract_code, v.extract_status, v.record, v.json],
                             api_name="save_schema_output")
    v.schema.change(schema_text, [v.schema, v.record], v.schema_view)
    v.record.change(schema_text, [v.schema, v.record], v.schema_view)
    v.annot.change(on_annot_change, [v.annot, v.base, v.record, out_dir],
                   [v.table, v.json, v.record])
    v.apply_btn.click(apply_table, [v.table, v.record, out_dir], v.outputs)
    v.clear_btn.click(clear, [v.record, out_dir], v.outputs)
    v.export_btn.click(export, [paths_state, out_dir], v.export_files,
                       api_name="export_structured")
