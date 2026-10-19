"""Structured-document tab: detect the layout sections of a page with a vision
model, read each section, and save the page as structured JSON.

The model family is picked with a preset (see `PROFILES`):

- PaddleOCR-VL-1.5 has no layout task, so sections are built from the
  `Spotting:` text lines, merged into blocks. Draw or retype tables, formulas
  and figures by hand; "Read sections" then uses the matching task prompt
  (`Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`).
- Qwen3-VL (e.g. qwen3-vl-4b-thinking) is a general VLM: one prompt returns
  every section with its type, box (0-1000 page coordinates) and content.

Each image is autosaved to <out>/structured/<image>.json holding the flat
`sections` list (reading order) and the nested `document` built from it.
"""

import json
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pandas as pd
from gradio_image_annotation import image_annotator

from common import ask_model, crop_box, make_cfg, open_image, parse_loc_lines, status_text

SECTION_TYPES = ["title", "heading", "paragraph", "list", "table", "formula",
                 "figure", "caption", "header", "footer", "page_number"]
TYPE_COLORS = {
    "title": (220, 38, 38), "heading": (234, 88, 12), "paragraph": (37, 99, 235),
    "list": (8, 145, 178), "table": (22, 163, 74), "formula": (147, 51, 234),
    "figure": (219, 39, 119), "caption": (202, 138, 4), "header": (100, 116, 139),
    "footer": (100, 116, 139), "page_number": (100, 116, 139),
}
OTHER_COLOR = (64, 64, 64)
TYPE_ALIASES = {
    "text": "paragraph", "plain_text": "paragraph", "section_header": "heading",
    "section_title": "heading", "subtitle": "heading", "doc_title": "title",
    "list_item": "list", "equation": "formula", "isolate_formula": "formula",
    "image": "figure", "picture": "figure", "chart": "figure",
    "figure_caption": "caption", "table_caption": "caption",
    "page_header": "header", "page_footer": "footer",
}
# Heading-like types whose read text is collapsed onto one line.
INLINE_TYPES = {"title", "heading", "paragraph", "caption", "header", "footer", "page_number"}
COORDS = ("xmin", "ymin", "xmax", "ymax")
TABLE_HEADERS = ["order", "type", "level", "content", *COORDS]
LABEL_RE = re.compile(r"\s*(\d+)\s*:\s*(.*)")
LIST_RE = re.compile(r"^\s*([•●▪◦·\-–*]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")

QWEN_LAYOUT_PROMPT = """\
Parse the layout of this document page. Find every layout element in reading \
order and return them as a JSON array, with no other text.

Each element is an object:
{"type": "<type>", "level": <1-3, headings only>, "bbox_2d": [x1, y1, x2, y2], "content": "<text>"}

- type is one of: title, heading, paragraph, list, table, formula, figure, \
caption, header, footer, page_number.
- bbox_2d is the element's box in page coordinates from 0 to 1000.
- content is the exact text of the element. Write tables as HTML, formulas as \
LaTeX, and describe figures in one sentence."""

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
    },
    "Qwen3-VL-4B-Thinking": {
        "match": ["qwen3-vl-4b-thinking", "qwen3-vl-4b", "qwen3-vl"],
        "model": "qwen/qwen3-vl-4b-thinking",
        "detect": "layout",
        "detect_prompt": QWEN_LAYOUT_PROMPT,
        "read_prompts": {
            "table": "Convert this table to HTML. Return only the <table> element.",
            "formula": "Convert this formula to LaTeX. Return only the LaTeX.",
            "figure": "Describe this figure in one sentence. If it is a chart, "
                      "also list its data.",
            "_default": "Read all text in this image exactly as written. "
                        "Return only the text.",
        },
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
    from its line height (headings) and leading bullets (lists)."""
    if not lines:
        return []
    med = statistics.median(y1 - y0 for _, _, y0, _, y1 in lines) or 1
    blocks = []
    for text, x0, y0, x1, y1 in sorted(lines, key=lambda l: (l[2], l[1])):
        lh = y1 - y0
        big = lh > 1.3 * med
        bullet = bool(LIST_RE.match(text))
        for b in reversed(blocks):
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
                           "big": big, "list": bullet, "lh": lh})

    tallest = max(b["lh"] for b in blocks)
    sections, has_title = [], False
    for b in reading_order(blocks, width):
        rect = clamp_rect(b["xmin"], b["ymin"], b["xmax"], b["ymax"], width, height)
        if b["big"] and not has_title and b["lh"] == tallest and b["ymin"] < height / 3:
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


def extract_json(text: str):
    """The first JSON value in `text`; if the output was cut off, every complete
    object found in it."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    dec = json.JSONDecoder()
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


def parse_layout(text: str, width: int, height: int) -> list[dict]:
    """Parse a Qwen-VL style JSON list of {type, bbox_2d, content, level}."""
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
    # Qwen3-VL answers in 0-1000 page coordinates; Qwen2.5-VL in pixels.
    pixels = max((v for b in boxes if b for v in b), default=0) > 1000
    sx, sy = (1, 1) if pixels else (width / 1000, height / 1000)

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
        try:
            level = int(d.get("level") or 1)
        except (TypeError, ValueError):
            level = 1
        sections.append(new_section(type_, rect, str(content or "").strip(), level))
    return sections


def build_document(sections: list[dict]) -> dict:
    """Nest the flat, reading-ordered sections under their headings."""
    doc = {"title": None, "page_header": [], "page_footer": [], "blocks": [], "sections": []}
    stack = [(0, doc)]
    for s in sections:
        bbox = [s[k] for k in COORDS]
        elem = {"type": s["type"], "bbox": bbox, "content": s["content"]}
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
        return {k: data.get(k) for k in ("image", "width", "height", "model")} | {
            "sections": data.get("sections", [])}
    return {"image": str(image_path), "width": width, "height": height,
            "model": None, "sections": []}


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
# UI
# --------------------------------------------------------------------------- #
def annot_value(image_path, sections, img=None):
    if not image_path:
        return None
    boxes = [{"label": f"{i}:{s['type']}", "color": TYPE_COLORS.get(s["type"], OTHER_COLOR),
              **{k: s[k] for k in COORDS}} for i, s in enumerate(sections)]
    return {"image": img or open_image(image_path), "boxes": boxes}


def view(record, img=None):
    """Values for StructureView.outputs: (annot, base, table, json, record)."""
    if not record:
        return None, [], sections_to_table([]), "", None
    sections = record["sections"]
    return (annot_value(record["image"], sections, img), sections,
            sections_to_table(sections), dump(record), record)


def show(image_path: str | None, out_dir: str):
    if not image_path:
        return view(None)
    img = open_image(image_path)
    return view(load_record(out_dir, image_path, *img.size), img)


@dataclass
class StructureView:
    annot: gr.components.Component
    base: gr.State
    table: gr.Dataframe
    json: gr.Code
    record: gr.State
    detect_btn: gr.Button
    read_mode: gr.Radio
    read_btn: gr.Button
    clear_btn: gr.Button
    apply_btn: gr.Button
    export_btn: gr.Button
    export_files: gr.File
    detect_prompt: gr.Textbox
    max_tokens: gr.Number

    @property
    def outputs(self):
        return [self.annot, self.base, self.table, self.json, self.record]


def build_tab(default_preset: str) -> StructureView:
    """Lay out the tab's components; call inside a `gr.Tab`."""
    base = gr.State([])
    record = gr.State(None)
    with gr.Row():
        detect_btn = gr.Button("🧭 Detect sections", variant="primary")
        read_mode = gr.Radio(["Empty sections", "All sections"], value="Empty sections",
                             show_label=False, container=False)
        read_btn = gr.Button("📖 Read sections")
        clear_btn = gr.Button("🗑 Clear sections", variant="stop")
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
            json_view = gr.Code(language="json", label="Structured JSON (autosaved)",
                                interactive=False, lines=30, max_lines=30,
                                buttons=["copy", "download"])
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
        detect_prompt = gr.Textbox(PROFILES[default_preset]["detect_prompt"], lines=4,
                                   label="Detect prompt (set by the model preset)")
        max_tokens = gr.Number(16384, precision=0,
                               label="Max tokens (thinking models need room to think)")
    return StructureView(annot, base, table, json_view, record, detect_btn, read_mode,
                         read_btn, clear_btn, apply_btn, export_btn, export_files,
                         detect_prompt, max_tokens)


def wire(v: StructureView, *, paths_state, idx_state, conn_inputs, preset, pad, out_dir,
         status):
    """Attach the tab's event handlers. `conn_inputs` = [provider, base_url, api_key, model]."""

    def detect(paths, idx, prov, url, key, mdl, preset_name, mt, prompt, out,
               progress=gr.Progress()):
        if not paths:
            raise gr.Error("Load an image first.")
        path = paths[idx]
        img = open_image(path)
        w, h = img.size
        progress(0.1, desc=f"Detecting sections with {mdl}…")
        raw = ask_model(img, prompt, make_cfg(prov, url, key, mdl, mt))
        if PROFILES[preset_name]["detect"] == "spotting":
            sections = group_lines(parse_loc_lines(raw, w, h), w, h)
        else:
            sections = parse_layout(raw, w, h)
        if not sections:
            hint = ("The model returned nothing — it may have used up Max tokens while "
                    "thinking." if not raw else f"Model output starts: {raw[:200]!r}")
            gr.Warning(f"No sections found for preset {preset_name}. {hint}")
            return *[gr.skip()] * 5, status_text(paths, idx, "Detected 0 sections.")
        record = {"image": str(path), "width": w, "height": h, "model": mdl,
                  "sections": sections}
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
            return *[gr.skip()] * 5, gr.skip()
        img = open_image(paths[idx])
        cfg = make_cfg(prov, url, key, mdl, mt)
        prompts = PROFILES[preset_name]["read_prompts"]
        for i in progress.tqdm(targets, desc="Reading sections"):
            s = sections[i]
            text = ask_model(crop_box(img, s, int(padding)),
                             prompts.get(s["type"], prompts["_default"]), cfg)
            s["content"] = " ".join(text.split()) if s["type"] in INLINE_TYPES else text
        record = {**record, "model": mdl, "sections": sections}
        save_record(out, record)
        return *view(record, img), status_text(paths, idx, f"Read {len(targets)} sections.")

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
            return [gr.skip()] * 5
        record = {**record, "sections": []}
        save_record(out, record)
        return view(record)

    def export(paths, out):
        if not paths:
            raise gr.Error("Load images first.")
        lines = []
        for path in paths:
            p = struct_path(out, path)
            if p.exists():
                lines.append(json.dumps(json.loads(p.read_text(encoding="utf-8")),
                                        ensure_ascii=False))
        if not lines:
            raise gr.Error("No structured annotations yet — detect sections first.")
        jsonl = Path(out) / "structured.jsonl"
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        gr.Info(f"Exported {len(lines)} documents to {jsonl.resolve()}")
        # Serve a copy from temp so any output folder works with Gradio's file sandbox.
        return [shutil.copy(jsonl, tempfile.mkdtemp(prefix="ppocr_struct_"))]

    model_inputs = [*conn_inputs, preset, v.max_tokens]
    v.detect_btn.click(detect, [paths_state, idx_state, *model_inputs, v.detect_prompt, out_dir],
                       [*v.outputs, status], api_name="detect_sections")
    v.read_btn.click(read, [paths_state, idx_state, v.annot, v.base, v.record, v.read_mode,
                            *model_inputs, pad, out_dir],
                     [*v.outputs, status], api_name="read_sections")
    v.annot.change(on_annot_change, [v.annot, v.base, v.record, out_dir],
                   [v.table, v.json, v.record])
    v.apply_btn.click(apply_table, [v.table, v.record, out_dir], v.outputs)
    v.clear_btn.click(clear, [v.record, out_dir], v.outputs)
    v.export_btn.click(export, [paths_state, out_dir], v.export_files,
                       api_name="export_structured")
