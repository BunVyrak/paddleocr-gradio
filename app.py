"""Document-annotation tool backed by PaddleOCR-VL-1.5 or Qwen3-VL (or another
vision model) served by LM Studio, Ollama or any OpenAI-compatible server — see
providers.py. Two tabs share the image list and model settings:

Text boxes
- "Auto-detect" runs the model's `Spotting:` task on the whole image and turns
  every returned text line into a labelled rectangle.
- Draw / move / resize boxes by hand, then "OCR boxes" crops each box and runs
  the `OCR:` task to fill in its label.
- Annotations are autosaved per image and can be exported in PPOCRLabel format
  (Label.txt for detection, crops + rec_gt.txt for recognition).

Structured document (structure.py)
- Detect the page's sections, read each one, and save the page as structured
  JSON (exported as structured.jsonl).
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
from gradio_image_annotation import image_annotator

import structure
from common import (ask_model, crop_box, make_cfg, open_image, parse_loc_lines,
                    status_text)
from providers import PROVIDERS, get_provider
from structure import PROFILES, pick_model, profile_for

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BOX_COLOR = (255, 64, 64)
TABLE_HEADERS = ["#", "label", "xmin", "ymin", "xmax", "ymax"]


# --------------------------------------------------------------------------- #
# Model output
# --------------------------------------------------------------------------- #
def parse_spotting(text: str, width: int, height: int) -> list[dict]:
    return [make_box(label, *rect, width, height)
            for label, *rect in parse_loc_lines(text, width, height)]


def make_box(label, xmin, ymin, xmax, ymax, width, height) -> dict:
    clamp = lambda v, hi: int(round(max(0, min(v, hi))))
    return {
        "label": label,
        "xmin": clamp(xmin, width), "ymin": clamp(ymin, height),
        "xmax": clamp(xmax, width), "ymax": clamp(ymax, height),
        "color": BOX_COLOR,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def ann_path(out_dir: str, image_path: str) -> Path:
    return Path(out_dir) / "annotations" / (Path(image_path).name + ".json")


def load_boxes(out_dir: str, image_path: str) -> list[dict]:
    p = ann_path(out_dir, image_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [{**b, "color": BOX_COLOR} for b in data.get("boxes", [])]


def save_boxes(out_dir: str, image_path: str, boxes: list[dict]) -> None:
    p = ann_path(out_dir, image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: b[k] for k in ("label", "xmin", "ymin", "xmax", "ymax")} for b in boxes]
    p.write_text(json.dumps({"image": str(image_path), "boxes": clean},
                            ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def boxes_to_table(boxes: list[dict]) -> pd.DataFrame:
    rows = [[i, b.get("label", ""), b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
            for i, b in enumerate(boxes)]
    return pd.DataFrame(rows, columns=TABLE_HEADERS)


def annot_value(image_path, boxes):
    if not image_path:
        return None
    # Send the decoded image (not the path) so folders outside cwd/temp work
    # and EXIF rotation matches the coordinates used for cropping.
    return {"image": open_image(image_path), "boxes": boxes}


def current_boxes(annot) -> list[dict]:
    if not annot:
        return []
    return [{**b, "color": BOX_COLOR} for b in annot.get("boxes", [])]


def show(paths, idx, out_dir, extra=""):
    """Return outputs for (annotator, table, status, selector, *structure tab)."""
    if not paths:
        return (None, boxes_to_table([]), status_text(paths, idx),
                gr.update(choices=[], value=None), *structure.show(None, out_dir))
    path = paths[idx]
    boxes = load_boxes(out_dir, path)
    return (
        annot_value(path, boxes),
        boxes_to_table(boxes),
        status_text(paths, idx, extra),
        gr.update(choices=[Path(p).name for p in paths], value=Path(path).name),
        *structure.show(path, out_dir),
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def build_app(default_provider: str, default_url: str, default_model: str,
              default_out: str) -> gr.Blocks:
    default_preset = profile_for(default_model) or next(iter(PROFILES))
    with gr.Blocks(title="PaddleOCR-VL Annotator") as demo:
        paths_state = gr.State([])
        idx_state = gr.State(0)

        gr.Markdown("## PaddleOCR-VL Document Annotator")

        with gr.Row():
            # ---------------- Sidebar ----------------
            with gr.Column(scale=1, min_width=280):
                with gr.Group():
                    folder = gr.Textbox(label="Image folder", placeholder="/path/to/images")
                    load_folder_btn = gr.Button("Load folder")
                uploads = gr.File(label="…or upload images", file_count="multiple",
                                  file_types=["image"])
                selector = gr.Dropdown(label="Jump to image", choices=[])
                with gr.Row():
                    prev_btn = gr.Button("◀ Prev")
                    next_btn = gr.Button("Next ▶")
                status = gr.Markdown("No images loaded.")

                with gr.Accordion("Model", open=True):
                    preset = gr.Radio(list(PROFILES), value=default_preset,
                                      label="Model preset")
                    provider = gr.Dropdown(list(PROVIDERS), value=default_provider,
                                           label="Provider")
                    base_url = gr.Textbox(
                        default_url or PROVIDERS[default_provider].default_url,
                        label="Base URL")
                    api_key = gr.Textbox("", label="API key (optional)", type="password")
                    with gr.Row(equal_height=True):
                        model = gr.Dropdown([default_model] if default_model else [],
                                            value=default_model or None,
                                            label="Model", allow_custom_value=True, scale=4)
                        refresh_btn = gr.Button("🔄", scale=1, min_width=40)

                with gr.Accordion("Settings", open=False):
                    spot_prompt = gr.Textbox("Spotting:", label="Detect prompt (must return "
                                             "<|LOC_n|> boxes, PaddleOCR-VL format)")
                    ocr_prompt = gr.Textbox(PROFILES[default_preset]["read_prompts"]["_default"],
                                            label="OCR prompt (per box)")
                    max_tokens = gr.Number(4096, label="Max tokens (text boxes)", precision=0)
                    pad = gr.Number(4, label="Crop padding (px)", precision=0)
                    out_dir = gr.Textbox(default_out, label="Output folder")

            # ---------------- Main ----------------
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Text boxes"):
                        with gr.Row():
                            detect_btn = gr.Button("🔍 Auto-detect (Spotting)", variant="primary")
                            ocr_mode = gr.Radio(["Unlabelled boxes", "All boxes"],
                                                value="Unlabelled boxes", show_label=False,
                                                container=False)
                            ocr_btn = gr.Button("🔤 OCR boxes")
                            save_btn = gr.Button("💾 Save")
                            clear_btn = gr.Button("🗑 Clear boxes", variant="stop")

                        annot = image_annotator(
                            label="Draw boxes (drag on image). Double-click a box to edit its label; "
                                  "select + Delete to remove.",
                            image_type="filepath",
                            sources=[],
                            boxes_alpha=0.15,
                            box_min_size=4,
                            handle_size=6,
                            show_clear_button=False,
                            show_remove_button=False,
                            height=720,
                        )

                        with gr.Row():
                            table = gr.Dataframe(
                                headers=TABLE_HEADERS,
                                datatype=["number", "str", "number", "number", "number", "number"],
                                interactive=True,
                                label="Boxes — edit labels/coords here, then click Apply",
                                wrap=True,
                                scale=4,
                            )
                            with gr.Column(scale=1, min_width=160):
                                apply_btn = gr.Button("✅ Apply table edits")
                                export_btn = gr.Button("📦 Export dataset", variant="primary")
                                export_files = gr.File(label="Export", file_count="multiple")

                    with gr.Tab("Structured document"):
                        sv = structure.build_tab(default_preset)

        cfg_inputs = [provider, base_url, api_key, model, max_tokens]

        # ---------------- Provider / model selection ----------------
        def list_models(prov, url, key):
            try:
                models = get_provider(prov, url, key).list_models()
            except Exception as e:
                gr.Warning(f"Could not list models from {prov} at {url}: {e}")
                return None
            if not models:
                gr.Warning(f"{prov} returned no models.")
            return models

        def fetch_models(prov, url, key, current, preset_name):
            models = list_models(prov, url, key)
            if models is None:
                return gr.update()
            # Keep the current choice if the server still has it; otherwise
            # prefer the preset's model.
            value = current if current in models else (
                pick_model(models, preset_name) or (models[0] if models else None))
            return gr.update(choices=models, value=value)

        def switch_preset(preset_name, prov, url, key):
            models = list_models(prov, url, key) or []
            value = pick_model(models, preset_name)
            if not value:
                value = PROFILES[preset_name]["model"]
                gr.Warning(f"{prov} has no {preset_name} model loaded; using {value!r}. "
                           "Load it on the server, then click 🔄.")
            return gr.update(choices=sorted({*models, value}), value=value)

        def on_model_change(mdl):
            # Follow a model picked by hand: switch preset and its prompts.
            name = profile_for(mdl)
            if not name:
                return gr.skip(), gr.skip(), gr.skip()
            p = PROFILES[name]
            return name, p["detect_prompt"], p["read_prompts"]["_default"]

        def on_provider_change(prov, key, preset_name):
            url = PROVIDERS[prov].default_url
            return url, fetch_models(prov, url, key, None, preset_name)

        provider.input(on_provider_change, [provider, api_key, preset], [base_url, model])
        refresh_btn.click(fetch_models, [provider, base_url, api_key, model, preset], model)
        preset.input(switch_preset, [preset, provider, base_url, api_key], model)
        model.change(on_model_change, model, [preset, sv.detect_prompt, ocr_prompt])
        demo.load(fetch_models, [provider, base_url, api_key, model, preset], model)
        view_outputs = [annot, table, status, selector, *sv.outputs]

        # ---------------- Loading / navigation ----------------
        def load_folder(folder_path, out):
            p = Path(folder_path or "").expanduser()
            if not p.is_dir():
                raise gr.Error(f"Not a folder: {folder_path}")
            paths = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if not paths:
                raise gr.Error("No images found in that folder.")
            return paths, 0, *show(paths, 0, out)

        def load_uploads(files, out):
            if not files:
                return gr.skip()
            paths = sorted(str(f) for f in files)
            return paths, 0, *show(paths, 0, out)

        def navigate(delta, paths, idx, annot_val, out):
            if not paths:
                return idx, *show(paths, idx, out)
            if annot_val:
                save_boxes(out, paths[idx], current_boxes(annot_val))
            new_idx = (idx + delta) % len(paths)
            return new_idx, *show(paths, new_idx, out)

        def jump(name, paths, idx, annot_val, out):
            names = [Path(p).name for p in paths]
            if not name or name not in names or names.index(name) == idx:
                return [gr.skip()] * (1 + len(view_outputs))
            return navigate(names.index(name) - idx, paths, idx, annot_val, out)

        load_folder_btn.click(load_folder, [folder, out_dir],
                              [paths_state, idx_state, *view_outputs], api_name="load_folder")
        uploads.upload(load_uploads, [uploads, out_dir],
                       [paths_state, idx_state, *view_outputs])
        prev_btn.click(lambda *a: navigate(-1, *a), [paths_state, idx_state, annot, out_dir],
                       [idx_state, *view_outputs])
        next_btn.click(lambda *a: navigate(1, *a), [paths_state, idx_state, annot, out_dir],
                       [idx_state, *view_outputs])
        selector.input(jump, [selector, paths_state, idx_state, annot, out_dir],
                       [idx_state, *view_outputs])

        # ---------------- Model actions ----------------
        def auto_detect(paths, idx, prov, url, key, mdl, mt, prompt, out,
                        progress=gr.Progress()):
            if not paths:
                raise gr.Error("Load an image first.")
            path = paths[idx]
            img = open_image(path)
            progress(0.1, desc=f"Running detection on {mdl}…")
            raw = ask_model(img, prompt, make_cfg(prov, url, key, mdl, mt))
            boxes = parse_spotting(raw, *img.size)
            if not boxes:
                gr.Warning("No boxes found in the model output. Auto-detect needs a model "
                           "that returns <|LOC_n|> coordinates (PaddleOCR-VL-1.5 'Spotting:'). "
                           "You can still draw boxes and use OCR boxes.")
                return gr.skip(), gr.skip(), status_text(paths, idx, "Detected 0 text regions.")
            save_boxes(out, path, boxes)
            return (annot_value(path, boxes), boxes_to_table(boxes),
                    status_text(paths, idx, f"Detected {len(boxes)} text regions."))

        def ocr_boxes(paths, idx, annot_val, mode, prov, url, key, mdl, mt, prompt, padding, out,
                      progress=gr.Progress()):
            if not paths or not annot_val:
                raise gr.Error("Load an image first.")
            path = paths[idx]
            boxes = current_boxes(annot_val)
            targets = [i for i, b in enumerate(boxes)
                       if mode == "All boxes" or not b.get("label", "").strip()]
            if not targets:
                gr.Info("No boxes to OCR.")
                return gr.skip(), gr.skip(), gr.skip()
            img = open_image(path)
            cfg = make_cfg(prov, url, key, mdl, mt)
            for i in progress.tqdm(targets, desc="OCR"):
                text = ask_model(crop_box(img, boxes[i], int(padding)), prompt, cfg)
                boxes[i]["label"] = " ".join(text.split())
            save_boxes(out, path, boxes)
            return (annot_value(path, boxes), boxes_to_table(boxes),
                    status_text(paths, idx, f"OCR'd {len(targets)} boxes."))

        detect_btn.click(auto_detect,
                         [paths_state, idx_state, *cfg_inputs, spot_prompt, out_dir],
                         [annot, table, status], api_name="auto_detect")
        ocr_btn.click(ocr_boxes,
                      [paths_state, idx_state, annot, ocr_mode, *cfg_inputs, ocr_prompt, pad,
                       out_dir],
                      [annot, table, status], api_name="ocr_boxes")

        # ---------------- Editing ----------------
        def sync_table(annot_val):
            return boxes_to_table(current_boxes(annot_val))

        def apply_table(df, paths, idx, out):
            if not paths:
                raise gr.Error("Load an image first.")
            path = paths[idx]
            w, h = open_image(path).size
            boxes = []
            for _, r in df.iterrows():
                try:
                    coords = [float(r[c]) for c in ("xmin", "ymin", "xmax", "ymax")]
                except (TypeError, ValueError):
                    continue
                label = "" if pd.isna(r["label"]) else str(r["label"])
                x0, y0, x1, y1 = coords
                boxes.append(make_box(label, min(x0, x1), min(y0, y1),
                                      max(x0, x1), max(y0, y1), w, h))
            save_boxes(out, path, boxes)
            return annot_value(path, boxes), boxes_to_table(boxes)

        def save_current(paths, idx, annot_val, out):
            if not paths:
                raise gr.Error("Load an image first.")
            save_boxes(out, paths[idx], current_boxes(annot_val))
            gr.Info(f"Saved {ann_path(out, paths[idx])}")

        def clear_current(paths, idx, out):
            if not paths:
                return gr.skip(), gr.skip()
            save_boxes(out, paths[idx], [])
            return annot_value(paths[idx], []), boxes_to_table([])

        annot.change(sync_table, annot, table)
        apply_btn.click(apply_table, [table, paths_state, idx_state, out_dir], [annot, table])
        save_btn.click(save_current, [paths_state, idx_state, annot, out_dir], None)
        clear_btn.click(clear_current, [paths_state, idx_state, out_dir], [annot, table])

        structure.wire(sv, paths_state=paths_state, idx_state=idx_state,
                       conn_inputs=[provider, base_url, api_key, model], preset=preset,
                       pad=pad, out_dir=out_dir, status=status)

        # ---------------- Export ----------------
        def export(paths, idx, annot_val, out, padding):
            if not paths:
                raise gr.Error("Load images first.")
            if annot_val:
                save_boxes(out, paths[idx], current_boxes(annot_val))
            out_p = Path(out)
            crops_dir = out_p / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            det_lines, rec_lines, n_boxes = [], [], 0
            for path in paths:
                boxes = load_boxes(out, path)
                if not boxes:
                    continue
                img = open_image(path)
                items = []
                for j, b in enumerate(boxes):
                    x0, y0, x1, y1 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
                    items.append({"transcription": b["label"],
                                  "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                  "difficult": False})
                    if b["label"].strip() and x1 > x0 and y1 > y0:
                        name = f"{Path(path).stem}_crop_{j}.png"
                        img.crop((x0, y0, x1, y1)).save(crops_dir / name)
                        rec_lines.append(f"crops/{name}\t{b['label']}")
                    n_boxes += 1
                det_lines.append(f"{path}\t{json.dumps(items, ensure_ascii=False)}")
            label_txt = out_p / "Label.txt"
            rec_txt = out_p / "rec_gt.txt"
            label_txt.write_text("\n".join(det_lines) + "\n", encoding="utf-8")
            rec_txt.write_text("\n".join(rec_lines) + "\n", encoding="utf-8")
            gr.Info(f"Exported {n_boxes} boxes from {len(det_lines)} images to {out_p.resolve()}")
            # Serve copies from temp so any output folder works with Gradio's file sandbox.
            dl_dir = Path(tempfile.mkdtemp(prefix="ppocr_export_"))
            return [shutil.copy(label_txt, dl_dir), shutil.copy(rec_txt, dl_dir)]

        export_btn.click(export, [paths_state, idx_state, annot, out_dir, pad], export_files, api_name="export")

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="LM Studio", choices=list(PROVIDERS))
    ap.add_argument("--base-url", default="", help="default: the provider's standard URL")
    ap.add_argument("--model", default="paddleocr-vl-1.5")
    ap.add_argument("--out", default="output")
    ap.add_argument("--port", type=int, default=None, help="default: first free port from 7860")
    args = ap.parse_args()
    build_app(args.provider, args.base_url, args.model, args.out).launch(
        server_port=args.port)
