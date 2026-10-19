# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gradio app for annotating document images with a vision-language model (VLM) served by LM Studio, Ollama or any OpenAI-compatible server (vLLM, ...). Two tabs share one image list and one model panel:

- **Text boxes** (`app.py`): PaddleOCR-VL `Spotting:`/`OCR:` boxes, exported in PPOCRLabel format.
- **Structured document** (`structure.py`): parse a page into sections, choose an output schema automatically (document / Q&A / MCQ / QCM), and export JSONL.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python app.py                                   # http://localhost:7860 (next free port if busy)
python app.py --provider "OpenAI-compatible" --base-url http://localhost:8000/v1 --model rednote-hilab/dots.ocr --out output --port 7861
python -m py_compile *.py                       # quick syntax check
```

There is no test suite and no linter config. To check changes without a real model:
- Pure functions (`schemas.extract/classify/validate`, `structure.parse_layout/group_lines`, `formulas.*`) can be called directly from `.venv/bin/python`.
- For end-to-end checks, run a small fake OpenAI-compatible HTTP server that returns canned output for `/v1/models` and `/v1/chat/completions`, start `app.py` against it, and drive it with `gradio_client.Client`. The named endpoints include `/load_folder`, `/parse_page`, `/parse_all_pages`, `/detect_sections`, `/read_sections`, `/convert_schema`, `/save_schema_output`, `/export_structured`, `/switch_preset`.
- `gradio_client` leaves `gr.State` outputs out of its results, and `image_annotator` inputs need `handle_file(...)` for the image and `"rgb(r, g, b)"` strings for box colours.

## Architecture

**Module roles**
- `providers.py`: model backends behind one interface, `list_models()` and `chat(img, prompt, model, max_tokens) -> str`. Every answer goes through `strip_thinking()`, which removes `<think>…</think>`.
- `common.py`: shared helpers: `ask_model` (turns errors into `gr.Error`), `parse_loc_lines` (PaddleOCR `<|LOC_n|>` 0–1000 quads → pixel rectangles), `crop_box`, `open_image` (applies EXIF rotation; always use it so box coordinates match), `status_text`.
- `app.py`: builds the UI, shared sidebar and Text boxes tab; calls `structure.build_tab()` / `structure.wire()`. `show()` returns the outputs for **both** tabs, so navigation outputs = `[annot, table, status, selector, *sv.outputs]`.
- `structure.py`: model presets (`PROFILES`), layout parsing, the parse pipeline, persistence and the tab's UI and event handlers.
- `schemas.py`: JSON Schemas (`SCHEMAS`, `SCHEMA_CHOICES` including `auto`), `validate`, rule-based `extract`, `classify` (picks a schema from page content), `label_sections`, and the model prompt builders.
- `formulas.py`: finds formulas wrapped in `\( \)`, `\[ \]`, `\{ \}`, `$ $` or `$$ $$` (first delimiter wins; `$` must hug its content).
- `prompts.py` + `prompt.md`: prompts for general VLMs live in `prompt.md`. They are loaded at import (**restart the app after editing**), each from the first ```` ```text ```` block under a ``## `name` `` heading, with `{{placeholders}}` filled by `render()`. Required names are listed in `prompts.REQUIRED`.

**Presets (`structure.PROFILES`)** decide how the model is used:
- `detect: "spotting"` (PaddleOCR-VL): lines are merged into blocks by `group_lines`, typed by line height and markers, and ordered by `reading_order`.
- `detect: "layout"`: the model returns a JSON list of elements, handled by `parse_layout`. Boxes are either `bbox: "norm1000"` (Qwen3-VL / Generic; values over 1000 are taken as pixels) or `"pixels"` of the image actually sent (dots.ocr, which is pre-resized with `smart_resize`/`page_for_model`).
- `can_fill`: the model may classify and fill schemas; otherwise rules are used, with a note.
- Qwen3-VL and Generic VLM share `GENERAL_VLM` (the `prompt.md` prompts). PaddleOCR-VL and dots.ocr keep their fixed task prompts; don't rewrite those.
- `match` name patterns let `profile_for` / `pick_model` link a preset and a server model in both directions; Generic has none and keeps the selected model.

**Parse pipeline** (`parse_page`): `detect_sections` → `read_sections` (empty sections only, per-type read prompt, `clean_read`) → `convert_record`: `mark_formulas` → `pick_schema` (rules `schemas.classify`, or the model) → `schemas.label_sections` → `schemas.extract` or model fill → `schemas.validate`. Every step must accept output from any preset, so new logic should work on plain section dicts.

**Data model**
- A section is `{type, level, content, xmin, ymin, xmax, ymax}`, stored in reading order. `level` is only set for headings.
- A record is `{image, width, height, model, sections, extraction}` and autosaves to `<out>/structured/<image>.json`. The nested `document` is rebuilt from `sections` on every save (`build_document`) and never edited directly. An `extraction` with `data: None` means the page's schema is `document`.
- Text-box annotations are separate: `<out>/annotations/<image>.json`.

**Section type conventions** (used by the prompts, the rules and the parser alike; keep them in sync with `prompt.md`'s "What the parser expects")
- Section types are listed in `SECTION_TYPES`, and other models' labels are mapped through `TYPE_ALIASES`.
- Question parts are their own types: `question`, `choice`, `answer`, `explanation`.
- A ticked option starts with `☑`; the rules read it as marked correct.
- A `formula` section is bare LaTeX. Text that mixes words and math keeps its type, and its document block gets `segments`.
- `extract_json` fixes unescaped LaTeX backslashes and recovers the complete objects from cut-off arrays; `fix_latex` repairs control characters produced by `\f`, `\t`, `\n`.

**Annotator sync (gotcha)**
- `gradio_image_annotator` boxes may only carry `label`, `color` and coordinates.
- Structure-tab boxes are therefore labelled `"<index>:<type>"`, and `merge_annot` maps edits back to sections using the `base` state, which is the sections as they were when last rendered.
- Every handler that re-renders the annotator must also set `base`, which is why handlers return the full `view()` tuple (`StructureView.outputs`).
- Don't update the annotator from its own `.change` handler.

**Adding things**
- New backend: subclass `Provider` and register it in `PROVIDERS`.
- New preset: add a `PROFILES` entry.
- New schema: add it to `SCHEMAS`/`SCHEMA_CHOICES`, add rules in `extract()`, and add a `guide-<name>` block in `prompt.md` plus its name in `prompts.REQUIRED`.
- `README.md` documents user-facing behaviour; update it alongside feature changes.
