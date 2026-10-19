# paddlocr-gradio

A Gradio web app for building document datasets. It has two tabs that share
the image list and model settings:

- **Text boxes:** draw boxes around text and
  [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) reads the
  text inside each box (PPOCRLabel export).
- **Structured document:** the model finds the page's sections (title,
  headings, paragraphs, lists, tables, formulas, figures, ...), reads each one,
  and the page is saved as structured JSON.

Switch between **PaddleOCR-VL-1.5** and **Qwen3-VL-4B-Thinking** with the
**Model preset** option. The model runs locally in LM Studio or Ollama, or on
any server with an OpenAI-compatible API.

## Features

### Text boxes

- **Auto-detect:** runs the model's `Spotting:` task on the whole image. Every
  line of text it finds becomes a labelled box.
- **Draw and edit boxes:** drag on the image to add a box, drag a box or its
  handles to move or resize it, double-click it to change its label, and select
  it and press Delete to remove it.
- **OCR boxes:** cuts out each box and runs the model's `OCR:` task to fill in
  its label, either for boxes with no label yet or for all of them.
- **Box table:** lists every box with its label and coordinates. Fix long text
  here, then click **Apply table edits**.
- **Folders of images:** load a folder or upload files and move between them
  with Prev/Next. Your work on each image is saved automatically and picked up
  again the next time you load that folder.
- **PaddleOCR export:** creates training files in PPOCRLabel format for
  text detection and text recognition.

### Structured document

- **Detect sections:** finds every section on the page, in reading order.
  - *PaddleOCR-VL-1.5* has no layout task, so the app runs `Spotting:` and
    merges the text lines into blocks. Taller lines become the title and
    headings, bulleted lines become lists, and two-column pages are read left
    column first. Tables, formulas and figures aren't found this way: draw
    them and name them by type.
  - *Qwen3-VL* gets one prompt and returns every section with its type, box
    and content as JSON.
- **Read sections:** crops each section and reads it with a prompt for its
  type: tables as HTML, formulas as LaTeX, figures as a description, the rest
  as text. For PaddleOCR-VL these are its own tasks (`Table Recognition:`,
  `Formula Recognition:`, `Chart Recognition:`, `OCR:`).
- **Edit sections:** boxes are labelled `<order>:<type>` and coloured by type.
  Move, resize, delete or draw boxes, and double-click one to change its type.
  Fix type, heading level, content and reading order in the table, then click
  **Apply table edits**.
- **Structured JSON:** headings nest the sections under them. The JSON is
  shown next to the page and saved automatically after every change.
- **JSONL export:** one line per page in `structured.jsonl`.

### Both tabs

- **Model preset:** switches between PaddleOCR-VL-1.5 and
  Qwen3-VL-4B-Thinking. It picks the matching model on the server and sets the
  prompts. Picking a model by hand also switches the preset.
- **Choice of provider:** LM Studio, Ollama or any OpenAI-compatible server
  (vLLM, llama.cpp server, ...), selected in the app.
- **Thinking models:** the `<think>...</think>` reasoning is removed from the
  answer.

## Requirements

- Python 3.10 or newer
- One of these, running PaddleOCR-VL-1.5, Qwen3-VL-4B-Thinking or another
  vision model:
  - [LM Studio](https://lmstudio.ai)
  - [Ollama](https://ollama.com)
  - any OpenAI-compatible server

## Setup

```bash
git clone https://github.com/BunVyrak/paddlocr-gradio.git
cd paddlocr-gradio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start the model server

**LM Studio**

Download `PaddleOCR-VL-1.5` in LM Studio, then start the server and load the
model. You can do this in the LM Studio app (Developer tab), or from the
terminal:

```bash
lms server start
lms load paddleocr-vl-1.5
lms load qwen/qwen3-vl-4b-thinking   # optional, for the Qwen3-VL preset
```

The server runs at `http://localhost:1234/v1`.

**Ollama**

```bash
ollama serve
ollama pull <paddleocr-vl-model>
```

The server runs at `http://localhost:11434`.

## Run the app

```bash
python app.py
```

Then open the URL it prints (normally <http://localhost:7860>; if that port is
busy, the app uses the next free one).

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--provider` | `LM Studio` | `LM Studio`, `Ollama` or `OpenAI-compatible` |
| `--base-url` | the provider's standard URL | Address of the model server |
| `--model` | `paddleocr-vl-1.5` | Model name. If the server doesn't have it, the app picks one. A Qwen3-VL name starts the app on the Qwen3-VL preset. |
| `--out` | `output` | Folder for annotations and exports |
| `--port` | first free port from 7860 | Port for the web app |

Example:

```bash
python app.py --provider Ollama --out my_dataset
```

The provider, server address, API key and model can also be changed in the
app's **Model** panel. 🔄 reloads the list of models from the server.

## How to annotate

### Text boxes

1. Type a folder path and click **Load folder**, or upload images.
2. Click **🔍 Auto-detect** to find and read all text on the page.
3. Fix the boxes: draw any that are missing, resize or delete wrong ones.
4. Click **🔤 OCR boxes** to read the text in new or changed boxes.
5. Correct labels in the table if needed, then click **✅ Apply table edits**.
6. Go to the next image with **Next ▶**. Your work is saved automatically.
7. When you're done, click **📦 Export dataset**.

### Structured document

1. Load images as above and open the **Structured document** tab.
2. Choose a **Model preset** in the Model panel.
3. Click **🧭 Detect sections**.
4. Fix the sections: draw missing ones (name the box by its type, e.g.
   `table`), retype, resize or delete wrong ones.
5. Click **📖 Read sections** to read new sections (or all of them).
6. Fix content, heading levels and order in the table, then click
   **✅ Apply table edits**.
7. When you're done, click **📦 Export JSONL**.

## Output

```
output/
├── annotations/<image>.json   # boxes and labels for each image (autosaved)
├── Label.txt                  # text detection, PPOCRLabel format
├── rec_gt.txt                 # text recognition: "crops/<file>\t<text>"
├── crops/                     # one cropped image per labelled box
├── structured/<image>.json    # sections and structured JSON (autosaved)
└── structured.jsonl           # structured export, one page per line
```

Each line of `Label.txt` is an image path, a tab, and a JSON list of boxes:

```
/path/img.jpg	[{"transcription": "Hello", "points": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "difficult": false}]
```

Each `structured/<image>.json` holds the flat `sections` list in reading
order (boxes in pixels) and the nested `document` built from it:

```json
{
  "image": "/path/page1.png", "width": 1000, "height": 1400,
  "model": "qwen/qwen3-vl-4b-thinking",
  "sections": [
    {"type": "heading", "level": 1, "content": "1 Introduction",
     "xmin": 50, "ymin": 140, "xmax": 500, "ymax": 182}
  ],
  "document": {
    "title": "Report",
    "page_header": [], "page_footer": [],
    "blocks": [],
    "sections": [
      {"heading": "1 Introduction", "level": 1, "bbox": [50, 140, 500, 182],
       "blocks": [{"type": "table", "bbox": [50, 448, 950, 840], "content": "<table>...</table>"}],
       "sections": []}
    ]
  }
}
```

Section types are `title`, `heading`, `paragraph`, `list`, `table`,
`formula`, `figure`, `caption`, `header`, `footer` and `page_number`; any
other name you give a box is kept as its type. `header`, `footer` and
`page_number` go to `page_header`/`page_footer`; the first `title` becomes
`document.title`.

## Notes

- **Auto-detect needs PaddleOCR-VL.** It relies on the model's box markers
  (`<|LOC_n|>`). Other vision models won't return them, but you can still draw
  boxes by hand and use **OCR boxes**. The prompts can be changed under
  **Settings**.
- **Section detection with PaddleOCR-VL is a heuristic.** Headings are found
  by line height only, so bold headings in body-sized text come out as
  paragraphs; retype them. Qwen3-VL finds sections and types directly but is
  slower, especially while thinking. Raise **Max tokens** under **Structure
  settings** if detection returns nothing.
- **Qwen3-VL boxes** are read as 0-1000 page coordinates. If any value is
  above 1000 they are read as pixels (Qwen2.5-VL style).
- **Boxes are upright rectangles.** The model returns 4-point boxes that can be
  angled; the app turns them into upright rectangles, so slanted text gets a
  looser box.
- **Adding a provider:** subclass `Provider` in `providers.py`, implement
  `list_models()` and `chat()`, and add it to `PROVIDERS`.
- **Adding a model preset:** add an entry to `PROFILES` in `structure.py` with
  its name patterns, detect prompt and per-type read prompts.

## Project structure

```
app.py            # Gradio UI, text-box tab, saving and export
structure.py      # structured-document tab and model presets
common.py         # helpers shared by both tabs (model calls, cropping)
providers.py      # model backends: LM Studio, Ollama, OpenAI-compatible
requirements.txt
```
