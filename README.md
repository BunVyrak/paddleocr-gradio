# paddlocr-gradio

A Gradio web app for building document datasets. It has two tabs that share
the image list and model settings:

- **Text boxes:** draw boxes around text and
  [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) reads the
  text inside each box (PPOCRLabel export).
- **Structured document:** the model finds the page's sections (title,
  headings, paragraphs, questions, choices, answers, tables, formulas,
  figures, ...), reads each one, and the page is saved as structured JSON. The
  page can then be converted to a **Q&A**, **MCQ** or **QCM** schema.

Switch between **PaddleOCR-VL-1.5**, **Qwen3-VL-4B-Thinking**, **dots.ocr**
and any other vision-language model (**Generic VLM**) with the **Model
preset** option. The prompts for general VLMs live in
[`prompt.md`](prompt.md). The model runs locally in LM Studio or Ollama, or on
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

- **⚡ Parse page:** runs the whole pipeline on the current page:
  1. detect the sections, in reading order;
  2. read any section that has no text yet;
  3. **decide the schema from the page's content**: Document, Q&A, MCQ or
     QCM (see *Automatic schema* below);
  4. **relabel the sections to fit it**: on a question page, `1. ...` becomes
     a `question`, `A. ...` a `choice`, `Answer: ...` an `answer`;
  5. fill the schema and validate it.

  The detected schema and the reasons for it are shown above the output.
- **⚡ Parse all pages:** the same for every loaded image, skipping pages
  that are already parsed unless you untick **Skip pages already parsed**.
  Each page gets its own schema, so a folder can mix exam pages and
  ordinary pages.
- **Manual steps:** **🧭 Detect sections**, **📖 Read sections** and
  **🧩 Convert** run single steps, e.g. after you fix the sections.
- **Detect sections:** finds every section on the page, in reading order.
  - *PaddleOCR-VL-1.5* has no layout task, so the app runs `Spotting:` and
    merges the text lines into blocks. Taller lines become the title and
    headings, bulleted lines become lists, and two-column pages are read left
    column first. Tables, formulas and figures aren't found this way: draw
    them and name them by type.
    On a question page, each question, choice, answer and explanation line
    starts its own section of that type.
  - *Qwen3-VL* and *Generic VLM* get the `layout` prompt from
    [`prompt.md`](prompt.md) and return every section with its type, box and
    content as JSON. The prompt defines each type, the box format, the
    reading order, marked answers (`☑`) and how to keep text and math apart
    (`\(...\)` in text, formulas and tables as their own sections).
  - *dots.ocr* gets its own layout prompt (unchanged, since the model is
    trained on that wording) and returns every section with its category, box
    and content. The page is resized the way dots.ocr's processor does before
    it is sent, so its pixel boxes can be mapped back to the original image.
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
- **Formulas:** text wrapped in `\(...\)`, `\[...\]` or `\{...\}` (and
  `$...$`, `$$...$$`) is a formula:
  - a section that is nothing but a formula (`\[ E = mc^2 \]`) becomes a
    `formula` section, stored as bare LaTeX (`E = mc^2`);
  - a sentence that mixes text and formulas keeps its text, and its block in
    the document JSON gets `segments` marking each part as `text` or
    `formula`. `\[...\]` and `$$...$$` are displayed equations
    (`"display": true`); the others are inline.

  This runs after detecting and reading, so it works for every model. The
  Qwen3-VL prompts ask for `\(...\)` and `\[...\]`.
- **Output schemas:** the parsed page is converted to one of these; see the
  **JSON Schema** tab for each one:
  - **Document:** the nested headings and blocks above.
  - **Q&A:** open questions, each with `question`, `answer` and `explanation`.
  - **MCQ:** multiple-choice questions with exactly one correct choice:
    `question`, `choices` (`label` + `text`), `answer` (one letter or null).
  - **QCM:** multiple-choice questions where several choices can be correct:
    same fields, but `answers` is a list of letters.

  Every question keeps its parts apart: the stem has no number and no
  options, each option is its own entry, and answers and explanations have
  their own fields. Math stays LaTeX in its delimiters (`\(...\)` inline,
  `\[...\]` displayed).
- **Automatic schema:** with **Output schema** set to *Auto* (the default),
  each page gets the schema that fits its content:
  - **MCQ** when at least half of the numbered questions have lettered
    choices;
  - **QCM** instead when a question has several choices marked correct, or
    the page says so ("select all that apply", "one or more", "plusieurs
    réponses", ...);
  - **Q&A** when the numbered items read as questions (question mark,
    question word such as *Explain* or *Calculer*, or an answer line);
  - **Document** otherwise, e.g. a report with a numbered list.

  Pick a schema by hand to override it.
- **Decide + fill with Rules or Model:**
  - *Rules* works with every preset and makes no extra model call. It uses
    section types (question, choice, answer, explanation) and recognises
    numbering (`1.`, `Q2`, `Question 3`, `Câu 4`), choice letters (`A.`,
    `(b)`, including several on one line), ticked boxes (`☑`, `[x]`) and
    `Answer:`/`Réponse :`/`Đáp án:` and `Explanation:` lines.
  - *Model* asks the model which schema fits (with Auto), then sends the page
    and its parsed text with the JSON Schema to fill it. It needs the
    Qwen3-VL preset; with PaddleOCR-VL and dots.ocr the rules are used
    instead, and the output says so.
- **Validation:** the output is checked against the JSON Schema, including
  that every answer is one of the choices. Problems are listed above the
  output. Fix the JSON in place and click **Validate + save edits**.
- **JSONL export:** one line per page in `structured.jsonl`, plus
  `document.jsonl`, `qa.jsonl`, `mcq.jsonl` and `qcm.jsonl` holding the pages
  of each schema.

### Both tabs

- **Model preset:** switches between PaddleOCR-VL-1.5,
  Qwen3-VL-4B-Thinking, dots.ocr and Generic VLM. It picks the matching model
  on the server and sets the prompts. Picking a model by hand also switches
  the preset. **Generic VLM** is for any other instruction-following model
  (InternVL, Gemma 3, MiniCPM-V, an API model, ...): choose the preset, then
  the model; it uses the same prompts as Qwen3-VL.
- **Prompts in one file:** [`prompt.md`](prompt.md) holds every prompt the
  general VLMs get (layout, reading a section, choosing and filling a
  schema), with what the parser expects from each. Edit it and restart the
  app. PaddleOCR-VL and dots.ocr keep their own fixed task prompts.
- **Choice of provider:** LM Studio, Ollama or any OpenAI-compatible server
  (vLLM, llama.cpp server, ...), selected in the app.
- **Thinking models:** the `<think>...</think>` reasoning is removed from the
  answer.

## Requirements

- Python 3.10 or newer
- One of these, running PaddleOCR-VL-1.5, Qwen3-VL-4B-Thinking, dots.ocr or
  another vision model:
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

**dots.ocr**

dots.ocr is served with [vLLM](https://github.com/rednote-hilab/dots.ocr), as
an OpenAI-compatible server:

```bash
vllm serve rednote-hilab/dots.ocr --trust-remote-code
```

Choose the **OpenAI-compatible** provider with `http://localhost:8000/v1`, then
the **dots.ocr** preset. If the server uses another model name (for example
`--served-model-name model`), choose the preset first, then pick the model by
hand.

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
2. Choose a **Model preset** in the Model panel. Leave **Output schema** on
   *Auto*.
3. Click **⚡ Parse page**, or **⚡ Parse all pages** for the whole folder.
4. Check the detected schema and its reasons in **Schema output**.
5. If needed, fix the sections: draw missing ones (name the box by its type,
   e.g. `table` or `choice`), retype, resize or delete wrong ones, and fix
   content and order in the table, then click **✅ Apply table edits**. Use
   **📖 Read sections** to read boxes you drew.
6. Click **🧩 Convert** to redo the schema after section changes (pick a
   schema by hand if *Auto* chose the wrong one). Fix any problems in the
   JSON and click **💾 Validate + save edits**.
7. When you're done, click **📦 Export JSONL**.

## Output

```
output/
├── annotations/<image>.json   # boxes and labels for each image (autosaved)
├── Label.txt                  # text detection, PPOCRLabel format
├── rec_gt.txt                 # text recognition: "crops/<file>\t<text>"
├── crops/                     # one cropped image per labelled box
├── structured/<image>.json    # sections, document and schema output (autosaved)
├── structured.jsonl           # structured export, one page per line
└── qa.jsonl, mcq.jsonl, qcm.jsonl  # schema output, one page per line
```

Each line of `Label.txt` is an image path, a tab, and a JSON list of boxes:

```
/path/img.jpg	[{"transcription": "Hello", "points": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "difficult": false}]
```

Each `structured/<image>.json` holds the flat `sections` list in reading
order (boxes in pixels), the nested `document` built from it, and the latest
schema output in `extraction`:

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
       "blocks": [
         {"type": "paragraph", "bbox": [50, 190, 950, 230], "content": "Solve \\(x^2 = 4\\) for x.",
          "segments": [{"type": "text", "content": "Solve"},
                       {"type": "formula", "content": "x^2 = 4", "display": false},
                       {"type": "text", "content": "for x."}]},
         {"type": "formula", "bbox": [300, 240, 700, 280], "content": "x = \\pm 2"},
         {"type": "table", "bbox": [50, 448, 950, 840], "content": "<table>...</table>"}],
       "sections": []}
    ]
  },
  "extraction": {"schema": "mcq", "method": "rules", "data": {...}, "errors": [], "notes": []}
}
```

A page in `mcq.jsonl` (`qcm.jsonl` has `"answers": ["B"]` instead of
`"answer"`; `qa.jsonl` has no `choices`):

```json
{"image": "/path/quiz.png", "schema": "mcq", "title": "Quiz", "instructions": "Choose one answer.",
 "items": [{"id": "1", "question": "Solve \\(x^2 = 4\\).",
            "choices": [{"label": "A", "text": "\\(x = 2\\)"}, {"label": "B", "text": "\\(x = \\pm 2\\)"}],
            "answer": "B", "explanation": null, "bbox": [50, 140, 950, 308]}]}
```

Section types are `title`, `heading`, `paragraph`, `list`, `question`,
`choice`, `answer`, `explanation`, `table`, `formula`, `figure`, `caption`,
`footnote`, `header`, `footer` and `page_number`; any
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
- **dots.ocr boxes** are pixels of the resized page the app sends. A server
  that resizes images again in its own way (for example a GGUF build in LM
  Studio) will give shifted boxes; use vLLM.
- **LaTeX in JSON:** models often write `\(`, `\pm` or `\frac` without
  escaping the backslash. The app repairs this when it reads the model's JSON.
- **`\{...\}` is a formula delimiter here.** Its braces are dropped from
  the LaTeX, so `\{1, 2\}` becomes the formula `1, 2`. Inside another
  formula (`\( x \in \{1, 2\} \)`) it stays part of that formula. A single
  `$` only counts when it hugs its content, so `$5 and $10` is plain text.
- **Rules are a heuristic.** They expect questions numbered and choices
  lettered. A line like `A. Smith wrote...` inside a question is read as a
  choice; fix it in the schema output, or set the section types and convert
  again.
- **Automatic schema is per page.** A question that continues onto the next
  page is not joined with it, and a page with only a few questions among
  ordinary text may be classed as Document; pick the schema by hand there.
- **Schema output isn't updated automatically.** After you change sections,
  click **Convert** again.
- **Boxes are upright rectangles.** The model returns 4-point boxes that can be
  angled; the app turns them into upright rectangles, so slanted text gets a
  looser box.
- **Adding a provider:** subclass `Provider` in `providers.py`, implement
  `list_models()` and `chat()`, and add it to `PROVIDERS`.
- **Adding a model preset:** add an entry to `PROFILES` in `structure.py` with
  its name patterns, detect prompt, box format and per-type read prompts. A
  general VLM can reuse `GENERAL_VLM` (the prompts from `prompt.md`).
- **Adding a schema:** add its JSON Schema to `SCHEMAS` and `SCHEMA_CHOICES`
  in `schemas.py`, rules in `extract()`, and a `guide-<name>` prompt in
  `prompt.md` (also add it to `REQUIRED` in `prompts.py`).

## Project structure

```
app.py            # Gradio UI, text-box tab, saving and export
structure.py      # structured-document tab and model presets
schemas.py        # output schemas (Q&A, MCQ, QCM), rules, validation
formulas.py       # finding \(...\), \[...\], \{...\} and $...$ formulas in text
prompts.py        # loads the prompts from prompt.md
prompt.md         # prompts for general VLMs (layout, read, classify, fill)
common.py         # helpers shared by both tabs (model calls, cropping)
providers.py      # model backends: LM Studio, Ollama, OpenAI-compatible
requirements.txt
```
