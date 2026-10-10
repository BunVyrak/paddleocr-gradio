# paddlocr-gradio

A Gradio web app for building OCR datasets. Draw boxes around text in an image,
and [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) reads
the text inside each box and fills in its label. The model runs locally in
LM Studio or Ollama, or on any server with an OpenAI-compatible API.

## Features

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
- **Choice of provider:** LM Studio, Ollama or any OpenAI-compatible server
  (vLLM, llama.cpp server, ...), selected in the app.

## Requirements

- Python 3.10 or newer
- One of these, running PaddleOCR-VL-1.5 (or another vision model):
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
| `--model` | `paddleocr-vl-1.5` | Model name. If the server doesn't have it, the app picks one. |
| `--out` | `output` | Folder for annotations and exports |
| `--port` | first free port from 7860 | Port for the web app |

Example:

```bash
python app.py --provider Ollama --out my_dataset
```

The provider, server address, API key and model can also be changed in the
app's **Model** panel. 🔄 reloads the list of models from the server.

## How to annotate

1. Type a folder path and click **Load folder**, or upload images.
2. Click **🔍 Auto-detect** to find and read all text on the page.
3. Fix the boxes: draw any that are missing, resize or delete wrong ones.
4. Click **🔤 OCR boxes** to read the text in new or changed boxes.
5. Correct labels in the table if needed, then click **✅ Apply table edits**.
6. Go to the next image with **Next ▶**. Your work is saved automatically.
7. When you're done, click **📦 Export dataset**.

## Output

```
output/
├── annotations/<image>.json   # boxes and labels for each image (autosaved)
├── Label.txt                  # text detection, PPOCRLabel format
├── rec_gt.txt                 # text recognition: "crops/<file>\t<text>"
└── crops/                     # one cropped image per labelled box
```

Each line of `Label.txt` is an image path, a tab, and a JSON list of boxes:

```
/path/img.jpg	[{"transcription": "Hello", "points": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "difficult": false}]
```

## Notes

- **Auto-detect needs PaddleOCR-VL.** It relies on the model's box markers
  (`<|LOC_n|>`). Other vision models won't return them, but you can still draw
  boxes by hand and use **OCR boxes**. The prompts can be changed under
  **Settings**.
- **Boxes are upright rectangles.** The model returns 4-point boxes that can be
  angled; the app turns them into upright rectangles, so slanted text gets a
  looser box.
- **Adding a provider:** subclass `Provider` in `providers.py`, implement
  `list_models()` and `chat()`, and add it to `PROVIDERS`.

## Project structure

```
app.py            # Gradio UI, box handling, saving and export
providers.py      # model backends: LM Studio, Ollama, OpenAI-compatible
requirements.txt
```
