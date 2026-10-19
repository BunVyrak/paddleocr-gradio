# VLM prompts

The prompts the app sends to general vision-language models: the
**Qwen3-VL-4B-Thinking** and **Generic VLM** presets. They are written to work
across instruction-following VLMs (Qwen2.5-VL / Qwen3-VL, InternVL, Gemma 3,
MiniCPM-V, GPT-4o-class APIs) and to produce exactly what the app's parser
expects.

PaddleOCR-VL-1.5 and dots.ocr are not instruction-following models: they are
trained on fixed task prompts (`Spotting:`, `OCR:`, dots.ocr's layout
prompt), so their presets keep those and don't use this file.

## How the app uses this file

`prompts.py` loads every `## \`name\`` section below and takes the first
` ```text ` block in it as that prompt. Text outside those blocks is
documentation and is never sent. Edit a block and restart the app to change
a prompt; the **Detect prompt** box under **Structure settings** can still
override `layout` for a single session.

`{{name}}` placeholders are filled in by the app:

| Placeholder | Filled with |
| --- | --- |
| `{{page}}` | the page's parsed sections as Markdown, in reading order |
| `{{guide}}` | the `guide-<schema>` prompt for the chosen schema |
| `{{schema}}` | the chosen schema's JSON Schema (see `schemas.py`) |

| Prompt | Used by | Step |
| --- | --- | --- |
| `layout` | ⚡ Parse page, 🧭 Detect sections | find every section, its type, box and text |
| `read-text` | 📖 Read sections, ⚡ Parse page | read a cropped section that has no text yet |
| `read-table` | same, for `table` sections | |
| `read-formula` | same, for `formula` sections | |
| `read-figure` | same, for `figure` sections | |
| `classify` | Auto schema, *Decide + fill with: Model* | pick document / qa / mcq / qcm |
| `fill` | Q&A, MCQ, QCM, *Decide + fill with: Model* | fill the chosen schema |
| `guide-qa`, `guide-mcq`, `guide-qcm` | inserted into `fill` | what the schema means |

## What the parser expects

Keep these in step with the code when you edit a prompt:

- **Layout output:** a JSON array of
  `{"type", "bbox_2d", "content"}` objects, plus `"level"` on headings
  (`parse_layout` in `structure.py`). Text around the array, Markdown fences
  and `<think>` blocks are removed; a cut-off array keeps its complete
  elements.
- **Types:** `title`, `heading`, `paragraph`, `list`, `question`, `choice`,
  `answer`, `explanation`, `table`, `formula`, `figure`, `caption`,
  `footnote`, `header`, `footer`, `page_number` (`SECTION_TYPES`). Common
  synonyms such as `text`, `section_header` or `picture` are mapped.
- **Boxes:** `[left, top, right, bottom]` normalised to 0-1000. If any value
  is above 1000 the boxes are read as pixels instead (Qwen2.5-VL habit).
- **Math:** `\( ... \)` inline and `\[ ... \]` displayed (`formulas.py`; `$`,
  `$$` and `\{ \}` are also recognised). A `formula` section is bare LaTeX.
- **Question parts:** numbers (`1.`, `Q2`), option letters (`A.`, `(b)`),
  ticked boxes (`☑`) and labels (`Answer:`, `Explanation:`) are kept as
  printed, because the rule-based conversion reads them (`schemas.py`).
- **JSON strings:** LaTeX backslashes must be escaped (`\\frac`); the app
  repairs the usual mistakes, but escaped output is safer.

Example `layout` answer:

```json
[{"type": "title", "bbox_2d": [180, 30, 820, 70], "content": "Algebra Quiz"},
 {"type": "paragraph", "bbox_2d": [60, 90, 940, 120], "content": "Choose one answer for each question."},
 {"type": "question", "bbox_2d": [60, 140, 940, 180], "content": "1. Solve \\(x^2 = 4\\)."},
 {"type": "choice", "bbox_2d": [80, 190, 300, 220], "content": "A. \\(x = 2\\)"},
 {"type": "choice", "bbox_2d": [320, 190, 560, 220], "content": "☑ B. \\(x = \\pm 2\\)"},
 {"type": "formula", "bbox_2d": [300, 240, 700, 290], "content": "x = \\pm \\sqrt{4}"},
 {"type": "page_number", "bbox_2d": [480, 960, 520, 985], "content": "1"}]
```

## `layout`: parse the page

```text
You are a document parser. Split this page image into layout elements and return them as a JSON array. Output only the JSON array: no Markdown fences, no comments, no explanation.

Each element is one object:
{"type": "<type>", "bbox_2d": [x1, y1, x2, y2], "content": "<text>"}
Headings also have "level": 1 for a top-level heading, 2 or 3 for a sub-heading.

Types:
- title: the document title (at most one per page)
- heading: a section or part heading
- paragraph: a block of body text, including instructions
- list: a bulleted or numbered list that is not a set of questions; one element for the whole list, one item per line
- question: a question or exercise stem, with its number ("1. Solve ...")
- choice: ONE answer option of a question ("A. 12"); one element per option, even when several options share a line
- answer: an answer or answer key ("Answer: B")
- explanation: a worked solution or the explanation of an answer
- table: a table
- formula: an equation displayed on its own line
- figure: a picture, diagram, chart or photo
- caption: the caption of a table or figure
- footnote: a footnote
- header, footer: running text at the top or bottom of the page
- page_number: the page number

Rules:
1. Boxes: bbox_2d is the tight box around the element as [left, top, right, bottom], in coordinates normalised to 0-1000 over the whole page: [0, 0] is the top-left corner and [1000, 1000] the bottom-right, whatever the image size.
2. Order: list the elements in reading order, top to bottom; on a multi-column page finish the left column before the right one.
3. Text: content is the text exactly as printed. Keep its language, spelling, punctuation, numbering, option letters and labels such as "Answer:". Do not translate, correct, summarise, or answer the questions.
4. Marked answers: if an option is marked as chosen or correct (ticked, circled, highlighted), start its content with ☑; write an empty tick box as ☐.
5. Math: math inside a sentence is LaTeX in \( ... \); an equation displayed inside a text block is LaTeX in \[ ... \]; the content of a formula element is bare LaTeX, without delimiters.
6. Tables and figures: a table's content is one HTML <table> with <tr>, <th> and <td>, using rowspan and colspan for merged cells; a figure's content is one sentence describing what it shows.
7. JSON: escape every backslash and double quote inside strings ("\\(", "\\frac", "\""), and write line breaks as \n.
8. Never merge two elements, never split one, and never output an empty element. If the page has no content, return [].

Example:
[{"type": "question", "bbox_2d": [60, 140, 940, 180], "content": "1. Solve \\(x^2 = 4\\)."},
 {"type": "choice", "bbox_2d": [80, 190, 300, 220], "content": "A. \\(x = 2\\)"},
 {"type": "choice", "bbox_2d": [320, 190, 560, 220], "content": "☑ B. \\(x = \\pm 2\\)"},
 {"type": "formula", "bbox_2d": [300, 240, 700, 290], "content": "x = \\pm \\sqrt{4}"}]
```

## `read-text`: read a text section

Used for every section type without a prompt of its own (paragraph, question,
choice, answer, heading, ...). The answer is plain text, not JSON.

```text
Transcribe the text in this image exactly as printed. Keep line breaks, numbering, option letters and labels such as "Answer:". If an option is marked as chosen or correct, start it with ☑; write an empty tick box as ☐. Write math as LaTeX: \( ... \) inside a sentence, \[ ... \] for a displayed equation. Do not translate or correct anything. Return only the text.
```

## `read-table`: read a table section

```text
Convert this table to HTML: one <table> with <tr>, <th> and <td>, using rowspan and colspan for merged cells. Copy the cell text exactly as printed and write math in cells as \( ... \). Return only the <table> element.
```

## `read-formula`: read a formula section

```text
Write this formula as LaTeX. Return only the LaTeX, without $, \( \) or \[ \] delimiters.
```

## `read-figure`: describe a figure section

```text
Describe this figure in one sentence. If it is a chart, also list its data. Return only the description.
```

## `classify`: pick the output schema

The answer is read with the same JSON parser; `schema` must be one of the four
names, otherwise the app falls back to its rules.

```text
Decide which format fits this page best:
- document: ordinary text such as an article, report, form, letter or notes; not a set of questions
- qa: questions or exercises answered in free text, without answer options
- mcq: multiple-choice questions where exactly ONE option is correct
- qcm: multiple-choice questions where SEVERAL options can be correct (e.g. "select all that apply", or several options ticked for one question)

Return only JSON: {"schema": "document" | "qa" | "mcq" | "qcm", "reason": "<one sentence>"}

The page text, already parsed in reading order:
<page>
{{page}}
</page>
```

## `fill`: fill the chosen schema

```text
Extract the {{guide}}

Return one JSON object that matches this JSON Schema, and nothing else:
{{schema}}

Rules:
1. Copy text exactly as printed. Do not translate, correct, or answer the questions yourself.
2. Keep the parts apart: the question is only the stem (no number, no options); options, answers and explanations go in their own fields.
3. id is the question number as printed ("1", "2a"...).
4. Write math as LaTeX: \( ... \) inside a sentence, \[ ... \] for a displayed equation. Escape every backslash for JSON ("\\(", "\\frac").
5. instructions is any text before the first question; bbox is null.

The page text, already parsed in reading order (use it to get the wording right):
<page>
{{page}}
</page>
```

## `guide-qa`: Q&A

```text
open questions and their answers. answer is the answer printed on the page, or null if there is none.
```

## `guide-mcq`: MCQ, one correct choice

```text
multiple-choice questions with exactly ONE correct choice. Put each option in choices with its letter as label (A, B, C...) and its text without the letter or tick box. answer is the letter of the option the page marks as correct (ticked box, answer key, bold), or null if none is marked.
```

## `guide-qcm`: QCM, several correct choices

```text
multiple-choice questions where SEVERAL choices may be correct. Put each option in choices with its letter as label (A, B, C...) and its text without the letter or tick box. answers lists the letters of every option the page marks as correct (ticked boxes, answer key), or [] if none are marked.
```
