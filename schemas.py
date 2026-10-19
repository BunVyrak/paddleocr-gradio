"""Output schemas for structured annotation.

A parsed page (the reading-ordered `sections` from structure.py) can be turned
into one of these JSON formats:

- document  the page as nested headings and blocks (see structure.build_document)
- qa        open questions with their answers
- mcq       multiple-choice questions with exactly one correct choice
- qcm       multiple-choice questions where any number of choices can be correct

Question parts are kept apart: the stem, each choice, the answer and the
explanation each have their own field (and their own section type: question,
choice, answer, explanation). Math stays LaTeX in its delimiters: `\\( ... \\)`
(or `\\{ ... \\}`, `$...$`) inside text, `\\[ ... \\]` (or `$$...$$`) for a displayed
equation; see formulas.py.

`classify()` picks the schema that fits a page from its content (or
`classify_prompt()` asks a general VLM), `label_sections()` retypes the
sections to match it, `extract()` converts them with rules (numbering, choice
letters, ticked boxes, "Answer:" lines), `fill_prompt()` asks a general VLM to
do it, and `validate()` checks the result against the JSON Schema.
"""

import json
import re

from jsonschema import Draft202012Validator

from formulas import bare_latex
from prompts import render

# Section types that carry a question's parts.
ROLE_TYPES = ["question", "choice", "answer", "explanation"]
# Sections that aren't part of the question text.
SKIP_TYPES = {"header", "footer", "page_number", "figure"}

# --------------------------------------------------------------------------- #
# JSON Schemas
# --------------------------------------------------------------------------- #
_bbox = {"type": ["array", "null"], "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
         "description": "[xmin, ymin, xmax, ymax] in image pixels"}
_text = {"type": "string", "minLength": 1,
         "description": "Text as printed; math as LaTeX in \\( ... \\) or \\[ ... \\]"}
_opt_text = {"type": ["string", "null"]}
_label = {"type": "string", "pattern": "^[A-Z]$", "description": "Choice letter: A, B, C..."}
_choice = {
    "type": "object",
    "required": ["label", "text"],
    "properties": {"label": _label, "text": {"type": "string"}},
    "additionalProperties": False,
}


def _questions(name: str, title: str, item_props: dict, required: list[str]) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": ["schema", "items"],
        "properties": {
            "schema": {"const": name},
            "title": _opt_text,
            "instructions": {**_opt_text, "description": "Text before the first question"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "question", *required],
                    "properties": {
                        "id": {"type": "string", "description": "Question number as printed"},
                        "question": {**_text, "description": "The question stem, without "
                                     "its number or choices"},
                        **item_props,
                        "explanation": _opt_text,
                        "bbox": _bbox,
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


_element = {
    "type": "object",
    "required": ["type", "content"],
    "properties": {"type": {"type": "string"}, "bbox": _bbox, "content": {"type": "string"}},
}

SCHEMAS = {
    "document": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Document",
        "type": "object",
        "required": ["title", "blocks", "sections"],
        "properties": {
            "title": _opt_text,
            "page_header": {"type": "array", "items": {"$ref": "#/$defs/element"}},
            "page_footer": {"type": "array", "items": {"$ref": "#/$defs/element"}},
            "blocks": {"type": "array", "items": {"$ref": "#/$defs/element"}},
            "sections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
        },
        "$defs": {
            "element": _element,
            "section": {
                "type": "object",
                "required": ["heading", "level", "blocks", "sections"],
                "properties": {
                    "heading": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1},
                    "bbox": _bbox,
                    "blocks": {"type": "array", "items": {"$ref": "#/$defs/element"}},
                    "sections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
                },
            },
        },
    },
    "qa": _questions("qa", "Q&A", {"answer": _opt_text}, ["answer"]),
    "mcq": _questions("mcq", "MCQ: one correct choice", {
        "choices": {"type": "array", "items": _choice, "minItems": 2},
        "answer": {**_label, "type": ["string", "null"],
                   "description": "The one correct choice's label, or null if not given"},
    }, ["choices", "answer"]),
    "qcm": _questions("qcm", "QCM: one or more correct choices", {
        "choices": {"type": "array", "items": _choice, "minItems": 2},
        "answers": {"type": "array", "items": _label, "uniqueItems": True,
                    "description": "Labels of every correct choice; empty if not given"},
    }, ["choices", "answers"]),
}

# Dropdown label -> schema name; "auto" picks one from the page content.
SCHEMA_CHOICES = {
    "Auto (from page content)": "auto",
    "Document": "document",
    "Q&A": "qa",
    "MCQ (one correct choice)": "mcq",
    "QCM (several correct choices)": "qcm",
}


def validate(name: str, data) -> list[str]:
    """Problems with `data` as schema `name`; empty if it is valid."""
    errors = [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
              for e in Draft202012Validator(SCHEMAS[name]).iter_errors(data)]
    if errors or name not in ("mcq", "qcm"):
        return errors
    for i, item in enumerate(data["items"]):
        labels = [c["label"] for c in item["choices"]]
        if len(set(labels)) != len(labels):
            errors.append(f"items/{i}/choices: duplicate labels {labels}")
        given = [item["answer"]] if name == "mcq" and item["answer"] else item.get("answers", [])
        for a in given:
            if a not in labels:
                errors.append(f"items/{i}: answer {a!r} is not one of the choices {labels}")
    return errors


# --------------------------------------------------------------------------- #
# Rule-based conversion
# --------------------------------------------------------------------------- #
QUESTION_RE = re.compile(
    r"^(?:(?:question|q|câu|exercice|exercise|problem)\s*\.?\s*(\d+)\s*[.):：-]?"
    r"|(\d{1,3})\s*[.):：](?=\s|$))\s*(.*)$", re.I | re.S)
CHOICE_RE = re.compile(r"^[\[(]?([A-Ha-h])[\]).:：](?=\s|$)\s*(.*)$", re.S)
# A choice letter inside a line, with any tick box in front of it ("☑ B.").
CHOICE_MARK_RE = re.compile(r"(?:(?<=\s)|^)(?:(?:[☐□○◯❍☑☒✓✔✅◉]|\[[ xX✓✔]\]|\([xX✓✔]\))\s*)?"
                            r"[\[(]?([A-Ha-h])[\]).](?=\s)")
ANSWER_RE = re.compile(
    r"^(?:correct\s+answers?|answers?|ans|key|réponses?|bonne\s+réponse|solution|đáp\s*án)"
    r"\s*[.:：-]\s*(.*)$", re.I | re.S)
EXPLANATION_RE = re.compile(
    r"^(?:explanation|explication|justification|rationale|giải\s*thích)\s*[.:：-]\s*(.*)$",
    re.I | re.S)
CHECKED_RE = re.compile(r"[☑☒✓✔✅◉]|\[[xX✓✔]\]|\([xX✓✔]\)")
BOX_RE = re.compile(r"[☐□○◯❍]|\[ \]")
MD_PREFIX_RE = re.compile(r"^\s*(?:#+\s+|[-*•]\s+|>\s*)")


# Section types that label_sections() may retype to a question part.
RELABEL_FROM = {"paragraph", "list", "text", "caption", "footnote"}


def clean_line(line: str) -> tuple[str, bool]:
    """Strip Markdown and tick-box glyphs; return (text, ticked)."""
    ticked = bool(CHECKED_RE.search(line))
    line = BOX_RE.sub("", CHECKED_RE.sub("", line)).replace("**", "")
    return MD_PREFIX_RE.sub("", line).strip(), ticked


def line_role(text: str) -> str | None:
    """question/choice/answer/explanation if the text starts like one."""
    line = clean_line(text.lstrip().split("\n", 1)[0])[0]
    for role, regex in (("answer", ANSWER_RE), ("explanation", EXPLANATION_RE),
                        ("question", QUESTION_RE), ("choice", CHOICE_RE)):
        if regex.match(line):
            return role
    return None


def split_inline_choices(line: str) -> list[str]:
    """Split "1. Pick one A. x B. y C. z" into the stem and one line per choice.
    Only splits on a run of consecutive letters starting at A."""
    starts, expect = [], "A"
    for m in CHOICE_MARK_RE.finditer(line):
        if m[1].upper() == expect:
            starts.append(m.start())
            expect = chr(ord(expect) + 1)
    if len(starts) < 2:
        return [line]
    cuts = ([0] if starts[0] > 0 else []) + starts + [len(line)]
    return [line[a:b].strip() for a, b in zip(cuts, cuts[1:]) if line[a:b].strip()]


def answer_labels(text: str, choices: list[dict]) -> list[str]:
    """Choice labels named by an answer line ("B", "A, C", "b", or a choice's text)."""
    labels = re.findall(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", text)
    if not labels and len(text) <= 15:
        labels = [l.upper() for l in re.findall(r"(?<![A-Za-z])([a-h])(?![A-Za-z])", text)]
    if not labels:
        low = text.lower()
        labels = [c["label"] for c in choices if c["text"] and c["text"].lower() in low]
    known = {c["label"] for c in choices}
    return list(dict.fromkeys(l for l in labels if not known or l in known))


def section_text(s: dict) -> str:
    """A section's content as it goes into question text: formulas as \\[ ... \\]."""
    if s["type"] == "formula":
        return f"\\[ {bare_latex(s['content'])} \\]"
    return s["content"]


def extract(name: str, sections: list[dict]) -> tuple[dict, list[str]]:
    """Convert reading-ordered sections into schema `name` (qa, mcq or qcm).
    Returns (data, notes): notes list what the rules couldn't decide.

    Section types question/choice/answer/explanation are used as given; other
    text is split into lines and recognised by numbering ("1.", "Q2", "Câu 3"),
    choice letters ("A.", "(b)"), ticked boxes and "Answer:"/"Explanation:" lines.
    """
    title, intro, items = None, [], []
    cur, field = None, None

    def new_item(qid, text, s):
        nonlocal cur, field
        cur = {"id": qid or str(len(items) + 1), "question": text, "choices": [],
               "ticked": [], "answer": "", "explanation": "", "boxes": []}
        items.append(cur)
        field = "question"

    def add(key, text):
        cur[key] = f"{cur[key]}\n{text}".strip() if cur[key] else text

    for s in sections:
        t = s["type"]
        if not s["content"].strip() or t in SKIP_TYPES:
            continue
        if t == "title" and title is None:
            title = s["content"].strip()
            continue
        if t == "heading" and cur is not None:
            cur, field = None, None  # e.g. "Part II": the question before it is done
        lines = [l for raw in section_text(s).splitlines() for l in split_inline_choices(raw)]
        for n, raw in enumerate(lines):
            line, ticked = clean_line(raw)
            if not line:
                continue
            # The section's own type decides what its first line is when the
            # line has no marker of its own.
            hint = t if n == 0 and t in ROLE_TYPES else None
            if m := ANSWER_RE.match(line):
                if cur is None:
                    new_item(None, "", s)
                field = "answer"
                add("answer", m[1].strip())
            elif m := EXPLANATION_RE.match(line):
                if cur is None:
                    new_item(None, "", s)
                field = "explanation"
                add("explanation", m[1].strip())
            elif (m := CHOICE_RE.match(line)) and cur is not None and name != "qa":
                cur["choices"].append({"label": m[1].upper(), "text": m[2].strip()})
                if ticked:
                    cur["ticked"].append(m[1].upper())
                field = "choice"
            elif m := QUESTION_RE.match(line):
                new_item(m[1] or m[2], m[3].strip(), s)
            elif hint == "question":
                new_item(None, line, s)
            elif hint == "choice" and cur is not None and name != "qa":
                label = chr(ord("A") + len(cur["choices"]))
                cur["choices"].append({"label": label, "text": line})
                field = "choice"
            elif hint in ("answer", "explanation") and cur is not None:
                field = hint
                add(hint, line)
            elif cur is None:
                intro.append(line)
                continue
            elif field == "choice" and cur["choices"] and t != "formula":
                cur["choices"][-1]["text"] = f"{cur['choices'][-1]['text']} {line}".strip()
            else:
                # A displayed formula after the choices belongs to the question.
                add("question" if field == "choice" else field, line)
            cur["boxes"].append(s)

    notes = []
    data = {"schema": name, "title": title, "instructions": "\n".join(intro) or None,
            "items": [_finish(name, it, notes) for it in items]}
    return data, notes


def _finish(name: str, it: dict, notes: list[str]) -> dict:
    boxes = it["boxes"]
    bbox = ([min(b["xmin"] for b in boxes), min(b["ymin"] for b in boxes),
             max(b["xmax"] for b in boxes), max(b["ymax"] for b in boxes)] if boxes else None)
    out = {"id": it["id"], "question": it["question"]}
    if name == "qa":
        out["answer"] = it["answer"] or None
    else:
        labels = it["ticked"] or answer_labels(it["answer"], it["choices"])
        out["choices"] = it["choices"]
        if name == "mcq":
            if len(labels) > 1:
                notes.append(f"Question {it['id']}: {', '.join(labels)} are all marked "
                             "correct; answer left empty (use QCM for several answers).")
            out["answer"] = labels[0] if len(labels) == 1 else None
        else:
            out["answers"] = labels
    out["explanation"] = it["explanation"] or None
    out["bbox"] = bbox
    return out


# --------------------------------------------------------------------------- #
# Model-based conversion
# --------------------------------------------------------------------------- #
def fill_prompt(name: str, parsed: str) -> str:
    """Prompt asking a general VLM to fill schema `name` from the page image,
    with the already-parsed page text as a reference (prompt.md: `fill`)."""
    schema = {k: v for k, v in SCHEMAS[name].items() if k != "$schema"}
    return render("fill", guide=render(f"guide-{name}"),
                  schema=json.dumps(schema, ensure_ascii=False), page=parsed)


# --------------------------------------------------------------------------- #
# Choosing the schema
# --------------------------------------------------------------------------- #
QA_HINT_RE = re.compile(
    r"\?|^\s*(?:solve|compute|calculate|find|explain|describe|prove|show|define|discuss|"
    r"why|how|what|which|who|when|where|résous|résoudre|calcule[rz]?|explique[rz]?|"
    r"montre[rz]?|démontre[rz]?|tính|giải)\b", re.I | re.M)
MULTI_HINT_RE = re.compile(
    r"select all that apply|all that apply|select all|one or more|more than one|multiple (?:correct )?answers|"
    r"several (?:correct )?answers|plusieurs (?:bonnes )?réponses|une ou plusieurs|"
    r"toutes les (?:bonnes )?réponses|cochez (?:la ou )?les", re.I)
SINGLE_HINT_RE = re.compile(
    r"one answer|single answer|only one|best answer|choose one|one correct|"
    r"une seule (?:bonne )?réponse|la bonne réponse", re.I)


def classify(sections: list[dict]) -> tuple[str, list[str]]:
    """The schema that fits the page best, with the reasons, from rules:
    - mcq/qcm: at least half the numbered questions have lettered choices;
      qcm when a question has several marked answers or the instructions say
      so ("select all that apply"), mcq otherwise;
    - qa: numbered items that read as questions (?, a question word, an answer);
    - document: anything else."""
    mc, _ = extract("qcm", sections)
    items = [it for it in mc["items"] if it["question"] or it["choices"]]
    if not items:
        return "document", ["No numbered questions or answer lines found."]
    with_choices = [it for it in items if len(it["choices"]) >= 2]
    if len(with_choices) * 2 >= len(items):
        reasons = [f"{len(with_choices)} of {len(items)} questions have lettered choices."]
        context = " ".join(filter(None, [mc["title"], mc["instructions"],
                                         *(s["content"] for s in sections
                                           if s["type"] in ("heading", "paragraph"))]))
        multi = [it["id"] for it in with_choices if len(it["answers"]) > 1]
        if multi:
            return "qcm", reasons + [f"Question(s) {', '.join(multi)} have several "
                                     "correct choices marked."]
        if m := MULTI_HINT_RE.search(context):
            return "qcm", reasons + [f"The page says “{m[0]}”."]
        if m := SINGLE_HINT_RE.search(context):
            return "mcq", reasons + [f"The page says “{m[0]}”."]
        return "mcq", reasons + ["Nothing says several choices can be correct."]
    qa, _ = extract("qa", sections)
    asked = [it for it in qa["items"] if it["answer"] or QA_HINT_RE.search(it["question"])]
    if asked and len(asked) * 2 >= len(qa["items"]):
        return "qa", [f"{len(asked)} of {len(qa['items'])} numbered items are questions "
                      "(question mark, question word or answer) without lettered choices."]
    return "document", [f"{len(qa['items'])} numbered items, but they read as a list, "
                        "not as questions."]


def classify_prompt(parsed: str) -> str:
    """Prompt asking a general VLM which schema fits the page (prompt.md: `classify`)."""
    return render("classify", page=parsed)


def label_sections(sections: list[dict], name: str) -> tuple[list[dict], int]:
    """On a question page, retype text sections by the part of a question they
    start with (question, choice, answer, explanation). Returns (sections, how
    many changed)."""
    if name not in ("qa", "mcq", "qcm"):
        return sections, 0
    out, changed = [], 0
    for s in sections:
        role = line_role(s["content"]) if s["type"] in RELABEL_FROM else None
        if name == "qa" and role == "choice":
            role = None
        if role and role != s["type"]:
            s = {**s, "type": role, "level": None}
            changed += 1
        out.append(s)
    return out, changed
