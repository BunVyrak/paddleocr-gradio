"""Prompts for general vision-language models, loaded from prompt.md.

Every "## `name`" section of prompt.md holds one prompt: the first ```text
block after the heading. `{{placeholder}}`s are filled in by `render()`.
"""

import re
from pathlib import Path

PROMPT_FILE = Path(__file__).with_name("prompt.md")
REQUIRED = {"layout", "read-text", "read-table", "read-formula", "read-figure", "classify",
            "fill", "guide-qa", "guide-mcq", "guide-qcm"}
HEADING_RE = re.compile(r"^## `([\w-]+)`", re.M)
BLOCK_RE = re.compile(r"^```text\n(.*?)\n```", re.S | re.M)


def load(path: Path = PROMPT_FILE) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    heads = list(HEADING_RE.finditer(text))
    prompts = {}
    for head, nxt in zip(heads, heads[1:] + [None]):
        chunk = text[head.end():nxt.start() if nxt else len(text)]
        if block := BLOCK_RE.search(chunk):
            prompts[head[1]] = block[1].strip()
    if missing := REQUIRED - prompts.keys():
        raise RuntimeError(f"{path} has no ```text block for: {', '.join(sorted(missing))}")
    return prompts


PROMPTS = load()


def render(name: str, **values: str) -> str:
    text = PROMPTS[name]
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text
