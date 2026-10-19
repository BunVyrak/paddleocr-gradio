"""Finding formulas in parsed text.

A span of text is a formula when it is wrapped in one of these delimiters:

    \\( ... \\)   \\{ ... \\}   $ ... $     inline formula
    \\[ ... \\]   $$ ... $$             displayed equation

The first delimiter found wins, so `\\{1, 2\\}` inside `\\( x \\in \\{1, 2\\} \\)` stays
part of the outer formula. A single `$` only counts when it hugs its content
(`$x$`, not `$5 and $10`), and `\\$` is a literal dollar sign.
"""

import re

# (opening, closing, displayed), longest first so "$$" wins over "$".
DELIMITERS = [("$$", "$$", True), ("\\[", "\\]", True), ("\\(", "\\)", False),
              ("\\{", "\\}", False), ("$", "$", False)]
# What may surround a section that is only a formula: "\[ x = 1 \]."
OUTER_RE = re.compile(r"^[\s.,;:]*$")


def _closing_dollar(text: str, start: int) -> int:
    """Index of the "$" closing an inline formula opened just before `start`, or -1."""
    if start >= len(text) or text[start].isspace():
        return -1
    k = start
    while (k := text.find("$", k)) != -1:
        before = text[k - 1]
        after = text[k + 1] if k + 1 < len(text) else ""
        if before != "\\" and not before.isspace() and not after.isdigit() and after != "$":
            return k
        k += 1
    return -1


def find_formulas(text: str) -> list[dict]:
    """Every formula in `text`, in order:
    {"start", "end", "latex", "display", "delimiter"} (end is exclusive)."""
    found, i, n = [], 0, len(text)
    while i < n:
        for opening, closing, display in DELIMITERS:
            if not text.startswith(opening, i):
                continue
            body = i + len(opening)
            end = (_closing_dollar(text, body) if opening == "$"
                   else text.find(closing, body))
            latex = text[body:end].strip() if end != -1 else ""
            if latex:
                found.append({"start": i, "end": end + len(closing), "latex": latex,
                              "display": display, "delimiter": opening + closing})
                i = end + len(closing)
                break
        else:
            # Skip an escaped character ("\$", "\\") so it can't open a formula.
            i += 2 if text[i] == "\\" else 1
    return found


def segments(text: str) -> list[dict] | None:
    """`text` split into text and formula parts, or None if it has no formula.
    A formula part is {"type": "formula", "content": <LaTeX>, "display": bool}."""
    found = find_formulas(text)
    if not found:
        return None
    parts, pos = [], 0
    for f in found:
        if text[pos:f["start"]].strip():
            parts.append({"type": "text", "content": text[pos:f["start"]].strip()})
        parts.append({"type": "formula", "content": f["latex"], "display": f["display"]})
        pos = f["end"]
    if text[pos:].strip():
        parts.append({"type": "text", "content": text[pos:].strip()})
    return parts


def whole_formula(text: str) -> str | None:
    """The LaTeX if `text` is a single formula and nothing else (bar
    punctuation), otherwise None."""
    found = find_formulas(text)
    if len(found) != 1:
        return None
    f = found[0]
    if OUTER_RE.match(text[:f["start"]]) and OUTER_RE.match(text[f["end"]:]):
        return f["latex"]
    return None


def bare_latex(text: str) -> str:
    """A formula section's content without its delimiters."""
    return whole_formula(text) or text.strip()
