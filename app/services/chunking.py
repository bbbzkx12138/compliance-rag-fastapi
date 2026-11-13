from __future__ import annotations

import re
from typing import Any


SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])")


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(
                text.rfind("。", start, end),
                text.rfind("；", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def split_section(
    section: dict[str, Any],
    *,
    max_chars: int = 500,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """Split long clauses while preserving headings and table-like rows."""

    text = str(section.get("text", "")).strip()
    if len(text) <= max_chars:
        return [
            {
                **section,
                "text": text,
                "part_index": 0,
            }
        ]

    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if len(blocks) == 1:
        blocks = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]

    pieces: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(_hard_split(block, max_chars, overlap))
            continue
        candidate = f"{current}\n{block}".strip() if current else block
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current.strip())
            current = block
    if current:
        pieces.append(current.strip())

    return [
        {
            **section,
            "heading": (
                section["heading"]
                if index == 0
                else f"{section['heading']}（片段 {index + 1}）"
            ),
            "text": piece,
            "part_index": index,
        }
        for index, piece in enumerate(pieces)
    ]
