from __future__ import annotations

import re


def clean_text(text: str, max_chars: int = 12000) -> str:
    """Normalize whitespace while keeping text readable for inspection."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = text.strip()

    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "\n\n[Document truncated for MVP processing.]"

    return cleaned
