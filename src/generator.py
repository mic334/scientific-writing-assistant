from __future__ import annotations

import re

from schemas import DraftOutput, DraftingContext


def _join_phrase(items: list[str], fallback: str) -> str:
    cleaned = [item.strip().rstrip(".") for item in items if item.strip()]
    if not cleaned:
        return fallback
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def generate_title(context: DraftingContext) -> str:
    user_input = context.user_input
    keywords = [keyword.strip() for keyword in user_input.keywords if keyword.strip()]
    method_hint = user_input.methods[0].strip() if user_input.methods else ""

    if keywords and method_hint:
        return f"{user_input.topic}: {keywords[0]} insights from {method_hint}"
    if keywords:
        return f"{user_input.topic}: {keywords[0]}-Guided Scientific Perspective"
    return f"{user_input.topic}: A Retrieval-Guided Scientific Draft"


def generate_abstract(context: DraftingContext) -> str:
    user_input = context.user_input
    rules = context.journal_rules

    findings = _join_phrase(user_input.main_findings, "the principal observations")
    methods = _join_phrase(user_input.methods, "the selected experimental and computational methods")
    keywords = _join_phrase(user_input.keywords, "the relevant scientific concepts")
    focus = str(rules.get("focus", "clear methods and results"))

    abstract = " ".join(
        [
            _sentence(
                f"This study addresses {user_input.topic}, with emphasis on {keywords}"
            ),
            _sentence(f"Using {methods}, the draft foregrounds these main results: {findings}"),
            _sentence(
                f"The resulting draft is framed for {user_input.journal}, prioritizing {focus}"
            ),
            _sentence(
                "Together, these elements support a coherent first abstract that can be inspected, revised, and expanded without attempting to generate a complete manuscript"
            ),
        ]
    )

    return _trim_words(abstract, int(rules.get("max_words", 220)))


def generate_outline(context: DraftingContext) -> list[str]:
    user_input = context.user_input
    methods = _join_phrase(user_input.methods, "core methods")
    findings = _join_phrase(user_input.main_findings, "main findings")

    return [
        f"1. Motivation and research gap: introduce {user_input.topic} and explain why the problem is scientifically relevant.",
        f"2. Study objective: state the central aim and connect it to {user_input.journal}.",
        f"3. Methods and materials: summarize {methods} at a level appropriate for an outline.",
        f"4. Key results: organize the draft around {findings}.",
        "5. Interpretation: explain how the findings advance the topic while staying close to the evidence.",
        "6. Journal-facing contribution: clarify novelty, scope, and limitations for the selected audience.",
    ]


def generate_draft(context: DraftingContext, prompt: str) -> DraftOutput:
    return DraftOutput(
        title=generate_title(context),
        abstract=generate_abstract(context),
        outline=generate_outline(context),
        retrieved_documents=[doc.filename for doc in context.retrieved_documents],
        prompt=prompt,
    )
