from __future__ import annotations

from schemas import DraftingContext, ReferenceDocument, UserInput


def build_query(user_input: UserInput) -> str:
    sections = [
        user_input.topic,
        " ".join(user_input.main_findings),
        " ".join(user_input.methods),
        " ".join(user_input.keywords),
        user_input.journal,
    ]
    return " ".join(section for section in sections if section).strip()


def make_reference_snippet(doc: ReferenceDocument, max_chars: int = 900) -> str:
    text = doc.cleaned_content or doc.content
    snippet = text[:max_chars].strip()
    if len(text) > max_chars:
        snippet += "..."

    score = ""
    if doc.similarity_score is not None:
        score = f" | similarity={doc.similarity_score:.3f}"

    return f"[{doc.filename}{score}]\n{snippet}"


def build_drafting_context(
    user_input: UserInput,
    journal_rules: dict[str, str | int],
    retrieved_documents: list[ReferenceDocument],
) -> DraftingContext:
    snippets = [make_reference_snippet(doc) for doc in retrieved_documents]
    return DraftingContext(
        user_input=user_input,
        journal_rules=journal_rules,
        retrieved_documents=retrieved_documents,
        reference_snippets=snippets,
    )


def build_structured_prompt(context: DraftingContext) -> str:
    user_input = context.user_input
    rules = context.journal_rules
    snippets = "\n\n".join(context.reference_snippets) or "No relevant snippets retrieved."

    return f"""You are a local retrieval-augmented scientific drafting assistant.

Task:
Generate only a first-pass title, abstract, and paper outline. Do not write a full paper, citations, references, or LaTeX formatting.

Target journal:
{user_input.journal}

Journal rules:
- Maximum abstract length: {rules.get("max_words", "not specified")} words
- Style: {rules.get("style", "neutral")}
- Focus: {rules.get("focus", "clear methods and results")}

User inputs:
- Topic: {user_input.topic}
- Main findings: {"; ".join(user_input.main_findings)}
- Methods: {"; ".join(user_input.methods)}
- Keywords: {", ".join(user_input.keywords)}

Retrieved style and theme references:
{snippets}

Output schema:
- title
- abstract
- outline
"""
