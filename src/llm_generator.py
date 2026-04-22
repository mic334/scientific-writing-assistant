from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from schemas import DraftOutput, DraftingContext


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("The local LLM did not return JSON.")

    return json.loads(match.group(0))


def _validate_output(data: dict) -> tuple[str, str, list[str]]:
    title = str(data.get("title", "")).strip()
    abstract = str(data.get("abstract", "")).strip()
    outline_value = data.get("outline", [])

    if isinstance(outline_value, str):
        outline = [line.strip("- ").strip() for line in outline_value.splitlines() if line.strip()]
    elif isinstance(outline_value, list):
        outline = [str(item).strip() for item in outline_value if str(item).strip()]
    else:
        outline = []

    if not title or not abstract or not outline:
        raise ValueError("The local LLM response is missing title, abstract, or outline.")

    return title, abstract, outline


def _build_llama_prompt(context: DraftingContext, prompt: str) -> str:
    max_words = context.journal_rules.get("max_words", 220)
    return f"""Use the following retrieval context to draft a scientific title, abstract, and outline.

Important constraints:
- Return valid JSON only.
- Use exactly these keys: title, abstract, outline.
- outline must be a list of strings.
- Do not write a full paper.
- Do not invent references or citations.
- Keep the abstract under {max_words} words.
- Use the retrieved documents only as local style and thematic references.

Retrieval context:
{prompt}
"""


def generate_draft_with_ollama(
    context: DraftingContext,
    prompt: str,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 180,
) -> DraftOutput:
    request_body = {
        "model": model,
        "prompt": _build_llama_prompt(context, prompt),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 900,
        },
    }
    request_data = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach local Ollama server at {ollama_url}. "
            "Start Ollama locally and make sure the selected Llama model is installed."
        ) from exc

    llm_text = str(response_data.get("response", "")).strip()
    title, abstract, outline = _validate_output(_extract_json(llm_text))

    return DraftOutput(
        title=title,
        abstract=abstract,
        outline=outline,
        retrieved_documents=[doc.filename for doc in context.retrieved_documents],
        prompt=prompt,
    )
