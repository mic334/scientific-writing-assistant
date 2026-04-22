# Scientific Writing Assistant

A local retrieval-based scientific drafting assistant for producing a first-pass:

- title
- abstract
- paper outline

The project is intentionally scoped as a drafting aid, not a full paper writer. It uses structured user input plus local reference documents written by the same author to retrieve stylistically and thematically relevant snippets. The MVP uses a deterministic rule-guided generator and also saves a structured prompt that can later be used with a local LLM.

## What It Does

1. Loads `.md`, `.txt`, and `.pdf` files from a local reference folder.
2. Cleans and preprocesses text.
3. Embeds documents with a local `sentence-transformers` model.
4. Builds a query from topic, findings, methods, keywords, and journal.
5. Retrieves the top-k most relevant reference documents using cosine similarity.
6. Builds a structured drafting context.
7. Generates a first draft of a title, abstract, and outline.
8. Saves outputs as JSON and Markdown.

## What It Does Not Do

- It does not call external APIs.
- It does not use OpenAI APIs.
- It does not fine-tune models.
- It does not generate a full scientific paper.
- It does not generate references, citations, or LaTeX formatting.
- It does not claim the output is publication-ready.

## Project Structure

```text
scientific-writing-assistant/
├── data/
│   └── references/
├── outputs/
├── src/
│   ├── loader.py
│   ├── preprocess.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── journal_adapter.py
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── schemas.py
│   └── main.py
├── requirements.txt
├── README.md
└── config.json
```

## Setup

Use Python 3.11 or newer.

```bash
cd scientific-writing-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model must be available locally or downloadable through Hugging Face when you first install/run it. After it is cached, the workflow can run locally.

## Add Reference Documents

Put the author's reference material in:

```text
data/references/
```

Supported file types:

- `.md`
- `.txt`
- `.pdf`

Empty files are skipped.

PDF support uses `pypdf` for local text extraction. Scanned/image-only PDFs are not OCRed in this MVP, so they may produce little or no text unless OCR text is already embedded in the file.

## Configure an Input

Edit `config.json`:

```json
{
  "journal": "Generic Scientific Journal",
  "topic": "photoinduced charge transfer in molecular materials",
  "main_findings": [
    "the system shows enhanced excited-state charge separation",
    "spectroscopic signatures indicate a long-lived intermediate state"
  ],
  "methods": [
    "steady-state spectroscopy",
    "time-resolved spectroscopy",
    "density functional theory"
  ],
  "keywords": [
    "charge transfer",
    "excited states",
    "molecular materials"
  ],
  "reference_docs_folder": "data/references",
  "top_k": 3
}
```

## Run

```bash
python src/main.py --config config.json
```

Outputs are saved to:

```text
outputs/draft_output.json
outputs/draft_output.md
```

You can also provide values from the command line:

```bash
python src/main.py \
  --journal "Angewandte Chemie" \
  --topic "photoinduced charge transfer in molecular materials" \
  --main-findings "enhanced excited-state charge separation; long-lived intermediate state" \
  --methods "time-resolved spectroscopy; density functional theory" \
  --keywords "charge transfer; excited states; molecular materials" \
  --references data/references \
  --top-k 3
```

## Supported Journal Rules

Current hardcoded journal profiles:

- `Chemistry - A European Journal`
- `Angewandte Chemie`
- `Journal of Physical Chemistry A`
- `Generic Scientific Journal`

To list them:

```bash
python src/main.py --list-journals
```

## Extension Points

- Add journal profiles in `src/journal_adapter.py`.
- Improve retrieval in `src/retriever.py`.
- Replace or extend template generation in `src/generator.py`.
- Add a local LLM backend later by consuming the structured prompt generated in `src/prompt_builder.py`.
