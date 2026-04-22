from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from generator import generate_draft
from journal_adapter import get_journal_rules, list_supported_journals
from loader import load_reference_documents
from preprocess import clean_text
from prompt_builder import build_drafting_context, build_query, build_structured_prompt
from schemas import DraftOutput, UserInput


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def load_user_input_from_config(config_path: Path) -> UserInput:
    config_path = config_path.expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    reference_folder = Path(data.get("reference_docs_folder", "data/references")).expanduser()
    if not reference_folder.is_absolute():
        reference_folder = config_path.parent / reference_folder

    return UserInput(
        journal=data["journal"],
        topic=data["topic"],
        main_findings=list(data.get("main_findings", [])),
        methods=list(data.get("methods", [])),
        keywords=list(data.get("keywords", [])),
        reference_docs_folder=str(reference_folder),
        top_k=int(data.get("top_k", 3)),
    )


def save_json(output: DraftOutput, path: Path) -> None:
    path.write_text(json.dumps(output.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown(output: DraftOutput, path: Path) -> None:
    outline = "\n".join(f"- {item}" for item in output.outline)
    retrieved = "\n".join(f"- {name}" for name in output.retrieved_documents) or "- None"
    markdown = f"""# Scientific Draft

## Title
{output.title}

## Abstract
{output.abstract}

## Paper Outline
{outline}

## Retrieved Documents
{retrieved}

## Structured Prompt for a Future Local LLM
```text
{output.prompt}
```
"""
    path.write_text(markdown, encoding="utf-8")


def run_pipeline(user_input: UserInput, output_dir: Path, model_name: str) -> DraftOutput:
    from embedder import embed_documents, embed_query, load_embedding_model
    from retriever import retrieve_top_k

    docs = load_reference_documents(user_input.reference_docs_folder)
    for doc in docs:
        doc.cleaned_content = clean_text(doc.content)

    model = load_embedding_model(model_name)
    embed_documents(docs, model)

    query = build_query(user_input)
    query_embedding = embed_query(query, model)
    retrieved_docs = retrieve_top_k(docs, query_embedding, user_input.top_k)

    journal_rules = get_journal_rules(user_input.journal)
    context = build_drafting_context(user_input, journal_rules, retrieved_docs)
    prompt = build_structured_prompt(context)
    output = generate_draft(context, prompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output, output_dir / "draft_output.json")
    save_markdown(output, output_dir / "draft_output.md")

    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local retrieval-based scientific title, abstract, and outline assistant."
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--list-journals", action="store_true")
    parser.add_argument("--journal")
    parser.add_argument("--topic")
    parser.add_argument("--main-findings", help="Semicolon-separated list.")
    parser.add_argument("--methods", help="Semicolon-separated list.")
    parser.add_argument("--keywords", help="Semicolon-separated list.")
    parser.add_argument("--references", help="Folder containing .md and .txt reference documents.")
    parser.add_argument("--top-k", type=int)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_journals:
        print("\n".join(list_supported_journals()))
        return

    if args.topic:
        user_input = UserInput(
            journal=args.journal or "Generic Scientific Journal",
            topic=args.topic,
            main_findings=parse_list(args.main_findings or ""),
            methods=parse_list(args.methods or ""),
            keywords=parse_list(args.keywords or ""),
            reference_docs_folder=args.references or "data/references",
            top_k=args.top_k or 3,
        )
    else:
        user_input = load_user_input_from_config(args.config)

    output = run_pipeline(user_input, args.output_dir, args.model)
    print(json.dumps({"saved": str(args.output_dir), **asdict(output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
