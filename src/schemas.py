from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReferenceDocument:
    filename: str
    content: str
    cleaned_content: str = ""
    embedding: Any | None = None
    similarity_score: float | None = None


@dataclass
class UserInput:
    journal: str
    topic: str
    main_findings: list[str]
    methods: list[str]
    keywords: list[str]
    reference_docs_folder: str
    writing_instructions: str = ""
    top_k: int = 3


@dataclass
class DraftOutput:
    title: str
    abstract: str
    outline: list[str]
    retrieved_documents: list[str]
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "outline": self.outline,
            "retrieved_documents": self.retrieved_documents,
            "prompt": self.prompt,
        }


@dataclass
class DraftingContext:
    user_input: UserInput
    journal_rules: dict[str, Any]
    retrieved_documents: list[ReferenceDocument] = field(default_factory=list)
    reference_snippets: list[str] = field(default_factory=list)
