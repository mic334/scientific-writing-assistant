from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from schemas import ReferenceDocument


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_documents(
    docs: list[ReferenceDocument],
    model: SentenceTransformer,
) -> list[ReferenceDocument]:
    texts = [doc.cleaned_content or doc.content for doc in docs]
    if not texts:
        return docs

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    for doc, embedding in zip(docs, embeddings, strict=True):
        doc.embedding = np.asarray(embedding)

    return docs


def embed_query(text: str, model: SentenceTransformer) -> np.ndarray:
    return np.asarray(model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0])
