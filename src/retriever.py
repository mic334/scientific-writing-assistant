from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from schemas import ReferenceDocument


def retrieve_top_k(
    docs: list[ReferenceDocument],
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> list[ReferenceDocument]:
    embedded_docs = [doc for doc in docs if doc.embedding is not None]
    if not embedded_docs:
        return []

    document_matrix = np.vstack([doc.embedding for doc in embedded_docs])
    scores = cosine_similarity(query_embedding.reshape(1, -1), document_matrix)[0]
    ranked_indices = np.argsort(scores)[::-1][: max(top_k, 0)]

    results: list[ReferenceDocument] = []
    for index in ranked_indices:
        doc = embedded_docs[int(index)]
        doc.similarity_score = float(scores[int(index)])
        results.append(doc)

    return results
