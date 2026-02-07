from __future__ import annotations
from typing import Any, Dict

from app.rag import RAGStore


def answer_with_citations(store: RAGStore, question: str, k: int = 5) -> Dict[str, Any]:
    hits = store.search(question, k=k)
    if not hits:
        return {"answer": "Nothing found in the knowledge base", "citations": []}

    citations = [{"source": h["source"], "id": h["id"], "distance": h["distance"]} for h in hits]
    context = "\n\n---\n\n".join(
        [f"[{i + 1}] source={h['source']} id={h['id']} dist={h['distance']:.4f}\n{h['text']}"
         for i, h in enumerate(hits)]
    )

    answer = (
            "Found relevant fragments. Below is the context (with sources). "
            "If you want a coherent final answer, connect an LLM on top of this.\n\n"
            + context
    )
    return {"answer": answer, "citations": citations}
