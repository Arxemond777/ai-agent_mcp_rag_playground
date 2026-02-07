from __future__ import annotations
import sys

from app.rag import RAGStore
from app.agent import answer_with_citations
from app.agent_llm import answer_with_tools


def main():
    if len(sys.argv) < 2:
        print("Usage:\n  python -m app.run index\n  python -m app.run ask 'your question'\n  python -m app.run ask_llm 'your question'")
        return

    cmd = sys.argv[1]
    store = RAGStore()

    if cmd == "index":
        n = store.index_folder("kb")
        print(f"Indexed chunks: {n}")
        return

    if cmd == "ask":
        q = " ".join(sys.argv[2:]).strip()
        out = answer_with_citations(store, q)
        print(out["answer"])
        return

    if cmd == "ask_llm":
        q = " ".join(sys.argv[2:]).strip()
        out = answer_with_tools(q)
        print(out["answer"])
        print("\nCITATIONS:")
        for c in out["citations"]:
            print(f"- {c['id']} ({c['source']})")
        return

    print("Unknown command")


if __name__ == "__main__":
    main()
