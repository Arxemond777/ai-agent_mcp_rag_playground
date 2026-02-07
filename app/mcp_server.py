from __future__ import annotations
import os
from typing import Any, Dict, List

from langchain_core.tools import tool

from app.rag import RAGStore

KB_ROOT = os.environ.get("KB_ROOT", "kb")

_store = RAGStore()


@tool("kb_index")
def kb_index() -> Dict[str, Any]:
    """Index all files in ./kb into vector store."""
    n = _store.index_folder(KB_ROOT)
    return {"indexed_chunks": n, "kb_root": KB_ROOT}


@tool("kb_search")
def kb_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search over KB. Returns top-k chunks with:
    id (chunk-id), source (relative path), text, distance
    """
    return _store.search(query, k=k)


@tool("read_file")
def read_file(path: str) -> Dict[str, Any]:
    """Read a file from KB by relative path."""
    safe_rel = os.path.normpath(path).lstrip("/\\")
    abs_root = os.path.abspath(KB_ROOT)
    abs_full = os.path.abspath(os.path.join(abs_root, safe_rel))

    if not (abs_full == abs_root or abs_full.startswith(abs_root + os.sep)):
        return {"error": "path_outside_kb"}

    if not os.path.exists(abs_full):
        return {"error": "not_found"}

    with open(abs_full, "r", encoding="utf-8", errors="ignore") as f:
        return {"path": safe_rel.replace("\\", "/"), "content": f.read()}
