from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    doc_id: str
    source: str
    text: str
    chunk_id: str


def iter_text_files(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".md", ".txt", ".py", ".go", ".js", ".ts", ".json", ".yaml", ".yml")):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        yield path, f.read()
                except Exception:
                    continue


def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def _validate_collection_name(name: str) -> str:
    """
    Chroma collection name rules (simplified, but matching your crash):
    - length: 3..512
    - allowed chars: [a-zA-Z0-9._-]
    - must start and end with [a-zA-Z0-9]
    """
    if not isinstance(name, str):
        raise ValueError("collection name must be a string")

    name = name.strip()
    if len(name) < 3 or len(name) > 512:
        raise ValueError(f"Invalid Chroma collection name length: {len(name)} (must be 3..512). Got: {name!r}")

    if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?", name):
        raise ValueError(
            "Invalid Chroma collection name. "
            "Allowed: [a-zA-Z0-9._-], must start/end with alnum. "
            f"Got: {name!r}"
        )

    if ".." in name:
        raise ValueError(f"Invalid Chroma collection name: must not contain '..'. Got: {name!r}")

    return name


class RAGStore:
    def __init__(
            self,
            persist_dir: str = "data/chroma",
            collection: Optional[str] = None,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        os.makedirs(persist_dir, exist_ok=True)

        # allow override via env without editing code
        if collection is None:
            collection = os.environ.get("CHROMA_COLLECTION", "kb_store")
        collection = _validate_collection_name(collection)

        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=collection)

    def index_folder(self, folder: str) -> int:
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for path, content in iter_text_files(folder):
            doc_id = os.path.relpath(path, folder).replace("\\", "/")
            parts = simple_chunk(content)
            for k, part in enumerate(parts):
                chunk_id = f"{doc_id}::chunk::{k}"
                ids.append(chunk_id)
                docs.append(part)
                metas.append({"source": doc_id})

        if not ids:
            return 0

        embs = self.model.encode(docs, normalize_embeddings=True).tolist()
        self.col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return len(ids)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        out: List[Dict[str, Any]] = []
        for i in range(len(res["ids"][0])):
            out.append(
                {
                    "id": res["ids"][0][i],
                    "source": (res["metadatas"][0][i] or {}).get("source"),
                    "text": res["documents"][0][i],
                    "distance": res["distances"][0][i],
                }
            )
        return out
