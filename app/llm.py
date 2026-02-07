from __future__ import annotations
import os
from langchain_groq import ChatGroq


def make_llm(model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it as an environment variable:\n"
            '  export GROQ_API_KEY="gsk_..."\n'
        )

    return ChatGroq(
        model=model,
        temperature=0.2,
        max_tokens=800,
    )
