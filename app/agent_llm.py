from __future__ import annotations

from typing import Any, Dict

from langchain.agents import create_agent
from app.llm import make_llm
from app.mcp_server import kb_search, read_file, kb_index


SYSTEM_PROMPT = """You are an AI agent with tools.

Goals:
- Answer the user's question using the knowledge base.
- Use tools when needed (kb_search, read_file, kb_index).
- Produce a coherent final answer.
- Add citations using chunk ids returned by kb_search.

Rules:
- If the KB may be empty, call kb_index first.
- Prefer kb_search first. Use read_file only when you need more context from a specific file.
- In the final answer, include citations inline like: (cite: <chunk_id>).
- Never invent citations. Only use ids you have actually seen in tool output.
"""


def answer_with_tools(question: str, k: int = 5) -> Dict[str, Any]:
    llm = make_llm()
    tools = [kb_index, kb_search, read_file]

    # create_agent builds a graph-based agent runtime (LangGraph under the hood)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # v1 agents are invoked with a messages payload
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # Result shape can vary; keep extraction simple and explicit
    # Commonly result contains "messages" with the final assistant message at the end
    final_answer = ""
    messages = result.get("messages") if isinstance(result, dict) else None
    if isinstance(messages, list) and messages:
        last = messages[-1]
        # last can be dict-like or a message object depending on runtime
        final_answer = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else "")
    else:
        final_answer = str(result)

    # Best-effort citations from top-k retrieval for the question
    hits = kb_search.invoke({"query": question, "k": k})
    citations = [{"id": h["id"], "source": h.get("source")} for h in hits]

    return {"answer": final_answer, "citations": citations}
