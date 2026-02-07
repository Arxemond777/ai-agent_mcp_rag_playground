import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def make_llm(model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it as an environment variable:\n"
            '  export GROQ_API_KEY="gsk_..."\n'
        )

    # ChatGroq reads GROQ_API_KEY from env by default, but we validate above so failures are obvious.
    return ChatGroq(
        model=model,
        temperature=0.2,
        max_tokens=512,
    )


def main() -> None:
    llm = make_llm(model="llama-3.3-70b-versatile")

    messages = [
        SystemMessage(content="You are a seasoned ML/AI engineer. Answer briefly."),
        HumanMessage(content="Answer briefly in 3 items about RAG."),
    ]

    r1 = llm.invoke(messages)
    print("A1:", r1.content)
    messages.append(AIMessage(content=r1.content))

    messages.append(HumanMessage(content="Now give an example of the architecture on FastAPI."))
    r2 = llm.invoke(messages)
    print("A2:", r2.content)


if __name__ == "__main__":
    main()
