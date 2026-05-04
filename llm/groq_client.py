from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq


@lru_cache(maxsize=4)
def get_groq_llm(model_name: str, api_key: str, max_tokens: int = 512) -> ChatGroq:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.0,
        max_retries=2,
        max_tokens=max_tokens,
        timeout=60,
    )
