from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_EMPTY_ANSWER = "The answer is not available in the provided documents."

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}\n\n"
            "Use the following context excerpts to answer the question.\n"
            "If the context is insufficient, reply exactly with:\n"
            f"{DEFAULT_EMPTY_ANSWER}\n\n"
            "Context:\n{context}",
        ),
        ("human", "Question: {question}"),
    ]
)


def format_context(documents: list[Document]) -> str:
    if not documents:
        return ""

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "?")
        sections.append(f"[{index}] Source: {source} | page {page}\n{document.page_content}")
    return "\n\n".join(sections)


def generate_answer(
    question: str,
    context_documents: list[Document],
    llm: BaseChatModel,
    system_prompt: str,
) -> str:
    if not context_documents:
        return DEFAULT_EMPTY_ANSWER

    chain = QA_PROMPT | llm | StrOutputParser()
    response = chain.invoke(
        {
            "system_prompt": system_prompt,
            "context": format_context(context_documents),
            "question": question,
        }
    )
    return response.strip() or DEFAULT_EMPTY_ANSWER
