from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def retrieve_documents(vectorstore: FAISS, query: str, top_k: int) -> list[Document]:
    if not query.strip():
        raise ValueError("Query cannot be empty for retrieval.")

    results = vectorstore.similarity_search_with_score(query, k=top_k)
    documents: list[Document] = []

    for document, score in results:
        documents.append(
            Document(
                page_content=document.page_content,
                metadata={**document.metadata, "retrieval_distance": float(score)},
            )
        )

    return documents
