from __future__ import annotations

from dataclasses import dataclass, field

import cohere
from langchain_core.documents import Document


@dataclass
class CohereReranker:
    api_key: str
    model_name: str
    _client: cohere.ClientV2 | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> cohere.ClientV2:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "COHERE_API_KEY is missing. Add it to your environment or .env file."
                )
            self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        if not query.strip():
            raise ValueError("Query cannot be empty for reranking.")
        if not documents:
            return []

        rerank_response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=[{"text": document.page_content} for document in documents],
            top_n=min(top_n, len(documents)),
        )

        reranked_documents: list[Document] = []
        for rank, result in enumerate(rerank_response.results, start=1):
            source_document = documents[result.index]
            reranked_documents.append(
                Document(
                    page_content=source_document.page_content,
                    metadata={
                        **source_document.metadata,
                        "rerank_rank": rank,
                        "rerank_score": float(result.relevance_score),
                    },
                )
            )

        return reranked_documents
