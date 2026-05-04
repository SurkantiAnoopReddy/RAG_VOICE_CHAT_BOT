from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def chunk_documents(
    documents: list[Document],
    tokenizer_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if not documents:
        return []

    tokenizer = get_tokenizer(tokenizer_model_name)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
