from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import VECTOR_CACHE_DIR
from rag.embeddings import get_embedding_model

LOGGER = logging.getLogger(__name__)


def get_vectorstore_path(cache_key: str) -> Path:
    return VECTOR_CACHE_DIR / cache_key


def vectorstore_exists(cache_key: str) -> bool:
    base_path = get_vectorstore_path(cache_key)
    return (base_path / "index.faiss").exists() and (base_path / "index.pkl").exists()


def write_metadata(
    cache_key: str,
    documents: list[Document],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    metadata_path = get_vectorstore_path(cache_key) / "metadata.json"
    payload: dict[str, Any] = {
        "chunk_count": len(documents),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sorted(
            {
                document.metadata.get("source", "unknown")
                for document in documents
            }
        ),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    metadata_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def read_metadata(cache_key: str) -> dict[str, Any] | None:
    metadata_path = get_vectorstore_path(cache_key) / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_vectorstore(cache_key: str, embedding_model_name: str) -> FAISS:
    if not vectorstore_exists(cache_key):
        raise FileNotFoundError(f"No cached FAISS index found for collection: {cache_key}")

    embeddings = get_embedding_model(embedding_model_name)
    path = get_vectorstore_path(cache_key)
    LOGGER.info("Loading cached vectorstore from %s", path)
    return FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def build_or_load_vectorstore(
    documents: list[Document],
    cache_key: str,
    embedding_model_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> FAISS:
    if not documents:
        raise ValueError("Cannot build a vectorstore without documents.")

    if vectorstore_exists(cache_key):
        return load_vectorstore(cache_key, embedding_model_name)

    embeddings = get_embedding_model(embedding_model_name)
    LOGGER.info("Building new vectorstore for collection %s", cache_key)
    vectorstore = FAISS.from_documents(documents, embeddings)

    target_dir = get_vectorstore_path(cache_key)
    target_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target_dir))
    write_metadata(cache_key, documents, extra_metadata=extra_metadata)

    return vectorstore
