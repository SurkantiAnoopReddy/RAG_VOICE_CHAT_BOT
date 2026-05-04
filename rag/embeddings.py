from __future__ import annotations

from functools import lru_cache

import torch
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": get_embedding_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
