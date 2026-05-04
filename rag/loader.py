from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import fitz
from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)


def normalize_lines(page_text: str) -> list[str]:
    return [line.strip() for line in page_text.splitlines() if line.strip()]


def detect_repeated_boundary_lines(page_lines: list[list[str]]) -> set[str]:
    first_lines = Counter()
    last_lines = Counter()

    for lines in page_lines:
        if lines:
            first_lines[lines[0]] += 1
            last_lines[lines[-1]] += 1

    threshold = max(2, len(page_lines) // 2) if page_lines else 2
    repeated_lines = {
        line
        for line, count in (first_lines + last_lines).items()
        if count >= threshold and len(line) < 120
    }
    return repeated_lines


def is_noise_line(line: str, repeated_lines: set[str]) -> bool:
    normalized = line.strip()
    if not normalized:
        return True
    if normalized in repeated_lines:
        return True
    if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", normalized, flags=re.IGNORECASE):
        return True
    return False


def finalize_text(lines: list[str]) -> str:
    if not lines:
        return ""

    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        LOGGER.info("Loading PDF: %s", pdf_path)
        with fitz.open(pdf_path) as pdf:
            raw_pages = [page.get_text("text") or "" for page in pdf]

        page_lines = [normalize_lines(page_text) for page_text in raw_pages]
        repeated_lines = detect_repeated_boundary_lines(page_lines)

        for page_number, lines in enumerate(page_lines, start=1):
            cleaned_lines = [line for line in lines if not is_noise_line(line, repeated_lines)]
            cleaned_text = finalize_text(cleaned_lines)
            if len(cleaned_text) < 40:
                continue

            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                        "path": str(pdf_path.resolve()),
                    },
                )
            )

    LOGGER.info("Loaded %s cleaned PDF pages.", len(documents))
    return documents
