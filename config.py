from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_CACHE_DIR = DATA_DIR / "vector_cache"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
TMP_DIR = DATA_DIR / "tmp"
LOG_DIR = BASE_DIR / "logs"

for directory in (DATA_DIR, UPLOAD_DIR, VECTOR_CACHE_DIR, AUDIO_CACHE_DIR, TMP_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are a precise and reliable AI assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT hallucinate.
- If answer is not present, say:
  "The answer is not available in the provided documents."
- Keep answers concise and structured.
"""

SUPPORTED_AUDIO_EXTENSIONS = ("wav", "mp3", "m4a", "mp4", "mpeg", "ogg", "webm", "flac")
SUPPORTED_PDF_EXTENSIONS = ("pdf",)


def discover_ffmpeg_executable() -> str | None:
    candidates: list[Path] = []

    env_path = os.getenv("FFMPEG_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        winget_packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages_dir.exists():
            candidates.extend(winget_packages_dir.glob("**/ffmpeg.exe"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return None


def bootstrap_ffmpeg_path() -> None:
    ffmpeg_executable = discover_ffmpeg_executable()
    if not ffmpeg_executable:
        return

    ffmpeg_dir = str(Path(ffmpeg_executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir


bootstrap_ffmpeg_path()


def configure_logging() -> None:
    if getattr(configure_logging, "_configured", False):
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ]

    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    configure_logging._configured = True


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    cohere_api_key: str
    groq_model: str
    cohere_rerank_model: str
    embedding_model_name: str
    whisper_model_size: str
    coqui_model_name: str
    tts_backend: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    rerank_top_n: int
    max_answer_tokens: int
    system_prompt: str = SYSTEM_PROMPT

    def missing_runtime_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return AppConfig(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        cohere_api_key=os.getenv("COHERE_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
        cohere_rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast").strip(),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base").strip(),
        coqui_model_name=os.getenv("COQUI_MODEL_NAME", "tts_models/en/vctk/vits").strip(),
        tts_backend=os.getenv("TTS_BACKEND", "auto").strip().lower(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "3")),
        max_answer_tokens=int(os.getenv("MAX_ANSWER_TOKENS", "512")),
    )
