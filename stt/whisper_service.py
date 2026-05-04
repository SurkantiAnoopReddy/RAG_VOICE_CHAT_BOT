from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch
import whisper

LOGGER = logging.getLogger(__name__)


def _candidate_ffmpeg_paths() -> list[str]:
    candidates: list[str] = []

    env_path = os.getenv("FFMPEG_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    resolved_from_path = shutil.which("ffmpeg")
    if resolved_from_path:
        candidates.append(resolved_from_path)

    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        winget_packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages_dir.exists():
            candidates.extend(
                str(path)
                for path in winget_packages_dir.glob("**/ffmpeg.exe")
            )

    program_files = os.getenv("ProgramFiles", "").strip()
    if program_files:
        candidates.extend(
            [
                str(Path(program_files) / "ffmpeg" / "bin" / "ffmpeg.exe"),
                str(Path(program_files) / "FFmpeg" / "bin" / "ffmpeg.exe"),
            ]
        )

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        normalized = str(Path(candidate))
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def get_ffmpeg_executable() -> str | None:
    for candidate in _candidate_ffmpeg_paths():
        if Path(candidate).exists():
            return candidate
    return None


def ensure_ffmpeg_available() -> str:
    ffmpeg_executable = get_ffmpeg_executable()
    if not ffmpeg_executable:
        raise RuntimeError(
            "FFmpeg is required for Whisper audio decoding but was not found. "
            "Install FFmpeg, or set FFMPEG_PATH to the full path of ffmpeg.exe, then restart the app."
        )
    return ffmpeg_executable


def ensure_ffmpeg_on_process_path(ffmpeg_executable: str) -> None:
    ffmpeg_dir = str(Path(ffmpeg_executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir
        LOGGER.info("Prepended FFmpeg directory to PATH: %s", ffmpeg_dir)


@lru_cache(maxsize=4)
def load_whisper_model(model_size: str):
    LOGGER.info("Loading Whisper model: %s", model_size)
    return whisper.load_model(model_size)


class WhisperService:
    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size

    @property
    def model(self):
        return load_whisper_model(self.model_size)

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ffmpeg_executable = ensure_ffmpeg_available()
        ensure_ffmpeg_on_process_path(ffmpeg_executable)
        LOGGER.info("Transcribing audio file: %s", audio_path)
        LOGGER.info("Using FFmpeg executable: %s", ffmpeg_executable)
        try:
            result = self.model.transcribe(
                str(audio_path),
                fp16=torch.cuda.is_available(),
                language=language,
                task="transcribe",
                condition_on_previous_text=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Whisper could not start FFmpeg. Make sure FFmpeg is installed and available on your PATH, "
                "then restart Streamlit."
            ) from exc
        return (result.get("text") or "").strip()
