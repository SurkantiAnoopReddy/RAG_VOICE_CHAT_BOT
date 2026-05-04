from __future__ import annotations

import logging
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path

import torch
try:
    from TTS.api import TTS
except ImportError as exc:
    TTS = None
    TTS_IMPORT_ERROR = exc
else:
    TTS_IMPORT_ERROR = None

try:
    import pyttsx3
except ImportError as exc:
    pyttsx3 = None
    PYTTSX3_IMPORT_ERROR = exc
else:
    PYTTSX3_IMPORT_ERROR = None

try:
    import pythoncom
except ImportError:
    pythoncom = None

from config import AUDIO_CACHE_DIR

LOGGER = logging.getLogger(__name__)


def configure_phonemizer_backend() -> None:
    library_path = os.getenv("PHONEMIZER_ESPEAK_LIBRARY", "").strip()
    legacy_path = os.getenv("PHONEMIZER_ESPEAK_PATH", "").strip()

    if not library_path and legacy_path:
        legacy_candidate = Path(legacy_path)
        if legacy_candidate.suffix.lower() == ".exe":
            guessed_library = legacy_candidate.with_name("libespeak-ng.dll")
            if guessed_library.exists():
                library_path = str(guessed_library)
        elif legacy_candidate.suffix.lower() == ".dll":
            library_path = str(legacy_candidate)

    if not library_path:
        return

    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        LOGGER.warning("phonemizer is not installed; eSpeak backend could not be configured.")
        return

    EspeakWrapper.set_library(library_path)
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = library_path
    LOGGER.info("Configured phonemizer eSpeak library: %s", library_path)


@lru_cache(maxsize=4)
def load_tts_model(model_name: str):
    if TTS is None:
        raise RuntimeError(
            "Coqui TTS could not be imported. Reinstall the pinned audio stack with "
            "`pip install -r requirements.txt`. If you are using PyTorch 2.9+ or newer, "
            "make sure `torchcodec` is installed as well."
        ) from TTS_IMPORT_ERROR

    configure_phonemizer_backend()
    LOGGER.info("Loading Coqui TTS model: %s", model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TTS(model_name=model_name, progress_bar=False).to(device)


class CoquiTTSService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def model(self):
        return load_tts_model(self.model_name)

    def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        output_path = output_path or AUDIO_CACHE_DIR / f"{uuid.uuid4()}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        speaker = None
        language = None

        if getattr(self.model, "speakers", None):
            speaker = self.model.speakers[0]
        if getattr(self.model, "languages", None):
            language = self.model.languages[0]

        LOGGER.info("Synthesizing answer audio to %s", output_path)
        try:
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker=speaker,
                language=language,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "No espeak backend found" in message:
                espeak_library = os.getenv("PHONEMIZER_ESPEAK_LIBRARY", "").strip()
                raise RuntimeError(
                    "Coqui TTS needs the phonemizer eSpeak backend for the current model. "
                    "Install `phonemizer`, install eSpeak NG on Windows, and set "
                    "`PHONEMIZER_ESPEAK_LIBRARY` in `.env` to the full path of `libespeak-ng.dll`."
                    + (
                        f" Current PHONEMIZER_ESPEAK_LIBRARY: {espeak_library}"
                        if espeak_library
                        else ""
                    )
                ) from exc
            raise
        return output_path


class Pyttsx3TTSService:
    def __init__(self) -> None:
        self.voice_name = os.getenv("PYTTSX3_VOICE_NAME", "").strip().lower()
        self.rate = int(os.getenv("PYTTSX3_RATE", "185"))

    def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        if sys.platform != "win32":
            raise RuntimeError("pyttsx3 fallback is currently configured only for Windows.")
        if pyttsx3 is None:
            raise RuntimeError(
                "pyttsx3 is not installed. Run `pip install -r requirements.txt` to enable the Windows TTS fallback."
            ) from PYTTSX3_IMPORT_ERROR

        output_path = output_path or AUDIO_CACHE_DIR / f"{uuid.uuid4()}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Synthesizing answer audio with pyttsx3 to %s", output_path)
        com_initialized = False
        engine = None
        try:
            if pythoncom is not None:
                pythoncom.CoInitialize()
                com_initialized = True

            engine = pyttsx3.init(driverName="sapi5")
            engine.setProperty("rate", self.rate)

            if self.voice_name:
                for voice in engine.getProperty("voices"):
                    voice_name = getattr(voice, "name", "").lower()
                    if self.voice_name in voice_name:
                        engine.setProperty("voice", voice.id)
                        break

            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
        finally:
            if engine is not None:
                engine.stop()
            if com_initialized and pythoncom is not None:
                pythoncom.CoUninitialize()

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("pyttsx3 did not create a usable audio file.")

        return output_path


class TextToSpeechService:
    def __init__(self, model_name: str, backend: str = "auto") -> None:
        self.model_name = model_name
        self.backend = backend
        self._coqui = CoquiTTSService(model_name=model_name)
        self._pyttsx3 = Pyttsx3TTSService()

    def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        if self.backend == "coqui":
            return self._coqui.synthesize(text, output_path=output_path)

        if self.backend == "pyttsx3":
            return self._pyttsx3.synthesize(text, output_path=output_path)

        if self.backend != "auto":
            raise ValueError("Unsupported TTS_BACKEND. Use `auto`, `coqui`, or `pyttsx3`.")

        try:
            return self._coqui.synthesize(text, output_path=output_path)
        except Exception as coqui_exc:
            LOGGER.warning("Coqui TTS failed in auto mode; falling back to pyttsx3. Error: %s", coqui_exc)
            return self._pyttsx3.synthesize(text, output_path=output_path)
