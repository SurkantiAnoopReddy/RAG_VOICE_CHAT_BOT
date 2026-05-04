# Voice RAG Assistant

Production-ready Streamlit Voice RAG application with:

- PDF ingestion via PyMuPDF
- Local speech-to-text via Whisper
- FAISS retrieval with HuggingFace embeddings
- Mandatory Cohere reranking
- Groq LLM generation via LangChain
- Local text-to-speech via Coqui TTS

## Project Structure

```text
project/
├── app.py
├── config.py
├── llm/
│   └── groq_client.py
├── rag/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── reranker.py
│   └── qa_chain.py
├── stt/
│   └── whisper_service.py
├── tts/
│   └── coqui_service.py
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Recommended Python version:

- Python `3.10` to `3.14`
- The project uses the maintained `coqui-tts` package. Do not install the legacy `TTS` PyPI package on Python 3.14.
- Coqui TTS is pinned to `transformers<5.0.0` because newer `transformers` releases break Coqui imports.
- Cohere reranking must use a valid model ID such as `rerank-v4.0-fast`, `rerank-v4.0-pro`, or `rerank-v3.5`. `rerank-v4.0` is not a valid model ID.
- With PyTorch 2.9+ on Windows, Coqui TTS also needs `torchcodec` for audio I/O.
- On Windows, the app can fall back to `pyttsx3` with native SAPI voices if Coqui is unavailable.

3. Install FFmpeg and make sure it is available on your `PATH`.
4. Copy `.env.example` to `.env` and fill in:

```env
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

## Run

```powershell
streamlit run app.py
```

## FFmpeg On Windows

Whisper needs `ffmpeg.exe` to read microphone uploads and audio files.

1. Install FFmpeg:

```powershell
winget install Gyan.FFmpeg
```

2. Close and reopen your terminal or IDE.
3. Verify it is available:

```powershell
ffmpeg -version
```

If `ffmpeg` is not found, add the FFmpeg `bin` folder to your Windows `PATH` and restart Streamlit.

## If You Already Installed `transformers` 5.x

Repair the environment with:

```powershell
python -m pip uninstall -y transformers
python -m pip install -r requirements.txt
```

## Notes

- The first run downloads local Whisper, embedding, tokenizer, and Coqui model weights.
- Reprocessing the same PDF set reuses the cached FAISS index from `data/vector_cache/`.
- Logs are written to `logs/app.log`.
