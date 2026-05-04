from __future__ import annotations

import base64
import html
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Iterable

import streamlit as st

from config import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    TMP_DIR,
    UPLOAD_DIR,
    configure_logging,
    get_settings,
)
from llm.groq_client import get_groq_llm
from rag.chunking import chunk_documents
from rag.loader import load_pdf_documents
from rag.qa_chain import generate_answer
from rag.reranker import CohereReranker
from rag.retriever import retrieve_documents
from rag.vectorstore import (
    build_or_load_vectorstore,
    load_vectorstore,
    read_metadata,
    vectorstore_exists,
)
from stt.whisper_service import WhisperService
from tts.coqui_service import TextToSpeechService

configure_logging()
LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()


def initialize_session_state() -> None:
    defaults = {
        "vectorstore_key": None,
        "vectorstore": None,
        "document_stats": None,
        "last_query": "",
        "last_answer": "",
        "last_audio_path": None,
        "last_audio_token": "",
        "last_sources": [],
        "last_tts_warning": "",
        "autoplayed_audio_token": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_stt_service(model_size: str) -> WhisperService:
    return WhisperService(model_size=model_size)


@st.cache_resource(show_spinner=False)
def get_tts_service(model_name: str, backend: str) -> TextToSpeechService:
    return TextToSpeechService(model_name=model_name, backend=backend)


@st.cache_resource(show_spinner=False)
def get_reranker(api_key: str, model_name: str) -> CohereReranker:
    return CohereReranker(api_key=api_key, model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_cached_vectorstore(collection_id: str, embedding_model_name: str):
    return load_vectorstore(collection_id, embedding_model_name)


def fingerprint_files(uploaded_files: Iterable[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    digest = hashlib.sha256()
    for uploaded_file in sorted(uploaded_files, key=lambda item: item.name):
        payload = uploaded_file.getvalue()
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()


def persist_uploaded_pdfs(uploaded_files, collection_id: str) -> list[Path]:
    target_dir = UPLOAD_DIR / collection_id
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix.lstrip(".") not in SUPPORTED_PDF_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")

        target_path = target_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(target_path)

    return saved_paths


def save_audio_input(audio_file) -> Path:
    original_name = getattr(audio_file, "name", "voice_query.wav")
    suffix = Path(original_name).suffix.lower() or ".wav"
    if suffix.lstrip(".") not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError("Unsupported audio format. Upload WAV, MP3, M4A, OGG, FLAC, or WEBM.")

    audio_path = TMP_DIR / f"{uuid.uuid4()}{suffix}"
    audio_path.write_bytes(audio_file.getvalue())
    return audio_path


def get_active_vectorstore():
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is not None:
        return vectorstore

    collection_id = st.session_state.get("vectorstore_key")
    if not collection_id:
        return None

    vectorstore = get_cached_vectorstore(collection_id, SETTINGS.embedding_model_name)
    st.session_state.vectorstore = vectorstore
    return vectorstore


def process_documents(uploaded_files) -> None:
    collection_id = fingerprint_files(uploaded_files)
    saved_pdf_paths = persist_uploaded_pdfs(uploaded_files, collection_id)

    if vectorstore_exists(collection_id):
        st.session_state.vectorstore_key = collection_id
        st.session_state.vectorstore = get_cached_vectorstore(collection_id, SETTINGS.embedding_model_name)
        cached_metadata = read_metadata(collection_id) or {}
        st.session_state.document_stats = {
            "files": cached_metadata.get("file_count", len(saved_pdf_paths)),
            "pages": cached_metadata.get("page_count", 0),
            "chunks": cached_metadata.get("chunk_count", 0),
            "cached": True,
        }
        return

    raw_documents = load_pdf_documents(saved_pdf_paths)
    if not raw_documents:
        raise ValueError("No extractable text was found in the uploaded PDF documents.")

    chunked_documents = chunk_documents(
        documents=raw_documents,
        tokenizer_model_name=SETTINGS.embedding_model_name,
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    if not chunked_documents:
        raise ValueError("Document chunking produced no usable chunks.")

    vectorstore = build_or_load_vectorstore(
        documents=chunked_documents,
        cache_key=collection_id,
        embedding_model_name=SETTINGS.embedding_model_name,
        extra_metadata={
            "file_count": len(saved_pdf_paths),
            "page_count": len(raw_documents),
        },
    )

    st.session_state.vectorstore_key = collection_id
    st.session_state.vectorstore = vectorstore
    st.session_state.document_stats = {
        "files": len(saved_pdf_paths),
        "pages": len(raw_documents),
        "chunks": len(chunked_documents),
        "cached": False,
    }


def run_voice_rag(audio_file) -> None:
    missing_keys = SETTINGS.missing_runtime_keys()
    if missing_keys:
        missing_keys_text = ", ".join(missing_keys)
        raise RuntimeError(
            f"Missing required environment variables: {missing_keys_text}. "
            "Add them to your .env file before asking questions."
        )

    vectorstore = get_active_vectorstore()
    if vectorstore is None:
        raise RuntimeError("Process at least one PDF before asking a question.")

    audio_path = save_audio_input(audio_file)
    stt_service = get_stt_service(SETTINGS.whisper_model_size)
    transcription = stt_service.transcribe(audio_path)
    if not transcription:
        raise ValueError("Whisper did not detect any speech in the provided audio.")

    candidate_documents = retrieve_documents(
        vectorstore=vectorstore,
        query=transcription,
        top_k=SETTINGS.retrieval_top_k,
    )
    reranked_documents = get_reranker(
        SETTINGS.cohere_api_key,
        SETTINGS.cohere_rerank_model,
    ).rerank(
        query=transcription,
        documents=candidate_documents,
        top_n=SETTINGS.rerank_top_n,
    )

    llm = get_groq_llm(
        model_name=SETTINGS.groq_model,
        api_key=SETTINGS.groq_api_key,
        max_tokens=SETTINGS.max_answer_tokens,
    )
    answer = generate_answer(
        question=transcription,
        context_documents=reranked_documents,
        llm=llm,
        system_prompt=SETTINGS.system_prompt,
    )

    audio_response_path = None
    tts_warning = ""
    try:
        audio_response_path = get_tts_service(
            SETTINGS.coqui_model_name,
            SETTINGS.tts_backend,
        ).synthesize(answer)
    except Exception as exc:
        LOGGER.warning("TTS generation failed; returning text-only answer. Error: %s", exc)
        tts_warning = "Text answer generated successfully, but audio playback is unavailable in this environment."

    st.session_state.last_query = transcription
    st.session_state.last_answer = answer
    st.session_state.last_audio_path = str(audio_response_path) if audio_response_path else None
    st.session_state.last_audio_token = str(uuid.uuid4()) if audio_response_path else ""
    st.session_state.last_sources = reranked_documents
    st.session_state.last_tts_warning = tts_warning


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 12% 12%, rgba(74, 222, 128, 0.10), transparent 24%),
                radial-gradient(circle at 88% 16%, rgba(56, 189, 248, 0.14), transparent 24%),
                radial-gradient(circle at 50% 85%, rgba(59, 130, 246, 0.12), transparent 30%),
                linear-gradient(180deg, #031127 0%, #071a38 38%, #0a2247 100%);
            color: #e5eefb;
            font-family: "DM Sans", sans-serif;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.7rem;
            padding-bottom: 3.4rem;
        }
        @keyframes floatIn {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.25); }
            70% { box-shadow: 0 0 0 16px rgba(56, 189, 248, 0); }
            100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }
        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }
        .top-ribbon {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 196, 255, 0.10);
            background: rgba(4, 17, 39, 0.62);
            backdrop-filter: blur(14px);
            animation: floatIn 0.55s ease;
        }
        .top-ribbon .brand {
            color: #eff6ff;
            font-family: "Space Grotesk", sans-serif;
            font-weight: 700;
            font-size: 0.98rem;
        }
        .top-ribbon .sub {
            color: #8fb3d9;
            font-size: 0.86rem;
        }
        .top-ribbon .live-dot {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: #dbeafe;
            font-size: 0.84rem;
        }
        .top-ribbon .live-dot::before {
            content: "";
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: #38bdf8;
            animation: pulseGlow 1.8s infinite;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 2.35rem 2.45rem 2.2rem;
            color: #f8fafc;
            background:
                linear-gradient(135deg, rgba(8, 24, 57, 0.98) 0%, rgba(9, 38, 81, 0.95) 46%, rgba(9, 78, 110, 0.92) 100%);
            border: 1px solid rgba(148, 196, 255, 0.14);
            box-shadow: 0 28px 80px rgba(2, 6, 23, 0.42);
            margin-bottom: 1.2rem;
            animation: floatIn 0.65s ease;
        }
        .hero-shell::before {
            content: "";
            position: absolute;
            inset: auto -10% -40% auto;
            width: 360px;
            height: 360px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(125, 211, 252, 0.24) 0%, rgba(125, 211, 252, 0.04) 58%, transparent 72%);
            pointer-events: none;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(191, 219, 254, 0.16);
            color: #bfe3ff;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 1rem;
        }
        .hero-shell h1 {
            margin: 0;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.02;
            max-width: 760px;
        }
        .hero-shell p {
            margin: 1rem 0 0;
            color: rgba(236, 245, 255, 0.82);
            max-width: 720px;
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.3rem;
        }
        .hero-chip {
            padding: 0.6rem 0.85rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(191, 219, 254, 0.12);
            color: #d8eafe;
            font-size: 0.92rem;
            font-weight: 500;
        }
        .surface-card {
            background: linear-gradient(180deg, rgba(10, 26, 55, 0.82) 0%, rgba(8, 22, 47, 0.92) 100%);
            border: 1px solid rgba(148, 196, 255, 0.10);
            border-radius: 24px;
            padding: 1.2rem 1.25rem 1.05rem;
            box-shadow: 0 22px 56px rgba(2, 6, 23, 0.28);
            backdrop-filter: blur(16px);
            animation: floatIn 0.72s ease;
        }
        .metric-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            margin-bottom: 1.2rem;
        }
        .status-track {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1.3rem;
        }
        .status-pill {
            position: relative;
            overflow: hidden;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 196, 255, 0.10);
            background: linear-gradient(180deg, rgba(7, 20, 45, 0.8) 0%, rgba(8, 22, 47, 0.95) 100%);
            animation: floatIn 0.8s ease;
        }
        .status-pill::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.05) 25%, transparent 55%);
            transform: translateX(-120%);
            animation: shimmer 4.5s infinite;
        }
        @keyframes shimmer {
            100% { transform: translateX(120%); }
        }
        .status-pill .title {
            display: block;
            color: #f8fbff;
            font-size: 0.94rem;
            font-weight: 700;
            margin-bottom: 0.22rem;
        }
        .status-pill .meta {
            color: #8fb3d9;
            font-size: 0.8rem;
            line-height: 1.4;
        }
        .metric-card {
            min-width: 165px;
            border-radius: 20px;
            padding: 0.95rem 1rem;
            background: linear-gradient(180deg, rgba(10, 26, 55, 0.74) 0%, rgba(8, 22, 47, 0.92) 100%);
            border: 1px solid rgba(148, 196, 255, 0.10);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }
        .metric-card .label {
            display: block;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            color: #8fb3d9;
            margin-bottom: 0.15rem;
        }
        .metric-card .value {
            font-size: 1.08rem;
            font-weight: 700;
            color: #f8fbff;
        }
        .section-kicker {
            margin-bottom: 0.35rem;
            color: #7dd3fc;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .answer-shell {
            background: linear-gradient(180deg, rgba(10, 26, 55, 0.86) 0%, rgba(8, 22, 47, 0.95) 100%);
            border: 1px solid rgba(148, 196, 255, 0.10);
            border-radius: 24px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 22px 56px rgba(2, 6, 23, 0.28);
            animation: floatIn 0.86s ease;
        }
        .response-thread {
            display: grid;
            gap: 1rem;
        }
        .chat-row {
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
        }
        .chat-row.user {
            justify-content: flex-end;
        }
        .chat-row.user .chat-avatar {
            order: 2;
            background: linear-gradient(135deg, #0f3a78 0%, #155eef 100%);
        }
        .chat-row.user .chat-bubble {
            order: 1;
            background: linear-gradient(135deg, rgba(20, 77, 171, 0.92) 0%, rgba(18, 91, 197, 0.98) 100%);
            color: #f8fbff;
            border: 1px solid rgba(191, 219, 254, 0.16);
        }
        .chat-row.bot .chat-bubble {
            background: linear-gradient(180deg, rgba(7, 20, 45, 0.88) 0%, rgba(6, 17, 37, 0.98) 100%);
            border: 1px solid rgba(148, 196, 255, 0.10);
            color: #e8f1ff;
        }
        .chat-avatar {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
            color: #f8fbff;
            font-weight: 800;
            flex: 0 0 auto;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.22);
        }
        .chat-bubble {
            max-width: min(800px, 100%);
            padding: 1rem 1.05rem;
            border-radius: 20px;
            box-shadow: 0 16px 36px rgba(2, 6, 23, 0.22);
        }
        .chat-bubble .bubble-label {
            display: block;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
            color: #9bc5f2;
            font-weight: 700;
        }
        .chat-row.user .chat-bubble .bubble-label {
            color: rgba(224, 236, 255, 0.88);
        }
        .audio-shell {
            margin-top: 0.95rem;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 196, 255, 0.10);
        }
        .surface-card .stSubheader,
        .answer-shell .stSubheader,
        .surface-card h3 {
            color: #f8fbff;
        }
        .stMarkdown, .stCaption, .stText, .stFileUploader, label, .stAudioInput {
            color: #d6e4f5;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px dashed rgba(147, 197, 253, 0.28);
            border-radius: 18px;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #d9e9fb !important;
        }
        div[data-testid="stAudioInput"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 196, 255, 0.10);
            border-radius: 18px;
            padding: 0.6rem;
        }
        .stButton > button {
            border-radius: 16px;
            border: 1px solid rgba(103, 232, 249, 0.22);
            background: linear-gradient(135deg, #0f3a78 0%, #155eef 100%);
            color: #f8fbff;
            font-weight: 700;
            padding: 0.72rem 1rem;
            box-shadow: 0 14px 30px rgba(21, 94, 239, 0.28);
        }
        .stButton > button:hover {
            border-color: rgba(186, 230, 253, 0.45);
            background: linear-gradient(135deg, #114089 0%, #1d67ff 100%);
            color: #ffffff;
        }
        div[data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid rgba(148, 196, 255, 0.10);
            background: rgba(5, 17, 39, 0.55);
        }
        div[data-testid="stExpander"] summary {
            color: #e6f1ff;
        }
        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: none;
        }
        @media (max-width: 900px) {
            .status-track {
                grid-template-columns: 1fr 1fr;
            }
            .hero-shell {
                padding: 1.7rem 1.35rem 1.6rem;
            }
            .hero-shell h1 {
                font-size: 2.25rem;
            }
            .chat-bubble {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="top-ribbon">
            <div>
                <div class="brand">RAG Chatbot</div>
                <div class="sub">A polished voice assistant for navigating your documents</div>
            </div>
            <div class="live-dot">Live product preview</div>
        </div>
        <div class="hero-shell">
            <div class="hero-badge">Voice-first document intelligence</div>
            <h1>Welcome to RAG Chatbot</h1>
            <p>Your navy-blue knowledge cockpit for talking to documents. Upload PDFs, ask naturally with your voice, and get grounded answers back as text and audio in one smooth flow.</p>
            <div class="hero-chip-row">
                <div class="hero-chip">PDF-powered answers</div>
                <div class="hero-chip">Voice in, voice out</div>
                <div class="hero-chip">Groq + Cohere retrieval stack</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_status() -> None:
    knowledge_state = "Ready" if st.session_state.document_stats else "Waiting"
    knowledge_meta = "Documents processed and searchable." if st.session_state.document_stats else "Upload PDFs to activate retrieval."
    st.markdown(
        f"""
        <div class="status-track">
            <div class="status-pill">
                <span class="title">1. Knowledge Base</span>
                <span class="meta">{knowledge_state} · {knowledge_meta}</span>
            </div>
            <div class="status-pill">
                <span class="title">2. Voice Intake</span>
                <span class="meta">Record or upload a question in natural language.</span>
            </div>
            <div class="status-pill">
                <span class="title">3. Grounded Reasoning</span>
                <span class="meta">Retrieval, reranking, and answer generation stay context-bound.</span>
            </div>
            <div class="status-pill">
                <span class="title">4. Audio Delivery</span>
                <span class="meta">Responses return as text plus playable voice output.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_metrics() -> None:
    stats = st.session_state.document_stats
    if not stats:
        return

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card"><span class="label">Status</span><span class="value">Ready to chat</span></div>
            <div class="metric-card"><span class="label">Documents</span><span class="value">{stats['files']}</span></div>
            <div class="metric-card"><span class="label">Pages</span><span class="value">{stats['pages']}</span></div>
            <div class="metric-card"><span class="label">Chunks</span><span class="value">{stats['chunks']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_response_thread(query: str, answer: str, audio_path: str | None, audio_token: str) -> None:
    safe_query = html.escape(query).replace("\n", "<br>") if query else ""
    safe_answer = html.escape(answer).replace("\n", "<br>") if answer else ""
    st.markdown('<div class="response-thread">', unsafe_allow_html=True)
    if query:
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="chat-avatar">U</div>
                <div class="chat-bubble">
                    <span class="bubble-label">You asked</span>
                    <div>{safe_query}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if answer:
        st.markdown(
            f"""
            <div class="chat-row bot">
                <div class="chat-avatar">AI</div>
                <div class="chat-bubble">
                    <span class="bubble-label">Mini Anoop</span>
                    <div>{safe_answer}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if audio_path:
        st.markdown('<div class="audio-shell">', unsafe_allow_html=True)
        st.caption("Voice reply")
        render_audio_response(audio_path, audio_token)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def format_user_error(exc: Exception) -> str:
    message = str(exc)
    if "Missing required environment variables" in message:
        return "Required API credentials are missing. Add them to the environment and try again."
    if "Process at least one PDF" in message:
        return "Upload and process at least one PDF before asking a question."
    if "Whisper did not detect any speech" in message:
        return "No clear speech was detected. Try again with a clearer recording."
    if "Unsupported audio format" in message:
        return "That audio file format is not supported."
    if "No extractable text" in message:
        return "The uploaded PDFs did not contain extractable text."
    return "Something went wrong while generating the response. Please try again."


def render_audio_response(audio_path: str, audio_token: str) -> None:
    path = Path(audio_path)
    if not path.exists():
        return

    audio_bytes = path.read_bytes()
    if audio_token and st.session_state.autoplayed_audio_token != audio_token:
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.markdown(
            f"""
            <audio autoplay style="display:none">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.autoplayed_audio_token = audio_token

    st.audio(audio_bytes, format="audio/wav")


def render_source_documents(source_documents) -> None:
    if not source_documents:
        return

    with st.expander("Source excerpts", expanded=False):
        for index, document in enumerate(source_documents, start=1):
            source = document.metadata.get("source", "Unknown source")
            page = document.metadata.get("page", "?")
            score = document.metadata.get("rerank_score")
            header = f"{index}. {source} | page {page}"
            if score is not None:
                header += f" | rerank score: {score:.4f}"

            st.markdown(f"**{header}**")
            st.write(document.page_content[:1200])


def main() -> None:
    st.set_page_config(
        page_title="Mini Anoop's RAG Chatbot",
        page_icon="💬",
        layout="wide",
    )
    initialize_session_state()
    inject_global_styles()
    render_hero()
    render_pipeline_status()
    render_document_metrics()

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Knowledge Vault</div>', unsafe_allow_html=True)
        st.subheader("Load your source documents")
        uploaded_pdfs = st.file_uploader(
            "Add one or more PDF files",
            type=list(SUPPORTED_PDF_EXTENSIONS),
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("Process Documents", type="primary", use_container_width=True):
            if not uploaded_pdfs:
                st.error("Add at least one PDF before processing.")
            else:
                try:
                    with st.spinner("Parsing PDFs, chunking text, and building the FAISS index..."):
                        process_documents(uploaded_pdfs)
                    stats = st.session_state.document_stats or {}
                    cache_note = "Knowledge base refreshed from cache." if stats.get("cached") else "Knowledge base created successfully."
                    st.success(cache_note)
                except Exception as exc:
                    LOGGER.exception("Document processing failed.")
                    st.error(format_user_error(exc))

        if st.session_state.document_stats:
            stats = st.session_state.document_stats
            st.caption(f"{stats['files']} document(s) indexed across {stats['pages']} page(s) and {stats['chunks']} retrieval chunks.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Live Ask</div>', unsafe_allow_html=True)
        st.subheader("Speak to your knowledge base")
        recorded_audio = st.audio_input("Record your question")
        uploaded_audio = st.file_uploader(
            "Or upload an audio file",
            type=list(SUPPORTED_AUDIO_EXTENSIONS),
            accept_multiple_files=False,
            key="uploaded_audio",
        )
        selected_audio = recorded_audio or uploaded_audio

        if st.button("Ask Question", use_container_width=True):
            if selected_audio is None:
                st.error("Record a question or upload an audio file before submitting.")
            else:
                try:
                    with st.spinner("Running Whisper, retrieval, reranking, answer generation, and audio synthesis..."):
                        run_voice_rag(selected_audio)
                    st.success("Answer generated successfully.")
                except Exception as exc:
                    LOGGER.exception("Voice RAG pipeline failed.")
                    st.error(format_user_error(exc))

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_query or st.session_state.last_answer:
        st.markdown('<div class="section-kicker" style="margin-top: 1.4rem;">Latest Response</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-shell">', unsafe_allow_html=True)
        render_response_thread(
            st.session_state.last_query,
            st.session_state.last_answer,
            st.session_state.last_audio_path,
            st.session_state.last_audio_token,
        )

        if st.session_state.last_tts_warning:
            st.info(st.session_state.last_tts_warning)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_sources:
        render_source_documents(st.session_state.last_sources)


if __name__ == "__main__":
    main()
