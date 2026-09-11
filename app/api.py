"""Phase 7: REST API wrapper around the Phase 3+4 agentic RAG pipeline, plus a per-session
file-upload feature (Phase 8) letting a user ask questions about their own CSV/PDF/DOCX/TXT/MD
files instead of the fixed NovaTech dataset.

Run with: uvicorn app.api:app --reload
"""
import logging
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google.genai import errors as genai_errors
from pydantic import BaseModel

from src.generation.agentic_pipeline import answer_question_with_context
from src.generation.session_pipeline import answer_question_for_session
from src.ingestion.file_extractors import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_TABULAR_EXTENSIONS,
    extract_text,
    load_csv,
)
from src.reranking import reranker
from src.retrieval import retriever, session_store

logger = logging.getLogger("novatech_rag")
app = FastAPI(title="NovaTech RAG API")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None  # if set, answer from this session's uploaded files instead


class AskResponse(BaseModel):
    question: str
    answer: str
    context_used: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    kind: str  # "document" or "table"
    summary: str


@app.on_event("startup")
def warm_up_models() -> None:
    """Loads the embedding model and cross-encoder once at boot, so the first real request
    doesn't pay the multi-second model-load cost."""
    logger.info("Warming up retrieval models...")
    retriever.warm_up()
    reranker.warm_up()
    logger.info("Models ready.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _handle_genai_errors(e: Exception):
    if isinstance(e, genai_errors.APIError):
        if getattr(e, "code", None) == 429:
            logger.warning("Gemini quota exceeded: %s", getattr(e, "message", e))
            raise HTTPException(
                status_code=503,
                detail="The assistant is temporarily over its request quota — please try again shortly.",
            ) from e
        logger.exception("Gemini API error")
        raise HTTPException(status_code=502, detail="Upstream model provider error.") from e
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal error while generating the answer.") from e


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        if request.session_id:
            result = answer_question_for_session(question, request.session_id)
        else:
            result = answer_question_with_context(question)
    except Exception as e:
        _handle_genai_errors(e)
        raise  # unreachable — _handle_genai_errors always raises

    return AskResponse(question=question, answer=result.answer, context_used=result.context_used)


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), session_id: str | None = Form(None)) -> UploadResponse:
    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_TABULAR_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: csv, pdf, docx, txt, md.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 20 MB).")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    session_id = session_id or uuid.uuid4().hex

    try:
        if ext in SUPPORTED_TABULAR_EXTENSIONS:
            df = load_csv(file_bytes)
            table_name = session_store.add_csv(session_id, filename, df)
            summary = f"Loaded {len(df)} rows, {len(df.columns)} columns into table `{table_name}`."
            kind = "table"
        else:
            text = extract_text(filename, file_bytes)
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="No extractable text found in this file (it may be scanned/empty).",
                )
            n_chunks = session_store.add_document(session_id, filename, text)
            summary = f"Indexed {n_chunks} chunks from '{filename}'."
            kind = "document"
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}") from e

    return UploadResponse(session_id=session_id, filename=filename, kind=kind, summary=summary)
