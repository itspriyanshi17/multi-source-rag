"""Agentic RAG pipeline over a user's own uploaded files, mirroring the fixed NovaTech
agentic_pipeline.py but binding its tools to one session's uploaded CSV/document data instead
of the fixed products.db / novatech_docs collection.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.generation.agentic_pipeline import AnsweredQuestion
from src.retrieval import session_store

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM_INSTRUCTION_TEMPLATE = """You are a document assistant answering questions about files a
user has uploaded in this session — nothing else. You have two tools:

- query_uploaded_data: for questions about any uploaded CSV data (filtering, sorting, counting,
  aggregating over rows/columns).
- search_uploaded_documents: for questions about uploaded PDF/DOCX/TXT/MD document content.

Use whichever tool(s) the question needs. Do not guess at data or document content — always
retrieve it via the tools before answering. If the uploaded data/documents don't contain the
answer, say so explicitly instead of guessing.

{schema}

Always cite which uploaded file(s) you used in your final answer.
"""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — check your .env file")
    return genai.Client(api_key=api_key)


def _build_session_tools(session_id: str):
    def query_uploaded_data(sql: str) -> str:
        """Runs a read-only SQL SELECT query against the user's uploaded CSV table(s) and
        returns the results.

        Args:
            sql: A single SQLite SELECT statement referencing one of the uploaded tables.
        """
        return session_store.query_data(session_id, sql)

    def search_uploaded_documents(query: str, top_k: int = 5) -> str:
        """Searches the user's uploaded documents (PDF/DOCX/TXT/MD) for passages relevant to
        the question.

        Args:
            query: The natural-language question or topic to search for.
            top_k: Number of passages to return (default 5).
        """
        return session_store.search_documents(session_id, query, top_k=top_k)

    return [query_uploaded_data, search_uploaded_documents]


def answer_question_for_session(question: str, session_id: str) -> AnsweredQuestion:
    if not session_store.has_any_uploads(session_id):
        return AnsweredQuestion(
            question=question,
            answer="No files have been uploaded yet for this session — upload a CSV, PDF, DOCX, "
            "TXT, or MD file first.",
            context_used="",
        )

    schema = session_store.describe_tables(session_id)
    doc_names = session_store.list_uploaded_document_names(session_id)
    doc_note = f"Uploaded documents: {', '.join(doc_names)}" if doc_names else "No documents uploaded."
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(schema=f"{schema}\n\n{doc_note}")

    client = _client()
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            tools=_build_session_tools(session_id),
            system_instruction=system_instruction,
        ),
    )
    response = chat.send_message(question)

    tool_results = []
    for content in chat.get_history():
        for part in content.parts:
            if part.function_response is not None:
                tool_results.append(str(part.function_response.response.get("result", "")))

    return AnsweredQuestion(
        question=question,
        answer=response.text,
        context_used="\n\n".join(tool_results),
    )
