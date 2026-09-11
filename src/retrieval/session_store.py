"""Per-session storage for user-uploaded files: a scratch Chroma collection for documents
(PDF/DOCX/TXT/MD) and a scratch SQLite database for CSVs. Each session is isolated from the
fixed NovaTech dataset and from every other session, keyed by a session_id the API hands out
on first upload.
"""
import re
import sqlite3
from pathlib import Path

import chromadb
import pandas as pd

from src.ingestion.chunker import chunk_text
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.sql_safety import run_validated_select

SESSIONS_ROOT = Path(__file__).resolve().parents[2] / "data" / "database" / "sessions"

_chroma_clients: dict[str, chromadb.ClientAPI] = {}
DOCS_COLLECTION = "uploaded_docs"


def _session_dir(session_id: str) -> Path:
    d = SESSIONS_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path(session_id: str) -> Path:
    return _session_dir(session_id) / "data.db"


def _get_chroma_client(session_id: str) -> chromadb.ClientAPI:
    if session_id not in _chroma_clients:
        _chroma_clients[session_id] = chromadb.PersistentClient(
            path=str(_session_dir(session_id) / "chroma")
        )
    return _chroma_clients[session_id]


def _get_docs_collection(session_id: str):
    return _get_chroma_client(session_id).get_or_create_collection(DOCS_COLLECTION)


def sanitize_table_name(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    name = re.sub(r"\W+", "_", stem).strip("_").lower()
    return name or "uploaded_table"


# ---------- Documents (PDF / DOCX / TXT / MD) ----------

def add_document(session_id: str, filename: str, text: str) -> int:
    """Chunks and embeds `text` into this session's document collection. Returns chunk count."""
    chunks = chunk_text(text, source=filename)
    if not chunks:
        return 0

    model = get_embedding_model()
    embeddings = model.encode([c.text for c in chunks]).tolist()
    collection = _get_docs_collection(session_id)
    existing_count = collection.count()
    collection.add(
        ids=[f"{filename}::{existing_count + i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"source": filename, "chunk_index": c.chunk_index} for c in chunks],
    )
    return len(chunks)


def search_documents(session_id: str, query: str, top_k: int = 5) -> str:
    collection = _get_docs_collection(session_id)
    if collection.count() == 0:
        return "No documents have been uploaded for this session."

    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=min(top_k, collection.count()))

    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        parts.append(f"[Source: {meta['source']}]\n{doc}")
    return "\n\n".join(parts) if parts else "No matching passages found."


def list_uploaded_document_names(session_id: str) -> list[str]:
    collection = _get_docs_collection(session_id)
    if collection.count() == 0:
        return []
    metadatas = collection.get()["metadatas"]
    return sorted({m["source"] for m in metadatas})


# ---------- Tabular (CSV) ----------

def add_csv(session_id: str, filename: str, df: pd.DataFrame) -> str:
    """Loads `df` into this session's SQLite database as a new table. Returns the table name."""
    table_name = sanitize_table_name(filename)
    with sqlite3.connect(_db_path(session_id)) as conn:
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        final_name = table_name
        suffix = 2
        while final_name in existing:
            final_name = f"{table_name}_{suffix}"
            suffix += 1
        df.to_sql(final_name, conn, if_exists="fail", index=False)
    return final_name


def describe_tables(session_id: str) -> str:
    """Builds a schema description string for every table uploaded in this session, in the
    same style as the fixed NovaTech SCHEMA_DESCRIPTION, so it can be dropped into a tool
    docstring or system prompt."""
    db_path = _db_path(session_id)
    if not db_path.exists():
        return "No CSV data has been uploaded for this session."

    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        if not tables:
            return "No CSV data has been uploaded for this session."

        blocks = []
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            col_lines = "\n".join(f"  {c[1]:<20} {c[2]}" for c in cols)
            blocks.append(f"Table: {table}\nColumns:\n{col_lines}")
    return "\n\n".join(blocks)


def query_data(session_id: str, sql: str) -> str:
    db_path = _db_path(session_id)
    if not db_path.exists():
        return "No CSV data has been uploaded for this session."
    return run_validated_select(db_path, sql)


def has_any_uploads(session_id: str) -> bool:
    db_path = _db_path(session_id)
    has_csv = False
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
            has_csv = count > 0
    has_docs = _get_docs_collection(session_id).count() > 0
    return has_csv or has_docs
