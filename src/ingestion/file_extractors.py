"""Extracts text/tabular content from user-uploaded files (PDF, DOCX, TXT/MD, CSV)."""
import io

import pandas as pd
from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_TABULAR_EXTENSIONS = {".csv"}
SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_TABULAR_EXTENSIONS


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extracts plain text from a PDF, DOCX, TXT, or MD file's raw bytes."""
    ext = _extension(filename)
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if ext in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {ext}")


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
