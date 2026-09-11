"""Loads markdown documents from data/documents and chunks them for embedding."""
from pathlib import Path

from src.ingestion.chunker import Chunk, chunk_text

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"


def load_and_chunk_documents(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=path.name))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}")
    for c in chunks[:3]:
        print(f"--- {c.source} #{c.chunk_index} ---\n{c.text[:200]}...\n")
