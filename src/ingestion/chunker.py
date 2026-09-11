"""Simple fixed-size word chunking with overlap."""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, source: str, chunk_size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Splits text into overlapping chunks of roughly `chunk_size` words each."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    index = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(Chunk(text=" ".join(chunk_words), source=source, chunk_index=index))
        index += 1
        if end == len(words):
            break
        start += step
    return chunks
