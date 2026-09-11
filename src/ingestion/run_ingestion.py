"""Orchestrates the full ingestion pipeline: CSV -> SQLite, and docs -> Chroma vector store.

Run with: python -m src.ingestion.run_ingestion
"""
from src.ingestion.build_vector_store import build_vector_store
from src.ingestion.load_products import build_products_db


def main():
    n_products = build_products_db()
    print(f"[1/2] Loaded {n_products} products into SQLite.")

    n_chunks = build_vector_store()
    print(f"[2/2] Embedded {n_chunks} document chunks into Chroma.")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
