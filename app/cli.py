"""Interactive CLI for the baseline RAG pipeline.

Run with: python -m app.cli
"""
from src.generation.rag_pipeline import answer_question


def main():
    print("NovaTech RAG assistant (baseline dense retrieval). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        result = answer_question(question)
        print(f"\nAssistant: {result.answer}\n")
        print("Retrieved chunks:")
        for c in result.chunks:
            print(f"  - {c.source} #{c.chunk_index} (dist={c.distance:.3f})")
        print()


if __name__ == "__main__":
    main()
