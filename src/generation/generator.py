"""Builds a grounded prompt from retrieved chunks and calls Gemini for an answer."""
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.retrieval.retriever import RetrievedChunk

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

PROMPT_TEMPLATE = """You are a support assistant for NovaTech Electronics. Answer the question \
using ONLY the context below. If the context does not contain enough information to answer, \
say so explicitly instead of guessing. Cite the source filename(s) you used in your answer.

Context:
{context}

Question: {question}

Answer:"""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — check your .env file")
    return genai.Client(api_key=api_key)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    prompt = build_prompt(question, chunks)
    client = _client()
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


if __name__ == "__main__":
    from src.retrieval.retriever import Retriever

    retriever = Retriever()
    question = "Can I return my laptop after 20 days if I already registered it?"
    chunks = retriever.retrieve(question, top_k=3)
    answer = generate_answer(question, chunks)
    print("ANSWER:\n", answer)
