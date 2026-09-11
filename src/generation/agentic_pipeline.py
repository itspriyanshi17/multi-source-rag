"""Phase 3: combined routing + retrieval + generation.

Instead of a separate router classification call, the LLM itself decides — via function
calling — whether to query the structured product database, search unstructured documents,
or both, then generates the final answer in the same call chain.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.retrieval.retriever import search_documents
from src.retrieval.sql_retriever import SCHEMA_DESCRIPTION, run_sql_query

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM_INSTRUCTION = f"""You are a support assistant for NovaTech Electronics.

You have two tools:
- run_sql_query: for structured product-catalog questions (price, stock, category, brand,
  rating, specs, counts, filters, sorting).
- search_documents: for policy, FAQ, manual, and support-ticket questions.

Use whichever tool(s) the question needs — call both if the question spans both product data
and policy/support information. Do not guess at product data or policy details; always retrieve
them via the tools before answering.

{SCHEMA_DESCRIPTION}

Always cite which tool/source you used (SQL query results and/or document filenames) in your
final answer. If neither tool returns enough information, say so explicitly instead of guessing.
"""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — check your .env file")
    return genai.Client(api_key=api_key)


@dataclass
class AnsweredQuestion:
    question: str
    answer: str
    context_used: str  # concatenated results of every tool call the model made to answer


def answer_question(question: str) -> str:
    return answer_question_with_context(question).answer


def answer_question_with_context(question: str) -> AnsweredQuestion:
    client = _client()
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            tools=[run_sql_query, search_documents],
            system_instruction=SYSTEM_INSTRUCTION,
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


if __name__ == "__main__":
    tests = [
        "How many laptops are under $1000 and in stock?",
        "What is the return window for accessories?",
        "Is the NovaView 27 QHD in stock, and is it covered under warranty for dead pixels?",
    ]
    for q in tests:
        print("Q:", q)
        print("A:", answer_question(q))
        print()
