"""LLM-as-judge scoring for generated answers — a hand-built equivalent of what a library like
RAGAS automates, kept in-house so it reuses the same Gemini setup as the rest of the pipeline
instead of adding a second LLM-integration stack.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

JUDGE_PROMPT = """You are an evaluation judge for a RAG (retrieval-augmented generation) system.
Score the GENERATED ANSWER on three dimensions, each from 0.0 to 1.0:

- faithfulness: Are ALL factual claims in the answer actually supported by the RETRIEVED CONTEXT?
  1.0 = every claim is grounded in the context. 0.0 = the answer contradicts or invents facts not
  present in the context. An answer that correctly says "not enough information" when the context
  truly lacks the answer should score 1.0 for faithfulness.
- relevancy: Does the answer directly address what the QUESTION actually asked, without being
  generic or off-topic?
- correctness: How well does the answer's content match the EXPECTED ANSWER (ground truth)?
  If no expected answer is provided, output null for this field.

Respond with ONLY a JSON object: {{"faithfulness": <float>, "relevancy": <float>,
"correctness": <float or null>, "reasoning": "<one sentence>"}}

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

EXPECTED ANSWER (ground truth, may be a loose summary rather than exact wording):
{expected_answer}
"""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — check your .env file")
    return genai.Client(api_key=api_key)


def judge(question: str, context: str, answer: str, expected_answer: str | None = None) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context or "(no context retrieved)",
        answer=answer,
        expected_answer=expected_answer or "(not provided)",
    )
    client = _client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)
