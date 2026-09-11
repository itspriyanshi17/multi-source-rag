# Project Report: Multi-Source RAG with Agentic Routing, Re-ranking & Evaluation

**Project:** NovaTech Electronics RAG Assistant
**Type:** Learning project / portfolio piece
**Repository:** https://github.com/itspriyanshi17/multi-source-rag

---

## 1. Overview

This project is a retrieval-augmented generation (RAG) system built to go beyond the standard
"embed some docs, ask questions" tutorial pattern. It answers questions over **two structurally
different data sources** — a structured product catalog (SQL) and unstructured policy/support
documents (vector search) — and adds the three components most RAG tutorials skip:

1. **Agentic multi-source routing** — no hand-written if/else or classifier decides which data
   source to query. The LLM itself decides, via native function calling, whether a question
   needs SQL, document search, or both — and can call both in a single turn.
2. **Cross-encoder re-ranking** — a second, more accurate scoring pass on top of fast-but-
   approximate dense vector retrieval, with results quantified before/after (not eyeballed).
3. **A real evaluation harness** — a labeled test set scored with retrieval metrics (Hit Rate,
   MRR) and an LLM-as-judge for generation quality (faithfulness, relevancy, correctness).

The project was later extended with a **bring-your-own-data mode**: users can upload their own
CSV/PDF/DOCX/TXT/MD files through a web UI and ask questions about them, using the identical
agentic routing and retrieval pattern, fully isolated per session from the fixed demo dataset.

No LangChain, LlamaIndex, or similar framework was used — every retrieval, routing, chunking,
and generation step is implemented directly against the vector store, database, and LLM SDK, so
the mechanics are actually learned rather than abstracted away.

---

## 2. Tech Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, free, fast enough for CPU |
| Vector store | ChromaDB (persistent, local) | Simple embedded vector DB, no server needed |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Standard, well-tested passage re-ranker |
| Structured data | SQLite, queried via LLM-generated SQL | Real relational querying, zero infra |
| LLM | Google Gemini (`gemini-3.5-flash-lite`) | Free tier, native function calling |
| API | FastAPI + Uvicorn | Async-capable, typed, auto-docs |
| Frontend | Streamlit | Fast to build a real chat UI in Python |
| File parsing | `pypdf`, `python-docx`, `pandas` | Standard, lightweight extractors |
| Evaluation | Hand-built LLM-as-judge + custom retrieval metrics | Full control, one fewer dependency stack |

---

## 3. Architecture

```
                              ┌─────────────────────┐
                              │   User question       │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Gemini (agentic)      │
                              │  decides which tool(s)  │
                              │  to call — no separate  │
                              │  router model needed    │
                              └──────┬─────────┬──────┘
                     ┌───────────────┘         └───────────────┐
                     ▼                                         ▼
        ┌────────────────────────┐             ┌──────────────────────────────┐
        │  run_sql_query           │             │  search_documents              │
        │  (text-to-SQL)           │             │  (dense retrieval + rerank)    │
        │                          │             │                                 │
        │  SELECT-only validated   │             │  bi-encoder top-20              │
        │  SQL against products.db │             │  → cross-encoder top-5          │
        └────────────┬─────────────┘             └───────────────┬─────────────────┘
                     │                                          │
                     └───────────────────┬──────────────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  Grounded answer with  │
                              │  cited sources          │
                              └──────────────────────┘
```

The file-upload mode runs the identical pattern against a per-session SQLite DB and Chroma
collection instead of the fixed `products.db` / `novatech_docs` store.

---

## 4. Phase-by-Phase Breakdown

### Phase 1 — Synthetic Dataset & Ingestion Pipeline

**Goal:** Build a realistic multi-source dataset with full control over ground truth, so
retrieval and evaluation results are verifiable rather than guessed at.

**What was built:**
- A synthetic "NovaTech Electronics" company: `src/ingestion/generate_products.py` procedurally
  generates 44 products across 10 categories (Laptops, Smartphones, Headphones, Monitors,
  Smartwatches, Tablets, Cameras, Speakers, Accessories, Gaming Consoles), each with price,
  stock, rating, release year, and specs.
- 18 hand-authored unstructured documents: company overview, return/shipping/warranty/payment
  policies, 3 product manuals, 5 support-ticket transcripts, store locations, loyalty program
  FAQ, privacy policy, and a press release.
- A deliberate **conflicting-rule test case**: the general return policy states a 30-day window,
  but the NovaBook Pro 15's product manual states a 14-day window once the device is registered
  (because pre-installed software activates on first boot). This was built specifically to
  stress-test whether retrieval and re-ranking surface the *specific* rule over the *general*
  one — a common real-world RAG failure mode.
- Ingestion pipeline (`src/ingestion/`): `chunker.py` (fixed-size word chunking, 220 words with
  40-word overlap), `load_documents.py`, `load_products.py` (CSV → SQLite), `build_vector_store.py`
  (chunks → embeddings → Chroma), orchestrated by `run_ingestion.py`.

**Result:** 44 products loaded into SQLite (`products.db`), 19 document chunks embedded into a
persistent Chroma collection (`novatech_docs`). Verified via direct query — asking about
"laptop return after registration" correctly surfaced both `return_policy.md` and the matching
support ticket about the exact same scenario.

### Phase 2 — Baseline Dense Retrieval + Generation

**Goal:** Get a working end-to-end RAG loop before adding any sophistication.

**What was built:**
- `src/retrieval/retriever.py` — embeds a query with the same model used at ingestion time,
  queries Chroma for top-k chunks by vector distance.
- `src/generation/generator.py` — builds a "stuffed" prompt (context + question), calls Gemini,
  with explicit instructions to cite sources and refuse to guess when context is insufficient.
- `src/generation/rag_pipeline.py` — ties retrieve → generate together.

**Key design decision:** query and chunk embeddings must come from the identical model, or
similarity scores become meaningless — this was called out explicitly since it's an easy mistake.

**Result:** Verified against the deliberately tricky NovaBook Pro 15 conflicting-policy question
— the pipeline correctly resolved the conflict (14-day exception overrides the general 30-day
policy) and pulled in the relevant support-ticket precedent, citing all three sources used.

### Phase 3 — Structured Data Retriever + Agentic Router

**Goal:** Answer questions vector search fundamentally can't — filtering, counting, aggregating
over the product catalog — and decide automatically which data source(s) a question needs.

**What was built:**
- `src/retrieval/sql_retriever.py` — a text-to-SQL tool. The LLM is given the table schema and
  writes its own SQL; a validator (`src/retrieval/sql_safety.py`) enforces SELECT-only, single-
  statement, no-forbidden-keyword queries before execution — never trust LLM-generated SQL
  blindly, even against a local read-only database.
- `src/retrieval/retriever.py` gained `search_documents` as a second LLM-callable tool.
- **Routing decision:** rather than a separate classifier call ("is this STRUCTURED,
  UNSTRUCTURED, or BOTH?"), the router was collapsed into the generation step itself using
  Gemini's native function calling — the model sees both tools and their docstrings, and decides
  which to call (including both, in the same turn) as part of normal generation. This avoids an
  extra LLM round-trip and extra prompt-engineering surface area.
- `src/generation/agentic_pipeline.py` — the combined pipeline, using `client.chats.create(...,
  config=types.GenerateContentConfig(tools=[run_sql_query, search_documents]))`.

**Result:** Three test cases confirmed the pattern works:
1. Structured-only ("laptops under $1000") → called `run_sql_query`, correct answer.
2. Unstructured-only ("return window for accessories") → called `search_documents`, correct
   answer including the NovaTech Plus 45-day exception.
3. **Combined** ("is the NovaView 27 QHD in stock, and is it covered under warranty for dead
   pixels?") → called **both tools in the same turn**, correctly merged SQL stock/price data
   with the warranty policy's 3-pixel threshold rule.

### Phase 4 — Cross-Encoder Re-ranking

**Goal:** Fix the accuracy gap between fast bi-encoder similarity and true relevance.

**Why it's needed:** a bi-encoder embeds the query and each chunk *independently*, then compares
vectors after the fact — it never actually looks at query and passage together. A cross-encoder
takes both as one input through a transformer, producing a much more accurate relevance score,
at the cost of not being precomputable (too slow to run over an entire corpus).

**What was built:**
- `src/reranking/reranker.py` — retrieves a wide candidate pool (top-20, via
  `CANDIDATE_POOL_SIZE`) from the bi-encoder, then re-scores with
  `cross-encoder/ms-marco-MiniLM-L-6-v2` and keeps the top-5.
- `search_documents` in `retriever.py` was updated to use this wide-retrieve → rerank-narrow
  pattern transparently, so both the Phase 2 and Phase 3 pipelines benefit automatically.
- `src/reranking/compare_rerank.py` — a before/after comparison script, so the improvement is
  demonstrated, not asserted.

**Result (quantified in Phase 6, see below):** Hit Rate@5 improved from 92.3% → 100%, MRR from
0.846 → 0.923. One test question was retrieved *nowhere* in the bi-encoder's top-5 and was
correctly recovered to rank #1 after re-ranking — a genuine miss recovered, not just reordering.

### Phase 5 — Generation with Citations & Hallucination Guard

**Goal:** Force every answer to cite its sources, and explicitly refuse to guess when the
retrieved context doesn't contain the answer.

**Note:** this phase turned out to already be satisfied by the prompt engineering done in
Phases 2 and 3 (`generator.py` and `agentic_pipeline.py` both instruct the model to cite sources
and say "not enough information" rather than guess) — but it still needed explicit verification,
since "the prompt says to do X" is not proof "the model actually does X."

**Verification:** asked a two-part question — "Who is the CEO of NovaTech Electronics, and do
you offer a lifetime warranty on laptops?" — where the CEO half has no answer anywhere in the
dataset. The system correctly said the documentation doesn't state who the CEO is, while
correctly answering the warranty half from `warranty_policy.md`, handling each half
independently rather than letting one failure contaminate the other.

### Phase 6 — Evaluation

**Goal:** Replace "looks right" with actual measured numbers — the phase that most separates a
demo project from one that demonstrates real understanding of RAG failure modes.

**What was built:**
- `data/eval/test_set.json` — 18 labeled test questions across four categories: unstructured
  (10), structured (4), combined/both (2), unanswerable (2) — each with an `expected_sources`
  and `expected_answer` field as ground truth.
- `src/evaluation/retrieval_eval.py` — computes **Hit Rate@5** (did the correct source appear
  anywhere in the top-5?) and **MRR** (rewards ranking it higher, not just present), run twice:
  bi-encoder-only vs. bi-encoder + re-ranking, to quantify Phase 4's real effect.
- `src/evaluation/llm_judge.py` — a hand-built LLM-as-judge (deliberately not the `ragas`
  library, to keep everything on the same Gemini setup instead of adding a second LLM-
  integration stack), scoring **faithfulness** (are all claims grounded in retrieved context),
  **relevancy** (does the answer address the actual question), and **correctness** (match
  against the expected answer).
- `src/evaluation/generation_eval.py` — runs the full agentic pipeline on every test question
  and scores it, extracting the *actual* context used by inspecting the Gemini chat history's
  function-response parts (not a re-derived guess at what was retrieved).

**Results:**

*Retrieval — before vs. after re-ranking:*

| Metric | Bi-encoder only | + Cross-encoder re-rank |
|---|---|---|
| Hit Rate@5 | 92.3% | **100%** |
| MRR | 0.846 | **0.923** |

*Generation — LLM-judged, by category:*

| Category | n | Faithfulness | Relevancy | Correctness |
|---|---|---|---|---|
| Unstructured (docs) | 10 | 0.980 | 1.000 | 0.990 |
| Structured (SQL) | 4 | 1.000 | 1.000 | 1.000 |
| Combined (both) | 2 | 1.000 | 1.000 | 1.000 |
| Unanswerable | 2 | 1.000 | 1.000 | 1.000 |
| **Overall** | 18 | **0.989** | **1.000** | — |

### Phase 7 — API / UI Wrapper

**Goal:** Turn the pipeline from "a folder of scripts" into something demoable/usable by others.

**What was built:**
- `app/api.py` — FastAPI wrapper with `GET /health` and `POST /ask`. Endpoints are plain `def`
  (not `async def`) so FastAPI runs them in a threadpool automatically — the pipeline's
  synchronous, blocking calls (embedding, cross-encoding, Gemini requests) would otherwise
  freeze the entire server for every other concurrent request.
- Startup pre-warming (`retriever.warm_up()`, `reranker.warm_up()`) so the embedding and
  cross-encoder models load once at boot, not on whichever user's request happens to be first.
- Error handling: Gemini `429` (quota exhausted) mapped to a clean `503` with a user-facing
  message instead of a raw traceback; unexpected errors mapped to `500` without leaking stack
  traces to the client.
- `GEMINI_MODEL` moved into `.env` rather than hardcoded, specifically because of a real problem
  hit during development (see Section 5).
- `app/cli.py` — interactive terminal chat, covering the CLI half of this phase.

**Verified live:** `/health` → 200; `/ask` with the combined NovaView monitor question → correct
grounded answer; `/ask` with a blank question → 400, rejected before reaching the pipeline; log
confirmed both models finished loading *before* "Application startup complete," proving the
pre-warm fix worked as intended.

### Phase 8 (extension) — Bring-Your-Own-Data Upload

**Goal:** Let a user upload their own CSV/PDF/DOCX/TXT/MD files and ask questions about them,
using the same agentic multi-source pattern, isolated from the fixed demo dataset.

**What was built:**
- `src/ingestion/file_extractors.py` — text extraction for PDF (`pypdf`), DOCX (`python-docx`),
  TXT/MD (plain decode), and CSV (`pandas`).
- `src/retrieval/session_store.py` — per-session storage: a scratch Chroma collection for
  documents and a scratch SQLite database for CSVs, keyed by a `session_id` and stored under
  `data/database/sessions/<session_id>/`, fully isolated from both the fixed dataset and other
  sessions. Includes `describe_tables()` which introspects the session's actual SQLite schema
  (via `PRAGMA table_info`) to build a live schema description, since uploaded CSV structure
  isn't known ahead of time the way the fixed `products` table's schema is.
- `src/generation/session_pipeline.py` — mirrors `agentic_pipeline.py`, but builds its two tools
  (`query_uploaded_data`, `search_uploaded_documents`) as closures bound to one `session_id` per
  request, so the LLM never has to be trusted to supply the correct session identifier itself.
- `app/api.py` gained `POST /upload` (multipart file upload, 20MB cap, extension whitelist) and
  `POST /ask` gained an optional `session_id` field to switch to session-scoped answering.
- `app/streamlit_app.py` gained a sidebar mode toggle ("NovaTech demo data" / "My uploaded
  files"), a file uploader, and per-session upload history display.
- A refactor for reuse: `src/retrieval/embeddings.py` (shared embedding-model singleton) and
  `src/retrieval/sql_safety.py` (shared SELECT-only SQL validator) were factored out so the
  fixed and session-scoped retrievers share code instead of duplicating it.

**Verified live (via direct API calls):**
- CSV upload ("region, product, units_sold, revenue" test data) → loaded into a session SQLite
  table; "which region had highest total revenue" → correctly computed via SQL `GROUP BY`
  ($4,200 for North, verified by hand).
- DOCX upload (a synthetic project-status report) → chunked and embedded; "who is the project
  lead and what's blocking it" → correctly answered with citation.
- **Combined question** across both uploaded sources in one session ("total revenue across all
  regions, and separately who is the Aquila project lead") → correctly called both tools in one
  turn, both answers correct ($11,250 total revenue, verified by hand).
- Edge cases: a blank/scanned PDF → clean `400` ("no extractable text") instead of silently
  indexing nothing or crashing; an unsupported file extension (`.xyz`) → clean `400` rejection.

---

## 5. Real Problems Hit During Development (and how they were handled)

Documented honestly because working through them was as much a part of the learning goal as the
RAG architecture itself.

- **Gemini free-tier daily quota exhaustion.** `gemini-3.6-flash` capped at just 20 requests/day
  on the free tier — hit mid-way through the Phase 6 evaluation run (`429 RESOURCE_EXHAUSTED`).
  Fixed by switching to `gemini-3.5-flash-lite`, which has a much higher free-tier quota and
  scored identically well in evaluation.
- **Mid-project model deprecation.** Google killed `gemini-2.5-flash` overnight (the API
  redirected to `gemini-3.6-flash` with a `404` and a message pointing to the replacement) —
  this happened twice during the build. Fixed by moving the model name into a `GEMINI_MODEL`
  environment variable read by all three modules that call Gemini, so a future deprecation is a
  one-line config change instead of a multi-file code edit.
- **Transient network errors weren't retried at the client layer.** A `429`-driven eval run
  later crashed on a plain `httpx.ConnectError` ("connection forcibly closed") that the SDK's
  built-in retry didn't cover — simply re-running the script succeeded. The API layer's own
  error handling (Section on Phase 7) was written specifically so a real user hitting this
  wouldn't see a raw traceback.
- **Blocking calls under FastAPI.** Recognized upfront (not hit as a live bug) that `async def`
  endpoints calling the fully synchronous pipeline (`SentenceTransformer.encode`,
  `CrossEncoder.predict`, Gemini SDK calls) would block the entire server's event loop per
  request. Endpoints were written as plain `def` so FastAPI threadpools them automatically.

---

## 6. Design Notes & Known Limitations

- The NovaTech dataset is entirely synthetic and deliberately includes a conflicting-rule case
  (laptop manual overrides general policy) specifically to stress-test whether retrieval and
  re-ranking surface the *specific* rule over the *general* one.
- The SQL tools (both fixed and per-session) only allow validated `SELECT` statements — no
  writes, no schema changes, no multi-statement queries — since the LLM generates the query text
  itself and cannot be trusted with anything broader.
- Uploaded-file sessions are tracked in-memory by the API process (a dict of Chroma clients
  keyed by `session_id`); restarting the API loses that in-memory cache, though the underlying
  on-disk data under `data/database/sessions/` survives and gets picked back up on next access.
  There is no session expiry/cleanup — acceptable for a local demo, not for multi-tenant
  production use.
- PDF text extraction works for text-based PDFs; scanned/image-only PDFs with no embedded text
  layer are rejected at upload time with a clear error rather than silently indexing nothing.
- No LangChain/LlamaIndex anywhere in the stack — a deliberate choice to actually implement and
  understand chunking, embedding, retrieval, routing, and tool-calling directly, rather than
  treat them as opaque framework behavior.

---

## 7. Summary

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Synthetic dataset + ingestion pipeline | ✅ 44 products, 18 documents, verified |
| 2 | Baseline dense retrieval + generation | ✅ Verified on conflicting-policy test case |
| 3 | SQL retriever + agentic multi-source routing | ✅ Verified structured/unstructured/both |
| 4 | Cross-encoder re-ranking | ✅ Hit Rate 92.3%→100%, MRR 0.846→0.923 |
| 5 | Citations + hallucination guard | ✅ Verified on unanswerable-question test |
| 6 | Evaluation harness | ✅ 0.989 faithfulness, 1.000 relevancy, 18 test cases |
| 7 | API + CLI wrapper | ✅ Live-tested, quota/error handling in place |
| 8 | File upload (bring-your-own-data) | ✅ CSV/DOCX/combined verified via live API tests |

This project demonstrates the full lifecycle of a production-shaped RAG system: dataset
construction with intentional edge cases, retrieval, agentic multi-source routing, re-ranking,
grounded generation, quantified evaluation, and a usable interface — plus the operational
realities (rate limits, model deprecation, error handling) that a from-scratch tutorial usually
skips.
