# Multi-Source RAG with Agentic Routing & Re-ranking

A retrieval-augmented generation system that answers questions over **two different data
sources** — a structured product catalog (SQL) and unstructured policy/support documents
(vector search) — by letting the LLM itself decide which source(s) a question needs, then
re-ranking retrieved passages with a cross-encoder before generating a grounded, cited answer.

Built as a learning project to go beyond a basic "embed docs, ask questions" RAG demo and
into the parts that are usually skipped: multi-source routing, re-ranking, and — critically —
**measuring** whether any of it actually works, instead of eyeballing a few example outputs.

## Why this is more than a chatbot demo

Most RAG tutorials stop at "retrieve top-k chunks, stuff them in a prompt." This project adds
the three things that actually separate a working RAG system from a fragile one:

1. **Agentic multi-source routing** — no hand-written classifier. The LLM decides via function
   calling whether a question needs SQL (`How many laptops are under $1000?`), document search
   (`What's the return policy for accessories?`), or both (`Is the NovaView 27 QHD in stock, and
   is it covered under warranty for dead pixels?`) — and can call both tools in a single turn.
2. **Cross-encoder re-ranking** — dense (bi-encoder) retrieval is fast but approximate. A
   cross-encoder re-scores a wider candidate pool before anything reaches the LLM, measurably
   improving retrieval accuracy (see results below).
3. **A real evaluation harness** — a labeled test set with known-correct sources/answers,
   scored with retrieval metrics (Hit Rate, MRR) and an LLM-judge for generation quality
   (faithfulness, relevancy, correctness). Numbers, not vibes.
4. **Bring-your-own-data mode** — upload a CSV, PDF, DOCX, TXT, or MD file through the web UI
   and the same agentic routing + retrieval pattern runs against your own data, fully isolated
   from the fixed demo dataset and from other users' sessions.

## Architecture

```
                              ┌─────────────────────┐
                              │   User question      │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Gemini (agentic)     │
                              │  decides which tool(s) │
                              │  to call — no separate │
                              │  router model needed   │
                              └──────┬─────────┬──────┘
                     ┌───────────────┘         └───────────────┐
                     ▼                                         ▼
        ┌────────────────────────┐             ┌──────────────────────────────┐
        │  run_sql_query          │             │  search_documents             │
        │  (text-to-SQL)          │             │  (dense retrieval + rerank)   │
        │                         │             │                                │
        │  SELECT-only validated  │             │  bi-encoder top-20            │
        │  SQL against products.db│             │  → cross-encoder top-5        │
        └────────────┬────────────┘             └───────────────┬────────────────┘
                     │                                          │
                     └───────────────────┬──────────────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  Grounded answer with  │
                              │  cited sources          │
                              └──────────────────────┘
```

The **file upload mode** runs the identical pattern (agentic tool-calling, dense retrieval +
re-ranking, SELECT-only SQL) against a per-session SQLite DB and Chroma collection built from
whatever you upload, instead of the fixed `products.db` / `novatech_docs` store — see
[Uploading your own files](#uploading-your-own-files).

## Results

Everything below is measured against a held-out set of 18 labeled test questions
(`data/eval/test_set.json`), not eyeballed.

**Retrieval quality — before vs. after cross-encoder re-ranking**

| Metric | Bi-encoder only | + Cross-encoder re-rank |
|---|---|---|
| Hit Rate@5 | 92.3% | **100%** |
| MRR | 0.846 | **0.923** |

One test question was retrieved *nowhere* in the bi-encoder's top-5 and was correctly recovered
to rank #1 after re-ranking — proof re-ranking isn't just reordering, it recovers real misses.

**Generation quality — LLM-judged across question categories**

| Category | n | Faithfulness | Relevancy | Correctness |
|---|---|---|---|---|
| Unstructured (docs) | 10 | 0.980 | 1.000 | 0.990 |
| Structured (SQL) | 4 | 1.000 | 1.000 | 1.000 |
| Combined (both) | 2 | 1.000 | 1.000 | 1.000 |
| Unanswerable | 2 | 1.000 | 1.000 | 1.000 |
| **Overall** | 18 | **0.989** | **1.000** | — |

*Faithfulness* = are all claims in the answer actually supported by retrieved context (no
hallucination). *Unanswerable* cases (e.g. "Who is the CEO?") are specifically tested to confirm
the system says "not in the available data" instead of guessing.

## Tech stack

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), local, free
- **Vector store**: ChromaDB (persistent, local)
- **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Structured data**: SQLite, queried via LLM-generated (validated, SELECT-only) SQL
- **LLM**: Google Gemini (`gemini-3.5-flash-lite`, free tier), via native function calling —
  model name is env-configurable
- **API**: FastAPI + Uvicorn
- **Frontend**: Streamlit (chat UI + file upload)
- **File parsing**: `pypdf` (PDF), `python-docx` (DOCX), `pandas` (CSV)
- **Evaluation**: hand-built LLM-as-judge (faithfulness/relevancy/correctness) + custom
  retrieval metrics (Hit Rate, MRR)

No LangChain/LlamaIndex — every retrieval, routing, and generation step is implemented directly
against the vector store, database, and LLM SDK, to actually learn the mechanics rather than
call a framework.

## Project structure

```
data/
  csv/products.csv          synthetic product catalog (44 products, 10 categories)
  documents/                 18 policy/FAQ/manual/support-ticket documents
  database/                  products.db (SQLite) + chroma/ (vector store)
                              sessions/<session_id>/ — per-upload SQLite + Chroma (gitignored)
  eval/test_set.json         18 labeled test questions for evaluation

src/
  ingestion/                 dataset generation, chunking, embedding, DB loading
                              file_extractors.py — PDF/DOCX/TXT/MD/CSV parsing for uploads
  retrieval/                 dense retriever + SQL retriever (both exposed as LLM tools)
                              embeddings.py — shared embedding-model singleton
                              sql_safety.py — shared SELECT-only SQL validator
                              session_store.py — per-session Chroma collection + SQLite DB
  reranking/                 cross-encoder re-ranking + before/after comparison script
  generation/                baseline RAG pipeline + the agentic multi-source pipeline
                              session_pipeline.py — agentic pipeline over one session's uploads
  evaluation/                retrieval metrics, LLM judge, generation eval orchestrator

app/
  cli.py                     interactive terminal chat
  api.py                     FastAPI wrapper (POST /ask, POST /upload, GET /health)
  streamlit_app.py           chat UI, with a demo-data / uploaded-files mode toggle
```

## Setup

```bash
cd multi-source-rag
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):
```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

Build the datastores once:
```bash
python -m src.ingestion.run_ingestion
```

## Usage

**Interactive CLI:**
```bash
python -m app.cli
```

**API server:**
```bash
uvicorn app.api:app --reload
```
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is the NovaView 27 QHD in stock, and is it covered under warranty for dead pixels?"}'
```

**Web UI:** (needs the API server running first)
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501 — a chat interface with a sidebar toggle between the NovaTech demo
data and your own uploaded files.

## Uploading your own files

Switch the sidebar to **"My uploaded files"**, upload a CSV, PDF, DOCX, TXT, or MD file, then
ask questions about it — the same agentic SQL + document routing runs against your upload
instead of the NovaTech dataset.

- **CSV** → loaded into a private, session-scoped SQLite table; questions run through the same
  validated text-to-SQL path as the product catalog (filtering, aggregation, sorting all work).
- **PDF / DOCX / TXT / MD** → chunked and embedded into a private, session-scoped Chroma
  collection; questions run through the same dense-retrieval path as the policy documents.
- Multiple files in one session are combined automatically — a question that needs both a CSV
  and a document gets answered by calling both tools in one turn, exactly like the NovaTech
  "both" category in the eval results above.
- Nothing you upload touches the fixed demo dataset or persists across sessions — each upload
  gets its own isolated `data/database/sessions/<session_id>/` directory (gitignored).

Via the API directly:
```bash
curl -X POST http://127.0.0.1:8000/upload -F "file=@sales_data.csv"
# → {"session_id": "...", "filename": "sales_data.csv", "kind": "table", "summary": "..."}

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which region had the highest revenue?", "session_id": "<from above>"}'
```

**Run the evaluation suite:**
```bash
python -m src.evaluation.retrieval_eval
python -m src.evaluation.generation_eval
```

**See re-ranking's effect directly:**
```bash
python -m src.reranking.compare_rerank
```

## Design notes / known limitations

- The dataset (NovaTech Electronics) is entirely synthetic and LLM-generated, deliberately
  including conflicting rules (e.g. a laptop's product manual overrides the general return
  policy) to stress-test whether retrieval and re-ranking surface the *specific* rule over the
  *general* one.
- The SQL tool only allows validated `SELECT` statements — no writes, no schema changes, no
  multi-statement queries — since the LLM generates the query text itself.
- Gemini's free tier has low daily request quotas that vary by model; `gemini-3.5-flash-lite`
  was chosen after `gemini-3.6-flash` hit a 20-request/day cap during development. The model
  name is a single env var (`GEMINI_MODEL`), not hardcoded, since Google has deprecated models
  mid-project more than once during this build.
- Uploaded-file sessions are tracked in-memory by the API process (a dict of Chroma clients
  keyed by `session_id`) — restarting the API loses that cache, though the underlying data on
  disk under `data/database/sessions/` survives and gets picked up again on next access.
  There's no session expiry/cleanup, since this is a local demo, not a multi-tenant deployment.
- PDF text extraction (`pypdf`) works on text-based PDFs; scanned/image-only PDFs with no
  embedded text layer are rejected at upload time with a clear error rather than silently
  indexing nothing.
