# Semantic Cache with Redis and LangChain

A project demonstrating semantic caching for LLM responses using Redis Vector DB, LangChain, and HuggingFace embeddings. The pipeline parses a PDF document, generates FAQ pairs using a Groq LLM, and stores them in a Redis semantic cache for fast, similarity-based retrieval.

## Overview

1. **PDF Parsing** — Uses the [LlamaCloud SDK](https://github.com/run-llama/llama-cloud-py) (`llama-cloud>=1.0`) to upload a PDF and parse it into structured markdown via an agentic OCR workflow.
2. **FAQ Generation** — Uses a Groq LLM (Llama-3.1-8B) via LangChain to extract question/answer pairs from each document section.
3. **Semantic Caching** — Embeds FAQ prompts with a HuggingFace sentence-transformer model and stores them in Redis using RedisVL's `SemanticCache`.
4. **Cache Lookup** — Queries the cache with natural language; returns cached responses for semantically similar questions without calling the LLM again.

## Requirements

- Python 3.9+
- A [Redis Cloud](https://redis.com/try-free/) instance (or local Redis Stack)
- A [Groq API key](https://console.groq.com)
- A [LlamaCloud API key](https://cloud.llamaindex.ai)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Create a `credentials.py` file in the project root with the following variables:

```python
LLAMA_CLOUD_API_KEY = "<your-llamacloud-api-key>"
GROQ_API_KEY        = "<your-groq-api-key>"
HF_TOKEN            = "<your-huggingface-token>"
REDIS_HOST          = "<your-redis-host>"
REDIS_PORT          = <your-redis-port>
REDIS_PASSWORD      = "<your-redis-password>"
```

| Variable | Description |
|---|---|
| `LLAMA_CLOUD_API_KEY` | LlamaCloud API key for PDF parsing |
| `GROQ_API_KEY` | Groq API key for the LLM |
| `HF_TOKEN` | HuggingFace token for embedding model access |
| `REDIS_HOST` | Redis instance hostname |
| `REDIS_PORT` | Redis instance port |
| `REDIS_PASSWORD` | Redis instance password |

## Usage

### Notebook

Open and run `semantic-cache-project.ipynb`:

1. Set credentials and environment variables
2. Initialize the Groq LLM and HuggingFace vectorizer
3. Download the sample PDF (2022 Chevrolet Colorado brochure)
4. Connect to Redis and optionally flush the database
5. Upload and parse the PDF using the LlamaCloud SDK
6. Split the parsed markdown into chunks with `MarkdownNodeParser`
7. Generate FAQ prompt/response pairs with the Groq LLM
8. Embed FAQ prompts and store them in the Redis semantic cache
9. Query the cache with natural language questions

### Streamlit App

```bash
streamlit run app.py
```

**Features:**
- Type any question and get an instant answer from the Redis semantic cache
- Cache misses fall back to the Groq LLM and can optionally be stored for next time
- Sidebar shows live Redis connection status, hit/miss stats, and a hit-rate progress bar
- Adjustable similarity threshold slider (tighter or looser matching)
- **Populate cache** panel — upload a new PDF or use the existing one to re-run the full parse → FAQ → embed → store pipeline
- **Upload your own PDF** directly from the UI — the app parses it with LlamaCloud, automatically extracts FAQ prompt/response pairs using the Groq LLM, and pre-populates the semantic cache, with no notebook or command-line steps required

## Project Structure

```
semantic-cache/
├── app.py                         # Streamlit web app
├── semantic-cache-project.ipynb   # Notebook walkthrough
├── credentials.py                 # API keys and secrets (not committed)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── data/                          # Downloaded PDF files
```

## Key Technologies

- [Streamlit](https://streamlit.io/) — Web app framework for the interactive UI
- [llama-cloud](https://github.com/run-llama/llama-cloud-py) — Official LlamaCloud Python SDK for agentic PDF parsing (`llama-cloud>=1.0`)
- [llama-index-core](https://docs.llamaindex.ai/) — `Document` and `MarkdownNodeParser` for document chunking
- [RedisVL](https://github.com/redis/redis-vl-python) — Vector layer for Redis, provides `SemanticCache`
- [LangChain](https://python.langchain.com/) — LLM orchestration framework
- [LangChain-Groq](https://python.langchain.com/docs/integrations/chat/groq/) — Groq LLM integration
- [sentence-transformers](https://www.sbert.net/) — HuggingFace embedding models

---

## Caching Strategies: Notebook vs. Streamlit App

Both the notebook and the app use the same underlying `SemanticCache` from RedisVL, but they apply fundamentally different caching strategies.

### Notebook — Offline / Batch pre-generated

The notebook follows a **write-then-read** pattern. The entire cache is built before any queries are made:

1. A PDF is parsed into structured markdown with LlamaCloud.
2. The markdown chunks are passed to the Groq LLM to generate FAQ prompt/response pairs in bulk.
3. All FAQ prompts are embedded in a single batch with `redisvl_vectorizer.embed_many(...)`.
4. Every entry is stored in Redis up-front via `cache.store(...)`.
5. Queries are then made with `cache.check(...)` — the cache is fully populated and **never written to again during querying**.
6. A cache miss simply returns an empty list; there is no LLM fallback at query time.
7. The distance threshold is fixed at `0.2` and cannot be changed without re-running cells.

This is best described as **static pre-generated**: the cache is a snapshot derived entirely from a known document, and its contents do not change based on what users ask.

### Streamlit App — Live / Interactive Online Caching

`app.py` follows a **read-through with write-on-miss** pattern. The cache evolves dynamically during user interactions:

1. On every query the cache is checked first (`cache.check(question)`).
2. A **cache hit** returns the stored answer instantly — no LLM call is made.
3. A **cache miss** falls through to the Groq LLM, and the response is optionally written back into the cache (`cache.store(...)`) so the same (or semantically similar) question hits the cache next time.
4. The cache grows organically as users ask new questions; it is not limited to content derived from a single document.
5. The distance threshold is **adjustable at runtime** via the sidebar slider — changing it clears and reinitialises the `SemanticCache` instance without restarting the server.
6. Hit/miss statistics and a hit-rate progress bar give immediate visibility into how well the cache is performing.
7. The app also exposes an optional **"Populate cache from PDF"** panel that re-runs the same batch pipeline as the notebook (parse → FAQ generation → embed → store), allowing you to seed the cache from a document before live queries begin.

### Side-by-Side Comparison

| | Notebook | Streamlit App (`app.py`) |
|---|---|---|
| **Cache population** | Batch, upfront, before any queries | Organic, on-miss writes during live use |
| **LLM at query time** | Never called during querying | Called on cache miss as a fallback |
| **Cache growth** | Static after population step | Grows with every uncached question |
| **Distance threshold** | Hardcoded (`0.2`) | Adjustable via UI slider at runtime |
| **Pre-Generated from PDF** | Primary workflow | Optional — upload any PDF via the sidebar UI; the app parses it and pre-generates FAQs into the cache automatically |
| **Observability** | Raw `cache.check()` output | Live hit/miss counters and hit-rate bar |
| **Use case** | Exploring and validating the pipeline | Production-ready interactive assistant |
