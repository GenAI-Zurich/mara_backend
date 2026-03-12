# MARA

MARA is a memory-augmented retail agent for lighting search. It combines vector retrieval, persistent user memory, and hard-constraint enforcement so recommendations stay relevant across sessions.

## Core Idea

Standard retrieval ranks products by similarity only. MARA adds:

- hard constraints that should never be violated
- soft preferences that influence ranking
- episodic browsing signals that decay quickly

## Architecture

- `Supabase` is the source of truth for the catalog
- `extract_supabase_catalog.py` normalizes that catalog into MARA's canonical schema
- `Qdrant` stores retrieval-optimized product vectors and user memory
- `FastAPI` orchestrates retrieval, memory lookup, and response generation
- `Groq / Llama 3.3` generates the natural-language reply

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env_example .env
python extract_supabase_catalog.py --output catalog_export.json
python setup_qdrant.py
python -m uvicorn main:app --reload --port 8001
```

## Required Environment Variables

```bash
QDRANT_URL=https://your-qdrant-cluster-url:6333
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=gsk_your_groq_api_key
HF_TOKEN=hf_your_huggingface_token
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
```

## API

- `POST /constraints` stores explicit hard constraints
- `POST /browse` stores browse events as episodic memory
- `POST /chat` returns the LLM reply, ranked results, and hydration metadata
- `GET /debug/constraints/{user_id}` inspects active constraints
- `GET /debug/history/{user_id}` inspects browse history
- `GET /debug/memory/{user_id}` inspects stored memory

## Key Files

- `main.py` FastAPI application
- `mara_engine.py` retrieval, filtering, and reranking
- `user_memory.py` user memory persistence and retrieval
- `embeddings.py` embedding helpers
- `extract_supabase_catalog.py` catalog extraction and normalization
- `setup_qdrant.py` Qdrant indexing
- `CATALOG_SCHEMA.md` canonical catalog schema
- `FRONTEND_CONTRACT.md` frontend integration contract

## Frontend Contract

The backend returns MARA-ranked results plus hydration metadata. Frontends should:

1. call `POST /chat`
2. preserve MARA rank order
3. hydrate product details from Supabase using `source_article_id`
4. merge Supabase product data with MARA scores and metadata
