# MARA — Memory-Augmented Retail Agent

> "Amazon remembers what you clicked. MARA remembers who you are."

## What is MARA?

MARA is an AI lighting assistant that learns from its users across sessions.
Most retail AI agents treat all memory equally — a hard budget constraint fades
just like a random browsing click. MARA solves this with a three-layer memory
architecture where different types of information decay at different rates.

A judge sets their constraints once. MARA never forgets them.
A user browses a product. MARA remembers it briefly, then lets it fade.
A user mentions they love scandinavian style. MARA holds that for weeks.

## The Core Innovation

Standard RAG scores products by similarity only:
```
Score = Similarity(product, query)
```

MARA reparameterizes the retrieval space:
```
Score = Similarity(product, query)
      × StructuralWeight(constraints)
      × DecayFunction(memory_type, time_elapsed)
```

This means hard constraints (budget, wattage, material) never get violated.
Soft preferences (style, mood, finish) influence results but fade over time.
Recent browsing has small influence and disappears within days.

## The Demo

Split-screen comparison showing baseline RAG vs MARA side by side.

**Judge sets constraints:** max 40W, under 200 CHF, no plastic, warm white only

**Baseline RAG recommends:**
- Chrome Ceiling Spot — 60W, 240 CHF, plastic ❌ violates everything

**MARA recommends:**
- Brass Wall Sconce — 35W, 175 CHF, metal, warm 2700K ✓ all constraints respected

A live violation counter updates in real time. The judge sets their own
constraints and watches MARA hold them across the entire conversation.

## Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn qdrant-client sentence-transformers groq python-dotenv

# 2. Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# 3. Create .env file
cp .env.example .env
# Fill in GROQ_API_KEY and HF_TOKEN

# 4. Index the product catalog
python3 setup_qdrant.py

# 5. Start the API
uvicorn main:app --reload --port 8001

# 6. Open API docs
open http://localhost:8001/docs
```

## Environment Variables

```bash
# .env
QDRANT_URL=http://localhost:6333      # local Docker
QDRANT_API_KEY=                       # leave empty for local
GROQ_API_KEY=gsk_xxxxx               # from console.groq.com (free)
HF_TOKEN=hf_xxxxx                    # from huggingface.co (free)
```

## File Structure

```
mara/
├── products.json       # 30 synthetic lighting products (Step 1)
├── setup_qdrant.py     # indexes catalog into Qdrant (Step 2)
├── mara_engine.py      # decay engine + constraint scoring (Step 3)
├── user_memory.py      # user learning across sessions (Step 4)
├── embeddings.py       # HuggingFace BGE model (Nursena)
├── main.py             # FastAPI endpoints, wires everything
├── README.md           # this file
├── ARCHITECTURE.md     # technical deep dive
└── CONTEXT.md          # current status + team roles
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/constraints` | Judge sets hard rules |
| POST | `/browse` | User clicks a product |
| POST | `/chat` | Main search + LLM reply |
| GET | `/debug/memory/{user_id}` | See what MARA remembers |

## Tech Stack

| Layer | Technology | Owner |
|-------|-----------|-------|
| Frontend | Lovable (React) | Lu |
| API | FastAPI (Python) | Nursena |
| Vector DB | Qdrant Cloud | Simeon |
| Embeddings | HuggingFace BGE-large | Nursena |
| LLM | Groq / Llama 3.3 70B | Nursena |
| Memory Engine | Custom decay engine | Simeon |

## Team

- **Simeon** — Agent Core, Memory Architecture, Qdrant
- **Nursena** — FastAPI, Embeddings, LLM (Groq)
- **Lu** — Frontend (Lovable), Product Catalog
- **Domenica** — Business Model, Investor Story
- **Ben** — Demo Video, Pitch, Submission
