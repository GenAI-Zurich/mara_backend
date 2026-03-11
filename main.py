"""
MARA — main.py (Final Version)
================================
The central hub. Wires everything together.

WHAT THIS FILE DOES:
  1. Receives requests from Lu's Lovable frontend
  2. Searches the product catalog via mara_engine.py
  3. Reads + writes user memory via user_memory.py
  4. Generates a natural language reply via Groq (Nursena's LLM)
  5. Returns everything to the frontend in one response

FILES THIS IMPORTS:
  mara_engine.py   → product search (Simeon)
  user_memory.py   → user learning (Simeon)
  embeddings.py    → text → vectors (Nursena) ← swap mock when ready

HOW TO RUN:
  uvicorn main:app --reload --port 8001

ENDPOINTS:
  POST /constraints → judge sets hard rules
  POST /browse      → user clicks a product
  POST /chat        → main search + LLM reply

FULL RESPONSE FROM /chat:
  baseline_results  → left panel  (standard RAG, no constraints)
  mara_results      → right panel (constraint-aware + memory)
  llm_reply         → natural language response from Groq
  user_context      → what MARA remembers about this user
  violation_count   → live counter for Lu's frontend
"""

import os
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Load environment variables ───────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── MARA modules ─────────────────────────────
from mara_engine import (
    run_baseline,
    run_mara,
    UserConstraints,
    UserPreferences,
    ScoredProduct,
)
from user_memory import (
    save_constraints_as_memory,
    save_browse_as_memory,
    save_chat_preference,
    get_user_context,
)
from embeddings import embed


# ─────────────────────────────────────────────
# GROQ LLM
# ─────────────────────────────────────────────

def call_groq(system_prompt: str, user_message: str) -> str:
    """
    Calls Groq API with Llama 3.3 70B.
    Returns a natural language response string.

    If GROQ_API_KEY is missing → returns a fallback message.
    API still works even without Groq configured.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "LLM not configured — add GROQ_API_KEY to .env. "
            "Product results are still available above."
        )

    try:
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            temperature = 0.6,
            max_tokens  = 400,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Groq error: {str(e)}"


def build_llm_prompt(
    user_context: dict,
    mara_products: list,
    baseline_products: list,
) -> str:
    """
    Builds the system prompt injected into Groq.
    Combines user memory + MARA results so the reply feels personal.
    """
    mara_lines = []
    for i, p in enumerate(mara_products[:3], 1):
        name  = p.get("name", "?")
        price = p.get("price_chf", "?")
        watt  = p.get("wattage", "?")
        mat   = p.get("material", "?")
        mara_lines.append(f"  {i}. {name} — {price} CHF, {watt}W, {mat}")

    baseline_top = baseline_products[0]["name"] if baseline_products else "unknown"

    return f"""You are MARA, a memory-augmented lighting assistant.
You remember this user's preferences and constraints across sessions.

WHAT YOU KNOW ABOUT THIS USER:
{user_context.get("summary", "No prior context.")}

YOUR TOP RECOMMENDATIONS (constraint-aware):
{chr(10).join(mara_lines) if mara_lines else "No matching products found."}

WITHOUT YOUR MEMORY (standard search would suggest):
  {baseline_top}

YOUR RULES:
1. Be warm, concise, and confident — max 3 sentences.
2. Reference the user's specific constraints or style preferences naturally.
3. Recommend from YOUR list above, not from the standard search result.
4. Never mention "MARA", "baseline", "vectors", or technical terms.
5. If no products match, say so honestly and suggest relaxing a constraint."""


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "MARA API",
    description = "Memory-Augmented Retail Agent — Qdrant-powered lighting recommendations",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# SESSION STORE (in-memory)
# Constraints and browsing history for this session.
# user_memory.py persists everything to Qdrant across sessions.
# ─────────────────────────────────────────────

constraints_store: dict[str, UserConstraints] = {}
browsing_store:    dict[str, list[dict]]       = {}


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class ConstraintsRequest(BaseModel):
    user_id:             str
    max_wattage:         Optional[float] = None
    max_price_chf:       Optional[float] = None
    forbidden_materials: list[str]       = []
    kelvin_min:          Optional[float] = None
    kelvin_max:          Optional[float] = None
    room_type:           Optional[str]   = None

class BrowseRequest(BaseModel):
    user_id:     str
    product_id:  str
    name:        str
    description: str

class ChatRequest(BaseModel):
    user_id:          str
    message:          str
    preferred_style:  Optional[str] = None
    preferred_finish: Optional[str] = None
    preferred_mood:   Optional[str] = None

class ProductResult(BaseModel):
    product_id:       str
    name:             str
    price_chf:        float
    wattage:          float
    kelvin:           float
    material:         str
    style:            str
    finish:           str
    mood:             str
    room_type:        str
    image_url:        str
    similarity_score: float
    final_score:      float
    violations:       list[str]

class ChatResponse(BaseModel):
    user_id:          str
    query:            str
    llm_reply:        str
    baseline_results: list[dict]
    mara_results:     list[ProductResult]
    violation_count:  int
    constraints_used: dict
    user_context:     dict


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_constraints(user_id: str) -> UserConstraints:
    return constraints_store.get(user_id, UserConstraints())


def get_preferences(user_id: str, overrides: dict) -> UserPreferences:
    history      = browsing_store.get(user_id, [])
    browsing_age = 0.0

    if history:
        last  = history[-1]["timestamp"]
        now   = datetime.now(timezone.utc)
        delta = now - datetime.fromisoformat(last)
        browsing_age = delta.total_seconds() / 86400

    return UserPreferences(
        preferred_style   = overrides.get("preferred_style"),
        preferred_finish  = overrides.get("preferred_finish"),
        preferred_mood    = overrides.get("preferred_mood"),
        style_age_days    = 0.0,
        browsing_age_days = browsing_age,
    )


def scored_to_model(p: ScoredProduct) -> ProductResult:
    return ProductResult(
        product_id       = p.product_id,
        name             = p.name,
        price_chf        = p.price_chf,
        wattage          = p.wattage,
        kelvin           = p.kelvin,
        material         = p.material,
        style            = p.style,
        finish           = p.finish,
        mood             = p.mood,
        room_type        = p.room_type,
        image_url        = p.image_url,
        similarity_score = p.similarity_score,
        final_score      = p.final_score,
        violations       = p.violations,
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "project":   "MARA",
        "version":   "2.0.0",
        "status":    "running",
        "endpoints": ["/constraints", "/browse", "/chat"],
    }


# ── ENDPOINT 1: /constraints ─────────────────
@app.post("/constraints")
def save_constraints(req: ConstraintsRequest):
    """
    Judge sets their hard rules.
    Saved in two places:
      1. constraints_store (in-memory) → mara_engine uses this session
      2. user_memory (Qdrant)          → persists as structural memory (λ=0.01)
    """
    constraints = UserConstraints(
        max_wattage         = req.max_wattage,
        max_price_chf       = req.max_price_chf,
        forbidden_materials = req.forbidden_materials,
        kelvin_min          = req.kelvin_min,
        kelvin_max          = req.kelvin_max,
        room_type           = req.room_type,
    )
    constraints_store[req.user_id] = constraints

    save_constraints_as_memory(req.user_id, {
        "max_wattage":         req.max_wattage,
        "max_price_chf":       req.max_price_chf,
        "forbidden_materials": req.forbidden_materials,
        "kelvin_min":          req.kelvin_min,
        "kelvin_max":          req.kelvin_max,
        "room_type":           req.room_type,
    })

    return {
        "status":      "saved",
        "user_id":     req.user_id,
        "memory":      "saved to Qdrant as structural (λ=0.01)",
        "constraints": {
            "max_wattage":         req.max_wattage,
            "max_price_chf":       req.max_price_chf,
            "forbidden_materials": req.forbidden_materials,
            "kelvin_min":          req.kelvin_min,
            "kelvin_max":          req.kelvin_max,
            "room_type":           req.room_type,
        },
    }


# ── ENDPOINT 2: /browse ──────────────────────
@app.post("/browse")
def log_browse(req: BrowseRequest):
    """
    User clicks a product.
    Saved in two places:
      1. browsing_store (in-memory) → preference timing this session
      2. user_memory (Qdrant)       → episodic memory (λ=0.30, fades fast)
    """
    if req.user_id not in browsing_store:
        browsing_store[req.user_id] = []

    browsing_store[req.user_id].append({
        "product_id": req.product_id,
        "name":       req.name,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    save_browse_as_memory(req.user_id, req.name, req.description)

    return {
        "status":        "logged",
        "user_id":       req.user_id,
        "product_id":    req.product_id,
        "history_count": len(browsing_store[req.user_id]),
        "memory":        "saved to Qdrant as episodic (λ=0.30)",
    }


# ── ENDPOINT 3: /chat ────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    The main endpoint. Runs in 6 steps:

      1. Embed user message
      2. Search product catalog — baseline + MARA in parallel
      3. Load user memory from Qdrant
      4. Call Groq LLM → natural language reply
      5. Save new preferences detected in this message
      6. Return everything for split-screen demo
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Step 1 — embed
    query_vector = embed(req.message)

    # Step 2 — search products
    constraints = get_constraints(req.user_id)
    preferences = get_preferences(req.user_id, {
        "preferred_style":  req.preferred_style,
        "preferred_finish": req.preferred_finish,
        "preferred_mood":   req.preferred_mood,
    })
    baseline = run_baseline(query_vector)
    mara     = run_mara(query_vector, constraints, preferences)

    # Step 3 — load user memory
    user_context = get_user_context(req.user_id, req.message)

    # Step 4 — LLM reply
    mara_dicts = [
        {"name": p.name, "price_chf": p.price_chf,
         "wattage": p.wattage, "material": p.material}
        for p in mara
    ]
    system_prompt = build_llm_prompt(user_context, mara_dicts, baseline)
    llm_reply     = call_groq(system_prompt, req.message)

    # Step 5 — save new preferences
    if req.preferred_style:
        save_chat_preference(req.user_id, f"prefers {req.preferred_style} style lighting")
    if req.preferred_mood:
        save_chat_preference(req.user_id, f"wants {req.preferred_mood} mood atmosphere")
    if req.preferred_finish:
        save_chat_preference(req.user_id, f"likes {req.preferred_finish} finish")

    # Step 6 — return
    return ChatResponse(
        user_id          = req.user_id,
        query            = req.message,
        llm_reply        = llm_reply,
        baseline_results = baseline,
        mara_results     = [scored_to_model(p) for p in mara],
        violation_count  = sum(len(p.violations) for p in mara),
        constraints_used = {
            "max_wattage":         constraints.max_wattage,
            "max_price_chf":       constraints.max_price_chf,
            "forbidden_materials": constraints.forbidden_materials,
            "kelvin_min":          constraints.kelvin_min,
            "kelvin_max":          constraints.kelvin_max,
            "room_type":           constraints.room_type,
        },
        user_context = {
            "structural_count": len(user_context["structural"]),
            "semantic_count":   len(user_context["semantic"]),
            "episodic_count":   len(user_context["episodic"]),
            "summary":          user_context["summary"],
        },
    )


# ─────────────────────────────────────────────
# DEBUG ENDPOINTS — remove before final demo
# ─────────────────────────────────────────────

@app.get("/debug/constraints/{user_id}")
def debug_constraints(user_id: str):
    c = constraints_store.get(user_id)
    if not c:
        return {"user_id": user_id, "constraints": None}
    return {
        "user_id": user_id,
        "constraints": {
            "max_wattage":         c.max_wattage,
            "max_price_chf":       c.max_price_chf,
            "forbidden_materials": c.forbidden_materials,
            "kelvin_min":          c.kelvin_min,
            "kelvin_max":          c.kelvin_max,
            "room_type":           c.room_type,
        }
    }

@app.get("/debug/history/{user_id}")
def debug_history(user_id: str):
    return {
        "user_id": user_id,
        "count":   len(browsing_store.get(user_id, [])),
        "history": browsing_store.get(user_id, []),
    }

@app.get("/debug/memory/{user_id}")
async def debug_memory(user_id: str):
    """
    Shows everything MARA remembers about a user.
    Demo moment — judge watches their memory grow in real time.
    """
    context = get_user_context(user_id, "show me everything")
    return {
        "user_id":    user_id,
        "structural": context["structural"],
        "semantic":   context["semantic"],
        "episodic":   context["episodic"],
        "summary":    context["summary"],
    }
