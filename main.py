"""FastAPI application for MARA product retrieval and memory orchestration."""

import os
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

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


def call_groq(system_prompt: str, user_message: str) -> str:
    """Return a natural-language answer from Groq or a fallback message."""
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
    """Build the prompt passed to the LLM."""
    mara_lines = []
    for i, p in enumerate(mara_products[:3], 1):
        name = p.get("name", "?")
        details = []

        price = p.get("price_chf")
        if price is not None:
            details.append(f"{price} CHF")

        watt = p.get("wattage")
        if watt is not None:
            details.append(f"{watt}W")

        kelvin = p.get("kelvin")
        if kelvin is not None:
            details.append(f"{int(kelvin) if float(kelvin).is_integer() else kelvin}K")

        finish = p.get("finish")
        if finish:
            details.append(str(finish))

        manufacturer = p.get("manufacturer")
        if manufacturer:
            details.append(str(manufacturer))

        mara_lines.append(f"  {i}. {name}" + (f" — {', '.join(details)}" if details else ""))

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

# In-memory state for the current process. Persistent memory lives in Qdrant.
constraints_store: dict[str, UserConstraints] = {}
browsing_store:    dict[str, list[dict]]       = {}


class ConstraintsRequest(BaseModel):
    user_id:             str
    max_wattage:         Optional[float] = None
    max_price_chf:       Optional[float] = None
    forbidden_materials: list[str]       = Field(default_factory=list)
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
    source_article_id: Optional[int] = None
    source_article_number: Optional[str] = None
    source_l_number:  Optional[int] = None
    name:             str
    manufacturer:     Optional[str] = None
    category:         Optional[str] = None
    family:           Optional[str] = None
    price_chf:        Optional[float] = None
    wattage:          Optional[float] = None
    kelvin:           Optional[float] = None
    material:         Optional[str] = None
    style:            Optional[str] = None
    finish:           Optional[str] = None
    mood:             Optional[str] = None
    room_type:        Optional[str] = None
    image_url:        Optional[str] = None
    tags:             list[str] = Field(default_factory=list)
    similarity_score: float
    final_score:      float
    violations:       list[str] = Field(default_factory=list)


class HydrationTarget(BaseModel):
    rank:                  int
    product_id:            str
    source_article_id:     Optional[int] = None
    source_article_number: Optional[str] = None
    source_l_number:       Optional[int] = None


class FrontendHydration(BaseModel):
    provider:              str
    preferred_key:         str
    ordered_article_ids:   list[int] = Field(default_factory=list)
    ranked_targets:        list[HydrationTarget] = Field(default_factory=list)


class FrontendContractInfo(BaseModel):
    version:               str
    primary_result_field:  str
    hydration_provider:    str
    hydration_key:         str
    baseline_available:    bool
    merge_strategy:        str


class ChatResponse(BaseModel):
    user_id:          str
    query:            str
    llm_reply:        str
    baseline_results: list[dict]
    mara_results:     list[ProductResult]
    violation_count:  int
    constraints_used: dict
    user_context:     dict
    hydration:        FrontendHydration
    frontend:         FrontendContractInfo


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
        source_article_id = p.source_article_id,
        source_article_number = p.source_article_number,
        source_l_number = p.source_l_number,
        name             = p.name,
        manufacturer     = p.manufacturer,
        category         = p.category,
        family           = p.family,
        price_chf        = p.price_chf,
        wattage          = p.wattage,
        kelvin           = p.kelvin,
        material         = p.material,
        style            = p.style,
        finish           = p.finish,
        mood             = p.mood,
        room_type        = p.room_type,
        image_url        = p.image_url,
        tags             = p.tags,
        similarity_score = p.similarity_score,
        final_score      = p.final_score,
        violations       = p.violations,
    )


def build_hydration_payload(products: list[ProductResult]) -> FrontendHydration:
    ranked_targets = [
        HydrationTarget(
            rank=index,
            product_id=product.product_id,
            source_article_id=product.source_article_id,
            source_article_number=product.source_article_number,
            source_l_number=product.source_l_number,
        )
        for index, product in enumerate(products, start=1)
    ]
    ordered_article_ids = [
        product.source_article_id
        for product in products
        if product.source_article_id is not None
    ]
    return FrontendHydration(
        provider="supabase",
        preferred_key="source_article_id",
        ordered_article_ids=ordered_article_ids,
        ranked_targets=ranked_targets,
    )

@app.get("/")
def root():
    return {
        "project":   "MARA",
        "version":   "2.0.0",
        "status":    "running",
        "endpoints": ["/constraints", "/browse", "/chat"],
    }


@app.post("/constraints")
def save_constraints(req: ConstraintsRequest):
    """Store explicit hard constraints for the user."""
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


@app.post("/browse")
def log_browse(req: BrowseRequest):
    """Record a browse event for session state and long-term memory."""
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


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run retrieval, load memory, generate the reply, and return results."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    query_vector = embed(req.message)

    constraints = get_constraints(req.user_id)
    preferences = get_preferences(req.user_id, {
        "preferred_style":  req.preferred_style,
        "preferred_finish": req.preferred_finish,
        "preferred_mood":   req.preferred_mood,
    })
    baseline = run_baseline(query_vector)
    mara     = run_mara(query_vector, constraints, preferences)

    user_context = get_user_context(req.user_id, req.message)

    mara_dicts = [
        {
            "name": p.name,
            "price_chf": p.price_chf,
            "wattage": p.wattage,
            "kelvin": p.kelvin,
            "finish": p.finish,
            "manufacturer": p.manufacturer,
        }
        for p in mara
    ]
    system_prompt = build_llm_prompt(user_context, mara_dicts, baseline)
    llm_reply     = call_groq(system_prompt, req.message)

    if req.preferred_style:
        save_chat_preference(req.user_id, f"prefers {req.preferred_style} style lighting")
    if req.preferred_mood:
        save_chat_preference(req.user_id, f"wants {req.preferred_mood} mood atmosphere")
    if req.preferred_finish:
        save_chat_preference(req.user_id, f"likes {req.preferred_finish} finish")

    mara_models = [scored_to_model(p) for p in mara]
    return ChatResponse(
        user_id          = req.user_id,
        query            = req.message,
        llm_reply        = llm_reply,
        baseline_results = baseline,
        mara_results     = mara_models,
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
        hydration = build_hydration_payload(mara_models),
        frontend = FrontendContractInfo(
            version="2026-03-12",
            primary_result_field="mara_results",
            hydration_provider="supabase",
            hydration_key="source_article_id",
            baseline_available=True,
            merge_strategy="hydrate_supabase_then_enrich_with_mara",
        ),
    )
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
    """Return the current memory context for a user."""
    context = get_user_context(user_id, "show me everything")
    return {
        "user_id":    user_id,
        "structural": context["structural"],
        "semantic":   context["semantic"],
        "episodic":   context["episodic"],
        "summary":    context["summary"],
    }
