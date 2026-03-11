"""
MARA — Step 3: Decay Engine + Constraint Scoring
==================================================
This is the core MARA logic. It does 3 things:

  1. DECAY    — different memory types fade at different rates
  2. FILTER   — hard constraints are never violated
  3. RERANK   — soft preferences boost/lower scores over time

This file is the brain Simeon owns.
Nursena imports run_mara() and run_baseline() into FastAPI (Step 4).

MARA FORMULA:
  FinalScore = Similarity(product, query)
             × StructuralWeight(constraints)
             × DecayFunction(memory_type, time_elapsed)
"""

import math
import json
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
import os
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTION_HARD = "hard_constraints"
COLLECTION_SOFT = "soft_preferences"
TOP_K           = 5   # how many results to return


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class UserConstraints:
    """
    Hard constraints — what the user absolutely requires.
    MARA will never recommend a product that violates these.

    Example:
        UserConstraints(
            max_wattage=40,
            max_price_chf=200,
            forbidden_materials=["plastic"],
            kelvin_min=2700,
            kelvin_max=3000,
        )
    """
    max_wattage:          Optional[float] = None
    max_price_chf:        Optional[float] = None
    forbidden_materials:  list[str]       = field(default_factory=list)
    kelvin_min:           Optional[float] = None
    kelvin_max:           Optional[float] = None
    room_type:            Optional[str]   = None


@dataclass
class UserPreferences:
    """
    Soft preferences — what the user tends to like.
    These influence scoring but never block results.
    Each carries a time_elapsed (in days) so decay can be applied.

    Example:
        UserPreferences(
            preferred_style="scandinavian",
            preferred_finish="brushed brass",
            preferred_mood="cozy",
            style_age_days=10,     # preference stated 10 days ago
            browsing_age_days=2,   # browsed something 2 days ago
        )
    """
    preferred_style:    Optional[str]  = None
    preferred_finish:   Optional[str]  = None
    preferred_mood:     Optional[str]  = None
    style_age_days:     float          = 0.0   # days since style was stated
    browsing_age_days:  float          = 0.0   # days since last browsing


@dataclass
class ScoredProduct:
    """A product with its final MARA score and violation report."""
    product_id:        str
    name:              str
    price_chf:         float
    wattage:           float
    kelvin:            float
    material:          str
    style:             str
    finish:            str
    mood:              str
    room_type:         str
    image_url:         str
    similarity_score:  float   # raw cosine similarity from Qdrant
    decay_score:       float   # after decay weighting
    final_score:       float   # after constraint weighting
    violations:        list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# 1. DECAY FUNCTION
# ─────────────────────────────────────────────

# Lambda rates per memory type
LAMBDA = {
    "hard":     0.01,   # hard constraints — almost never fade
    "soft":     0.10,   # style/finish preferences — slow drift
    "episodic": 0.30,   # recent browsing — fades fast
}

def decay(initial_score: float, memory_type: str, time_elapsed_days: float) -> float:
    """
    Exponential decay: score × e^(-λ × t)

    memory_type:        "hard" | "soft" | "episodic"
    time_elapsed_days:  how many days ago this memory was created

    Examples:
        decay(1.0, "hard",     180) → 0.165  (still strong after 6 months)
        decay(1.0, "soft",      30) → 0.050  (noticeable fade after 1 month)
        decay(1.0, "episodic",   7) → 0.117  (nearly gone after 1 week)
    """
    lam = LAMBDA.get(memory_type, 0.10)
    return initial_score * math.exp(-lam * time_elapsed_days)


# ─────────────────────────────────────────────
# 2. CONSTRAINT WEIGHT
# ─────────────────────────────────────────────

def constraint_weight(product: dict, constraints: UserConstraints) -> tuple[float, list[str]]:
    """
    Returns a weight between 0.0 and 1.0 based on hard constraint compliance.
    Also returns a list of violations for the demo split-screen.

    Weight = 1.0  → all constraints satisfied
    Weight = 0.0  → at least one hard constraint violated
    (We use 0.0 so violating products always rank below compliant ones.)
    """
    violations = []

    if constraints.max_wattage is not None:
        if product.get("wattage", 0) > constraints.max_wattage:
            violations.append(
                f"× {product['wattage']}W exceeds limit of {constraints.max_wattage}W"
            )

    if constraints.max_price_chf is not None:
        if product.get("price_chf", 0) > constraints.max_price_chf:
            violations.append(
                f"× {product['price_chf']} CHF exceeds budget of {constraints.max_price_chf} CHF"
            )

    if constraints.forbidden_materials:
        mat = product.get("material", "").lower()
        for forbidden in constraints.forbidden_materials:
            if forbidden.lower() in mat:
                violations.append(f"× material '{mat}' is forbidden")

    if constraints.kelvin_min is not None:
        if product.get("kelvin", 0) < constraints.kelvin_min:
            violations.append(
                f"× {product['kelvin']}K below minimum {constraints.kelvin_min}K"
            )

    if constraints.kelvin_max is not None:
        if product.get("kelvin", 0) > constraints.kelvin_max:
            violations.append(
                f"× {product['kelvin']}K above maximum {constraints.kelvin_max}K"
            )

    if constraints.room_type is not None:
        if product.get("room_type", "").lower() != constraints.room_type.lower():
            violations.append(
                f"× room type '{product.get('room_type')}' doesn't match '{constraints.room_type}'"
            )

    weight = 0.0 if violations else 1.0
    return weight, violations


# ─────────────────────────────────────────────
# 3. SOFT PREFERENCE BOOST
# ─────────────────────────────────────────────

def preference_boost(product: dict, preferences: UserPreferences) -> float:
    """
    Adds a small boost (0.0 to 0.3) based on how well the product
    matches soft preferences, weighted by their decay.

    This is additive on top of similarity — it nudges the ranking
    without overriding hard constraint filtering.
    """
    boost = 0.0

    if preferences.preferred_style:
        if product.get("style", "").lower() == preferences.preferred_style.lower():
            # Style preference decays slowly
            boost += 0.15 * decay(1.0, "soft", preferences.style_age_days)

    if preferences.preferred_finish:
        if product.get("finish", "").lower() == preferences.preferred_finish.lower():
            boost += 0.10 * decay(1.0, "soft", preferences.style_age_days)

    if preferences.preferred_mood:
        if product.get("mood", "").lower() == preferences.preferred_mood.lower():
            # Mood is more like episodic — fades slightly faster
            boost += 0.05 * decay(1.0, "episodic", preferences.browsing_age_days)

    return boost


# ─────────────────────────────────────────────
# 4. QDRANT SEARCH HELPERS
# ─────────────────────────────────────────────

def get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def build_qdrant_filter(constraints: UserConstraints) -> Optional[Filter]:
    """
    Builds a Qdrant Filter to pre-filter products before vector search.
    This is more efficient than post-filtering — Qdrant skips non-matching
    vectors entirely.

    Only numeric/exact filters go here.
    Material filtering is done in Python (more flexible for partial matches).
    """
    conditions = []

    if constraints.max_wattage is not None:
        conditions.append(FieldCondition(
            key="wattage",
            range=Range(lte=constraints.max_wattage)
        ))

    if constraints.max_price_chf is not None:
        conditions.append(FieldCondition(
            key="price_chf",
            range=Range(lte=constraints.max_price_chf)
        ))

    if constraints.kelvin_min is not None:
        conditions.append(FieldCondition(
            key="kelvin",
            range=Range(gte=constraints.kelvin_min)
        ))

    if constraints.kelvin_max is not None:
        conditions.append(FieldCondition(
            key="kelvin",
            range=Range(lte=constraints.kelvin_max)
        ))

    if not conditions:
        return None

    return Filter(must=conditions)


# ─────────────────────────────────────────────
# 5. MAIN SEARCH FUNCTIONS
# ─────────────────────────────────────────────

def run_baseline(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """
    Standard RAG — pure cosine similarity, no constraints, no decay.
    This is what the LEFT side of the demo split-screen shows.
    Nursena calls this from FastAPI.
    """
    client = get_client()

    results = client.query_points(
        collection_name=COLLECTION_SOFT,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    output = []
    for r in results:
        p = r.payload
        output.append({
            "product_id":       p.get("product_id"),
            "name":             p.get("name"),
            "style":            p.get("style"),
            "finish":           p.get("finish"),
            "mood":             p.get("mood"),
            "similarity_score": round(r.score, 4),
            "method":           "baseline",
        })

    return output


def run_mara(
    query_vector: list[float],
    constraints: UserConstraints,
    preferences: UserPreferences,
    top_k: int = TOP_K,
) -> list[ScoredProduct]:
    """
    MARA search — constraint-filtered, decay-weighted, preference-boosted.
    This is what the RIGHT side of the demo split-screen shows.
    Nursena calls this from FastAPI.

    Steps:
      1. Pre-filter in Qdrant (wattage, price, kelvin)
      2. Retrieve top candidates from hard_constraints collection
      3. Apply constraint weight (hard violations → score = 0)
      4. Apply decay to soft preferences
      5. Add preference boost
      6. Re-rank and return top_k
    """
    client = get_client()

    # Step 1+2 — Qdrant pre-filter + vector search
    qdrant_filter = build_qdrant_filter(constraints)

    hard_results = client.query_points(
        collection_name=COLLECTION_HARD,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k * 3,    # fetch more, we'll rerank and trim
        with_payload=True,
    ).points

    # Also fetch soft preferences for the same products
    soft_results = client.query_points(
        collection_name=COLLECTION_SOFT,
        query=query_vector,
        limit=top_k * 3,
        with_payload=True,
    ).points

    # Build a lookup: product_id → soft payload
    soft_lookup = {
        r.payload.get("product_id"): r.payload
        for r in soft_results
    }

    scored = []

    for r in hard_results:
        hp = r.payload   # hard payload
        sp = soft_lookup.get(hp.get("product_id"), {})  # soft payload

        similarity = r.score

        # Step 3 — constraint weight (1.0 or 0.0)
        c_weight, violations = constraint_weight(hp, constraints)

        # Step 4 — decay on similarity based on memory age
        # Hard constraints are always fresh (age = 0)
        decayed_similarity = decay(similarity, "hard", 0)

        # Step 5 — soft preference boost (decayed)
        boost = preference_boost(sp, preferences)

        # Final score
        final = (decayed_similarity + boost) * c_weight

        scored.append(ScoredProduct(
            product_id       = hp.get("product_id", ""),
            name             = hp.get("name", ""),
            price_chf        = hp.get("price_chf", 0),
            wattage          = hp.get("wattage", 0),
            kelvin           = hp.get("kelvin", 0),
            material         = hp.get("material", ""),
            style            = sp.get("style", ""),
            finish           = sp.get("finish", ""),
            mood             = sp.get("mood", ""),
            room_type        = hp.get("room_type", ""),
            image_url        = hp.get("image_url", ""),
            similarity_score = round(similarity, 4),
            decay_score      = round(decayed_similarity, 4),
            final_score      = round(final, 4),
            violations       = violations,
        ))

    # Re-rank by final score
    scored.sort(key=lambda x: x.final_score, reverse=True)

    return scored[:top_k]


# ─────────────────────────────────────────────
# 6. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from embeddings import embed

    print("=" * 55)
    print("  MARA Engine — Quick Test")
    print("=" * 55)

    # Simulate judge query
    query = "warm light for a reading corner, cozy scandinavian feel"
    query_vector = embed(query)
    print(f"\nQuery: '{query}'\n")

    # Define hard constraints (what judge sets)
    constraints = UserConstraints(
        max_wattage       = 40,
        max_price_chf     = 200,
        forbidden_materials = ["plastic"],
        kelvin_min        = 2700,
        kelvin_max        = 3200,
    )

    # Define soft preferences (built up over session)
    preferences = UserPreferences(
        preferred_style    = "scandinavian",
        preferred_finish   = "brushed brass",
        preferred_mood     = "cozy",
        style_age_days     = 5,
        browsing_age_days  = 2,
    )

    # ── Baseline ────────────────────────────────
    print("─── BASELINE RAG (no constraints) ──────────────────")
    baseline = run_baseline(query_vector)
    for i, p in enumerate(baseline, 1):
        print(f"  {i}. {p['name']:<35} score: {p['similarity_score']}")

    # ── MARA ────────────────────────────────────
    print("\n─── MARA (constraint-aware + decay) ────────────────")
    mara_results = run_mara(query_vector, constraints, preferences)
    for i, p in enumerate(mara_results, 1):
        status = "✓" if not p.violations else "✗"
        print(f"  {i}. {status} {p.name:<33} final: {p.final_score}  sim: {p.similarity_score}")
        for v in p.violations:
            print(f"       {v}")

    # ── Decay demo ──────────────────────────────
    print("\n─── Decay Demo ─────────────────────────────────────")
    print("  How fast does each memory type fade?\n")
    for days in [0, 7, 30, 90, 180]:
        h = decay(1.0, "hard",     days)
        s = decay(1.0, "soft",     days)
        e = decay(1.0, "episodic", days)
        print(f"  Day {days:>3}  │  hard: {h:.3f}  soft: {s:.3f}  episodic: {e:.4f}")
