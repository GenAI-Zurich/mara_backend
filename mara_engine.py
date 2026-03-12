"""Retrieval and reranking logic for MARA."""

import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTION_HARD = "hard_constraints"
COLLECTION_SOFT = "soft_preferences"
TOP_K           = 5   # how many results to return


@dataclass
class UserConstraints:
    """Strict filters applied during retrieval and reranking."""
    max_wattage:          Optional[float] = None
    max_price_chf:        Optional[float] = None
    forbidden_materials:  list[str]       = field(default_factory=list)
    kelvin_min:           Optional[float] = None
    kelvin_max:           Optional[float] = None
    room_type:            Optional[str]   = None


@dataclass
class UserPreferences:
    """Soft signals that can boost ranking without filtering results."""
    preferred_style:    Optional[str]  = None
    preferred_finish:   Optional[str]  = None
    preferred_mood:     Optional[str]  = None
    style_age_days:     float          = 0.0
    browsing_age_days:  float          = 0.0


@dataclass
class ScoredProduct:
    """A product with its final MARA score and violation report."""
    product_id:        str
    source_article_id: Optional[int]
    source_article_number: Optional[str]
    source_l_number:   Optional[int]
    name:              str
    manufacturer:      Optional[str]
    category:          Optional[str]
    family:            Optional[str]
    price_chf:         Optional[float]
    wattage:           Optional[float]
    kelvin:            Optional[float]
    material:          Optional[str]
    style:             Optional[str]
    finish:            Optional[str]
    mood:              Optional[str]
    room_type:         Optional[str]
    image_url:         Optional[str]
    similarity_score:  float
    decay_score:       float
    final_score:       float
    tags:              list[str] = field(default_factory=list)
    violations:        list[str] = field(default_factory=list)

LAMBDA = {
    "hard":     0.01,   # hard constraints — almost never fade
    "soft":     0.10,   # style/finish preferences — slow drift
    "episodic": 0.30,   # recent browsing — fades fast
}


def decay(initial_score: float, memory_type: str, time_elapsed_days: float) -> float:
    """Apply exponential decay to a score."""
    lam = LAMBDA.get(memory_type, 0.10)
    return initial_score * math.exp(-lam * time_elapsed_days)


def constraint_weight(product: dict, constraints: UserConstraints) -> tuple[float, list[str]]:
    """Return the constraint weight and any violations for a product."""
    violations = []

    wattage = product.get("wattage")
    price = product.get("price_chf")
    kelvin = product.get("kelvin")
    material = product.get("material")
    room_type = product.get("room_type")

    if constraints.max_wattage is not None:
        if wattage is None:
            violations.append("× wattage unknown")
        elif wattage > constraints.max_wattage:
            violations.append(
                f"× {wattage}W exceeds limit of {constraints.max_wattage}W"
            )

    if constraints.max_price_chf is not None:
        if price is None:
            violations.append("× price unknown")
        elif price > constraints.max_price_chf:
            violations.append(
                f"× {price} CHF exceeds budget of {constraints.max_price_chf} CHF"
            )

    if constraints.forbidden_materials:
        if material is None:
            violations.append("× material unknown")
        else:
            mat = str(material).lower()
            for forbidden in constraints.forbidden_materials:
                if forbidden.lower() in mat:
                    violations.append(f"× material '{mat}' is forbidden")

    if constraints.kelvin_min is not None:
        if kelvin is None:
            violations.append("× color temperature unknown")
        elif kelvin < constraints.kelvin_min:
            violations.append(
                f"× {kelvin}K below minimum {constraints.kelvin_min}K"
            )

    if constraints.kelvin_max is not None:
        if kelvin is None:
            violations.append("× color temperature unknown")
        elif kelvin > constraints.kelvin_max:
            violations.append(
                f"× {kelvin}K above maximum {constraints.kelvin_max}K"
            )

    if constraints.room_type is not None:
        if room_type is None:
            violations.append("× room type unknown")
        elif str(room_type).lower() != constraints.room_type.lower():
            violations.append(
                f"× room type '{room_type}' doesn't match '{constraints.room_type}'"
            )

    weight = 0.0 if violations else 1.0
    return weight, violations


def preference_boost(product: dict, preferences: UserPreferences) -> float:
    """Return a ranking boost based on soft preference matches."""
    boost = 0.0

    if preferences.preferred_style:
        if product.get("style", "").lower() == preferences.preferred_style.lower():
            boost += 0.15 * decay(1.0, "soft", preferences.style_age_days)

    if preferences.preferred_finish:
        if product.get("finish", "").lower() == preferences.preferred_finish.lower():
            boost += 0.10 * decay(1.0, "soft", preferences.style_age_days)

    if preferences.preferred_mood:
        if product.get("mood", "").lower() == preferences.preferred_mood.lower():
            boost += 0.05 * decay(1.0, "episodic", preferences.browsing_age_days)

    return boost


def get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def build_qdrant_filter(constraints: UserConstraints) -> Optional[Filter]:
    """Build a Qdrant pre-filter from strict numeric constraints."""
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


def run_baseline(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """Return similarity-only search results from the soft collection."""
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
            "source_article_id": p.get("source_article_id"),
            "source_article_number": p.get("source_article_number"),
            "source_l_number": p.get("source_l_number"),
            "name":             p.get("name"),
            "manufacturer":     p.get("manufacturer"),
            "category":         p.get("category"),
            "family":           p.get("family"),
            "style":            p.get("style"),
            "finish":           p.get("finish"),
            "mood":             p.get("mood"),
            "image_url":        p.get("image_url"),
            "tags":             p.get("tags", []),
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
    """Return MARA-ranked results using filters, decay, and preference boosts."""
    client = get_client()

    qdrant_filter = build_qdrant_filter(constraints)

    hard_results = client.query_points(
        collection_name=COLLECTION_HARD,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k * 3,
        with_payload=True,
    ).points

    soft_results = client.query_points(
        collection_name=COLLECTION_SOFT,
        query=query_vector,
        limit=top_k * 3,
        with_payload=True,
    ).points

    soft_lookup = {
        r.payload.get("product_id"): r.payload
        for r in soft_results
    }

    scored = []

    for r in hard_results:
        hp = r.payload   # hard payload
        sp = soft_lookup.get(hp.get("product_id"), {})  # soft payload

        similarity = r.score

        c_weight, violations = constraint_weight(hp, constraints)
        decayed_similarity = decay(similarity, "hard", 0)
        boost = preference_boost(sp, preferences)
        final = (decayed_similarity + boost) * c_weight

        scored.append(ScoredProduct(
            product_id       = hp.get("product_id", ""),
            source_article_id = hp.get("source_article_id"),
            source_article_number = hp.get("source_article_number"),
            source_l_number = hp.get("source_l_number"),
            name             = hp.get("name", ""),
            manufacturer     = hp.get("manufacturer") or sp.get("manufacturer"),
            category         = hp.get("category") or sp.get("category"),
            family           = hp.get("family") or sp.get("family"),
            price_chf        = hp.get("price_chf"),
            wattage          = hp.get("wattage"),
            kelvin           = hp.get("kelvin"),
            material         = hp.get("material"),
            style            = sp.get("style"),
            finish           = sp.get("finish"),
            mood             = sp.get("mood"),
            room_type        = hp.get("room_type"),
            image_url        = hp.get("image_url") or sp.get("image_url"),
            tags             = sp.get("tags", []),
            similarity_score = round(similarity, 4),
            decay_score      = round(decayed_similarity, 4),
            final_score      = round(final, 4),
            violations       = violations,
        ))

    scored.sort(key=lambda x: x.final_score, reverse=True)

    return scored[:top_k]
