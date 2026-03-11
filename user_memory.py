"""
MARA — user_memory.py
======================
Stores and retrieves everything the user has ever told MARA.

This is what makes MARA a LEARNING system — not just a search engine.
Every message, every constraint, every click is remembered here
with the correct decay rate so the right things persist.

THREE MEMORY TYPES:
  structural  (λ = 0.01) — hard constraints, almost never fade
  semantic    (λ = 0.10) — style preferences, slow drift
  episodic    (λ = 0.30) — recent browsing/clicks, fades fast

COLLECTION: "user_memory" (single collection, type stored in payload)

WHO USES THIS FILE:
  Simeon  → owns this file, adds memory after each interaction
  main.py → calls save_memory() and get_user_context()
  No one else needs to touch this file.
"""

import os
import time
import math
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "user_memory"
VECTOR_SIZE     = 1024  # must match embedding model

# Lambda decay rates — same as mara_engine.py
LAMBDA = {
    "structural": 0.01,  # hard constraints — almost never fade
    "semantic":   0.10,  # style/finish preferences — slow drift
    "episodic":   0.30,  # recent clicks/browsing — fades fast
}

# How many memories to retrieve per type when building context
TOP_STRUCTURAL = 5   # always retrieve all hard constraints
TOP_SEMANTIC   = 5   # top style preferences
TOP_EPISODIC   = 3   # only very recent browsing matters


# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    A single memory stored for a user.

    Examples:
        MemoryEntry("user_01", "structural", "max budget 200 CHF")
        MemoryEntry("user_01", "semantic",   "loves scandinavian style")
        MemoryEntry("user_01", "episodic",   "just clicked brass wall sconce")
    """
    user_id:     str
    memory_type: str   # "structural" | "semantic" | "episodic"
    text:        str   # the raw text that gets embedded
    source:      str = "chat"   # "chat" | "browse" | "constraint"


from embeddings import embed


# ─────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────

def _get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


# ─────────────────────────────────────────────
# SETUP — run once
# ─────────────────────────────────────────────

def setup_collection():
    """
    Creates the user_memory collection in Qdrant.
    Run this once — safe to re-run, it checks first.

    Called automatically by save_memory() if collection doesn't exist.
    """
    client = _get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")


# ─────────────────────────────────────────────
# DECAY
# ─────────────────────────────────────────────

def _decay(memory_type: str, timestamp: float) -> float:
    """
    Calculates how much a memory has faded since it was created.

    score × e^(-λ × days_elapsed)

    Returns a multiplier between 0.0 and 1.0.
    1.0 = fresh memory, full weight
    0.0 = completely faded, no influence
    """
    days_elapsed = (time.time() - timestamp) / 86400  # seconds → days
    lam = LAMBDA.get(memory_type, 0.10)
    return math.exp(-lam * days_elapsed)


# ─────────────────────────────────────────────
# SAVE MEMORY
# ─────────────────────────────────────────────

def save_memory(entry: MemoryEntry) -> str:
    """
    Saves a single memory to Qdrant.
    Returns the memory ID.

    Called from main.py after every:
      - /constraints call  → memory_type = "structural"
      - /browse call       → memory_type = "episodic"
      - /chat call         → memory_type = "semantic" (if preference detected)

    Example:
        save_memory(MemoryEntry(
            user_id     = "judge_01",
            memory_type = "structural",
            text        = "maximum wattage 40W",
            source      = "constraint",
        ))
    """
    setup_collection()
    client  = _get_client()
    vector  = embed(entry.text)
    mem_id  = str(uuid.uuid4())
    now     = time.time()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id      = abs(hash(mem_id)) % (2**63),  # Qdrant needs int ID
                vector  = vector,
                payload = {
                    "mem_id":      mem_id,
                    "user_id":     entry.user_id,
                    "memory_type": entry.memory_type,
                    "text":        entry.text,
                    "source":      entry.source,
                    "timestamp":   now,
                    "lambda":      LAMBDA[entry.memory_type],
                },
            )
        ],
    )
    return mem_id


def save_many(entries: list[MemoryEntry]) -> list[str]:
    """Save multiple memories at once. More efficient than calling save_memory in a loop."""
    return [save_memory(e) for e in entries]


# ─────────────────────────────────────────────
# RETRIEVE USER CONTEXT
# ─────────────────────────────────────────────

def get_user_context(user_id: str, query: str) -> dict:
    """
    The main function called from /chat.

    Returns a structured context dict with:
      - structural: hard constraints (never violated)
      - semantic:   style preferences (decay-weighted)
      - episodic:   recent browsing (decay-weighted, fades fast)
      - summary:    plain text for the LLM system prompt

    Example output:
        {
            "structural": [
                {"text": "max budget 200 CHF", "decay_weight": 0.99},
                {"text": "no plastic",          "decay_weight": 0.99},
            ],
            "semantic": [
                {"text": "loves scandinavian style", "decay_weight": 0.87},
            ],
            "episodic": [
                {"text": "clicked brass wall sconce", "decay_weight": 0.43},
            ],
            "summary": "..."   ← injected into LLM system prompt
        }
    """
    client       = _get_client()
    query_vector = embed(query)

    # ── Fetch each memory type separately ────────
    def fetch(memory_type: str, limit: int) -> list[dict]:
        type_filter = Filter(
            must=[
                FieldCondition(key="user_id",     match=MatchValue(value=user_id)),
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type)),
            ]
        )
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=type_filter,
            limit=limit,
            with_payload=True,
        ).points

        memories = []
        for r in results:
            p           = r.payload
            raw_score   = r.score
            decay_w     = _decay(memory_type, p.get("timestamp", time.time()))
            final_score = raw_score * decay_w

            memories.append({
                "text":         p.get("text", ""),
                "memory_type":  memory_type,
                "source":       p.get("source", "chat"),
                "raw_score":    round(raw_score, 4),
                "decay_weight": round(decay_w, 4),
                "final_score":  round(final_score, 4),
                "timestamp":    p.get("timestamp", 0),
            })

        # Re-rank by final score (similarity × decay)
        return sorted(memories, key=lambda x: x["final_score"], reverse=True)

    structural = fetch("structural", TOP_STRUCTURAL)
    semantic   = fetch("semantic",   TOP_SEMANTIC)
    episodic   = fetch("episodic",   TOP_EPISODIC)

    # ── Build plain text summary for LLM ─────────
    lines = []

    if structural:
        lines.append("HARD CONSTRAINTS (must never be violated):")
        for m in structural:
            lines.append(f"  - {m['text']}")

    if semantic:
        lines.append("USER PREFERENCES (learned over time):")
        for m in semantic:
            lines.append(f"  - {m['text']} (confidence: {m['decay_weight']:.0%})")

    if episodic:
        lines.append("RECENT ACTIVITY (fades quickly):")
        for m in episodic:
            lines.append(f"  - {m['text']}")

    summary = "\n".join(lines) if lines else "No prior context for this user."

    return {
        "structural": structural,
        "semantic":   semantic,
        "episodic":   episodic,
        "summary":    summary,
    }


# ─────────────────────────────────────────────
# HELPERS FOR main.py
# ─────────────────────────────────────────────

def save_constraints_as_memory(user_id: str, constraints: dict):
    """
    Called from POST /constraints in main.py.
    Converts constraint fields into structural memories.

    Example:
        save_constraints_as_memory("judge_01", {
            "max_wattage": 40,
            "max_price_chf": 200,
            "forbidden_materials": ["plastic"],
        })
        → saves 3 structural memories
    """
    entries = []

    if constraints.get("max_wattage"):
        entries.append(MemoryEntry(
            user_id="judge_01", memory_type="structural",
            text=f"maximum wattage {constraints['max_wattage']}W",
            source="constraint",
        ))
        entries[-1].user_id = user_id

    if constraints.get("max_price_chf"):
        entries.append(MemoryEntry(
            user_id=user_id, memory_type="structural",
            text=f"maximum budget {constraints['max_price_chf']} CHF",
            source="constraint",
        ))

    for mat in constraints.get("forbidden_materials", []):
        entries.append(MemoryEntry(
            user_id=user_id, memory_type="structural",
            text=f"forbidden material: {mat}",
            source="constraint",
        ))

    if constraints.get("kelvin_min") or constraints.get("kelvin_max"):
        kmin = constraints.get("kelvin_min", "any")
        kmax = constraints.get("kelvin_max", "any")
        entries.append(MemoryEntry(
            user_id=user_id, memory_type="structural",
            text=f"color temperature range {kmin}K to {kmax}K",
            source="constraint",
        ))

    if constraints.get("room_type"):
        entries.append(MemoryEntry(
            user_id=user_id, memory_type="structural",
            text=f"room type: {constraints['room_type']}",
            source="constraint",
        ))

    saved = save_many(entries)
    print(f"  Saved {len(saved)} structural memories for {user_id}")
    return saved


def save_browse_as_memory(user_id: str, product_name: str, product_description: str):
    """
    Called from POST /browse in main.py.
    Saves a product click as an episodic memory.
    """
    entry = MemoryEntry(
        user_id     = user_id,
        memory_type = "episodic",
        text        = f"browsed: {product_name} — {product_description}",
        source      = "browse",
    )
    return save_memory(entry)


def save_chat_preference(user_id: str, preference_text: str):
    """
    Called from POST /chat in main.py when a style preference is detected.
    Example: user says "I love warm cozy lights" → semantic memory.
    """
    entry = MemoryEntry(
        user_id     = user_id,
        memory_type = "semantic",
        text        = preference_text,
        source      = "chat",
    )
    return save_memory(entry)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MARA — User Memory Test")
    print("=" * 55)

    USER = "test_judge"

    # 1. Save structural constraints
    print("\n[1] Saving structural constraints...")
    save_constraints_as_memory(USER, {
        "max_wattage":         40,
        "max_price_chf":       200,
        "forbidden_materials": ["plastic"],
        "kelvin_min":          2700,
        "kelvin_max":          3200,
    })

    # 2. Save semantic preferences
    print("\n[2] Saving semantic preferences...")
    save_chat_preference(USER, "loves scandinavian minimalist style")
    save_chat_preference(USER, "prefers warm light tones, cozy atmosphere")
    save_chat_preference(USER, "likes brushed brass and natural materials")

    # 3. Save episodic browsing
    print("\n[3] Saving episodic browsing...")
    save_browse_as_memory(USER, "Auro Brass Wall Sconce",
        "brushed brass wall sconce warm 2700K scandinavian")
    save_browse_as_memory(USER, "Linen Dome Floor Lamp",
        "fabric floor lamp warm cozy bedroom light")

    # 4. Retrieve context
    print("\n[4] Retrieving user context for query...")
    query   = "I need a light for my reading corner"
    context = get_user_context(USER, query)

    print(f"\nQuery: '{query}'\n")
    print("─── Structural (hard constraints) ──────────────")
    for m in context["structural"]:
        print(f"  [{m['decay_weight']:.2f}] {m['text']}")

    print("\n─── Semantic (preferences) ──────────────────────")
    for m in context["semantic"]:
        print(f"  [{m['decay_weight']:.2f}] {m['text']}")

    print("\n─── Episodic (recent browsing) ──────────────────")
    for m in context["episodic"]:
        print(f"  [{m['decay_weight']:.2f}] {m['text']}")

    print("\n─── LLM Summary ─────────────────────────────────")
    print(context["summary"])
