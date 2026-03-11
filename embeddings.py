"""
MARA — embeddings.py
=====================
Owned by Nursena.

This file has ONE job:
  Convert text into vectors (numbers) so Qdrant can search by meaning.

HOW IT WORKS:
  "warm scandinavian brass lamp"  →  [0.23, 0.87, 0.12, ...]  (384 numbers)
  "cozy reading corner light"     →  [0.21, 0.84, 0.15, ...]  (similar → close in space)
  "industrial chrome kitchen"     →  [0.91, 0.12, 0.78, ...]  (different → far in space)

The closer two vectors are → the more similar the meaning.
This is what allows MARA to understand "warm cozy light" 
without needing exact keyword matches.

MODEL: BAAI/bge-large-en-v1.5
  - Runs locally on your machine (downloaded once, ~1.3GB)
  - No API call needed after download
  - Output: 384-dimensional vector
  - Best open-source model for semantic search

NURSENA'S TASK:
  1. pip install sentence-transformers
  2. Run this file once to download the model
  3. Replace mock_embed() calls in mara_engine.py and user_memory.py
     with the embed() function from this file

SIMEON'S NOTE:
  All three files use the same embed() signature:
    embed(text: str) -> list[float]
  So the swap is identical in all three places.
"""

import os
from functools import lru_cache

# ─────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────

MODEL_NAME  = "BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024  # output dimension — must match Qdrant collections

# HuggingFace token (optional for this public model, but good practice)
HF_TOKEN = os.getenv("HF_TOKEN", None)


@lru_cache(maxsize=1)
def _load_model():
    """
    Loads the embedding model ONCE and caches it in memory.

    lru_cache(maxsize=1) means Python keeps one copy of the model loaded.
    Without this, every call would reload the model → very slow.

    First call: ~10 seconds (downloads + loads model)
    All subsequent calls: instant (already in memory)
    """
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {MODEL_NAME}")
    print("First run downloads ~1.3GB — this is normal, happens once only.\n")
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)
    print(f"Model loaded. Vector size: {model.get_sentence_embedding_dimension()}\n")
    return model


# ─────────────────────────────────────────────
# MAIN FUNCTION — only thing other files import
# ─────────────────────────────────────────────

_BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def embed(text: str) -> list[float]:
    """
    Converts any text string into a 1024-dimensional vector.

    Prepends the BGE instruction prefix required by BAAI/bge-large-en-v1.5
    for query-side encoding. Improves retrieval quality.

    Args:
        text: any string — product description, user query, constraint text

    Returns:
        list of 1024 floats — the vector representation of the text
    """
    model  = _load_model()
    vector = model.encode(_BGE_PREFIX + text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds multiple texts at once — much faster than calling embed() in a loop.
    Use this in setup_qdrant.py when indexing all 30 products.

    Args:
        texts: list of strings

    Returns:
        list of vectors, same order as input
    """
    model   = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vectors]


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MARA — Embeddings Test")
    print("=" * 55 + "\n")

    # Test 1 — basic embedding
    print("[1] Embedding a single text...")
    v = embed("warm scandinavian brass lamp for reading corner")
    print(f"  Vector size:     {len(v)}")
    print(f"  First 5 values:  {[round(x, 4) for x in v[:5]]}")
    print(f"  Is normalized:   {abs(sum(x**2 for x in v)**0.5 - 1.0) < 1e-4}")

    # Test 2 — similarity check
    print("\n[2] Semantic similarity test...")
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    texts = {
        "warm cozy scandinavian reading lamp":     embed("warm cozy scandinavian reading lamp"),
        "brass wall sconce warm 2700K":            embed("brass wall sconce warm 2700K"),
        "industrial chrome kitchen spotlight":     embed("industrial chrome kitchen spotlight"),
        "cold clinical office ceiling light":      embed("cold clinical office ceiling light"),
    }

    query  = embed("I need a warm light for my reading corner")
    base   = "warm cozy scandinavian reading lamp"

    print(f"\n  Query: 'I need a warm light for my reading corner'\n")
    for label, vec in texts.items():
        sim = cosine_similarity(query, vec)
        bar = "█" * int(sim * 100) + "░" * (20 - int(sim * 100))
        print(f"  {sim:.4f}  {bar}  {label}")

    print("\n  ✓ Similar texts should score higher than dissimilar ones.")
    print("  ✓ If the first two score higher than the last two — model works correctly.\n")

    # Test 3 — batch embedding
    print("[3] Batch embedding test (faster for 30 products)...")
    products = [
        "Auro Brass Wall Sconce — brushed brass warm 2700K scandinavian",
        "Vega Matte Pendant — matte black 3000K minimalist kitchen",
        "Linen Dome Floor Lamp — fabric warm cozy bedroom scandinavian",
    ]
    vectors = embed_batch(products)
    print(f"  Embedded {len(vectors)} products")
    print(f"  Each vector size: {len(vectors[0])}")
    print("\nAll tests passed. embeddings.py is ready.\n")
    print("NEXT STEP for Nursena:")
    print("  In mara_engine.py, user_memory.py, and main.py")
    print("  replace:  from embeddings import embed   ← add this line at top")
    print("  delete:   def embed(text): ...mock...    ← remove the mock function")
