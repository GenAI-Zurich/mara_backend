"""
MARA — Step 2: Qdrant Collections Setup
========================================
This script does 3 things:
  1. Connects to your Qdrant instance (local Docker or Qdrant Cloud)
  2. Creates two collections: hard_constraints and soft_preferences
  3. Loads all 30 products from products.json and indexes them

Run this ONCE before starting the API.
After this, your products are searchable in Qdrant.

HOW TO RUN:
  Local Docker:  python3 setup_qdrant.py
  Qdrant Cloud:  Set QDRANT_URL and QDRANT_API_KEY in the config below
"""

import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

# ─────────────────────────────────────────────
# CONFIG — change these for Qdrant Cloud
# ─────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)   # None = no auth (local)

# Collection names
COLLECTION_HARD = "hard_constraints"   # budget, wattage, material — never fades
COLLECTION_SOFT = "soft_preferences"   # style, mood, finish — decays over time

# Vector size — must match embedding model output
# BGE-large-en-v1.5 outputs 1024-dimensional vectors
VECTOR_SIZE = 1024

# Product catalog file (built in Step 1)
PRODUCTS_FILE = "products.json"


from embeddings import embed


# ─────────────────────────────────────────────
# CONNECT
# ─────────────────────────────────────────────
def connect() -> QdrantClient:
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)
    print("Connected.\n")
    return client


# ─────────────────────────────────────────────
# CREATE COLLECTIONS
# ─────────────────────────────────────────────
def create_collections(client: QdrantClient):
    """
    Creates two collections with cosine distance.
    Deletes existing ones first so this script is safe to re-run.
    """
    for name in [COLLECTION_HARD, COLLECTION_SOFT]:
        if client.collection_exists(name):
            print(f"  Deleting existing collection: {name}")
            client.delete_collection(name)

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"  Created collection: {name}")

    print()


# ─────────────────────────────────────────────
# BUILD PAYLOADS
# ─────────────────────────────────────────────
def build_hard_payload(product: dict) -> dict:
    """
    Hard constraint fields — what MARA must NEVER forget.
    These have a very low decay rate (lambda = 0.01).
    Even after 6 months, these constraints stay strong.
    """
    return {
        "product_id":   product["id"],
        "name":         product["name"],
        "price_chf":    product["price_chf"],
        "wattage":      product["wattage"],
        "kelvin":       product["kelvin"],
        "material":     product["material"],
        "room_type":    product["room_type"],
        "image_url":    product["image_url"],
        # Memory metadata
        "memory_type":  "hard",
        "lambda":       0.01,           # very slow decay
    }


def build_soft_payload(product: dict) -> dict:
    """
    Soft preference fields — what MARA adapts over time.
    Style and finish decay slowly (lambda = 0.10).
    Recent browsing decays fast (lambda = 0.30).
    """
    return {
        "product_id":   product["id"],
        "name":         product["name"],
        "style":        product["style"],
        "finish":       product["finish"],
        "mood":         product["mood"],
        "description":  product["description"],
        "image_url":    product["image_url"],
        # Memory metadata
        "memory_type":  "soft",
        "lambda":       0.10,           # slow decay for style
    }


# ─────────────────────────────────────────────
# INDEX PRODUCTS
# ─────────────────────────────────────────────
def index_products(client: QdrantClient, products: list[dict]):
    """
    Embeds each product and inserts it into both collections.

    Collection A (hard_constraints):
      - Text embedded = name + material + room_type
      - Payload = price, wattage, kelvin, material

    Collection B (soft_preferences):
      - Text embedded = description (rich natural language)
      - Payload = style, finish, mood
    """

    hard_points = []
    soft_points = []

    for i, product in enumerate(products):
        print(f"  Embedding product {i+1:02d}/30 — {product['name']}")

        # ── Hard constraint text ─────────────────
        hard_text = (
            f"{product['name']} "
            f"{product['material']} "
            f"{product['room_type']} "
            f"{product['wattage']}W "
            f"{product['kelvin']}K "
            f"{product['price_chf']} CHF"
        )
        hard_vector = embed(hard_text)

        # ── Soft preference text ─────────────────
        soft_text = product["description"]
        soft_vector = embed(soft_text)

        # ── Build Qdrant points ──────────────────
        hard_points.append(PointStruct(
            id=i,
            vector=hard_vector,
            payload=build_hard_payload(product),
        ))

        soft_points.append(PointStruct(
            id=i,
            vector=soft_vector,
            payload=build_soft_payload(product),
        ))

    # ── Upload to Qdrant ─────────────────────────
    print("\n  Uploading to hard_constraints collection ...")
    client.upsert(collection_name=COLLECTION_HARD, points=hard_points)

    print("  Uploading to soft_preferences collection ...")
    client.upsert(collection_name=COLLECTION_SOFT, points=soft_points)

    print(f"\n  Indexed {len(products)} products into both collections.")


# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────
def verify(client: QdrantClient):
    """Quick sanity check — print collection stats."""
    print("\n─── Verification ───────────────────────────")
    for name in [COLLECTION_HARD, COLLECTION_SOFT]:
        info = client.get_collection(name)
        count = info.points_count
        print(f"  {name}: {count} points")
    print("────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  MARA — Qdrant Setup")
    print("=" * 50 + "\n")

    # 1. Connect
    client = connect()

    # 2. Load products
    print(f"Loading products from {PRODUCTS_FILE} ...")
    with open(PRODUCTS_FILE) as f:
        products = json.load(f)
    print(f"  Loaded {len(products)} products.\n")

    # 3. Create collections
    print("Creating collections ...")
    create_collections(client)

    # 4. Index products
    print("Indexing products ...")
    index_products(client, products)

    # 5. Verify
    verify(client)

    print("Setup complete. Qdrant is ready for MARA.\n")
    print("Next step: run the FastAPI server with  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
