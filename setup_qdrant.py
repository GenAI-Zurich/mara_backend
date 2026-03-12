"""
MARA — Qdrant setup
===================

This script creates the MARA product collections in Qdrant and indexes a
retrieval-optimized product representation.

Source:
  canonical real-catalog export from extract_supabase_catalog.py

Usage:
  python3 setup_qdrant.py
  python3 setup_qdrant.py --catalog-file catalog_export.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embeddings import embed_batch

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTION_HARD = "hard_constraints"
COLLECTION_SOFT = "soft_preferences"
VECTOR_SIZE = 1024
UPSERT_BATCH_SIZE = 128

DEFAULT_CATALOG_FILE = "catalog_export.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and populate MARA Qdrant collections.")
    parser.add_argument(
        "--catalog-file",
        default=DEFAULT_CATALOG_FILE,
        help=f"Canonical catalog JSON file. Default: {DEFAULT_CATALOG_FILE}.",
    )
    return parser.parse_args()


def connect() -> QdrantClient:
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)
    print("Connected.\n")
    return client


def create_collections(client: QdrantClient) -> None:
    for name in [COLLECTION_HARD, COLLECTION_SOFT]:
        if client.collection_exists(name):
            print(f"  Deleting existing collection: {name}")
            client.delete_collection(name)

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created collection: {name}")

    print()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Catalog file must contain a JSON array: {path}")

    print(f"Loaded {len(payload)} records from {path}")
    return payload


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_canonical_product(product: dict[str, Any]) -> dict[str, Any]:
    source = product.get("source") or {}
    identity = product.get("identity") or {}
    pricing = product.get("pricing") or {}
    technical = product.get("technical") or {}
    classification = product.get("classification") or {}
    semantic = product.get("semantic") or {}
    content = product.get("content") or {}
    media = product.get("media") or {}

    kelvin_values = technical.get("kelvin_values") or []
    return {
        "product_id": product.get("product_id"),
        "source_article_id": source.get("article_id"),
        "source_article_number": source.get("article_number"),
        "source_l_number": source.get("l_number"),
        "source_version": source.get("version"),
        "name": identity.get("name"),
        "manufacturer": identity.get("manufacturer"),
        "category": identity.get("category"),
        "family": identity.get("family"),
        "price_chf": as_float(pricing.get("price_chf")),
        "price_type": pricing.get("price_type"),
        "wattage": as_float(technical.get("wattage")),
        "kelvin": as_float(technical.get("kelvin_primary")),
        "kelvin_values": kelvin_values,
        "material": technical.get("material"),
        "material_code": technical.get("material_code"),
        "ip_rating": technical.get("ip_rating"),
        "ik_rating": technical.get("ik_rating"),
        "cri": technical.get("cri"),
        "light_output": technical.get("light_output"),
        "inside": classification.get("inside"),
        "outside": classification.get("outside"),
        "mounting": classification.get("mounting") or [],
        "luminaire_types": classification.get("luminaire_types") or [],
        "style": semantic.get("style"),
        "finish": semantic.get("finish"),
        "mood": semantic.get("mood"),
        "room_type": semantic.get("room_type"),
        "tags": semantic.get("tags") or [],
        "description": content.get("semantic_description") or "",
        "image_url": media.get("hero_image_url") or media.get("hero_image_path"),
    }


def flatten_product(product: dict[str, Any]) -> dict[str, Any]:
    if "source" not in product or "identity" not in product:
        raise ValueError("Catalog records must use the canonical MARA schema. Run extract_supabase_catalog.py first.")
    return flatten_canonical_product(product)


def build_hard_text(product: dict[str, Any]) -> str:
    parts: list[str] = []

    for value in [
        product.get("name"),
        product.get("manufacturer"),
        product.get("category"),
        product.get("family"),
        product.get("material"),
    ]:
        if value:
            parts.append(str(value))

    if product.get("inside") is True:
        parts.append("inside")
    if product.get("outside") is True:
        parts.append("outside")

    if product.get("mounting"):
        parts.append("mounting " + ", ".join(product["mounting"]))

    if product.get("luminaire_types"):
        parts.append("type " + ", ".join(product["luminaire_types"]))

    if product.get("wattage") is not None:
        wattage = product["wattage"]
        parts.append(f"{int(wattage) if float(wattage).is_integer() else wattage}W")

    if product.get("kelvin") is not None:
        kelvin = product["kelvin"]
        parts.append(f"{int(kelvin) if float(kelvin).is_integer() else kelvin}K")

    if product.get("price_chf") is not None:
        price = product["price_chf"]
        parts.append(f"{int(price) if float(price).is_integer() else price} CHF")

    return " ".join(parts).strip()


def build_soft_text(product: dict[str, Any]) -> str:
    parts = [product.get("description") or ""]

    optional_bits: list[str] = []
    for label, value in [
        ("manufacturer", product.get("manufacturer")),
        ("category", product.get("category")),
        ("family", product.get("family")),
        ("style", product.get("style")),
        ("finish", product.get("finish")),
        ("mood", product.get("mood")),
        ("room_type", product.get("room_type")),
    ]:
        if value:
            optional_bits.append(f"{label} {value}")

    if product.get("tags"):
        optional_bits.append("tags " + ", ".join(product["tags"]))

    if optional_bits:
        parts.append(". ".join(optional_bits))

    return " ".join(part for part in parts if part).strip()


def build_hard_payload(product: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "product_id": product.get("product_id"),
        "name": product.get("name"),
        "price_chf": product.get("price_chf"),
        "wattage": product.get("wattage"),
        "kelvin": product.get("kelvin"),
        "kelvin_values": product.get("kelvin_values"),
        "material": product.get("material"),
        "room_type": product.get("room_type"),
        "image_url": product.get("image_url"),
        "manufacturer": product.get("manufacturer"),
        "category": product.get("category"),
        "family": product.get("family"),
        "inside": product.get("inside"),
        "outside": product.get("outside"),
        "mounting": product.get("mounting"),
        "luminaire_types": product.get("luminaire_types"),
        "source_article_id": product.get("source_article_id"),
        "source_article_number": product.get("source_article_number"),
        "source_l_number": product.get("source_l_number"),
        "source_version": product.get("source_version"),
        "memory_type": "hard",
        "lambda": 0.01,
    }
    return {key: value for key, value in payload.items() if value not in (None, [], "")}


def build_soft_payload(product: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "product_id": product.get("product_id"),
        "name": product.get("name"),
        "style": product.get("style"),
        "finish": product.get("finish"),
        "mood": product.get("mood"),
        "description": product.get("description"),
        "image_url": product.get("image_url"),
        "manufacturer": product.get("manufacturer"),
        "category": product.get("category"),
        "family": product.get("family"),
        "tags": product.get("tags"),
        "source_article_id": product.get("source_article_id"),
        "source_article_number": product.get("source_article_number"),
        "source_l_number": product.get("source_l_number"),
        "source_version": product.get("source_version"),
        "memory_type": "soft",
        "lambda": 0.10,
    }
    return {key: value for key, value in payload.items() if value not in (None, [], "")}


def batched_points(points: list[PointStruct], batch_size: int) -> list[list[PointStruct]]:
    return [points[i:i + batch_size] for i in range(0, len(points), batch_size)]


def upload_points(client: QdrantClient, collection_name: str, points: list[PointStruct]) -> None:
    batches = batched_points(points, UPSERT_BATCH_SIZE)
    for index, batch in enumerate(batches, start=1):
        print(
            f"  Uploading batch {index}/{len(batches)} to {collection_name} "
            f"({len(batch)} points) ..."
        )
        client.upsert(collection_name=collection_name, points=batch)


def index_products(client: QdrantClient, products: list[dict[str, Any]]) -> None:
    flattened = [flatten_product(product) for product in products]

    for product in flattened[:3]:
        print(f"  Prepared {product['product_id']} — {product['name']}")

    hard_texts = [build_hard_text(product) for product in flattened]
    soft_texts = [build_soft_text(product) for product in flattened]

    print(f"\nEmbedding {len(flattened)} hard-view texts ...")
    hard_vectors = embed_batch(hard_texts)
    print(f"Embedding {len(flattened)} soft-view texts ...")
    soft_vectors = embed_batch(soft_texts)

    hard_points: list[PointStruct] = []
    soft_points: list[PointStruct] = []

    for i, product in enumerate(flattened):
        hard_points.append(
            PointStruct(
                id=i,
                vector=hard_vectors[i],
                payload=build_hard_payload(product),
            )
        )
        soft_points.append(
            PointStruct(
                id=i,
                vector=soft_vectors[i],
                payload=build_soft_payload(product),
            )
        )

    print("\nUploading to hard_constraints collection ...")
    upload_points(client, COLLECTION_HARD, hard_points)

    print("Uploading to soft_preferences collection ...")
    upload_points(client, COLLECTION_SOFT, soft_points)

    print(f"\nIndexed {len(flattened)} products into both collections.")


def verify(client: QdrantClient) -> None:
    print("\n─── Verification ───────────────────────────")
    for name in [COLLECTION_HARD, COLLECTION_SOFT]:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points")
    print("────────────────────────────────────────────\n")


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("  MARA — Qdrant Setup")
    print("=" * 50 + "\n")

    client = connect()

    catalog_path = Path(args.catalog_file)
    print(f"Loading catalog from {catalog_path} ...")
    products = load_catalog(catalog_path)
    print()

    print("Creating collections ...")
    create_collections(client)

    print("Indexing products ...")
    index_products(client, products)

    verify(client)

    print("Setup complete. Qdrant is ready for MARA.\n")
    print("Next step: run the FastAPI server with uvicorn main:app --reload")


if __name__ == "__main__":
    main()
