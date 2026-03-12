"""
Validate a canonical MARA catalog export and print field coverage statistics.

Usage:
  python3 validate_catalog.py
  python3 validate_catalog.py --catalog-file catalog_export.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_CATALOG_FILE = "catalog_export.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the canonical MARA catalog export.")
    parser.add_argument(
        "--catalog-file",
        default=DEFAULT_CATALOG_FILE,
        help=f"Canonical catalog JSON file. Default: {DEFAULT_CATALOG_FILE}.",
    )
    return parser.parse_args()


def pct(count: int, total: int) -> str:
    return f"{(count / total):.1%}" if total else "0.0%"


def main() -> int:
    args = parse_args()
    path = Path(args.catalog_file)
    items = json.loads(path.read_text())

    if not isinstance(items, list):
        raise ValueError("Catalog file must contain a JSON array")

    total = len(items)
    print(f"Catalog records: {total}")
    print()

    checks = {
        "price_chf": lambda x: x["pricing"]["price_chf"] is not None,
        "wattage": lambda x: x["technical"]["wattage"] is not None,
        "kelvin_values": lambda x: bool(x["technical"]["kelvin_values"]),
        "hero_image_url": lambda x: x["media"]["hero_image_url"] is not None,
        "manufacturer": lambda x: x["identity"]["manufacturer"] is not None,
        "category": lambda x: x["identity"]["category"] is not None,
        "family": lambda x: x["identity"]["family"] is not None,
        "finish": lambda x: x["semantic"]["finish"] is not None,
        "inside_or_outside": lambda x: (
            x["classification"]["inside"] is not None
            or x["classification"]["outside"] is not None
        ),
        "mounting": lambda x: bool(x["classification"]["mounting"]),
        "luminaire_types": lambda x: bool(x["classification"]["luminaire_types"]),
        "hydration_ready": lambda x: (
            x["source"]["article_id"] is not None
            and x["media"]["hero_image_url"] is not None
        ),
        "full_hard_filter_ready": lambda x: (
            x["pricing"]["price_chf"] is not None
            and x["technical"]["wattage"] is not None
            and bool(x["technical"]["kelvin_values"])
        ),
    }

    print("Coverage")
    for name, fn in checks.items():
        count = sum(1 for item in items if fn(item))
        print(f"- {name}: {count}/{total} ({pct(count, total)})")

    print()

    prices = [x["pricing"]["price_chf"] for x in items if x["pricing"]["price_chf"] is not None]
    watts = [x["technical"]["wattage"] for x in items if x["technical"]["wattage"] is not None]
    kelvins = [x["technical"]["kelvin_primary"] for x in items if x["technical"]["kelvin_primary"] is not None]

    print("Ranges")
    if prices:
        print(f"- price_chf: {min(prices)} -> {max(prices)}")
    if watts:
        print(f"- wattage: {min(watts)} -> {max(watts)}")

    print()
    print("Top Values")
    print(f"- kelvin: {Counter(kelvins).most_common(10)}")
    print(f"- mounting: {Counter(m for x in items for m in x['classification']['mounting']).most_common(10)}")
    print(f"- luminaire_types: {Counter(t for x in items for t in x['classification']['luminaire_types']).most_common(10)}")
    print(f"- finish: {Counter(x['semantic']['finish'] for x in items if x['semantic']['finish']).most_common(10)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
