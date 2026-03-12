"""
Extract the real lighting catalog from Supabase and normalize it into the
canonical MARA product schema.

Usage:
  python3 extract_supabase_catalog.py
  python3 extract_supabase_catalog.py --limit 25
  python3 extract_supabase_catalog.py --output catalog_export.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_OUTPUT = "catalog_export.json"
DEFAULT_PAGE_SIZE = 200
MAX_LONG_DESCRIPTION_CHARS = 1200

MOUNTING_FIELD_MAP = {
    "mounting_method_wall": "wall",
    "mounting_method_ceiling": "ceiling",
    "mounting_method_floor": "floor",
    "mounting_method_table": "table",
    "mounting_method_power_rail": "power_rail",
}

LUMINAIRE_TYPE_FIELD_MAP = {
    "luminaire_type_surface_mounted": "surface_mounted",
    "luminaire_type_semi_recessed": "semi_recessed",
    "luminaire_type_down_light": "down_light",
    "luminaire_type_cable_pendant": "cable_pendant",
    "luminaire_type_outdoor": "outdoor",
    "luminaire_type_recessed": "recessed",
    "luminaire_type_suspended": "suspended",
    "luminaire_type_rope_pendant": "rope_pendant",
    "luminaire_type_tube_pendant": "tube_pendant",
    "luminaire_type_chains_pendant": "chains_pendant",
    "luminaire_type_spot_light": "spot_light",
    "luminaire_type_power_rail": "power_rail",
    "luminaire_type_profile_luminaire": "profile_luminaire",
    "luminaire_type_system_luminaire": "system_luminaire",
    "luminaire_type_light_bands": "light_bands",
    "luminaire_type_light_strip": "light_strip",
    "luminaire_type_special_luminaire": "special_luminaire",
    "luminaire_type_bollard_luminaire": "bollard_luminaire",
    "luminaire_type_head_light": "head_light",
    "luminaire_type_acoustic_luminaire": "acoustic_luminaire",
    "luminaire_type_clean_room": "clean_room",
    "luminaire_type_light_management": "light_management",
    "luminaire_type_higher_protection_class": "higher_protection_class",
    "luminaire_type_high_bay_luminaire": "high_bay_luminaire",
    "luminaire_type_safety_lighting": "safety_lighting",
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def fetch_json(url: str, headers: dict[str, str]) -> Any:
    request = Request(url, headers=headers)
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_price(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    cleaned = text.replace("CHF", "").replace("'", "").replace(" ", "")
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[int] = []
    for item in items:
        if item in (None, ""):
            continue
        try:
            normalized.append(int(float(item)))
        except (TypeError, ValueError):
            continue

    return sorted(set(normalized))


def parse_first_number_from_text(text: str | None, suffix: str) -> float | None:
    if not text:
        return None

    pattern = rf"(\d+(?:[.,]\d+)?)\s*{re.escape(suffix)}\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def truncate_text(text: str | None, max_chars: int) -> str | None:
    if not text:
        return None

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def collect_true_flags(record: dict[str, Any] | None, mapping: dict[str, str]) -> list[str]:
    if not record:
        return []

    values = [label for field, label in mapping.items() if record.get(field) is True]
    return sorted(values)


def derive_finish(character: dict[str, Any] | None) -> str | None:
    if not character:
        return None

    finish_labels = [
        ("housing_glossy", "glossy"),
        ("housing_mat", "matte"),
        ("housing_brushed", "brushed"),
        ("housing_textured", "textured"),
        ("housing_anodized", "anodized"),
        ("housing_metallic", "metallic"),
    ]
    values = [label for field, label in finish_labels if character.get(field) is True]
    if not values:
        return None
    return ", ".join(values)


def derive_light_output(character: dict[str, Any] | None) -> int | None:
    if not character:
        return None

    fluxes = character.get("luminaire_fluxes")
    if isinstance(fluxes, list):
        for item in fluxes:
            try:
                return int(float(item))
            except (TypeError, ValueError):
                continue

    fallback = character.get("light_output")
    try:
        return int(fallback) if fallback is not None else None
    except (TypeError, ValueError):
        return None


def build_semantic_description(
    article: dict[str, Any],
    manufacturer: str | None,
    category: str | None,
    family: str | None,
    mounting: list[str],
    luminaire_types: list[str],
    wattage: float | None,
    kelvin_values: list[int],
    finish: str | None,
) -> str:
    parts: list[str] = []

    name = article.get("very_short_description_de")
    if name:
        parts.append(str(name).strip())

    short_description = article.get("short_description_de")
    if short_description:
        parts.append(str(short_description).strip())

    long_description = truncate_text(article.get("long_description_de"), MAX_LONG_DESCRIPTION_CHARS)
    if long_description:
        parts.append(str(long_description).strip())

    facts: list[str] = []
    if manufacturer:
        facts.append(f"manufacturer {manufacturer}")
    if category:
        facts.append(f"category {category}")
    if family:
        facts.append(f"family {family}")
    if mounting:
        facts.append("mounting " + ", ".join(mounting))
    if luminaire_types:
        facts.append("luminaire types " + ", ".join(luminaire_types))
    if wattage is not None:
        facts.append(f"{int(wattage) if wattage.is_integer() else wattage}W")
    if kelvin_values:
        facts.append("kelvin " + ", ".join(str(v) for v in kelvin_values))
    if finish:
        facts.append(f"finish {finish}")

    if facts:
        parts.append(". ".join(facts))

    text = " ".join(part for part in parts if part)
    return re.sub(r"\s+", " ", text).strip()


def derive_wattage(article: dict[str, Any], technical: dict[str, Any] | None) -> float | None:
    technical = technical or {}
    wattage_raw = technical.get("electrical_power")
    if wattage_raw is not None:
        return float(wattage_raw)

    for field in ("short_description_de", "long_description_de"):
        fallback = parse_first_number_from_text(article.get(field), "W")
        if fallback is not None:
            return fallback

    return None


def derive_kelvin_values(article: dict[str, Any], character: dict[str, Any] | None) -> list[int]:
    character = character or {}
    values = normalize_int_list(character.get("light_color_colors"))
    if values:
        return values

    for field in ("short_description_de", "long_description_de"):
        fallback = parse_first_number_from_text(article.get(field), "K")
        if fallback is not None:
            return [int(fallback)]

    return []


def resolve_hero_image_url(supabase_url: str, hero_image_path: str | None) -> str | None:
    if not hero_image_path:
        return None

    path = hero_image_path.lstrip("/")
    storage_path = f"opt/LO/LOwebserver/{path}"
    return f"{supabase_url}/storage/v1/object/public/pim/{storage_path}"


def normalize_record(article: dict[str, Any], supabase_url: str) -> dict[str, Any]:
    classification = article.get("article_classifications") or {}
    technical = article.get("article_technical_profiles") or {}
    character = article.get("article_character_profiles") or {}

    manufacturer = ((article.get("manufacturers") or {}).get("man_name"))
    category = ((article.get("light_categories") or {}).get("name_de"))
    family = ((article.get("light_families") or {}).get("name_de"))

    price_sp = parse_price(article.get("price_sp_chf"))
    price_pp = parse_price(article.get("price_pp_chf"))
    price_chf = price_sp if price_sp is not None else price_pp
    price_type = "price_sp_chf" if price_sp is not None else "price_pp_chf" if price_pp is not None else None

    wattage = derive_wattage(article, technical)
    kelvin_values = derive_kelvin_values(article, character)
    kelvin_primary = kelvin_values[0] if kelvin_values else None

    mounting = collect_true_flags(classification, MOUNTING_FIELD_MAP)
    luminaire_types = collect_true_flags(classification, LUMINAIRE_TYPE_FIELD_MAP)
    finish = derive_finish(character)
    light_output = derive_light_output(character)

    hero_image_path = article.get("hero_image_url")
    hero_image_url = resolve_hero_image_url(supabase_url, hero_image_path)

    return {
        "product_id": f"article_{article['id']}",
        "source": {
            "article_id": article["id"],
            "article_number": article.get("article_number"),
            "l_number": article.get("l_number"),
            "version": article.get("version"),
        },
        "identity": {
            "name": article.get("very_short_description_de") or article.get("article_number"),
            "manufacturer": manufacturer,
            "category": category,
            "family": family,
        },
        "pricing": {
            "price_chf": price_chf,
            "price_type": price_type,
        },
        "technical": {
            "wattage": wattage,
            "kelvin_primary": kelvin_primary,
            "kelvin_values": kelvin_values,
            "material": None,
            "material_code": character.get("housing_material"),
            "ip_rating": technical.get("ip_rating"),
            "ik_rating": technical.get("ik_rating"),
            "cri": character.get("cri"),
            "light_output": light_output,
        },
        "classification": {
            "inside": classification.get("inside"),
            "outside": classification.get("outside"),
            "mounting": mounting,
            "luminaire_types": luminaire_types,
        },
        "semantic": {
            "finish": finish,
            "style": None,
            "mood": None,
            "room_type": None,
            "tags": sorted({*mounting, *luminaire_types, *(["inside"] if classification.get("inside") else []), *(["outside"] if classification.get("outside") else []), *( [finish] if finish else [] )}),
        },
        "content": {
            "short_description": article.get("short_description_de"),
            "long_description": article.get("long_description_de"),
            "semantic_description": build_semantic_description(
                article=article,
                manufacturer=manufacturer,
                category=category,
                family=family,
                mounting=mounting,
                luminaire_types=luminaire_types,
                wattage=wattage,
                kelvin_values=kelvin_values,
                finish=finish,
            ),
        },
        "media": {
            "hero_image_path": hero_image_path,
            "hero_image_url": hero_image_url,
        },
    }


def build_select_clause() -> str:
    return ",".join(
        [
            "id",
            "l_number",
            "version",
            "is_current",
            "article_number",
            "price_pp_chf",
            "price_sp_chf",
            "hero_image_url",
            "very_short_description_de",
            "short_description_de",
            "long_description_de",
            "manufacturers(man_name)",
            "light_categories(name_de)",
            "light_families(name_de)",
            "article_classifications("
            + ",".join(MOUNTING_FIELD_MAP.keys())
            + ","
            + ",".join(LUMINAIRE_TYPE_FIELD_MAP.keys())
            + ",inside,outside"
            + ")",
            "article_technical_profiles(electrical_power,ip_rating,ik_rating)",
            "article_character_profiles("
            "housing_material,"
            "housing_glossy,housing_mat,housing_brushed,housing_textured,"
            "housing_anodized,housing_metallic,"
            "light_color_colors,cri,luminaire_fluxes,light_output"
            ")",
        ]
    )


def fetch_articles(supabase_url: str, anon_key: str, page_size: int, limit: int | None) -> list[dict[str, Any]]:
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Accept-Profile": "public",
    }
    select = build_select_clause()
    base_url = f"{supabase_url.rstrip('/')}/rest/v1/articles"

    records: list[dict[str, Any]] = []
    offset = 0

    while True:
        remaining = None if limit is None else max(limit - len(records), 0)
        if remaining == 0:
            break

        batch_size = page_size if remaining is None else min(page_size, remaining)
        params = {
            "select": select,
            "is_current": "eq.true",
            "order": "id.asc",
            "limit": str(batch_size),
            "offset": str(offset),
        }
        url = f"{base_url}?{urlencode(params)}"
        batch = fetch_json(url, headers=headers)
        if not batch:
            break

        records.extend(batch)
        offset += len(batch)

        if len(batch) < batch_size:
            break

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the real Supabase catalog into the MARA canonical schema.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of articles to extract.")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help=f"Supabase page size. Default: {DEFAULT_PAGE_SIZE}")
    return parser.parse_args()


def main() -> int:
    load_env_file(Path(__file__).parent / ".env")
    args = parse_args()

    try:
        supabase_url = require_env("SUPABASE_URL")
        anon_key = require_env("SUPABASE_ANON_KEY")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        raw_articles = fetch_articles(
            supabase_url=supabase_url,
            anon_key=anon_key,
            page_size=args.page_size,
            limit=args.limit,
        )
        normalized = [normalize_record(article, supabase_url) for article in raw_articles]
    except HTTPError as exc:
        print(f"Supabase HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Supabase connection error: {exc.reason}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True) + "\n")

    print(f"Extracted {len(normalized)} products to {output_path}")
    if normalized:
        first = normalized[0]
        print(f"First product: {first['product_id']} | {first['identity']['name']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
