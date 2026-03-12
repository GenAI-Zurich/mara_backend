# MARA Catalog Schema

## Purpose

This document defines the canonical product schema MARA should use for the real
Supabase catalog.

This is the phase 1 output.

The goal is to avoid coupling the MARA codebase directly to the raw Supabase
relational structure. Supabase remains the source of truth, but MARA should work
against one stable, retrieval-oriented product representation.


## Source Tables

Minimum required Supabase tables:

- `articles`
- `article_technical_profiles`
- `article_character_profiles`
- `article_classifications`

Supporting tables:

- `manufacturers`
- `light_categories`
- `light_families`

Optional supporting assets:

- `article_media`
- hero image storage resolution


## Design Principles

1. Supabase is the full catalog source of truth.
2. MARA should consume a canonical schema, not raw table rows.
3. Qdrant should index a retrieval-optimized subset of that schema.
4. The frontend should hydrate full product detail from Supabase using stable
   product IDs.
5. Fields that are weak or unavailable in the real catalog should be optional,
   derived, or deferred rather than guessed carelessly.


## Canonical MARA Product Schema

Each product record should follow this shape:

```json
{
  "product_id": "article_7",
  "source": {
    "article_id": 7,
    "article_number": "2.306848",
    "l_number": 120001,
    "version": 1
  },
  "identity": {
    "name": "TRY 2A/M 50-840-88 EM",
    "manufacturer": "Performance In Lighting - Italy",
    "category": null,
    "family": null
  },
  "pricing": {
    "price_chf": 549.0,
    "price_type": "price_sp_chf"
  },
  "technical": {
    "wattage": 53.0,
    "kelvin_primary": 4000,
    "kelvin_values": [4000],
    "material": null,
    "material_code": 1,
    "ip_rating": 16,
    "ik_rating": 5,
    "cri": 80,
    "light_output": 8169
  },
  "classification": {
    "inside": true,
    "outside": false,
    "mounting": ["ceiling"],
    "luminaire_types": []
  },
  "semantic": {
    "finish": null,
    "style": null,
    "mood": null,
    "room_type": null,
    "tags": []
  },
  "content": {
    "short_description": "LED-Lichteinsätze 4000K 53W statisch Notlicht 3h asymmetrische mittlere Optik",
    "long_description": "Serien von LED Innenleuchten ...",
    "semantic_description": "TRY 2A/M 50-840-88 EM. LED-Lichteinsätze 4000K 53W ..."
  },
  "media": {
    "hero_image_path": "images/TRY_2AM_angle_01.jpg",
    "hero_image_url": null
  }
}
```


## Required Canonical Fields

These fields are required for MARA v1 with real data:

- `product_id`
- `source.article_id`
- `source.article_number`
- `source.l_number`
- `identity.name`
- `pricing.price_chf`
- `technical.wattage`
- `technical.kelvin_values`
- `content.semantic_description`
- `media.hero_image_path` or `media.hero_image_url`

These are the minimum fields needed for:

- product identity
- retrieval
- filtering
- frontend hydration


## Field Mapping

## Identity

### `product_id`

- Type: `string`
- Required: yes
- Rule: stable MARA identifier derived from Supabase article identity
- Recommended mapping: `article_{articles.id}`

### `source.article_id`

- Type: `int`
- Required: yes
- Source: `articles.id`

### `source.article_number`

- Type: `string`
- Required: yes
- Source: `articles.article_number`

### `source.l_number`

- Type: `int`
- Required: yes
- Source: `articles.l_number`

### `source.version`

- Type: `int`
- Required: yes
- Source: `articles.version`

### `identity.name`

- Type: `string`
- Required: yes
- Source: `articles.very_short_description_de`
- Fallback: `articles.article_number`

### `identity.manufacturer`

- Type: `string | null`
- Required: no
- Source: `manufacturers.man_name`

### `identity.category`

- Type: `string | null`
- Required: no
- Source: `light_categories.name_de`

### `identity.family`

- Type: `string | null`
- Required: no
- Source: `light_families.name_de`


## Pricing

### `pricing.price_chf`

- Type: `float | null`
- Required: yes
- Preferred source: `articles.price_sp_chf`
- Fallback source: `articles.price_pp_chf`
- Rule: parse from text into numeric CHF value

### `pricing.price_type`

- Type: `string | null`
- Required: no
- Values:
  - `price_sp_chf`
  - `price_pp_chf`
  - `null`


## Technical

### `technical.wattage`

- Type: `float | null`
- Required: yes
- Source: `article_technical_profiles.electrical_power`

### `technical.kelvin_values`

- Type: `list[int]`
- Required: yes
- Source: `article_character_profiles.light_color_colors`
- Rule: normalize JSON array to a sorted list of integer Kelvin values

### `technical.kelvin_primary`

- Type: `int | null`
- Required: no
- Rule: first value from `kelvin_values` when present

### `technical.material_code`

- Type: `int | null`
- Required: no
- Source: `article_character_profiles.housing_material`

### `technical.material`

- Type: `string | null`
- Required: no
- Rule: derived label from `material_code` or finish/material flags
- Status: deferred until material code mapping is confirmed

### `technical.ip_rating`

- Type: `int | null`
- Required: no
- Source: `article_technical_profiles.ip_rating`

### `technical.ik_rating`

- Type: `int | null`
- Required: no
- Source: `article_technical_profiles.ik_rating`

### `technical.cri`

- Type: `int | null`
- Required: no
- Source: `article_character_profiles.cri`

### `technical.light_output`

- Type: `int | null`
- Required: no
- Preferred source: `article_character_profiles.luminaire_fluxes[0]`
- Fallback source: `article_character_profiles.light_output`


## Classification

### `classification.inside`

- Type: `bool | null`
- Required: no
- Source: `article_classifications.inside`

### `classification.outside`

- Type: `bool | null`
- Required: no
- Source: `article_classifications.outside`

### `classification.mounting`

- Type: `list[string]`
- Required: no
- Rule: derive from true booleans such as:
  - `mounting_method_wall`
  - `mounting_method_ceiling`
  - `mounting_method_floor`
  - `mounting_method_table`
  - `mounting_method_power_rail`

### `classification.luminaire_types`

- Type: `list[string]`
- Required: no
- Rule: derive from true luminaire type flags such as:
  - `luminaire_type_down_light`
  - `luminaire_type_recessed`
  - `luminaire_type_suspended`
  - `luminaire_type_spot_light`
  - `luminaire_type_profile_luminaire`
  - `luminaire_type_high_bay_luminaire`
  - `luminaire_type_outdoor`
  - others if true


## Semantic

These are the fields most affected by the migration from the original flat
prototype schema to the real catalog.

### `semantic.finish`

- Type: `string | null`
- Required: no
- Rule: derive from finish-related booleans such as:
  - `housing_glossy`
  - `housing_mat`
  - `housing_brushed`
  - `housing_textured`
  - `housing_anodized`
  - `housing_metallic`
- Status: derived field

### `semantic.style`

- Type: `string | null`
- Required: no
- Rule: not directly present in Supabase
- Status: deferred unless high-confidence derivation is added

### `semantic.mood`

- Type: `string | null`
- Required: no
- Rule: not directly present in Supabase
- Status: deferred unless derived from text or offline enrichment

### `semantic.room_type`

- Type: `string | null`
- Required: no
- Rule: not directly present in the catalog
- Status: deferred

### `semantic.tags`

- Type: `list[string]`
- Required: no
- Rule: optional derived tags from:
  - category
  - family
  - mounting
  - inside/outside
  - luminaire types
  - finish


## Content

### `content.short_description`

- Type: `string | null`
- Required: no
- Source: `articles.short_description_de`

### `content.long_description`

- Type: `string | null`
- Required: no
- Source: `articles.long_description_de`

### `content.semantic_description`

- Type: `string`
- Required: yes
- Rule: composed retrieval text for embeddings

Recommended composition order:

1. product name
2. short description
3. long description
4. manufacturer
5. category / family
6. mounting tags
7. luminaire type tags
8. technical facts:
   - wattage
   - kelvin
   - indoor/outdoor
   - finish
   - material

This field is the most important input for semantic search quality.


## Media

### `media.hero_image_path`

- Type: `string | null`
- Required: yes
- Source: `articles.hero_image_url`

### `media.hero_image_url`

- Type: `string | null`
- Required: no
- Rule: resolved public or signed URL if needed
- Status: can be hydrated later by frontend if path is enough


## Direct vs Derived vs Deferred

## Direct fields

These can be mapped immediately:

- article identity fields
- name
- manufacturer
- category
- family
- price text
- wattage
- kelvin raw values
- inside / outside
- mounting flags
- luminaire type flags
- descriptions
- hero image path

## Derived fields

These can be built with deterministic logic:

- `product_id`
- `price_chf`
- `kelvin_primary`
- `classification.mounting`
- `classification.luminaire_types`
- `semantic.finish`
- `content.semantic_description`
- optional `semantic.tags`

## Deferred fields

These should not block the migration, but they are not clean direct mappings:

- `technical.material`
- `semantic.style`
- `semantic.mood`
- `semantic.room_type`


## MARA v1 Compatibility Rules

To migrate safely to real data, MARA v1 should rely first on
the fields that are strong in Supabase:

- product identity
- price
- wattage
- kelvin
- indoor/outdoor classification
- mounting type
- product descriptions

Fields that were convenient in the earlier prototype but weak in the real
catalog should become optional rather than required.

This means:

- hard constraints should stay strong
- semantic retrieval should stay strong
- weak lifestyle attributes should not be forced into the ranking pipeline yet


## Recommended Qdrant Usage Based On This Schema

This document does not define the final Qdrant payloads, but it does constrain
them.

### Hard/spec retrieval view

Should primarily use:

- `product_id`
- `identity.name`
- `pricing.price_chf`
- `technical.wattage`
- `technical.kelvin_values`
- `technical.material`
- `classification.inside`
- `classification.outside`
- `classification.mounting`
- `classification.luminaire_types`

### Soft/semantic retrieval view

Should primarily use:

- `product_id`
- `identity.name`
- `content.semantic_description`
- `identity.manufacturer`
- `identity.category`
- `identity.family`
- `semantic.finish`
- `semantic.tags`


## Open Questions

These need to be answered before phase 2 or phase 3 implementation is finalized:

1. What is the exact label mapping for `housing_material` codes?
2. Should `price_sp_chf` always be the preferred price, or should pricing logic
   vary by business rule?
3. Should hero image URLs be resolved during extraction or during frontend
   hydration?
4. Do we want offline enrichment for style and mood, or should those fields be
   removed from MARA ranking until trustworthy data exists?
5. Should product filtering include additional technical fields beyond wattage
   and kelvin in the first real-data version?


## Phase 1 Decision Summary

The canonical MARA schema should be based on real Supabase products but remain
retrieval-oriented.

The migration should preserve:

- stable product identity
- strong constraint fields
- rich semantic text
- clear separation between source of truth and retrieval layer

The migration should not depend on prototype-only fields being perfectly
available in the real catalog.
