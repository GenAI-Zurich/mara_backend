# MARA Frontend Contract

## Purpose

This document defines how Lovable should consume the MARA API once the real
catalog flow is active.

The key principle is:

- MARA decides which products are relevant
- Supabase remains the source of truth for full product data
- Lovable hydrates the final UI from Supabase using stable identifiers returned
  by MARA


## Core Flow

1. User sends a request in Lovable.
2. Lovable sends the query to FastAPI.
3. FastAPI embeds the message, retrieves from Qdrant, applies MARA logic, and
   returns ranked results.
4. Each result includes stable product identifiers from the real catalog.
5. Lovable uses those identifiers to fetch the full product data from Supabase.
6. Lovable renders the product cards and detail views from the Supabase records.


## Why This Contract Exists

Qdrant should not be treated as the full product database.

Qdrant is for:

- retrieval
- ranking
- filtering
- memory-aware recommendation logic

Supabase is for:

- full product detail
- rich technical data
- full image and document context
- the canonical product record shown to users


## Endpoints

### `POST /constraints`

Purpose:
- store hard user rules for the session and in memory

Request shape:

```json
{
  "user_id": "user_01",
  "max_wattage": 40,
  "max_price_chf": 300,
  "forbidden_materials": ["plastic"],
  "kelvin_min": 2700,
  "kelvin_max": 3200,
  "room_type": null
}
```

### `POST /browse`

Purpose:
- store episodic memory about clicked products

Request shape:

```json
{
  "user_id": "user_01",
  "product_id": "article_123",
  "name": "Product Name",
  "description": "Short product description"
}
```

### `POST /chat`

Purpose:
- retrieve baseline and MARA results
- generate the natural-language answer

Request shape:

```json
{
  "user_id": "user_01",
  "message": "I need a warm ceiling light under 300 CHF",
  "preferred_style": null,
  "preferred_finish": null,
  "preferred_mood": null
}
```


## `POST /chat` Response Contract

MARA returns two result lists:

- `baseline_results`
- `mara_results`

Both should include stable hydration identifiers.
The response also includes:

- `hydration`
- `frontend`


## Baseline Result Shape

```json
{
  "product_id": "article_123",
  "source_article_id": 123,
  "source_article_number": "2.815711233001",
  "source_l_number": 100467,
  "name": "DL150LED SB 12 C/EW 830 WH9016 OP",
  "manufacturer": "Performance In Lighting - Italy",
  "category": "Einbauspot",
  "family": "DLSB",
  "style": null,
  "finish": "textured",
  "mood": null,
  "image_url": "https://...",
  "tags": ["ceiling", "down_light", "inside"],
  "similarity_score": 0.8123,
  "method": "baseline"
}
```

Baseline results are mainly for comparison and debugging. Lovable may choose not
to show them in the final UI if the product direction changes.


## MARA Result Shape

```json
{
  "product_id": "article_123",
  "source_article_id": 123,
  "source_article_number": "2.815711233001",
  "source_l_number": 100467,
  "name": "DL150LED SB 12 C/EW 830 WH9016 OP",
  "manufacturer": "Performance In Lighting - Italy",
  "category": "Einbauspot",
  "family": "DLSB",
  "price_chf": 199.0,
  "wattage": 13.0,
  "kelvin": 3000,
  "material": null,
  "style": null,
  "finish": "textured",
  "mood": null,
  "room_type": null,
  "image_url": "https://...",
  "tags": ["ceiling", "down_light", "inside"],
  "similarity_score": 0.8012,
  "final_score": 0.9012,
  "violations": []
}
```


## Fields Lovable Should Trust Directly

Lovable can use these directly from MARA for fast rendering or fallback UI:

- `product_id`
- `source_article_id`
- `source_article_number`
- `source_l_number`
- `name`
- `manufacturer`
- `category`
- `family`
- `price_chf`
- `wattage`
- `kelvin`
- `image_url`
- `tags`
- `violations`
- `similarity_score`
- `final_score`


## Frontend Integration Metadata

`POST /chat` also returns a frontend-focused hydration block:

```json
{
  "hydration": {
    "provider": "supabase",
    "preferred_key": "source_article_id",
    "ordered_article_ids": [248, 441, 300],
    "ranked_targets": [
      {
        "rank": 1,
        "product_id": "article_248",
        "source_article_id": 248,
        "source_article_number": "2.305673",
        "source_l_number": 103021
      }
    ]
  },
  "frontend": {
    "version": "2026-03-12",
    "primary_result_field": "mara_results",
    "hydration_provider": "supabase",
    "hydration_key": "source_article_id",
    "baseline_available": true,
    "merge_strategy": "hydrate_supabase_then_enrich_with_mara"
  }
}
```

Lovable should use this block as the stable integration contract:

- `hydration.ordered_article_ids` preserves MARA rank order
- `hydration.ranked_targets` preserves rank plus fallback identifiers
- `frontend.hydration_key` tells Lovable which Supabase key to use


## Fields Lovable Should Hydrate From Supabase

Lovable should fetch the full product detail from Supabase using:

- `source_article_id` as the preferred join key

Recommended Supabase fetch target:

- `articles`
- related technical profile
- related character profile
- related classification
- manufacturer/category/family
- media/documents if needed


## Recommended Hydration Key

Primary key for hydration:

- `source_article_id`

Secondary display identifiers:

- `product_id`
- `source_article_number`
- `source_l_number`

Rule:
- MARA owns the retrieval identity
- Supabase owns the full product record


## Recommended Lovable Rendering Flow

1. Call `POST /chat`
2. Receive ranked MARA results
3. Extract `source_article_id` values
4. Fetch matching `articles` rows and related records from Supabase
5. Merge MARA metadata onto the Supabase results
6. Render:
   - product cards from Supabase data
   - ranking / reason / violations from MARA data


## Merge Strategy

MARA fields should enrich the hydrated product records, not replace them.

Recommended merge:

- Supabase provides:
  - canonical product content
  - images
  - detailed technical fields
  - document links
- MARA provides:
  - ranking order
  - scores
  - constraint violations
  - recommendation reasoning context


## Why This Is Better

This keeps the architecture clean:

- FastAPI + Qdrant decide relevance
- Supabase provides full truth
- Lovable renders real data

This avoids:

- duplicating the entire product database inside Qdrant responses
- using the LLM as the product source of truth
- coupling frontend rendering to retrieval storage
