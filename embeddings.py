"""Embedding helpers used for product search and memory retrieval."""

import os
from functools import lru_cache

MODEL_NAME  = "BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024

HF_TOKEN = os.getenv("HF_TOKEN", None)


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {MODEL_NAME}")
    print("First run downloads ~1.3GB — this is normal, happens once only.\n")
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)
    print(f"Model loaded. Vector size: {model.get_sentence_embedding_dimension()}\n")
    return model

_BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def embed(text: str) -> list[float]:
    """Encode a single text into a normalized embedding vector."""
    model  = _load_model()
    vector = model.encode(_BGE_PREFIX + text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode multiple texts into normalized embedding vectors."""
    model   = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vectors]
