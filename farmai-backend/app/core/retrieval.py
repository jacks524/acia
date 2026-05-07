import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import CHUNKS_META_PATH, DEVICE, ENCODER_NAME, FAISS_INDEX_PATH

if not FAISS_INDEX_PATH.exists():
    raise FileNotFoundError(f"Index FAISS introuvable : {FAISS_INDEX_PATH}")
if not CHUNKS_META_PATH.exists():
    raise FileNotFoundError(f"Métadonnées chunks introuvables : {CHUNKS_META_PATH}")

print(f"Chargement de l'encodeur : {ENCODER_NAME}")
encoder = SentenceTransformer(ENCODER_NAME, device=DEVICE)
print(f"Encodeur pret sur {DEVICE}")

index = faiss.read_index(str(FAISS_INDEX_PATH))
with CHUNKS_META_PATH.open("r", encoding="utf-8") as f:
    all_chunks = json.load(f)

texts = [chunk["text"] for chunk in all_chunks]
embeddings = encoder.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

LANG_INDICES: dict[str, list[int]] = {}
for i, chunk in enumerate(all_chunks):
    LANG_INDICES.setdefault(chunk["lang"], []).append(i)


def retrieve(query: str, lang: str | None = None, k: int = 3) -> list[tuple[float, dict]]:
    """
    Cherche les k chunks les plus pertinents.
    Si lang est fourni, la recherche est filtrée sur cette langue.
    """
    q_emb = encoder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    if lang is None:
        scores, indices = index.search(q_emb, k)
        return [
            (float(score), all_chunks[i])
            for score, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    candidate_idxs = LANG_INDICES.get(lang, [])
    if not candidate_idxs:
        return []

    candidate_embs = embeddings[candidate_idxs]
    sims = (candidate_embs @ q_emb.T).flatten()
    top_k_local = sims.argsort()[::-1][:k]

    return [
        (float(sims[local_idx]), all_chunks[candidate_idxs[local_idx]])
        for local_idx in top_k_local
    ]
