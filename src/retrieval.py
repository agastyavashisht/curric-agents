"""
Lightweight RAG retriever for the Content agent.

Builds an in-memory TF-IDF index over the domain corpus
(data/corpus/<domain>.md), split into paragraph-level chunks.

Pure numpy — no scipy, no sentence-transformers, no FAISS.
Tokenisation is shared via src.utils.tokenize.

The index is built once per process and cached in memory.

Usage:
    from src.retrieval import retrieve
    chunks = retrieve("python_programming", "explain for loops", top_k=3)
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List

import numpy as np

from src.utils import tokenize

_CORPUS_DIR = "data/corpus"


def _corpus_path(domain: str) -> str:
    return os.path.join(_CORPUS_DIR, f"{domain}.md")


def _load_chunks(domain: str) -> List[str]:
    """Split the corpus markdown into paragraph-level chunks (min 30 chars)."""
    path = _corpus_path(domain)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = [c.strip() for c in re.split(r"\n\n+", raw) if len(c.strip()) > 30]
    return chunks


# ── pure-numpy TF-IDF ─────────────────────────────────────────────────────────

def _build_tfidf(corpus: List[str]):
    """Returns (tfidf_matrix, vocab_index, idf_vector)."""
    tokenized = [tokenize(doc) for doc in corpus]
    vocab = sorted({tok for doc in tokenized for tok in doc})
    v2i   = {w: i for i, w in enumerate(vocab)}
    V, N  = len(vocab), len(corpus)

    tf = np.zeros((N, V), dtype=np.float32)
    for di, tokens in enumerate(tokenized):
        for tok in tokens:
            tf[di, v2i[tok]] += 1
        if tf[di].sum() > 0:
            tf[di] /= tf[di].sum()

    df  = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log((N + 1) / (df + 1)) + 1.0

    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms
    return tfidf, v2i, idf


@lru_cache(maxsize=8)
def _build_index(domain: str):
    chunks = _load_chunks(domain)
    if not chunks:
        return [], None, None, None
    tfidf, v2i, idf = _build_tfidf(chunks)
    return chunks, tfidf, v2i, idf


def _query_vec(query: str, v2i: dict, idf: np.ndarray) -> np.ndarray:
    tokens = tokenize(query)
    qv = np.zeros(len(v2i), dtype=np.float32)
    for tok in tokens:
        if tok in v2i:
            qv[v2i[tok]] += 1
    qv *= idf
    norm = np.linalg.norm(qv)
    if norm > 0:
        qv /= norm
    return qv


def retrieve(domain: str, query: str, top_k: int = 3) -> List[str]:
    """Return top_k most relevant corpus chunks for the query."""
    chunks, tfidf, v2i, idf = _build_index(domain)
    if tfidf is None:
        return []
    qv    = _query_vec(query, v2i, idf)
    scores = tfidf @ qv
    top_k  = min(top_k, len(chunks))
    top_i  = np.argsort(scores)[::-1][:top_k].tolist()
    return [chunks[i] for i in top_i]
