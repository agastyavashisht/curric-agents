"""
Shared utility functions used across multiple src modules.

Keeping these here avoids private cross-module dependencies
(e.g. assessment.py importing _tokenize from retrieval.py).
"""
import re
from typing import List


def tokenize(text: str) -> List[str]:
    """Lowercase word tokeniser — returns alphanumeric tokens only."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tfidf_cosine(a: str, b: str) -> float:
    """
    Simple bag-of-words cosine similarity between two strings.
    Used for short-answer embedding cross-check in the Assessment agent.
    Pure Python + numpy only — no scipy, no sentence-transformers.
    """
    import numpy as np
    ta, tb = tokenize(a), tokenize(b)
    vocab  = list({*ta, *tb})
    if not vocab:
        return 0.0
    v2i = {w: i for i, w in enumerate(vocab)}
    va  = np.zeros(len(vocab), dtype=np.float32)
    vb  = np.zeros(len(vocab), dtype=np.float32)
    for t in ta:
        va[v2i[t]] += 1
    for t in tb:
        vb[v2i[t]] += 1
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va / na, vb / nb))
