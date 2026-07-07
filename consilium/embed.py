"""Deterministic, dependency-free hashing embedder + cosine.

Vendored from a standard hashing-trick embedder (the hashing trick over a
stable BLAKE2b digest, fixed ``dim``, L2-normalized) so Consilium is publishable
standalone: no numpy, no network, no model download. Identical text always yields
the identical vector, so snapshots, the demo, and the eval are reproducible across
processes and platforms.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

__all__ = ["HashEmbedder", "cosine", "tokenize"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# A modest English stopword list. Dropping these sharpens short queries (a 4-word
# question is otherwise ~75% function words that only add hash-collision noise) and
# keeps vectors content-driven, which is what makes routing + abstention separable.
_STOP = frozenset("""
a an the and or of to in on for is are was were be been being do does did done what which who whom
how when where why our your my we you i it its this that these those with as at by from about into
per each any all some so than then there here not no can could would should may might will shall
has have had having me us them they he she his her their if our out over under again more most
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric content tokens (stopwords removed)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


class HashEmbedder:
    """Hashing-trick bag-of-words embedder: pure python, fixed dim, deterministic."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t or "") for t in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Inputs from :class:`HashEmbedder` are already
    L2-normalized, so this is a dot product; we normalize defensively anyway."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
