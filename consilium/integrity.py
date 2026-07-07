"""The integrity gate: bind every claim to a source span, or drop it.

Each candidate claim is matched against the retrieved chunks; the best match must
clear ``floor`` or the claim is **unsupported** and dropped. An answer with no
surviving claim **abstains**. This is what makes the switchboard trustworthy: it
cannot emit a sentence that no source supports (the failure mode of a plain RAG
chatbot), and it refuses rather than guesses.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embed import cosine


@dataclass
class Citation:
    claim: str
    module: str
    doc: str
    chunk_id: str
    score: float


def bind_claim(claim: str, embedder, candidate_chunks, floor: float,
               query_vec=None, query_floor: float = 0.0):
    """Return the best-supporting Citation for ``claim`` or None if unsupported.

    ``candidate_chunks`` is a list of ``(module_name, Chunk)``.

    A supporting chunk must satisfy TWO conditions, not one:
      1. it supports the claim  (cosine(claim, chunk) >= ``floor``), and
      2. it is RELEVANT TO THE QUERY (cosine(query, chunk) >= ``query_floor``),
         when ``query_vec`` is supplied.
    Condition (2) closes the tautological self-support hole: an extractive claim is
    byte-identical to its own chunk (cosine 1.0), so claim-support alone always
    passes -- the chunk must ALSO be relevant to what was actually asked, or the
    claim is dropped as query-irrelevant rather than emitted with a confident cite.
    """
    cv = embedder.embed_one(claim)
    best = None
    best_s = -1.0
    for mod_name, chunk in candidate_chunks:
        if query_vec is not None and cosine(query_vec, chunk.vec) < query_floor:
            continue  # chunk not relevant to the query -> not a valid support
        s = cosine(cv, chunk.vec)
        if s > best_s:
            best_s = s
            best = (mod_name, chunk)
    if best is None or best_s < floor:
        return None
    mod_name, chunk = best
    return Citation(claim=claim, module=mod_name, doc=chunk.doc,
                    chunk_id=chunk.id, score=round(best_s, 3))


def gate(claims, embedder, candidate_chunks, floor: float = 0.5,
         query_vec=None, query_floor: float = 0.0):
    """Bind each claim; keep the supported ones, drop the rest, abstain if none.

    When ``query_vec`` is supplied, a claim is kept only if its supporting chunk is
    also relevant to the query (see :func:`bind_claim`)."""
    kept = []
    dropped = []
    for claim in claims:
        cit = bind_claim(claim, embedder, candidate_chunks, floor,
                         query_vec=query_vec, query_floor=query_floor)
        if cit is not None:
            kept.append(cit)
        else:
            dropped.append(claim)
    return {"kept": kept, "dropped": dropped, "abstain": len(kept) == 0}
