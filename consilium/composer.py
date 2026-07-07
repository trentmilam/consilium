"""The composer: assemble a cited answer across the selected modules.

v1 is extractive: the retrieved supporting chunks ARE the claims, so every line
of the answer is bound to a real source span (the citation-coverage guarantee).
``extra_claims`` lets a caller feed an LLM-proposed / test-injected claim through
the same integrity gate -- an unsupported one is dropped.

v2 (``harden=True``) adds the integrity-hardening layer: per-module poison
quarantine before gating, and cross-module conflict detection (resolved by module
trust) surfaced on the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .integrity import gate
from .hardening import quarantine_poison, detect_conflicts


@dataclass
class Answer:
    query: str
    text: str
    citations: list
    modules_used: list
    abstained: bool
    dropped: list
    conflicts: list = field(default_factory=list)
    quarantined: list = field(default_factory=list)


def compose(query, route_result, registry, embedder,
            k_per_module: int = 2, support_floor: float = 0.5, extra_claims=None,
            harden: bool = False, query_relevance_floor: float = 0.05) -> Answer:
    if route_result.abstained:
        return Answer(query, "(no module in scope -- abstained)", [], [], True, [])

    query_vec = embedder.embed_one(query)
    candidate_chunks = []
    claims = []
    quarantined = []
    trust_map = {}
    for name in route_result.selected:
        m = registry.by_name(name)
        if m is None:
            continue
        trust_map[name] = m.descriptor.trust_tier
        retrieved = m.retrieve(query_vec, k_per_module)
        if harden:
            kept_ch, quar_ch, _flagged = quarantine_poison(m, retrieved, embedder)
            quarantined.extend(f"{name}/{c.chunk_id if hasattr(c, 'chunk_id') else c.id}" for c in quar_ch)
            use = [(c, None) for c in kept_ch]
        else:
            use = retrieved
        for chunk, _score in use:
            candidate_chunks.append((name, chunk))
            claims.append(chunk.text)

    if extra_claims:
        claims.extend(extra_claims)

    result = gate(claims, embedder, candidate_chunks, support_floor,
                  query_vec=query_vec, query_floor=query_relevance_floor)
    kept = result["kept"]
    if not kept:
        return Answer(query, "(no supported claims -- abstained)", [], [], True, result["dropped"])

    seen = set()
    unique = []
    for c in kept:
        key = (c.module, c.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    conflicts = []
    if harden and unique:
        conflicts = detect_conflicts(unique, embedder, trust_of=lambda c: trust_map.get(c.module, 0.5))

    modules_used = sorted({c.module for c in unique})
    lines = [f"- {c.claim}  [{c.module}/{c.chunk_id}, support={c.score}]" for c in unique]
    if conflicts:
        lines.append("")
        for cf in conflicts:
            losers = "; ".join(f"{l.module} (trust-resolved out)" for l in cf.losers)
            lines.append(f"  [conflict] resolved to {cf.winner.module}; contested by: {losers}")
    return Answer(query, "\n".join(lines), unique, modules_used, False, result["dropped"],
                  conflicts=conflicts, quarantined=quarantined)
