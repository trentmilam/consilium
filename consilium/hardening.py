"""Consilium v2 integrity hardening: cross-corpus conflict detection + trust
resolution, and corroboration-based poison quarantine.

Two failure modes a plain multi-RAG switchboard cannot handle:

* **Cross-corpus conflict** — two different modules assert contradictory facts
  about the same topic. Resolve by the higher-trust source; *surface* the loser.
* **Intra-corpus poison** — an injected/outlier chunk asserts a value that
  contradicts the corroborated majority of its module. Quarantine it — fail-safe.

Both reduce to one primitive: two claims that are the **same topic** (high cosine)
but carry **disjoint magnitude values** are in conflict.

The value comparison is scale/currency-aware: ``$4.2 billion`` and ``$4,200
million`` normalize to the same magnitude (agree), while ``$4.2 billion`` vs
``$4.2 million`` do not (a magnitude conflict a naive digit-set test would miss).
Bare numbers (years, percents, counts, identifiers) are ignored, so an incidental
"in 2025" does not manufacture a conflict.

Corroboration is counted over DISTINCT sources: near-duplicate chunks (a poison
claim re-injected N times) collapse to one, so naive copy-flooding cannot invert
the defense. Quarantine is fail-safe: it removes only an *uncorroborated* claim
that conflicts with a *corroborated* one; anything ambiguous is surfaced (flagged),
never silently dropped.

Deterministic v2 heuristic. Honest limits (see README): word-spelled numbers,
non-numeric conflicts (e.g. "Delaware" vs "Nevada"), same-magnitude different-metric
claims, and paraphrase-flooding are NOT handled here — a future NLI /
trust-provenance layer would be needed to close those gaps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .embed import cosine

# number optionally preceded by $ and optionally followed by a scale word
_VAL_RE = re.compile(r"(\$)?\s*(\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|tn|bn|mn)?", re.IGNORECASE)
_SCALE = {"trillion": 1e12, "tn": 1e12, "billion": 1e9, "bn": 1e9,
          "million": 1e6, "mn": 1e6, "thousand": 1e3}
_DUP = 0.97  # cosine at/above which two chunks are "the same source re-injected"


def _salient_values(text: str) -> set:
    """Canonical magnitudes for numbers that carry a currency or a scale word.
    Bare numbers (years/percents/counts/identifiers) are intentionally excluded."""
    out = set()
    for cur, num, scale in _VAL_RE.findall(text or ""):
        scale = scale.lower()
        mult = _SCALE.get(scale)
        if mult is None and not cur:
            continue  # bare number -> not a salient magnitude
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        out.add("%.4g" % (val * (mult if mult is not None else 1.0)))
    return out


def _same_topic(a: str, b: str, embedder, floor: float) -> bool:
    return cosine(embedder.embed_one(a), embedder.embed_one(b)) >= floor


def _conflict(a: str, b: str) -> bool:
    """Both carry magnitude values and share NONE (genuine disagreement)."""
    sa, sb = _salient_values(a), _salient_values(b)
    return bool(sa) and bool(sb) and not (sa & sb)


def _agree(a: str, b: str) -> bool:
    """Share at least one magnitude value (positive corroboration)."""
    return bool(_salient_values(a) & _salient_values(b))


@dataclass
class Conflict:
    topic: str
    winner: object          # Citation
    losers: list            # [Citation]
    reason: str


def detect_conflicts(citations, embedder, trust_of, sim_floor: float = 0.6):
    """Group same-topic / conflicting-value citations from DIFFERENT modules and
    pick a winner by module trust (``trust_of(citation) -> float``). Losers are
    kept so the composer can surface the disagreement (never hide it)."""
    conflicts = []
    used = set()
    for i in range(len(citations)):
        if i in used:
            continue
        group = [citations[i]]
        for j in range(i + 1, len(citations)):
            if j in used:
                continue
            if (citations[j].module != citations[i].module
                    and _same_topic(citations[i].claim, citations[j].claim, embedder, sim_floor)
                    and _conflict(citations[i].claim, citations[j].claim)):
                group.append(citations[j])
                used.add(j)
        if len(group) > 1:
            used.add(i)
            ranked = sorted(group, key=lambda c: trust_of(c), reverse=True)
            conflicts.append(Conflict(
                topic=ranked[0].claim[:70],
                winner=ranked[0],
                losers=ranked[1:],
                reason="same topic, conflicting values; resolved by module trust",
            ))
    return conflicts


def _vec(chunk, embedder):
    return chunk.vec if getattr(chunk, "vec", None) else embedder.embed_one(chunk.text)


def _corroboration(chunk, module, embedder, sim_floor: float) -> int:
    """Count DISTINCT module chunks that agree with ``chunk`` (same topic, shared
    magnitude value). Near-duplicates (cosine >= _DUP) are skipped as the same
    source re-injected, so copy-flooding cannot manufacture corroboration."""
    cv = _vec(chunk, embedder)
    cnt = 0
    for other in module.chunks:
        if other.id == chunk.id:
            continue
        cos = cosine(cv, _vec(other, embedder))
        if cos >= _DUP:
            continue  # duplicate injection, not independent corroboration
        if cos >= sim_floor and _agree(chunk.text, other.text):
            cnt += 1
    return cnt


def quarantine_poison(module, retrieved, embedder, sim_floor: float = 0.6):
    """Within one module's retrieved ``[(chunk, score), ...]`` classify each chunk:

    * **keep**       — corroborated, or not in any conflict.
    * **quarantine** — uncorroborated AND conflicts with a corroborated claim
                       (a likely injected outlier).
    * **flag**       — ambiguous conflict (both sides uncorroborated); surfaced,
                       never silently dropped — the fail-safe direction.

    Returns ``(kept, quarantined, flagged)``.

    PERFORMANCE: the same-topic filter below compares ``ch`` against every chunk
    in ``module`` (all of ``module.chunks`` -- the whole module, which grows with
    corpus size). It therefore uses ``_vec()`` (reuses each chunk's already-computed
    ``.vec``, cached once at module load) rather than ``_same_topic()`` (which takes
    raw text and always calls ``embedder.embed_one()`` -- an uncached embed call per
    comparison, cheap only when the candidate set is small, as it is in
    ``detect_conflicts()`` above, which compares a handful of already-selected
    citations rather than a whole module's chunks). This function instead runs one
    such comparison per *retrieved chunk x module chunk* pair, so an uncached
    embed call here scales with corpus size rather than with a small fixed
    candidate set; using the cached vector keeps it at zero embedding calls
    regardless of how large a module's corpus grows.
    """
    chunks = [c for c, _ in retrieved]
    kept, quar, flagged = [], [], []
    for ch in chunks:
        cc = _corroboration(ch, module, embedder, sim_floor)
        ch_vec = _vec(ch, embedder)
        conflict_cos = [
            _corroboration(other, module, embedder, sim_floor)
            for other in module.chunks
            if other.id != ch.id
            and cosine(ch_vec, _vec(other, embedder)) >= sim_floor
            and _conflict(ch.text, other.text)
        ]
        if not conflict_cos:
            kept.append(ch)
        elif cc >= 1:
            kept.append(ch)                       # this chunk is itself corroborated
        elif max(conflict_cos) >= 1:
            quar.append(ch)                       # uncorroborated outlier vs a corroborated claim
        else:
            flagged.append(ch)                    # both uncorroborated -> surface, do not drop
    return kept, quar, flagged
