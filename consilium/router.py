"""The switchboard: route a query to the right module(s), or abstain.

A module's routing score blends three signals, all cheap and deterministic:

    score = 0.45 * descriptor-centroid cosine   (clean: ~0 for out-of-scope)
          + 0.20 * best-chunk cosine            (a peek into the corpus; noisiest)
          + 0.35 * subject-keyword overlap      (clean: 0 for out-of-scope)

The clean signals (centroid, subject overlap) are weighted over the noisy
best-chunk term, because a single hash-bucket collision can give an unrelated
query a spurious best-chunk score -- but never a centroid or subject-overlap one.

Selection is by an **absolute floor**: every module scoring >= ``floor`` is
selected (that is the fan-out for cross-domain queries -- a relevant secondary
module sits well above the floor, an out-of-scope one well below). If NO module
clears the floor, the query is out-of-scope and the router **abstains**.

Clearing the floor is necessary but not sufficient: the selection must also
contain at least one **anchor** module -- one the query genuinely engages
(>=2 subject tokens, OR a strong descriptor-centroid, OR a strong best-chunk
match). A query that clears the floor purely on ONE incidental subject keyword
(e.g. "stock" in "race my stock car", "market" in "the farmers market") yields no
anchor and the router **abstains** rather than emit a confidently-cited but
query-irrelevant answer. A genuine query always has an anchor; a marginal
cross-domain secondary only rides along a real anchor primary.

An optional LLM-router upgrade (``use_llm``) is a thin hook that falls back to
this deterministic path if no gateway is reachable; v1 defaults to deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embed import cosine, tokenize


@dataclass
class RouteResult:
    query: str
    ranked: list      # [(module_name, score, breakdown)]
    selected: list    # module names selected (>=1 unless abstained)
    abstained: bool
    trace: dict


class Router:
    def __init__(self, registry, embedder, floor: float = 0.11,
                 anchor_min_subjects: int = 2, anchor_centroid: float = 0.25,
                 anchor_best_chunk: float = 0.25) -> None:
        self.registry = registry
        self.embedder = embedder
        self.floor = floor      # a module must score >= floor to be selected; none => abstain
        # A selection must contain at least one ANCHOR module, or the query is
        # treated as out-of-scope and the router abstains. A lone module that
        # clears the floor purely on ONE incidental subject keyword (a low
        # descriptor-centroid, single subject-token match) is NOT an anchor -- this
        # closes the fail-open where e.g. "race my stock car" or "the farmers
        # market" scored just over the floor on the single word stock/market and
        # got a confidently-cited, query-irrelevant answer. A genuine query always
        # has an anchor; a marginal cross-domain secondary only rides along a real
        # anchor primary. A module is an anchor when the query genuinely engages
        # its subject: >=2 subject tokens, OR a strong descriptor-centroid match,
        # OR a strong best-chunk corpus match. A single incidental keyword produces
        # none of these.
        self.anchor_min_subjects = anchor_min_subjects
        self.anchor_centroid = anchor_centroid
        self.anchor_best_chunk = anchor_best_chunk

    def _module_score(self, module, query_vec, q_tokens):
        cen = cosine(query_vec, module.centroid())
        best = max((cosine(query_vec, c.vec) for c in module.chunks), default=0.0)
        subj_tokens = set()
        for s in module.descriptor.subjects:
            subj_tokens |= set(tokenize(s))
        n_subj = len(q_tokens & subj_tokens)
        overlap = (n_subj / len(q_tokens)) if q_tokens else 0.0
        score = 0.45 * cen + 0.20 * best + 0.35 * overlap
        anchor = ((n_subj >= self.anchor_min_subjects)
                  or (cen >= self.anchor_centroid)
                  or (best >= self.anchor_best_chunk))
        return score, {
            "centroid": round(cen, 3),
            "best_chunk": round(best, 3),
            "subject_overlap": round(overlap, 3),
            "n_subj": n_subj,
            "anchor": anchor,
        }

    def route(self, query: str) -> RouteResult:
        query_vec = self.embedder.embed_one(query)
        q_tokens = set(tokenize(query))
        ranked = []
        for m in self.registry.modules:
            s, bd = self._module_score(m, query_vec, q_tokens)
            ranked.append((m.name, round(s, 4), bd))
        ranked.sort(key=lambda t: t[1], reverse=True)

        cleared = [(name, s, bd) for name, s, bd in ranked if s >= self.floor]
        if not cleared:
            return RouteResult(query, ranked, [], True,
                               {"reason": "no module cleared floor", "floor": self.floor})
        if not any(bd["anchor"] for _, _, bd in cleared):
            # Every module that cleared the floor did so on a lone incidental
            # keyword -- out of scope. Abstain rather than emit an irrelevant answer.
            return RouteResult(query, ranked, [], True,
                               {"reason": "no anchor module (only incidental-keyword matches)",
                                "floor": self.floor})
        selected = [name for name, _, _ in cleared]
        return RouteResult(query, ranked, selected, False,
                           {"floor": self.floor, "top": ranked[0][1]})
