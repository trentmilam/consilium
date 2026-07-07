"""Consilium v3 — heterogeneous modules: a COMPUTE module alongside retrieval modules.

A ComputeModule is routable like any other (it has a descriptor the router scores),
but instead of retrieving text it PARSES the query and returns an *audited*,
deterministic computation (a no-arithmetic-by-LLM audited compute pattern). The
switchboard can now answer a finance-math query with an exact number AND a text
query with a cited passage — one router over both.

Additive: v1/v2 retrieval modules and their evals are untouched.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from .module import Descriptor


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(spot, strike, t, r, vol, kind="call"):
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "call":
        return spot * _cdf(d1) - strike * math.exp(-r * t) * _cdf(d2)
    return strike * math.exp(-r * t) * _cdf(-d2) - spot * _cdf(-d1)


_NUM = r"([-+]?\d*\.?\d+)"


def _parse_params(query: str) -> dict:
    q = query.lower()

    def grab(keys):
        for k in keys:
            m = re.search(k + r"[^0-9+\-]{0,14}" + _NUM, q)
            if m:
                return float(m.group(1))
        return None

    return {
        "spot": grab([r"spot", r"underlying", r"price of"]),
        "strike": grab([r"strike"]),
        "t": grab([r"expiry", r"maturity", r"years?", r"\bt\b"]),
        "r": grab([r"risk-free rate", r"rate", r"\br\b"]),
        "vol": grab([r"volatility", r"\bvol", r"sigma"]),
        "kind": "put" if "put" in q else "call",
    }


@dataclass
class ComputeModule:
    """Generic BASE for a routable compute capability: no text corpus (``chunks``
    stays empty so the router's best-chunk term is 0), ``retrieve()`` is a no-op,
    and ``centroid()`` is memoized from the descriptor -- so the router scores it
    exactly like a retrieval :class:`~consilium.module.Module`. Subclasses (here
    :class:`FinanceComputeModule`; disk-registered adapters via
    ``Registry.load``'s ``kind: "compute"`` path) implement ``compute(query)``.
    """
    name: str
    descriptor: Descriptor
    embedder: object
    chunks: list = field(default_factory=list)   # empty -> router best-chunk term is 0
    _centroid: list = field(default=None, repr=False)

    def centroid(self):
        if self._centroid is None:
            self._centroid = self.embedder.embed_one(self.descriptor.profile_text())
        return self._centroid

    def retrieve(self, query_vec, k: int = 3):
        return []

    def compute(self, query: str) -> dict:
        """Parse ``query`` into this capability's own params and return the
        audited envelope ``{ok, tool, inputs, method, deterministic, result}``
        (or ``{ok: False, tool, error}`` on unparseable input). The generic base
        has no domain logic; subclasses implement it."""
        raise NotImplementedError


@dataclass
class FinanceComputeModule(ComputeModule):
    """The original Black-Scholes compute capability, now a concrete subclass of
    the generic :class:`ComputeModule` base (``make_finance_module()`` behavior
    is unchanged)."""

    def compute(self, query: str) -> dict:
        p = _parse_params(query)
        req = ["spot", "strike", "t", "r", "vol"]
        missing = [k for k in req if p.get(k) is None]
        if missing:
            return {"ok": False, "tool": "black_scholes", "error": f"missing params: {missing}"}
        invalid = [k for k in ("spot", "strike", "t", "vol") if p[k] <= 0]
        if invalid:
            return {"ok": False, "tool": "black_scholes",
                     "error": f"out-of-domain params (must be > 0): {invalid}"}
        price = black_scholes(p["spot"], p["strike"], p["t"], p["r"], p["vol"], p["kind"])
        return {
            "ok": True, "tool": "black_scholes",
            "inputs": {k: p[k] for k in req + ["kind"]},
            "method": "closed-form Black-Scholes (normal CDF via math.erf)",
            "deterministic": True, "result": round(price, 6),
        }


def make_finance_module(embedder, name: str = "quant") -> ComputeModule:
    desc = Descriptor(
        name=name,
        subjects=["option pricing", "black-scholes valuation", "call option", "put option", "greeks",
                  "implied volatility", "strike price", "spot price", "risk-free rate", "quantitative finance math"],
        example_queries=["price a European call option", "what is the Black-Scholes value", "value this option"],
        trust_tier=0.9,
    )
    return FinanceComputeModule(name=name, descriptor=desc, embedder=embedder)


def answer_v3(query: str, registry, embedder, router) -> dict:
    """Route across retrieval + compute modules, fanning out across ALL selected
    compute modules and (if any text modules are also selected) the hardened
    retrieval compose over just those.

    Back-compat: if the selection is a single compute module and no text module,
    return the original v3 shape (``{"kind":"compute","module":<name>,"audited":
    <dict>,"routing":...}``) so ``eval_v3.py`` stays green. Otherwise return
    ``{"kind": "compute"|"retrieval"|"mixed", "computed": [...], "routing": ...}``
    plus (when any text module was selected) ``module``/``answer``/``citations``
    from the retrieval compose.
    """
    rr = router.route(query)
    if rr.abstained:
        return {"kind": "abstain", "routing": rr.ranked[:3]}

    selected = [(name, registry.by_name(name)) for name in rr.selected]
    computes = [m for _, m in selected if isinstance(m, ComputeModule)]
    texts = [name for name, m in selected if not isinstance(m, ComputeModule)]

    if computes and not texts and len(computes) == 1:
        c = computes[0]
        return {"kind": "compute", "module": c.name, "audited": c.compute(query), "routing": rr.ranked[:3]}

    computed = [{"module": c.name, "audited": c.compute(query)} for c in computes]
    answer = None
    if texts:
        from .composer import compose
        # compose() only reads .abstained + .selected (composer.py) -- a fresh
        # RouteResult over just the text subset, not a mutation of the shared rr.
        text_rr = replace(rr, selected=texts, abstained=False)
        answer = compose(query, text_rr, registry, embedder, harden=True)

    kind = "mixed" if computed and answer else ("compute" if computed else "retrieval")
    out = {"kind": kind, "computed": computed, "routing": rr.ranked[:3]}
    if answer is not None:
        out["module"] = answer.modules_used
        out["answer"] = answer.text
        out["citations"] = len(answer.citations)
    return out
