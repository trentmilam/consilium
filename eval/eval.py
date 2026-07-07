"""Consilium v1 scripted eval — measured metrics, exits 0 on thresholds.

    python eval/eval.py

Deterministic (pure-python hashing embedder; no randomness, no network).

Metrics:
  routing_accuracy   single-domain query's top module == expected      (target >= 0.80)
  fanout_rate        cross-domain query selects all expected modules    (target == 1.00)
  abstention_rate    out-of-scope query is refused, not answered        (target == 1.00)
  oos_keyword_rate   OOS query w/ a lone incidental subject keyword abstains (target == 1.00)
  citation_coverage  every emitted claim bound to a real source span    (target == 1.00)
  fabricated_dropped an injected unsupported claim is dropped by gate    (target == True)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)

from consilium.embed import HashEmbedder          # noqa: E402
from consilium.registry import Registry           # noqa: E402
from consilium.router import Router               # noqa: E402
from consilium.composer import compose            # noqa: E402


def main() -> int:
    emb = HashEmbedder(1024)
    reg = Registry.load(os.path.join(ROOT, "modules"), emb)
    router = Router(reg, emb)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json"), encoding="utf-8") as f:
        cases = json.load(f)

    # --- routing accuracy (single-domain) ---
    single = cases["single"]
    correct = 0
    for c in single:
        rr = router.route(c["query"])
        top = rr.ranked[0][0] if rr.ranked else None
        ok = (not rr.abstained) and top == c["expect"]
        correct += 1 if ok else 0
        print(f"[single] {'OK  ' if ok else 'MISS'} top={top:<9} expect={c['expect']:<9} :: {c['query']}")
    routing_acc = correct / len(single)

    # --- fan-out (cross-domain) ---
    cross = cases["crossdomain"]
    fan_ok = 0
    for c in cross:
        rr = router.route(c["query"])
        sel = set(rr.selected)
        ok = (not rr.abstained) and set(c["expect"]).issubset(sel) and len(sel) >= 2
        fan_ok += 1 if ok else 0
        print(f"[cross]  {'OK  ' if ok else 'MISS'} selected={rr.selected} expect={c['expect']} :: {c['query']}")
    fanout_rate = fan_ok / len(cross)

    # --- abstention (out-of-scope) ---
    oos = cases["oos"]
    abs_ok = 0
    for c in oos:
        rr = router.route(c["query"])
        ok = rr.abstained
        abs_ok += 1 if ok else 0
        print(f"[oos]    {'OK  ' if ok else 'MISS'} abstained={rr.abstained} :: {c['query']}")
    abstention_rate = abs_ok / len(oos)

    # --- OOS lone-incidental-keyword abstention (red-case for the fail-open fix) ---
    # An out-of-scope query whose ONLY overlap with a module is a single incidental
    # subject keyword ("stock" in "race my stock car"; "market" in "farmers market")
    # MUST abstain -- both at the router AND end-to-end (no cited answer emitted).
    # Before the fix these cleared floor=0.11 and returned confidently-cited but
    # query-irrelevant answers.
    oosk = cases["oos_keyword"]
    oosk_ok = 0
    for c in oosk:
        rr = router.route(c["query"])
        ans = compose(c["query"], rr, reg, emb)
        ok = rr.abstained and ans.abstained and not ans.citations
        oosk_ok += 1 if ok else 0
        print(f"[oos-kw] {'OK  ' if ok else 'MISS'} route_abstained={rr.abstained} "
              f"answer_abstained={ans.abstained} cites={len(ans.citations)} :: {c['query']}")
    oos_keyword_rate = oosk_ok / len(oosk)

    # --- citation coverage (all answered cases) ---
    answered = 0
    covered = 0
    for c in single + cross:
        rr = router.route(c["query"])
        ans = compose(c["query"], rr, reg, emb)
        if ans.abstained:
            continue
        answered += 1
        covered += 1 if (ans.citations and all(ci.chunk_id for ci in ans.citations)) else 0
    citation_coverage = covered / answered if answered else 0.0

    # --- fabricated-claim rejection ---
    fc = cases["fabricated_claim"]
    rr = router.route(fc["query"])
    ans = compose(fc["query"], rr, reg, emb, extra_claims=[fc["injected"]])
    fab_dropped = (fc["injected"] in ans.dropped) and all(ci.claim != fc["injected"] for ci in ans.citations)
    print(f"[gate]   {'OK  ' if fab_dropped else 'MISS'} fabricated_claim_dropped={fab_dropped}")

    print("\n=== METRICS (measured) ===")
    print(f"routing_accuracy   = {routing_acc:.3f}   target >= 0.80")
    print(f"fanout_rate        = {fanout_rate:.3f}   target == 1.00")
    print(f"abstention_rate    = {abstention_rate:.3f}   target == 1.00")
    print(f"oos_keyword_rate   = {oos_keyword_rate:.3f}   target == 1.00  (lone-incidental-keyword)")
    print(f"citation_coverage  = {citation_coverage:.3f}   target == 1.00  ({covered}/{answered} answered)")
    print(f"fabricated_dropped = {fab_dropped}   target == True")

    passed = (routing_acc >= 0.80 and fanout_rate == 1.0 and abstention_rate == 1.0
              and oos_keyword_rate == 1.0 and citation_coverage == 1.0 and fab_dropped)
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
