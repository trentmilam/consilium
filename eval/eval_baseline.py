"""Consilium A/B eval — MEASURED head-to-head vs a naive no-gate RAG. Exit 0 on the gap.

    python eval/eval_baseline.py

The whole Consilium wedge is one claim: *it abstains / drops instead of confidently
hallucinating.* Every other eval self-grades Consilium alone (perfect scores on a tiny N),
which a reviewer discounts. This eval turns the wedge from an ASSERTION into PROOF: it runs
a FAIR naive incumbent RAG on the SAME cases (eval/cases.json) and MEASURES where the naive
baseline emits a confident cited answer to an out-of-scope query, or emits an uncited /
fabricated claim, that Consilium abstains-on or drops.

FAIR BASELINE (not a strawman): ``naive_rag`` is exactly what a competent engineer's minimal
RAG does -- embed the query, retrieve the global top-k chunks by cosine across ALL modules,
and return them as the cited answer, passing any LLM-proposed claims straight through. It uses
the SAME embedder, the SAME corpus, and the SAME retrieval as Consilium. The ONLY variables
removed are Consilium's three gates:
  * the router's abstention floor + anchor gate (router.py),
  * the integrity gate's claim<->source binding (integrity.py),
  * the query-relevance floor (integrity.bind_claim query_floor).
So the measured gap is attributable to the gates and nothing else. The baseline is NOT
crippled: on IN-SCOPE queries it answers with real citations, at parity with Consilium (proved
below) -- it only fails on the out-of-scope / fabricated surface, which is the point.

Deterministic (pure-python hashing embedder; no randomness, no network).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)

from consilium.embed import HashEmbedder, cosine   # noqa: E402
from consilium.registry import Registry            # noqa: E402
from consilium.router import Router                 # noqa: E402
from consilium.composer import compose             # noqa: E402


def naive_rag(query, registry, embedder, k=3, extra_claims=None):
    """A fair, minimal incumbent RAG: global top-k cosine retrieve -> answer.

    No abstention floor, no anchor gate, no integrity binding, no query-relevance
    check -- i.e. Consilium MINUS its gates. Always answers if the corpus is non-empty
    (a naive RAG has nothing that would make it abstain). Any ``extra_claims`` (modelling
    an LLM that hallucinates an addition on top of the retrieved context) are emitted
    verbatim -- with no source binding, they are *uncited*.

    Returns dict: abstained, cited (list of (module, chunk, cos_to_query)), uncited
    (list of str -- emitted claims with no supporting span).
    """
    qv = embedder.embed_one(query)
    scored = []
    for m in registry.modules:
        for c in m.chunks:
            scored.append((cosine(qv, c.vec), m.name, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]
    cited = [(name, c, round(s, 3)) for s, name, c in top]
    uncited = []
    for claim in (extra_claims or []):
        # naive RAG has no gate that could bind or reject an LLM-proposed claim:
        # if no retrieved chunk is byte-identical, it ships as an uncited assertion.
        if not any(c.text == claim for _n, c, _s in cited):
            uncited.append(claim)
    abstained = len(cited) == 0 and not uncited
    return {"abstained": abstained, "cited": cited, "uncited": uncited}


def main() -> int:
    emb = HashEmbedder(1024)
    reg = Registry.load(os.path.join(ROOT, "modules"), emb)
    router = Router(reg, emb)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json"),
              encoding="utf-8") as f:
        cases = json.load(f)

    inscope = cases["single"] + cases["crossdomain"]
    oos = cases["oos"] + cases["oos_keyword"]
    fab = cases["fabricated_claim"]

    # relevance floor used ONLY to label a naive citation as query-irrelevant in the
    # report (it is Consilium's own default query_relevance_floor). It does not gate
    # the baseline -- the baseline has no gate.
    REL = 0.05

    # ---------- IN-SCOPE: both must answer (proves the baseline is not crippled) ----------
    naive_inscope_answered = 0
    cons_inscope_answered = 0
    for c in inscope:
        n = naive_rag(c["query"], reg, emb)
        if not n["abstained"]:
            naive_inscope_answered += 1
        rr = router.route(c["query"])
        ans = compose(c["query"], rr, reg, emb)
        if not ans.abstained and ans.citations:
            cons_inscope_answered += 1
    naive_inscope_rate = naive_inscope_answered / len(inscope)
    cons_inscope_rate = cons_inscope_answered / len(inscope)

    # ---------- OUT-OF-SCOPE: naive confidently answers; Consilium must abstain ----------
    naive_oos_confident = 0     # naive emitted a cited answer to an OOS query
    naive_oos_irrelevant = 0    # ... and its top citation is not even relevant to the query
    cons_oos_abstained = 0
    print("--- OUT-OF-SCOPE cases (naive vs Consilium) ---")
    for c in oos:
        n = naive_rag(c["query"], reg, emb)
        emitted = (not n["abstained"]) and bool(n["cited"])
        top_rel = n["cited"][0][2] if n["cited"] else 0.0
        if emitted:
            naive_oos_confident += 1
            if top_rel < REL:
                naive_oos_irrelevant += 1
        rr = router.route(c["query"])
        ans = compose(c["query"], rr, reg, emb)
        cons_abstain = ans.abstained and not ans.citations
        if cons_abstain:
            cons_oos_abstained += 1
        top = n["cited"][0] if n["cited"] else None
        cite_str = f"{top[0]}/{top[1].id} rel={top[2]}" if top else "-"
        print(f"  naive={'ANSWERS' if emitted else 'abstains':7} [{cite_str}]  "
              f"consilium={'ABSTAINS' if cons_abstain else 'ANSWERS'}  :: {c['query']}")
    naive_oos_answer_rate = naive_oos_confident / len(oos)
    cons_oos_abstain_rate = cons_oos_abstained / len(oos)

    # ---------- FABRICATED CLAIM: naive emits it uncited; Consilium drops it ----------
    n = naive_rag(fab["query"], reg, emb, extra_claims=[fab["injected"]])
    naive_emits_fabricated = fab["injected"] in n["uncited"]
    rr = router.route(fab["query"])
    ans = compose(fab["query"], rr, reg, emb, extra_claims=[fab["injected"]])
    cons_drops_fabricated = (fab["injected"] in ans.dropped) and all(
        ci.claim != fab["injected"] for ci in ans.citations)
    print("\n--- FABRICATED-CLAIM case ---")
    print(f"  naive emits injected uncited = {naive_emits_fabricated}   "
          f"consilium drops it = {cons_drops_fabricated}")

    # ---------- head-to-head 'unsafe outputs' tally ----------
    # unsafe = a confident OOS answer, OR an emitted fabricated claim.
    naive_unsafe = naive_oos_confident + (1 if naive_emits_fabricated else 0)
    cons_unsafe = (len(oos) - cons_oos_abstained) + (0 if cons_drops_fabricated else 1)
    unsafe_surface = len(oos) + 1
    hallucinations_eliminated = naive_unsafe - cons_unsafe

    print("\n=== A/B TABLE (measured, same cases, same corpus/embedder) ===")
    print(f"{'case-set':22} {'N':>3}  {'naive':>10}  {'consilium':>10}")
    print(f"{'in-scope (answer)':22} {len(inscope):>3}  "
          f"{naive_inscope_answered:>10}  {cons_inscope_answered:>10}")
    print(f"{'out-of-scope (answer)':22} {len(oos):>3}  "
          f"{naive_oos_confident:>10}  {len(oos) - cons_oos_abstained:>10}")
    print(f"{'  ..query-irrelevant':22} {len(oos):>3}  "
          f"{naive_oos_irrelevant:>10}  {'0':>10}")
    print(f"{'fabricated (emitted)':22} {1:>3}  "
          f"{int(naive_emits_fabricated):>10}  {int(not cons_drops_fabricated):>10}")
    print(f"{'UNSAFE outputs (total)':22} {unsafe_surface:>3}  "
          f"{naive_unsafe:>10}  {cons_unsafe:>10}")

    print("\n=== METRICS (measured) ===")
    print(f"naive_inscope_answer_rate  = {naive_inscope_rate:.3f}   (fairness: baseline not crippled)")
    print(f"cons_inscope_answer_rate   = {cons_inscope_rate:.3f}   (parity: no recall lost to the gates)")
    print(f"naive_oos_answer_rate      = {naive_oos_answer_rate:.3f}   (baseline hallucinates on OOS)")
    print(f"naive_oos_irrelevant       = {naive_oos_irrelevant}/{len(oos)}   (cited chunk not even query-relevant)")
    print(f"cons_oos_abstain_rate      = {cons_oos_abstain_rate:.3f}   (gates abstain instead)")
    print(f"naive_emits_fabricated     = {naive_emits_fabricated}")
    print(f"cons_drops_fabricated      = {cons_drops_fabricated}")
    print(f"UNSAFE  naive={naive_unsafe}  consilium={cons_unsafe}  of {unsafe_surface}   "
          f"-> hallucinations_eliminated = {hallucinations_eliminated}")

    # ---------- the measuring assertions (the GAP must be real, not asserted) ----------
    checks = {
        # fairness: the naive baseline answers in-scope queries just like Consilium.
        "baseline_answers_inscope": naive_inscope_rate == 1.0,
        "consilium_parity_inscope": cons_inscope_rate == 1.0,
        # the gap: naive is genuinely unsafe on the OOS + fabricated surface.
        "naive_answers_all_oos": naive_oos_answer_rate == 1.0,
        "naive_emits_fabricated": naive_emits_fabricated,
        # consilium closes that surface.
        "consilium_abstains_all_oos": cons_oos_abstain_rate == 1.0,
        "consilium_drops_fabricated": cons_drops_fabricated,
        # the measured head-to-head margin is strictly positive.
        "measured_gap_positive": hallucinations_eliminated == unsafe_surface and cons_unsafe == 0,
    }
    print()
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")
    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
