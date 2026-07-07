"""Consilium v2 integrity-hardening eval — adversarial cases, exits 0 on thresholds.

    python eval/eval_v2.py

Deterministic. Builds small in-memory modules with injected conflicts / poison,
then asserts detection, resolution, quarantine, flooding-resistance, and a set of
negative controls (the detector must DISCRIMINATE, not flag everything).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from consilium.embed import HashEmbedder                                  # noqa: E402
from consilium.module import Module, Descriptor, Chunk                    # noqa: E402
from consilium.integrity import Citation                                  # noqa: E402
from consilium.hardening import detect_conflicts, quarantine_poison       # noqa: E402

EMB = HashEmbedder(1024)

LEGIT  = "Acme total revenue was $4.2 billion in fiscal 2025."
CORROB = "Acme total revenue for fiscal 2025 was $4.2 billion, confirmed in the annual report."
POISON = "Acme total revenue was $99 billion in fiscal 2025."
MILLION = "Acme total revenue was $4.2 million in fiscal 2025."
EQUIV  = "Acme total revenue was $4,200 million in fiscal 2025."
YEARVAR = "Acme total revenue was $4.2 billion, up from the prior 2024 fiscal year."
UNRELATED = "Full-time employees accrue twenty (20) days of paid time off per year."


def mk_module(name, trust, docs):
    chunks = []
    for cid, text in docs:
        ch = Chunk(id=cid, doc="d", text=text)
        ch.vec = EMB.embed_one(text)
        chunks.append(ch)
    desc = Descriptor(name=name, subjects=[name], example_queries=[], trust_tier=trust)
    return Module(name=name, descriptor=desc, chunks=chunks, embedder=EMB)


def cite(module_name, chunk):
    return Citation(claim=chunk.text, module=module_name, doc=chunk.doc, chunk_id=chunk.id, score=1.0)


def conflicts_between(name_a, trust_a, text_a, name_b, trust_b, text_b):
    ca = cite(name_a, mk_module(name_a, trust_a, [("x", text_a)]).chunks[0])
    cb = cite(name_b, mk_module(name_b, trust_b, [("y", text_b)]).chunks[0])
    trust = {name_a: trust_a, name_b: trust_b}
    return detect_conflicts([ca, cb], EMB, trust_of=lambda c: trust[c.module])


def main() -> int:
    checks = {}

    # 1) cross-corpus conflict + trust resolution + loser surfaced
    cx = conflicts_between("markets", 0.8, LEGIT, "altsource", 0.4, POISON)
    checks["cross_conflict_detected"] = len(cx) == 1
    checks["resolved_to_higher_trust"] = bool(cx) and cx[0].winner.module == "markets"
    checks["loser_surfaced"] = bool(cx) and any(l.module == "altsource" for l in cx[0].losers)
    print(f"[cross]  conflicts={len(cx)} winner={cx[0].winner.module if cx else None}")

    # 2) magnitude conflict ($4.2 billion vs $4.2 million) must be caught
    mag = conflicts_between("markets", 0.8, LEGIT, "altmag", 0.4, MILLION)
    checks["magnitude_conflict_detected"] = len(mag) == 1
    print(f"[mag]    billion-vs-million conflicts={len(mag)}")

    # 3) unit-equivalence must NOT be a conflict ($4.2 billion == $4,200 million)
    checks["no_false_on_unit_equiv"] = len(conflicts_between("markets", 0.8, LEGIT, "equiv", 0.5, EQUIV)) == 0
    # 4) incidental year must NOT be a conflict
    checks["no_false_on_incidental_year"] = len(conflicts_between("markets", 0.8, LEGIT, "yv", 0.5, YEARVAR)) == 0
    # 5) agreement must NOT be a conflict
    checks["no_false_on_agreement"] = len(conflicts_between("markets", 0.8, LEGIT, "m2", 0.5, CORROB)) == 0
    # 6) different topic must NOT be a conflict
    checks["no_false_on_diff_topic"] = len(conflicts_between("markets", 0.8, LEGIT, "hr", 0.5, UNRELATED)) == 0
    print(f"[neg]    equiv/year/agree/difftopic all-zero="
          f"{all(checks[k] for k in ['no_false_on_unit_equiv','no_false_on_incidental_year','no_false_on_agreement','no_false_on_diff_topic'])}")

    # 7) intra-corpus poison quarantine (single outlier)
    m = mk_module("markets", 0.8, [("m0", LEGIT), ("m1", CORROB), ("p0", POISON)])
    kept, quar, flagged = quarantine_poison(m, [(c, 1.0) for c in m.chunks], EMB)
    kid, qid = {c.id for c in kept}, {c.id for c in quar}
    checks["poison_quarantined"] = "p0" in qid
    checks["legit_kept"] = {"m0", "m1"} <= kid
    print(f"[intra]  kept={sorted(kid)} quarantined={sorted(qid)} flagged={sorted(c.id for c in flagged)}")

    # 8) poison-by-flooding: 3 exact poison copies must NOT invert the defense
    mf = mk_module("markets", 0.8, [("m0", LEGIT), ("m1", CORROB), ("p0", POISON), ("p1", POISON), ("p2", POISON)])
    keptf, quarf, flaggedf = quarantine_poison(mf, [(c, 1.0) for c in mf.chunks], EMB)
    kidf, qidf = {c.id for c in keptf}, {c.id for c in quarf}
    checks["flood_resisted"] = {"p0", "p1", "p2"} <= qidf and {"m0", "m1"} <= kidf
    print(f"[flood]  kept={sorted(kidf)} quarantined={sorted(qidf)} flagged={sorted(c.id for c in flaggedf)}")

    print("\n=== METRICS (measured) ===")
    for k, v in checks.items():
        print(f"{k:30} = {v}")
    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
