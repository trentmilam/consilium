"""Consilium v3 eval — heterogeneous routing (retrieval + compute). Exit 0 on pass.

    python eval/eval_v3.py

A finance-math query must route to the COMPUTE module and return the exact audited
value; a text query must still route to a retrieval module (no v1/v2 regression);
an out-of-scope query must abstain.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from consilium.embed import HashEmbedder                    # noqa: E402
from consilium.registry import Registry                     # noqa: E402
from consilium.router import Router                         # noqa: E402
from consilium.compute import make_finance_module, answer_v3  # noqa: E402


def main() -> int:
    emb = HashEmbedder(1024)
    text_modules = Registry.load(os.path.join(ROOT, "modules"), emb).modules
    reg = Registry([*text_modules, make_finance_module(emb)])
    router = Router(reg, emb)
    checks = {}

    fin_q = "Price a European call option with spot 100, strike 100, expiry 1 year, risk-free rate 0.05, volatility 0.2"
    a = answer_v3(fin_q, reg, emb, router)
    checks["finance_routes_to_compute"] = a["kind"] == "compute" and a["module"] == "quant"
    checks["compute_price_correct"] = a["kind"] == "compute" and a["audited"].get("ok") and abs(a["audited"]["result"] - 10.4506) < 1e-3
    checks["compute_is_audited"] = a["kind"] == "compute" and a["audited"].get("deterministic") is True
    print(f"[finance] kind={a['kind']} module={a.get('module')} result={a.get('audited', {}).get('result')}")

    t = answer_v3("How many PTO days do full-time employees accrue per year?", reg, emb, router)
    checks["text_routes_to_retrieval"] = t["kind"] == "retrieval" and "handbook" in (t.get("module") or [])
    print(f"[text]    kind={t['kind']} module={t.get('module')}")

    o = answer_v3("What is the best recipe for sourdough bread?", reg, emb, router)
    checks["oos_abstains"] = o["kind"] == "abstain"
    print(f"[oos]     kind={o['kind']}")

    print("\n=== METRICS (measured) ===")
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")
    passed = all(checks.values())
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
