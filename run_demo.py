"""Consilium v1 demo: route sample queries across modules, show cited answers.

    python run_demo.py

    # also register the finance compute module (heterogeneous routing) so a
    # finance-math query routes to an audited computation instead of abstaining
    python run_demo.py --with-compute --query "Black-Scholes call price for spot 100 strike 100 expiry 1 vol 0.2 rate 0.05"

Deterministic and gateway-independent (pure-python hashing embedder).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from consilium.embed import HashEmbedder          # noqa: E402
from consilium.registry import Registry           # noqa: E402
from consilium.router import Router               # noqa: E402
from consilium.composer import compose            # noqa: E402
from consilium.compute import ComputeModule, make_finance_module  # noqa: E402


def show(router, reg, emb, q, extra=None):
    rr = router.route(q)
    computed = []
    for n in rr.selected:
        m = reg.by_name(n)
        if isinstance(m, ComputeModule):
            computed.append(m)
    print("=" * 78)
    print("Q:", q)
    top3 = ", ".join(f"{n}={s}" for n, s, _ in rr.ranked[:3])
    print("  routing scores:", top3)
    print("  selected      :", rr.selected if not rr.abstained else "(ABSTAINED - out of scope)")
    if computed:
        for cm in computed:
            aud = cm.compute(q)
            if aud.get("ok"):
                print(f"  answer (from {cm.name}, computed):")
                print(f"      {aud['method']}")
                print(f"      inputs={aud['inputs']}")
                print(f"      result={aud['result']}  (audited, deterministic={aud['deterministic']})")
            else:
                print(f"  answer (from {cm.name}, computed): could not parse query - {aud['error']}")
        return
    ans = compose(q, rr, reg, emb, extra_claims=extra)
    if not ans.abstained:
        print(f"  answer (from {', '.join(ans.modules_used)}):")
        for line in ans.text.splitlines():
            print("     ", line)
    else:
        print("  answer        :", ans.text)
    if ans.dropped:
        print("  DROPPED by integrity gate (unsupported):")
        for d in ans.dropped:
            print("      x", d)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="ask a single question instead of running the sample queries")
    parser.add_argument("--with-compute", action="store_true",
                         help="also register the finance compute module (heterogeneous routing) so a "
                              "finance-math query (e.g. Black-Scholes) routes to an audited computation")
    args = parser.parse_args()

    emb = HashEmbedder(1024)
    reg = Registry.load(os.path.join(ROOT, "modules"), emb)
    if args.with_compute:
        reg = Registry([*reg.modules, make_finance_module(emb)])
    router = Router(reg, emb)
    print(f"Consilium - loaded {len(reg.modules)} modules: {[m.name for m in reg.modules]}\n")

    if args.query is not None:
        show(router, reg, emb, args.query)
        return 0

    show(router, reg, emb, "What was Acme's total revenue and which segment grew fastest?")
    show(router, reg, emb, "What is the confidentiality term length in the mutual NDA?")
    show(router, reg, emb, "How many PTO days do full-time employees accrue per year?")
    show(router, reg, emb, "For a new hire, what is the expense reimbursement policy and what does EBITDA mean?")
    show(router, reg, emb, "What is the best recipe for sourdough bread?")
    print("\n--- integrity gate: an LLM-proposed claim with no source is dropped ---")
    show(router, reg, emb, "What was Acme's total revenue?",
         extra=["Acme Corp secretly plans to acquire Globex Industries for $9 billion next month."])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
