"""Run all five Consilium evals and print one aggregate PASS/FAIL. Exit 0 iff all pass.

    python eval/run_all.py

Imports and calls each eval script's own ``main()`` in sequence (v1, baseline A/B,
v2 integrity hardening, v3 heterogeneous modules, and the Phase-C core interfaces),
so there is one command a reviewer needs to run instead of five.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval as eval_v1       # noqa: E402
import eval_baseline         # noqa: E402
import eval_v2               # noqa: E402
import eval_v3               # noqa: E402
import eval_core             # noqa: E402

SCRIPTS = [
    ("eval.py (v1)", eval_v1),
    ("eval_baseline.py (A/B vs naive RAG)", eval_baseline),
    ("eval_v2.py (integrity hardening)", eval_v2),
    ("eval_v3.py (heterogeneous modules)", eval_v3),
    ("eval_core.py (Phase-C core interfaces)", eval_core),
]


def main() -> int:
    results = {}
    for label, mod in SCRIPTS:
        print("=" * 78)
        print(f"RUNNING: {label}")
        print("=" * 78)
        results[label] = (mod.main() == 0)
        print()

    print("=" * 78)
    print("AGGREGATE RESULT")
    print("=" * 78)
    for label, ok in results.items():
        print(f"{'OK  ' if ok else 'FAIL'} {label}")
    passed = all(results.values())
    print("\nOVERALL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
