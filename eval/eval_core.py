"""Consilium CORE eval — Phase C additive interfaces (capability.py, generalized
ComputeModule, Registry.load "kind" dispatch, answer_v3 fan-out). Exit 0 on pass.

    python eval/eval_core.py

Three red-cases (the generalized compute-module interface -- capability protocol,
Registry.load "kind" dispatch, answer_v3 fan-out):
  (a) back-compat: a query that anchors exactly one compute module (no text module)
      still returns the ORIGINAL v3 shape ({"kind":"compute","module":...,"audited":...})
      -- eval_v3.py:30-32 must keep passing.
  (b) mixed fan-out: a query that anchors a compute module AND a retrieval module
      returns kind=="mixed" with BOTH computed[] and a retrieval answer populated.
  (c) generic disk registration: a throwaway kind:"compute" module (descriptor.json
      + an "adapter" class, no consilium/sibling-repo coupling) loads through
      Registry.load(..., allow_compute_adapters=True) -- the explicit opt-in a
      caller must give before Registry.load will import an adapter class.

Deterministic (pure-python hashing embedder; no randomness, no network).
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)

from consilium.capability import Capability, ComputeCapability            # noqa: E402,F401 (item 1: must exist + import cleanly)
from consilium.embed import HashEmbedder                                  # noqa: E402
from consilium.registry import Registry                                   # noqa: E402
from consilium.router import Router                                       # noqa: E402
from consilium.compute import ComputeModule, make_finance_module, answer_v3  # noqa: E402


FIN_Q = ("Price a European call option with spot 100, strike 100, expiry 1 year, "
         "risk-free rate 0.05, volatility 0.2")

MIXED_Q = ("What is the black-scholes valuation for a call option with spot 100 "
           "strike 100 expiry 1 year risk-free rate 0.05 volatility 0.2, and what "
           "were Acme quarterly earnings and EPS?")


def check_backcompat_single_compute(reg, emb, router):
    """(a) single-compute, no-text query -> the ORIGINAL v3 shape (eval_v3.py:30-32)."""
    a = answer_v3(FIN_Q, reg, emb, router)
    ok = (
        a.get("kind") == "compute"
        and a.get("module") == "quant"
        and isinstance(a.get("audited"), dict)
        and a["audited"].get("ok") is True
        and abs(a["audited"]["result"] - 10.4506) < 1e-3
        and "computed" not in a          # proves the OLD shape, not the new general one
    )
    return ok, {"kind": a.get("kind"), "module": a.get("module"), "keys": sorted(a.keys())}


def check_mixed_fanout(reg, emb, router):
    """(b) a query anchoring quant (compute) AND markets (retrieval) -> kind=="mixed"
    with BOTH computed[] and a real retrieval answer populated."""
    a = answer_v3(MIXED_Q, reg, emb, router)
    computed = a.get("computed") or []
    ok = (
        a.get("kind") == "mixed"
        and len(computed) >= 1
        and any(c["module"] == "quant" and c["audited"].get("ok") for c in computed)
        and bool(a.get("answer"))
        and a.get("citations", 0) > 0
        and "markets" in (a.get("module") or [])
    )
    return ok, {"kind": a.get("kind"), "computed_modules": [c["module"] for c in computed],
                "answer_module": a.get("module"), "citations": a.get("citations")}


def check_generic_disk_registration(emb):
    """(c) a throwaway kind:"compute" module (descriptor.json + adapter class) loads
    through Registry.load via the GENERIC mechanism -- no sibling-repo coupling."""
    with tempfile.TemporaryDirectory() as tmp:
        modules_dir = os.path.join(tmp, "modules")
        mod_dir = os.path.join(modules_dir, "fixture_compute")
        os.makedirs(mod_dir)
        with open(os.path.join(mod_dir, "descriptor.json"), "w", encoding="utf-8") as f:
            json.dump({
                "name": "fixture_compute",
                "subjects": ["fixture compute widget", "throwaway generic adapter test"],
                "example_queries": ["run the fixture compute widget"],
                "authority": "eval_core throwaway fixture",
                "freshness": "n/a",
                "trust_tier": 0.5,
                "kind": "compute",
                "adapter": "_eval_core_fixture_adapter:FixtureAdapter",
            }, f)
        adapter_src = (
            "from consilium.compute import ComputeModule\n\n"
            "class FixtureAdapter(ComputeModule):\n"
            "    def __init__(self, embedder, descriptor):\n"
            "        super().__init__(name=descriptor.name, descriptor=descriptor, embedder=embedder)\n\n"
            "    def compute(self, query):\n"
            "        return {'ok': True, 'tool': 'fixture_echo', 'inputs': {'query': query},\n"
            "                'method': 'throwaway eval_core fixture', 'deterministic': True,\n"
            "                'result': len(query)}\n"
        )
        with open(os.path.join(tmp, "_eval_core_fixture_adapter.py"), "w", encoding="utf-8") as f:
            f.write(adapter_src)

        sys.path.insert(0, tmp)
        sys.modules.pop("_eval_core_fixture_adapter", None)
        try:
            reg = Registry.load(modules_dir, emb, allow_compute_adapters=True)
            mod = reg.by_name("fixture_compute")
            ok = (
                mod is not None
                and isinstance(mod, ComputeModule)
                and mod.chunks == []
                and mod.retrieve([0.0] * emb.dim) == []
                and isinstance(mod.centroid(), list) and len(mod.centroid()) == emb.dim
                and mod.compute("hello world") == {
                    "ok": True, "tool": "fixture_echo", "inputs": {"query": "hello world"},
                    "method": "throwaway eval_core fixture", "deterministic": True, "result": 11,
                }
            )
            return ok, {"loaded": mod is not None,
                        "is_compute_module": isinstance(mod, ComputeModule) if mod else False}
        finally:
            sys.path.remove(tmp)
            sys.modules.pop("_eval_core_fixture_adapter", None)


def main() -> int:
    emb = HashEmbedder(1024)
    text_modules = Registry.load(os.path.join(ROOT, "modules"), emb).modules
    reg = Registry([*text_modules, make_finance_module(emb)])
    router = Router(reg, emb)

    checks = {}

    ok, info = check_backcompat_single_compute(reg, emb, router)
    checks["backcompat_single_compute_shape"] = ok
    print(f"[a] {'OK  ' if ok else 'MISS'} backcompat shape  :: {info}")

    ok, info = check_mixed_fanout(reg, emb, router)
    checks["mixed_fanout_compute_and_retrieval"] = ok
    print(f"[b] {'OK  ' if ok else 'MISS'} mixed fan-out     :: {info}")

    ok, info = check_generic_disk_registration(emb)
    checks["generic_disk_registration"] = ok
    print(f"[c] {'OK  ' if ok else 'MISS'} generic disk-load :: {info}")

    print("\n=== METRICS (measured) ===")
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")
    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
