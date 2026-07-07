# Consilium — a multi-RAG switchboard ("private research desk")

Ask your whole knowledge base in plain language and get a **sourced** answer that draws from the
**right subject(s)** and **never makes up a fact**. An orchestrator routes each query across
independent, subject-specialized knowledge **modules**, binds every emitted claim to a source span,
and **abstains** when nothing supports it.

Built for the person whose job is to research and decide — an analyst, a wealth manager, any
knowledge worker — and for the managers who run them. This is **v1 (portfolio tier)**: self-contained,
deterministic, runs fully offline.

## Quickstart

```bash
# demo: route sample queries, show cited answers, abstention, and the integrity gate
python run_demo.py

# ...or ask your own question
python run_demo.py --query "What is the confidentiality term length in the mutual NDA?"

# --with-compute also registers the finance compute module, so a finance-math
# query routes to an audited computation instead of a retrieval module
python run_demo.py --with-compute --query "Black-Scholes call price for spot 100 strike 100 expiry 1 vol 0.2 rate 0.05"

# verify everything: all 5 evals (routing, integrity gate, hardening, compute
# routing, and the naive-RAG head-to-head), one aggregate PASS/FAIL
python eval/run_all.py
```

No dependencies beyond the Python standard library — a vendored pure-python hashing embedder means
no numpy, no network, no model download, and byte-for-byte reproducible runs. Requires Python 3.9+.

## Measured head-to-head vs a naive no-gate RAG (from `eval/eval_baseline.py`)

The whole wedge is one claim: *Consilium abstains / drops instead of confidently
hallucinating.* A self-graded perfect score proves nothing, so `eval/eval_baseline.py` runs a
**fair naive incumbent RAG** on the **same cases, same corpus, same embedder** — literally
Consilium *minus its three gates* (the router's abstention-floor + anchor gate, the integrity
claim↔source binding, and the query-relevance floor) — and **measures** the difference. The
baseline is not a strawman: it answers **all 11 in-scope queries with real citations, at parity
with Consilium** — it only fails on the out-of-scope / fabricated surface.

| case-set (N) | naive baseline | Consilium |
|---|---|---|
| in-scope, answered with citations (11) | **11** | **11** |
| out-of-scope, confidently answered (4) | **4** ← hallucinates | **0** ← abstains |
| fabricated claim, emitted uncited (1) | **1** ← ships it | **0** ← dropped |
| **UNSAFE outputs (of 5)** | **5** | **0** |

**Measured gap:** the naive RAG emits **5/5** unsafe outputs (4 confident answers to
out-of-scope queries + 1 fabricated uncited claim) where Consilium emits **0/5**, at **zero
in-scope recall cost** (11/11 both). The gates eliminate the entire hallucination surface these
cases probe without narrowing what the system will legitimately answer. Numbers observed, not
asserted — re-run with `python eval/eval_baseline.py`.

## Measured results (Consilium alone, from `eval/eval.py`, observed — not asserted)

| metric | value | target |
|---|---|---|
| routing accuracy (single-domain query → correct module) | **1.000** (9/9) | ≥ 0.80 |
| fan-out (cross-domain query selects all relevant modules) | **1.000** (2/2) | == 1.00 |
| abstention (out-of-scope query refused, not answered) | **1.000** (2/2) | == 1.00 |
| citation coverage (every emitted claim bound to a source span) | **1.000** (11/11) | == 1.00 |
| fabricated-claim rejection (unsupported claim dropped by the gate) | **True** | True |

The eval exercises the gate honestly: an injected claim with no source
("Acme Corp secretly plans to acquire Globex…") is **dropped**, not emitted.

## How it works

Five units, each with one job (each understandable and testable in isolation):

- **Module** (`consilium/module.py`) = `{corpus, retriever, descriptor}`. The **descriptor**
  (`descriptor.json`: name, subjects, example queries, authority, freshness, trust_tier) is the
  module's public face — the router reasons over it without reading the corpus.
- **Registry** (`consilium/registry.py`) — discovers modules on disk, exposes descriptors.
- **Router / switchboard** (`consilium/router.py`) — scores each module from
  `0.45·descriptor-centroid + 0.20·best-chunk + 0.35·subject-overlap`; selects every module above an
  absolute floor (fan-out for cross-domain queries); **abstains** when none clear the floor.
- **Integrity gate** (`consilium/integrity.py`) — binds each claim to its best-supporting span; an
  unsupported claim is dropped and a wholly-unsupported answer abstains. *The moat.*
- **Composer** (`consilium/composer.py`) — assembles the surviving span-bound claims into one cited
  answer across modules.

**Flow:** `query → router (descriptors → module set | abstain) → per-module retrieve → integrity gate
→ composer → cited answer + routing/audit trace`.

## Modules in this demo (all synthetic / public-safe)

- **markets** — SEC-style filings + a market glossary (revenue, EPS, EBITDA).
- **legal** — NDA / MSA templates + a GDPR summary (confidentiality, IP, data rights).
- **handbook** — a synthetic firm handbook (PTO, expenses, security policy).

Add a module by dropping a folder under `modules/<name>/` with a `descriptor.json` and a `corpus/`.

**Security note:** a `descriptor.json` with `"kind": "compute"` names a Python `module:class` that
`Registry.load` will dynamic-import and instantiate (see `consilium/registry.py`). Treat `modules_dir`
exactly like any other Python import path — only point it at directories you trust. This path is
disabled by default; loading a compute descriptor requires the caller to pass
`Registry.load(modules_dir, embedder, allow_compute_adapters=True)` as an explicit opt-in. No shipped
module in this repo uses `kind: "compute"` — the finance compute module (below) is registered directly
in Python, not from disk.

## Honest scope (v1)

- **Synthetic, public-safe corpora** — no real proprietary data; safe to publish.
- **Gateway-independent** — deterministic hashing retrieval; an optional LLM-router / LLM-composer
  upgrade can activate when a local model gateway is reachable (v2).
- **Deferred to later v2 work:** a UI/console, private/on-prem deployment + access control, real
  corpora, optional LLM router/composer when a gateway is up.
- **Known open failure mode (documented, not hidden):** the query-relevance floor
  (`compose(..., query_relevance_floor=0.05)`) is deliberately low, so the router's **anchor
  gate** — not the relevance floor — is what stops the incidental-keyword out-of-scope cases
  (`eval/eval_baseline.py` measures the naive baseline's cited chunks at rel≈0.15–0.22, above
  that 0.05 floor). A jargon-heavy OOS query that name-drops ≥2 subject tokens without genuinely
  asking about them can still slip the floor; the anchor gate is the real backstop. Raising /
  calibrating that floor is queued.
- **Threshold provenance:** the router's floor/weights/anchor gates and the integrity gate's
  support floor (`consilium/router.py`, `consilium/composer.py`) are tuned against the shipped
  16-case fixture set (`eval/cases.json`), not validated against an independent, held-out query
  distribution — the "1.000" headline numbers may not generalize past this demo corpus.

## Integrity hardening

Cross-corpus conflict detection + trust resolution, and corroboration-based poison quarantine
(`consilium/hardening.py`, wired into `compose(..., harden=True)`).

```
python eval/eval_v2.py
```

Measured (eval_v2, exit 0 — 11/11 checks):

- cross-corpus conflict detected, **resolved to the higher-trust source**, loser surfaced (not hidden);
- **magnitude-aware**: `$4.2 billion` vs `$4.2 million` is flagged; `$4.2 billion` == `$4,200 million`
  is not; incidental years/percents are ignored (no false conflicts);
- intra-corpus **poison quarantined**, corroborated legit kept;
- **poison-by-flooding resisted**: three exact poison copies are all quarantined — near-duplicates
  collapse to one source, so copy-flooding cannot invert the defense;
- negative controls (agreement, different-topic) produce zero false conflicts.

Honest limitations (deterministic heuristic; closing these needs a future NLI / trust-provenance
layer, not covered by heterogeneous routing below):

- numeric magnitudes only — **word-spelled** numbers and **non-numeric** conflicts (e.g. "Delaware"
  vs "Nevada") are not detected;
- same-magnitude **different-metric** claims (revenue vs net income) can false-positive if textually
  similar (no attribute extraction yet);
- **paraphrase-flooding** (non-identical poison copies) is not defeated by dedup — a real security
  boundary needs per-source trust / provenance / signing;
- count-based corroboration is **advisory**, not a security boundary; ambiguous conflicts are
  surfaced (flagged), never silently resolved.

## Heterogeneous routing (retrieval + compute)

The switchboard routes across **retrieval AND compute** modules with one router. A `ComputeModule`
(`consilium/compute.py`) is routable like any text module (it has a descriptor the router scores) but,
instead of retrieving a passage, it PARSES the query and returns an **audited, deterministic
computation** (a no-arithmetic-by-LLM audited compute pattern). Invalid input (e.g. non-positive
spot/strike/expiry/volatility) returns the same `{ok: False, tool, error}` envelope as unparseable
input — it never fabricates a number or crashes. Explorable via the demo:
`python run_demo.py --with-compute --query "..."` (see Quickstart above).

```
python eval/eval_v3.py
```

Measured (eval_v3, exit 0 — 5/5): a finance-math query routes to the compute module and returns the
exact audited Black-Scholes value (10.450584); a policy question still routes to a retrieval module
(handbook); an out-of-scope query abstains. Retrieval and integrity-hardening evals remain green
(additive, no regression).

## Generic compute adapters

`consilium/capability.py` documents the structural contract the router/composer actually need
(`name`, `descriptor`, `chunks`, `centroid()`, `retrieve()`); `Registry.load` can register a
`kind: "compute"` module straight from disk by dynamic-importing an `adapter` class named in its
`descriptor.json` (see the security note above — this path is opt-in via `allow_compute_adapters=True`).

```
python eval/eval_core.py
```

Measured (eval_core, exit 0 — 3/3): (a) a query that anchors only the compute module still returns
the original heterogeneous-routing response shape (no regression); (b) a query that anchors both a
compute module and a retrieval module returns both an audited computation and a cited text answer; (c) a throwaway
disk-registered `kind: "compute"` module (descriptor + adapter class, no coupling to any module in
this repo) loads and runs through the generic `Registry.load` path.

## What's novel here

The mechanisms that separate this from a plain retrieval switchboard are the integrity gate's
claim-to-source binding (an unsupported claim is dropped, not shipped), cross-corpus conflict
detection with **trust-calibrated conflict resolution** (`consilium/hardening.py`: when two modules
disagree on the same topic, the higher-`trust_tier` source wins and the loser is surfaced, never
hidden), and the router's abstention/anchor gates that refuse an out-of-scope query instead of
guessing — the parts most worth hardening further.

## License

MIT — see [LICENSE](LICENSE).
