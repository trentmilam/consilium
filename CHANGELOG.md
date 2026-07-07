# Changelog

## v1.0.0 — initial release

Initial public release. Built up incrementally during development (milestone
labels below are historical build order, not separate shipped versions — the
released artifact is a single v1.0.0):

- **Core switchboard** — router + integrity gate + composer: subject-routed,
  citation-bound answers, with abstention when nothing supports the query.
- **Integrity hardening** — cross-corpus conflict detection with trust-tier
  resolution, and corroboration-based poison quarantine.
- **Heterogeneous routing** — the router selects across retrieval AND compute
  modules; a Black-Scholes finance module ships as the reference compute
  capability (`python run_demo.py --with-compute --query "..."`).
- **Generic compute adapters** — a structural adapter contract so third-party
  compute modules can be registered from disk (`kind: "compute"` descriptors,
  opt-in via `allow_compute_adapters=True`).
