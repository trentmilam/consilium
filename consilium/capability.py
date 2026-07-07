"""Structural typing for what the router (router.py) and composer (composer.py)
actually touch on a module: name / descriptor / chunks / centroid() / retrieve().
A ``ComputeCapability`` adds ``compute()``.

``typing.Protocol`` only -- no runtime coupling (no ``@runtime_checkable``, no
``isinstance`` dispatch against these Protocols anywhere in consilium; dispatch
uses the concrete ``ComputeModule`` base instead, see compute.py). This is a
documentation-grade contract, not a base class: the existing ``Module``
(module.py) and ``ComputeModule`` (compute.py) already satisfy it without any
changes, and it never imports or names a sibling capability repo.
"""
from __future__ import annotations

from typing import Protocol

from .module import Descriptor


class Capability(Protocol):
    name: str
    descriptor: Descriptor            # name/subjects/example_queries/authority/freshness/trust_tier
    chunks: list                      # [] for compute

    def centroid(self) -> list[float]: ...

    def retrieve(self, query_vec, k: int = 3) -> list[tuple]: ...   # [] for compute


class ComputeCapability(Capability, Protocol):
    def compute(self, query: str) -> dict: ...
