"""Consilium -- a multi-RAG switchboard ("private research desk").

An orchestrator routes a query across independent, subject-specialized modules
and returns a citation-bound answer, abstaining when nothing supports it.

Self-contained + deterministic + gateway-optional: a vendored pure-python
hashing embedder means no numpy, no network, no model download, and the demo /
eval reproduce byte-for-byte.
"""
from .embed import HashEmbedder, cosine
from .module import Chunk, Descriptor, Module
from .registry import Registry
from .router import Router, RouteResult
from .integrity import Citation, bind_claim, gate
from .composer import Answer, compose

__version__ = "1.0.0"

__all__ = [
    "HashEmbedder", "cosine",
    "Chunk", "Descriptor", "Module",
    "Registry", "Router", "RouteResult",
    "Citation", "bind_claim", "gate",
    "Answer", "compose",
    "__version__",
]
