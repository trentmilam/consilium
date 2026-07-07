"""Registry: discover modules on disk and expose their descriptors.

SECURITY: a ``kind: "compute"`` descriptor's ``adapter`` field names a Python
module:class to dynamic-import and instantiate (see ``_load_compute`` below).
``modules_dir`` (and anything importable via the caller's own ``sys.path`` /
``PYTHONPATH``) must therefore be trusted exactly like any other Python import
path -- loading a descriptor from an untrusted directory is equivalent to
running arbitrary code. ``Registry.load`` refuses ``kind: "compute"``
descriptors unless the caller explicitly opts in via ``allow_compute_adapters=True``.
"""
from __future__ import annotations

import importlib
import json
import os

from .module import Descriptor, Module


class Registry:
    def __init__(self, modules: list) -> None:
        self.modules = modules

    @classmethod
    def load(cls, modules_dir: str, embedder, allow_compute_adapters: bool = False) -> "Registry":
        """Discover modules under ``modules_dir``. Retrieval modules (no ``kind``
        or ``kind`` != ``"compute"``) always load. A ``kind: "compute"`` descriptor
        dynamic-imports and instantiates an arbitrary adapter class (see the
        module-level SECURITY note) and is therefore skipped unless the caller
        passes ``allow_compute_adapters=True`` -- an explicit statement that
        ``modules_dir`` is trusted."""
        mods: list = []
        for name in sorted(os.listdir(modules_dir)):
            p = os.path.join(modules_dir, name)
            descriptor_path = os.path.join(p, "descriptor.json")
            if not (os.path.isdir(p) and os.path.exists(descriptor_path)):
                continue
            with open(descriptor_path, encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("kind") == "compute":
                if not allow_compute_adapters:
                    continue
                mods.append(cls._load_compute(raw, embedder))
            else:
                mods.append(Module.from_dir(p, embedder))
        return cls(mods)

    @staticmethod
    def _load_compute(raw: dict, embedder):
        """Generic ``kind: "compute"`` disk registration: build the Descriptor
        from descriptor.json exactly like a retrieval Module, then dynamic-import
        the ``adapter`` class (``"pkg.mod:Class"``) and instantiate it as
        ``adapter_cls(embedder, descriptor)``. No sibling-repo coupling -- the
        adapter module is whatever the caller's own PYTHONPATH resolves. Only
        reached when the caller has opted in via ``allow_compute_adapters=True``
        (see the module-level SECURITY note)."""
        if "name" not in raw:
            raise ValueError("compute descriptor.json is missing required field 'name'")
        descriptor = Descriptor(
            name=raw["name"],
            subjects=list(raw.get("subjects", [])),
            example_queries=list(raw.get("example_queries", [])),
            authority=raw.get("authority", ""),
            freshness=raw.get("freshness", ""),
            trust_tier=float(raw.get("trust_tier", 0.5)),
        )
        mod_path, cls_name = raw["adapter"].split(":")
        adapter_cls = getattr(importlib.import_module(mod_path), cls_name)
        return adapter_cls(embedder, descriptor)

    def descriptors(self) -> list:
        return [m.descriptor for m in self.modules]

    def by_name(self, name: str):
        for m in self.modules:
            if m.name == name:
                return m
        return None
