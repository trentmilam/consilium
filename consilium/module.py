"""The Module abstraction: {corpus, retriever, descriptor}.

A Module is one subject-specialized knowledge unit. Its **descriptor** is the
public face the router reasons over without reading the corpus. A Module loads
from a directory:

    <module_dir>/
      descriptor.json      # name, subjects, example_queries, authority, freshness, trust_tier
      corpus/*.md|*.txt    # documents, split into paragraph chunks
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

from .embed import cosine


@dataclass
class Chunk:
    id: str
    doc: str
    text: str
    vec: list = field(default_factory=list)


@dataclass
class Descriptor:
    name: str
    subjects: list
    example_queries: list
    authority: str = ""
    freshness: str = ""
    trust_tier: float = 0.5

    def profile_text(self) -> str:
        """The text the router embeds to represent this module's subject."""
        return " ".join([self.name, *self.subjects, *self.example_queries])


def _split_paragraphs(text: str) -> list[str]:
    """Split into paragraph chunks, folding a lone markdown heading into the
    body paragraph that follows it (so a bare '# Title' is never its own chunk /
    citation, while its keywords stay available for retrieval)."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    out: list[str] = []
    pending = ""
    for p in parts:
        is_heading = p.startswith("#") and "\n" not in p
        if is_heading:
            pending = p.lstrip("#").strip()
            continue
        if pending:
            p = f"{pending}. {p}"
            pending = ""
        out.append(p)
    if pending:
        out.append(pending)
    return out


@dataclass
class Module:
    name: str
    descriptor: Descriptor
    chunks: list
    embedder: object
    _centroid: list = field(default=None, repr=False)

    @classmethod
    def from_dir(cls, path: str, embedder) -> "Module":
        with open(os.path.join(path, "descriptor.json"), encoding="utf-8") as f:
            d = json.load(f)
        if "name" not in d:
            raise ValueError(f"descriptor.json in {path} is missing required field 'name'")
        descriptor = Descriptor(
            name=d["name"],
            subjects=list(d.get("subjects", [])),
            example_queries=list(d.get("example_queries", [])),
            authority=d.get("authority", ""),
            freshness=d.get("freshness", ""),
            trust_tier=float(d.get("trust_tier", 0.5)),
        )
        chunks: list = []
        for fp in sorted(glob.glob(os.path.join(path, "corpus", "*"))):
            if not os.path.isfile(fp):
                continue
            doc = os.path.basename(fp)
            with open(fp, encoding="utf-8") as f:
                text = f.read()
            for i, para in enumerate(_split_paragraphs(text)):
                chunks.append(Chunk(id=f"{doc}#{i}", doc=doc, text=para))
        vecs = embedder.embed([c.text for c in chunks])
        for c, v in zip(chunks, vecs):
            c.vec = v
        return cls(name=descriptor.name, descriptor=descriptor, chunks=chunks, embedder=embedder)

    def centroid(self) -> list:
        if self._centroid is None:
            self._centroid = self.embedder.embed_one(self.descriptor.profile_text())
        return self._centroid

    def retrieve(self, query_vec, k: int = 3):
        scored = sorted(
            ((cosine(query_vec, c.vec), c) for c in self.chunks),
            key=lambda t: t[0],
            reverse=True,
        )
        return [(c, s) for s, c in scored[:k]]
