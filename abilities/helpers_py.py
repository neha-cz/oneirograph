"""Shared numerics for Jac helpers (cosine, skeleton embedding, mean). No numpy."""

from __future__ import annotations

import math

TYPE_VOCAB = [
    "threshold",
    "unknown_presence",
    "enclosed_space",
    "authority",
    "transformation",
    "obstacle",
    "guide",
    "void",
    "self_variant",
    "threshold_guardian",
]


def _as_floats(seq: list) -> list[float]:
    return [float(x) for x in seq]


def _dot(u: list[float], v: list[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def _norm(u: list[float]) -> float:
    return math.sqrt(sum(x * x for x in u))


def cosine_similarity(a: list, b: list) -> float:
    u = _as_floats(list(a))
    v = _as_floats(list(b))
    if not u or not v:
        return 0.0
    n = min(len(u), len(v))
    u, v = u[:n], v[:n]
    denom = _norm(u) * _norm(v)
    if denom == 0.0:
        return 0.0
    return float(_dot(u, v) / denom)


def compute_skeleton_embedding(node_types: list, edges: list) -> list:
    freq = [float(node_types.count(t)) for t in TYPE_VOCAB]
    n = max(len(node_types), 1)
    edge_density = float(len(edges)) / float(n**2)
    return freq + [edge_density]


def mean(values: list) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def top_by_similarity(recurring: list, k: int = 5) -> list:
    return sorted(recurring, key=lambda x: float(x["similarity"]), reverse=True)[:k]
