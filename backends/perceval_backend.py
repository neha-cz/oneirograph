"""
QUBO skeleton extraction (classical; no Perceval in this build).
"""

from __future__ import annotations

import itertools
import math


def build_qubo_matrix(node_features: list, edges: list) -> list[list[float]]:
    n = len(node_features)
    Q = [[0.0] * n for _ in range(n)]

    for i, feat in enumerate(node_features):
        feat_list = list(feat) if feat is not None else []
        valence = abs(float(feat_list[0])) if len(feat_list) > 0 else 0.0
        extra = float(feat_list[6]) if len(feat_list) > 6 else 0.5
        importance = valence * 0.5 + extra * 0.5
        Q[i][i] = -importance

    for pair in edges:
        if len(pair) != 2:
            continue
        i, j = int(pair[0]), int(pair[1])
        if 0 <= i < n and 0 <= j < n and i != j:
            Q[i][j] -= 0.5
            Q[j][i] -= 0.5

    return Q


def _energy(Q: list[list[float]], config: list[int]) -> float:
    n = len(config)
    e = 0.0
    for i in range(n):
        for j in range(n):
            e += Q[i][j] * config[i] * config[j]
    return e


def run_qubo(node_features: list, edges: list, n_samples: int = 1000) -> list:
    del n_samples
    n = len(node_features)
    if n == 0:
        return []
    if n == 1:
        return [0]

    Q = build_qubo_matrix(node_features, edges)

    if n <= 14:
        best_cfg = None
        best_e = math.inf
        for bits in itertools.product([0, 1], repeat=n):
            if sum(bits) == 0:
                continue
            en = _energy(Q, list(bits))
            if en < best_e:
                best_e = en
                best_cfg = list(bits)
        if best_cfg is None:
            return list(range(n))
        return [i for i, v in enumerate(best_cfg) if v == 1]

    keep = [1] * n
    improved = True
    while improved:
        improved = False
        base_e = _energy(Q, keep)
        for i in range(n):
            if keep[i] == 0:
                continue
            trial = keep.copy()
            trial[i] = 0
            if sum(trial) == 0:
                continue
            if _energy(Q, trial) <= base_e:
                keep = trial
                improved = True
                break
    return [i for i, v in enumerate(keep) if v == 1]
