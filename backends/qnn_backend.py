"""
Dream-symbol classification (classical heuristic; no PennyLane in this build).
"""

from __future__ import annotations

SYMBOL_TYPES = [
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


def _scores(feat: list[float]) -> list[float]:
    v = [0.0] * len(SYMBOL_TYPES)
    valence = float(feat[0]) if len(feat) > 0 else 0.0
    scale = float(feat[1]) if len(feat) > 1 else 0.5
    agency = float(feat[2]) if len(feat) > 2 else 0.5
    familiarity = float(feat[3]) if len(feat) > 3 else 0.5
    threat = float(feat[4]) if len(feat) > 4 else 0.0
    transform = float(feat[5]) if len(feat) > 5 else 0.0
    centrality = float(feat[6]) if len(feat) > 6 else 0.5
    boundary = float(feat[7]) if len(feat) > 7 else 0.5

    v[0] = boundary * 1.2 + (1.0 - familiarity) * 0.3
    v[1] = (1.0 - familiarity) * 0.9 + threat * 0.5
    v[2] = scale * 0.9 + (1.0 - agency) * 0.2
    v[3] = familiarity * 0.6 + agency * 0.5
    v[4] = transform * 1.1
    v[5] = threat * 0.7 + (1.0 - agency) * 0.2
    v[6] = agency * 0.5 + (1.0 - threat) * 0.2
    v[7] = (1.0 - centrality) * 0.8 + abs(valence) * 0.2
    v[8] = familiarity * 0.5 + centrality * 0.5
    v[9] = boundary * 0.8 + threat * 0.6
    return v


def _pad_features(features: list) -> list[float]:
    raw = [float(x) for x in list(features)]
    if len(raw) == 0:
        return [0.5] * 8
    if len(raw) < 8:
        return raw + [0.5] * (8 - len(raw))
    return raw[:8]


def classify_node(features: list) -> dict:
    feat_arr = _pad_features(features)
    scores = _scores(feat_arr)
    idx = max(range(len(scores)), key=lambda i: scores[i])
    total = sum(scores) or 1.0
    probs = [s / total for s in scores]
    confidence = float(probs[idx])
    return {
        "type": SYMBOL_TYPES[idx],
        "confidence": confidence,
        "all_probs": {SYMBOL_TYPES[i]: float(probs[i]) for i in range(len(SYMBOL_TYPES))},
    }
