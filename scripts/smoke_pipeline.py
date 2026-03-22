#!/usr/bin/env python3
"""End-to-end Python check: parse → classify → QUBO → embedding (no Jac runtime)."""

from __future__ import annotations

import os
import sys

# Project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from abilities.helpers_py import compute_skeleton_embedding  # noqa: E402
from backends.perceval_backend import run_qubo  # noqa: E402
from backends.qnn_backend import classify_node  # noqa: E402
from backends.tebd_backend import global_tebd_policy  # noqa: E402
from parsers.dream_parser import parse_dream  # noqa: E402


def _edge_indices(nodes: list, edges: list) -> list[list[int]]:
    id_to_i = {n["id"]: i for i, n in enumerate(nodes)}
    out: list[list[int]] = []
    for e in edges:
        a = id_to_i.get(e["from"])
        b = id_to_i.get(e["to"])
        if a is not None and b is not None:
            out.append([a, b])
    return out


def main() -> None:
    os.environ.pop("OPENAI_API_KEY", None)
    text = "I was trapped in a house. A faceless figure stood by the door I could not open."
    parsed = parse_dream(text)
    assert "nodes" in parsed and len(parsed["nodes"]) >= 1

    types = []
    for n in parsed["nodes"]:
        r = classify_node(n["features"])
        types.append(r["type"])
        assert r["confidence"] >= 0.0

    edges = _edge_indices(parsed["nodes"], parsed.get("edges", []))
    feats = [n["features"] for n in parsed["nodes"]]
    skel = run_qubo(feats, edges)
    emb = compute_skeleton_embedding([types[i] for i in skel], [[e[0], e[1]] for e in edges if e[0] in skel and e[1] in skel])
    assert len(emb) == 11

    traj = [
        {"embedding": emb, "node_types": [types[i] for i in skel], "is_anomaly": False},
        {"embedding": emb, "node_types": [types[i] for i in skel], "is_anomaly": False},
    ]
    global_tebd_policy.tebd_update(traj, 0.5)
    print("smoke_pipeline_ok", {"n_nodes": len(parsed["nodes"]), "skeleton": skel, "types": types})


if __name__ == "__main__":
    main()
