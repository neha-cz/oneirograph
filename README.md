# oneirograph

A dream journaling app that maps the recurring symbolic structures in your dream life using quantum graph algorithms.

You log dreams. Over time, Oneirograph builds a personal symbolic map — not through interpretation, but through structural analysis. It finds the patterns underneath your dreams that you can't see yourself.

---

## How it works

Every dream you log passes through three stages.

**1. Graph construction**
Your dream report (voice or text) is parsed by an LLM into a symbolic graph. Symbols, locations, emotions, and figures become nodes. Relationships and transitions become edges. Each node carries a feature vector encoding emotional charge, scale, familiarity, threat level, and other dimensions. The LLM abstracts upward — "a door I couldn't open" becomes a `threshold` node, not a door node.

**2. Skeleton extraction**
A photonic QUBO optimizer (running on Quandela's Perceval simulator via Fock state sampling) finds the minimum-energy subgraph of that dream — the irreducible structural core. Remove any node from this skeleton and the dream loses coherence. This is the load-bearing symbolic structure underneath the surface content.

A post-variational quantum neural network classifies each node into a functional symbolic type (threshold, unknown presence, enclosed space, void, guide, etc.). The QNN handles the sparse, fuzzy feature distributions of symbolic data better than classical alternatives at small sample sizes.

**3. Pattern learning across time**
Each skeleton becomes a node in a meta-graph connecting all your dreams. A TEBD deep Q-learning walker traverses this meta-graph. Its policy is a Matrix Product State — a tensor network representation that efficiently captures correlations across your dream history without exponential blowup. Over time it learns your personal symbolic grammar: which skeleton types recur, which sequences precede phase transitions, which dreams are structurally anomalous.

---

## What it surfaces

- **Recurring skeletons** — the same structural pattern appearing across dreams with completely different surface content
- **Phase transitions** — moments where your symbolic grammar reorganized
- **Anomalies** — dreams that don't fit your established patterns, flagged for attention

---

## Stack

- **[Jac](https://www.jac-lang.org/)** — graph runtime, walkers, persistence, API server, React frontend
- **[Perceval](https://perceval.quandela.net/)** — photonic QUBO via Fock state sampling
- **[PennyLane](https://pennylane.ai/)** — post-variational quantum neural network
- **[TensorNetwork](https://github.com/google/TensorNetwork)** — TEBD deep Q-learning
- **[React Flow](https://reactflow.dev/)** — graph visualization

---

## Running

```bash
pip install jaseci pennylane tensornetwork perceval-quandela openai anthropic

export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

jac start main.jac
```
