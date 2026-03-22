# oneirograph 🪬🌙

A dream journaling app that maps the recurring symbolic structures in your dream life using quantum graph algorithms. 

Users log dreams every night. Over time, Oneirograph builds a personal symbolic map through structural analysis, finding underlying patterns underneath your dreams that you can't see yourself.

---

## How it works

Every dream you log passes through three stages.

**1. Graph construction**
Your dream report is parsed by an LLM into a symbolic graph. Symbols, locations, emotions, and figures become nodes. Relationships and transitions become edges. Each node carries a feature vector encoding emotional charge, scale, familiarity, threat level, and other dimensions. 

**2. Skeleton extraction**
A photonic quadratic unconstrained binary optimizer (QUBO) finds the minimum-energy subgraph of that dream — the irreducible structural core. It does so running on Quandela's Perceval simulator via Fock state sampling. This is the load-bearing symbolic structure underneath the surface content.

A post-variational quantum neural network then classifies each node into a functional symbolic type (threshold, unknown presence, enclosed space, void, guide, etc.). The QNN handles the sparse, fuzzy feature distributions of symbolic data better than classical alternatives at small sample sizes.

**3. Pattern learning across time**
Each skeleton becomes a node in a meta-graph connecting all your dreams. A TEBD-inspired deep Q-learning walker traverses this meta-graph. Its policy is a Matrix Product State — a tensor network representation that efficiently captures correlations across your dream history without exponential blowup. Over time it learns your personal symbolic grammar: which skeleton types recur, which sequences precede phase transitions, which dreams are structurally anomalous.

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

