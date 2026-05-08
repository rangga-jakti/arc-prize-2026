# ARC Prize 2026 — Symbolic ARC Solver

A research-oriented ARC-AGI solver combining:

- Symbolic DSL program synthesis
- Beam search
- Heuristic-guided reasoning
- Object-centric transformations
- Dynamic pattern mining

Built for the ARC Prize 2026 competition.

---

# Features

## Symbolic DSL

Custom domain-specific language (DSL) for ARC transformations:

- Rotations
- Flips
- Scaling
- Tiling
- Alternating tile patterns
- Object movement operations

Example:

```python
[
    "MOVE_RIGHT",
    "ROT90",
    "ROW_ALT_TILE3"
]
```

---

# Beam Search Program Synthesis

The solver searches over symbolic programs using:

- Heuristic-guided beam search
- Candidate ranking
- Program composition
- Transformation scoring

---

# Object-Centric Reasoning

The system detects connected objects and reasons about:

- Object size
- Bounding boxes
- Spatial relationships
- Relative positioning
- Object movement

---

# Solver Architecture

```text
ARC Task
   ↓
Primitive Miner
   ↓
Pattern Analysis
   ↓
Heuristic Generator
   ↓
Beam Search
   ↓
DSL Program Search
   ↓
Prediction
```

---

# Current Capabilities

- Geometric transformations
- Dynamic tiling detection
- Alternating row pattern detection
- Object movement reasoning
- Symbolic program synthesis

---

# Example Result

Program discovered automatically:

```python
["ROW_ALT_TILE3"]
```

Output score:

```text
100% exact match
```

---

# Benchmark

| System | Solved Tasks |
|---|---|
| Rule-based baseline | 18 |
| Beam Search DSL Solver | In Progress |
| Object DSL Solver | In Progress |

---

# Tech Stack

- Python
- NumPy
- Symbolic AI
- Beam Search
- Object Detection
- ARC-AGI reasoning

---

# Future Work

- Object copy operations
- Compositional object planning
- Better heuristics
- Neural-guided search
- Memory-augmented reasoning
- Hybrid LLM + symbolic solving

---

# Competition

ARC Prize 2026 (ARC-AGI-2)

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

---

# Author

Rangga Jakti