# ARC Prize 2026 — Experiment Log

## Experiment 001 — Rule-Based Solver v1
**Date:** 2026-05-07
**Training score:** 3.07% (33/1076 perfect)
**Kaggle submission:** v1-rule-based (scoring...)

### Transforms Implemented (49 total)
- Geometric: rotate 90/180/270, flip H/V/diagonal
- Tiling: NxM, alternating flip variants  
- Color: single recolor, multi recolor, geom+color
- Object: fill interior, fill enclosed
- Pattern: symmetry completion, outline, crop
- Gravity: up/down/left/right
- Rectangle: complete corners

### Next Steps After Score
- Implement line extrapolation
- Implement object copy/move
- Add ML scoring layer