# ARC Prize 2026 - Rule-Based ARC-AGI Solver
Solving the [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) competition using a rule-based transformation search engine.
## Approach
ARC-AGI tasks require inferring a transformation rule from 2-5 input/output examples, then applying it to a test input. Our solver:
1. **Generates candidate transforms** from training pairs (geometric, color, object-based)
2. **Scores each transform** against all training pairs (pixel accuracy)
3. **Only predicts** when a transform achieves perfect score (1.0) on training - otherwise falls back to input
4. **Key insight**: 100% precision when score = 1.0, vs ~5% when using partial matches
## Architecture
\\\
solver/
+-- search/
¦   +-- engine.py          # Main orchestrator - generates + scores all transforms
¦   +-- scorer.py          # Pixel accuracy scoring
¦   +-- chainer.py         # 2-transform chaining
+-- rules/
    +-- transformations.py  # Geometric: rotate, flip, scale, tile (50+ transforms)
    +-- object_solver.py    # Object-based: fill_interior, gravity
    +-- color_solver.py     # Color: recolor, multi-recolor
    +-- pattern_solver.py   # Pattern: symmetry completion
    +-- line_solver.py      # Line: complete diagonal/straight lines
    +-- rectangle_solver.py # Rectangle: complete corners, draw border
\\\
## Results
| Dataset | Tasks | Score |
|---------|-------|-------|
| Training (1000 tasks) | 1000 | ~5% |
| Kaggle hidden test | 240 | TBD |
**Key finding**: Transforms with perfect training score (1.0) are 100% reliable on training data.
## Transform Categories
- **Geometric**: rotate 90/180/270, flip horizontal/vertical/diagonal, scale 2x/3x
- **Tiling**: tile NxM, alternating flip patterns
- **Color**: dynamic recolor (learned from training pairs)
- **Object**: fill enclosed interior (learned fill color), gravity down/up/left/right
- **Pattern**: symmetry completion, line completion
## Setup
\\\powershell
git clone https://github.com/rangga-jakti/arc-prize-2026
cd arc-prize-2026
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
\\\
Download competition data from [Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data) into \data/\.
## Usage
\\\powershell
# Test on training data
python test_full.py
# Generate submission for test set
python generate_submission_v3.py
\\\
## Lessons Learned
- Rule-based solvers plateau around 5% - ARC tasks are designed to require general intelligence
- High pixel accuracy (0.9+) does not mean the prediction is correct - only 1.0 is reliable
- Dynamic transforms (learned from training pairs) outperform static transforms
- The fill_interior pattern (enclosed region + learned fill color) is one of the most common ARC patterns
## Next Steps
- [ ] LLM integration (Ollama local / OpenRouter)
- [ ] Program synthesis via DSL search
- [ ] Test-time training
## Competition
- Platform: [Kaggle ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
- Prize pool: \,000
