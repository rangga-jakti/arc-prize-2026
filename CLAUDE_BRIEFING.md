# CLAUDE BRIEFING - ARC Prize 2026
## PROJECT
- Repo: https://github.com/rangga-jakti/arc-prize-2026
- Competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
- Tools: PowerShell + Python 3.14 + venv
- Path: D:\Projects\arc-prize-2026
- Komunikasi: Bahasa Indonesia informal
## TUJUAN
1. Naik score Kaggle ARC-AGI-2 (sekarang 0.00%, target 5%+)
2. Bikin portfolio GitHub yang impressive
## SCORE SAAT INI
- Training (1000 tasks) : 3.07% (33/1076 perfect)
- Evaluation (120 tasks): 0.00%
- Kaggle hidden test    : 0.00%
## PHASE YANG SUDAH SELESAI
- Phase 0  : Setup project, venv, requirements
- Phase 1  : Data loader + visualizer
- Phase 2  : Taxonomy 49 transformasi
- Phase 3-8: Rule-based solver 50+ transforms
- Phase 9  : Prediction pipeline
- Phase 10 : Kaggle submission (submitted v1)
- Phase 14 : GitHub portfolio
## STRUKTUR PROJECT
D:\Projects\arc-prize-2026\
+-- data\training, evaluation, test
+-- src\config.py, utils\data_loader.py, visualization\grid_viz.py
+-- solver\
¦   +-- rules\transformations.py    (rotate, flip, tile, scale, recolor)
¦   +-- rules\object_detector.py    (connected components)
¦   +-- rules\object_solver.py      (fill interior, gravity)
¦   +-- rules\color_solver.py       (multi recolor, geom+color)
¦   +-- rules\pattern_solver.py     (symmetry, outline, crop)
¦   +-- rules\rectangle_solver.py   (complete corners)
¦   +-- rules\line_solver.py        (complete diagonal/straight)
¦   +-- rules\llm_solver.py         (LLM integration - Groq)
¦   +-- search\engine.py            (main orchestrator)
¦   +-- search\scorer.py            (scoring functions)
¦   +-- search\chainer.py           (2-transform chaining)
+-- experiments\failure_analysis.py, dataset_analysis.py
+-- outputs\submissions\submission.json
+-- test_full.py, test_solver.py, test_eval.py
+-- solve_with_llm.py
+-- generate_submission.py
## MASALAH SAAT INI
- Rule-based mentok di 3%, 0% di Kaggle
- LLM integration: Groq rate limit parah, Gemini quota habis
- Perlu LLM yang bisa jalan smooth untuk 240 test tasks
## NEXT STEPS
1. Coba Ollama (lokal, unlimited) atau OpenRouter ($1 free)
2. Build proper LLM pipeline untuk 240 test tasks
3. Submit v2 ke Kaggle
4. Target: 5%+ score
## CARA SETUP
```powershell
cd D:\Projects\arc-prize-2026
.\venv\Scripts\Activate.ps1
# PYTHONPATH sudah di-set permanent
python test_full.py  # eval training
python generate_submission.py  # buat submission
```
## CATATAN PENTING
- Kaggle competition tidak boleh internet access saat submit
- Jadi tidak bisa call external API di notebook
- Solusi: pre-compute answers di local, upload sebagai dataset
- Groq API key ada di solve_with_llm.py (jangan di-commit!)
