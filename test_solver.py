# test_solver.py
from src.utils.data_loader import load_training_data
from solver.search.engine import solve_task, evaluate_solver
from src.visualization.grid_viz import plot_task
import matplotlib.pyplot as plt
import numpy as np

tasks = load_training_data()

# ── 1. Test solver di 1 task dulu ──────────────────────────
task_id = '00576224'  # task tiling yang udah kita lihat
task    = tasks[task_id]

print(f"Solving task: {task_id}")
results = solve_task(task, top_k=3)

print(f"\nTop {len(results)} kandidat transformasi:")
for i, r in enumerate(results):
    print(f"  #{i+1}  [{r['score']*100:.1f}%]  {r['transform']}")

best = results[0]
print(f"\nBest transform : {best['transform']}")
print(f"Score          : {best['score']*100:.1f}%")
print(f"\nPredicted output:")
for row in best['predictions'][0]:
    print(' ', row)

print(f"\nActual solution:")
for row in task['test_solutions'][0]:
    print(' ', row)

match = best['predictions'][0] == task['test_solutions'][0]
print(f"\nPerfect match? {'✅ YA!' if match else '❌ Belum'}")

# ── 2. Evaluasi di 100 tasks ───────────────────────────────
print("\n" + "="*45)
print("Evaluasi 100 tasks pertama...")
evaluate_solver(tasks, max_tasks=100)