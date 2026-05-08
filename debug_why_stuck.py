# debug_why_stuck.py
import numpy as np
from src.utils.data_loader import load_training_data
from solver.search.engine import search_single_transforms
from solver.search.scorer import score_prediction

tasks = load_training_data()
task_ids = list(tasks.keys())

perfect_tasks  = []
high_partial   = []  # score > 0.9
stuck_tasks    = []  # best score < 0.5

for tid in task_ids[:200]:
    task      = tasks[tid]
    solutions = task.get('test_solutions', [])
    if not solutions:
        continue

    results = search_single_transforms(task)
    if not results:
        stuck_tasks.append((tid, 0.0, 'no_transform'))
        continue

    best       = results[0]
    best_score = best['score']
    pred       = best['predictions'][0] if best['predictions'] else []
    actual_score = score_prediction(pred, solutions[0])

    if actual_score == 1.0:
        perfect_tasks.append((tid, best['transform']))
    elif best_score >= 0.9:
        high_partial.append((tid, best_score, best['transform']))
    elif best_score < 0.5:
        stuck_tasks.append((tid, best_score, best['transform']))

print(f"Perfect  : {len(perfect_tasks)}")
print(f"High partial (>90%): {len(high_partial)}")
print(f"Stuck (<50%)       : {len(stuck_tasks)}")

print(f"\n--- HIGH PARTIAL (paling mungkin di-fix) ---")
for tid, score, transform in high_partial[:10]:
    task = tasks[tid]
    inp  = np.array(task['train'][0]['input'])
    out  = np.array(task['train'][0]['output'])
    sol  = np.array(task['test_solutions'][0])
    print(f"{tid}: {score:.3f} via {transform}")
    print(f"  train: {inp.shape}->{out.shape} | test_sol: {sol.shape}")
    in_colors  = sorted(np.unique(inp).tolist())
    out_colors = sorted(np.unique(out).tolist())
    sol_colors = sorted(np.unique(sol).tolist())
    print(f"  colors in={in_colors} out={out_colors} sol={sol_colors}")