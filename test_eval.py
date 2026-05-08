# experiments/eval_analysis.py
import numpy as np
from collections import defaultdict
from src.utils.data_loader import load_evaluation_data
from solver.search.engine import solve_task
from solver.search.scorer import score_prediction

tasks    = load_evaluation_data()
task_ids = list(tasks.keys())

perfect  = []
high_par = []  # > 0.85
low_par  = []  # 0 < score < 0.85
failed   = []  # score == 0

for tid in task_ids:
    task      = tasks[tid]
    solutions = task.get('test_solutions', [])
    if not solutions:
        continue

    results = solve_task(task, top_k=1)
    best    = results[0]
    pred    = best['predictions'][0] if best['predictions'] else []
    score   = score_prediction(pred, solutions[0])

    entry = {
        'tid'      : tid,
        'score'    : score,
        'transform': best['transform'],
        'inp_shape': np.array(task['test'][0]['input']).shape,
        'sol_shape': np.array(solutions[0]).shape,
        'n_colors' : len(set(np.array(task['train'][0]['input']).flatten().tolist())),
        'n_pairs'  : len(task['train']),
    }

    if score == 1.0:   perfect.append(entry)
    elif score > 0.85: high_par.append(entry)
    elif score > 0:    low_par.append(entry)
    else:              failed.append(entry)

print("=" * 50)
print("  EVALUATION DATASET FAILURE ANALYSIS")
print("=" * 50)
print(f"Perfect  (100%) : {len(perfect)}")
print(f"High     (>85%) : {len(high_par)}")
print(f"Low      (>0%)  : {len(low_par)}")
print(f"Failed   (0%)   : {len(failed)}")
total = len(perfect)+len(high_par)+len(low_par)+len(failed)
print(f"Score           : {len(perfect)/total*100:.1f}%")

print(f"\n--- TOP HIGH PARTIAL (easiest to fix) ---")
high_par.sort(key=lambda x: -x['score'])
for e in high_par[:8]:
    print(f"  {e['tid']}: {e['score']:.3f} via {e['transform']}")
    print(f"    shape: {e['inp_shape']}→{e['sol_shape']} | colors:{e['n_colors']} pairs:{e['n_pairs']}")

print(f"\n--- TRANSFORM SUCCESS RATE ---")
all_entries = perfect + high_par + low_par + failed
transform_scores = defaultdict(list)
for e in all_entries:
    transform_scores[e['transform']].append(e['score'])

ranked = sorted(transform_scores.items(),
                key=lambda x: sum(x[1])/len(x[1]), reverse=True)
for name, scores in ranked[:12]:
    avg = sum(scores)/len(scores)
    perfect_n = sum(1 for s in scores if s==1.0)
    print(f"  {name:<30} avg={avg:.2f} perfect={perfect_n}/{len(scores)}")