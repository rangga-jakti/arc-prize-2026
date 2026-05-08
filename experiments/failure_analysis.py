
# experiments/failure_analysis.py
import numpy as np
from collections import defaultdict
from src.utils.data_loader import load_training_data
from solver.search.engine import solve_task, search_single_transforms
from solver.search.scorer import score_prediction
from solver.rules.transformations import to_np

tasks = load_training_data()
task_ids = list(tasks.keys())[:100]

results_log = []

for tid in task_ids:
    task = tasks[tid]
    solutions = task.get('test_solutions', [])
    if not solutions:
        continue

    candidates = solve_task(task, top_k=1)
    best = candidates[0]
    pred = best['predictions'][0] if best['predictions'] else []
    sol  = solutions[0]
    score = score_prediction(pred, sol)

    pred_np = to_np(pred) if pred else None
    sol_np  = to_np(sol)
    inp_np  = to_np(task['test'][0]['input'])

    # Kategorikan kegagalan
    failure_type = []

    if score == 1.0:
        failure_type.append('PERFECT')
    else:
        # Cek apakah ukuran output beda
        if pred_np is None or pred_np.shape != sol_np.shape:
            failure_type.append('WRONG_SIZE')
        else:
            # Seberapa beda warnanya?
            pred_colors = set(np.unique(pred_np).tolist())
            sol_colors  = set(np.unique(sol_np).tolist())

            if pred_colors != sol_colors:
                failure_type.append('WRONG_COLORS')
            else:
                # Sama warna tapi posisi beda
                failure_type.append('WRONG_POSITION')

        # Cek apakah input == output (identity)
        if np.array_equal(inp_np, sol_np):
            failure_type.append('IDENTITY_TASK')

        # Cek apakah output lebih besar dari input
        if sol_np.shape[0] > inp_np.shape[0] or sol_np.shape[1] > inp_np.shape[1]:
            failure_type.append('UPSCALE_NEEDED')
        elif sol_np.shape[0] < inp_np.shape[0] or sol_np.shape[1] < inp_np.shape[1]:
            failure_type.append('DOWNSCALE_NEEDED')

    results_log.append({
        'task_id'      : tid,
        'score'        : score,
        'best_transform': best['transform'],
        'failure_types': failure_type,
        'inp_shape'    : inp_np.shape,
        'sol_shape'    : sol_np.shape,
        'n_train_pairs': len(task['train']),
    })

# ── Ringkasan ──────────────────────────────────────────────
print("=" * 50)
print("         FAILURE ANALYSIS REPORT")
print("=" * 50)

perfect  = [r for r in results_log if 'PERFECT' in r['failure_types']]
failed   = [r for r in results_log if 'PERFECT' not in r['failure_types']]

print(f"Perfect   : {len(perfect)}")
print(f"Failed    : {len(failed)}")
print()

# Hitung frekuensi tiap failure type
freq = defaultdict(int)
for r in failed:
    for ft in r['failure_types']:
        freq[ft] += 1

print("Failure type breakdown:")
for ft, count in sorted(freq.items(), key=lambda x: -x[1]):
    pct = count / len(failed) * 100
    print(f"  {ft:<22} : {count:3d} tasks ({pct:.0f}%)")

print()
print("=" * 50)
print("Sample WRONG_SIZE tasks (perlu transformasi baru):")
wrong_size = [r for r in failed if 'WRONG_SIZE' in r['failure_types']][:5]
for r in wrong_size:
    print(f"  {r['task_id']}  inp={r['inp_shape']}  sol={r['sol_shape']}  best={r['best_transform']}")

print()
print("Sample WRONG_COLORS tasks:")
wrong_col = [r for r in failed if 'WRONG_COLORS' in r['failure_types']][:5]
for r in wrong_col:
    print(f"  {r['task_id']}  score={r['score']:.2f}  best={r['best_transform']}")

print()
print("Sample WRONG_POSITION tasks:")
wrong_pos = [r for r in failed if 'WRONG_POSITION' in r['failure_types']][:5]
for r in wrong_pos:
    print(f"  {r['task_id']}  score={r['score']:.2f}  best={r['best_transform']}")

print()
print("=" * 50)
print("Top transforms yang berhasil (dari partial):")
transform_scores = defaultdict(list)
for r in results_log:
    transform_scores[r['best_transform']].append(r['score'])
ranked = sorted(transform_scores.items(),
                key=lambda x: sum(x[1])/len(x[1]), reverse=True)
for name, scores in ranked[:10]:
    avg = sum(scores)/len(scores)
    print(f"  {name:<30} avg={avg:.2f}  n={len(scores)}")