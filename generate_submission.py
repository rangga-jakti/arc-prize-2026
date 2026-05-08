# generate_submission.py
import json
import numpy as np
from src.utils.data_loader import load_test_data, load_json
from solver.search.engine import solve_task
from tqdm import tqdm
import os

print("Loading test data...")
test_tasks = load_test_data()
print(f"Total test tasks: {len(test_tasks)}")

submission = {}

for task_id, task in tqdm(test_tasks.items(), desc="Solving"):
    try:
        results  = solve_task(task, top_k=2)
        n_tests  = len(task['test'])
        task_attempts = []

        for test_idx in range(n_tests):
            # Attempt 1: best transform
            if results and test_idx < len(results[0]['predictions']):
                attempt_1 = results[0]['predictions'][test_idx]
            else:
                attempt_1 = task['test'][test_idx]['input']

            # Attempt 2: second best atau sama dengan attempt 1
            if len(results) > 1 and test_idx < len(results[1]['predictions']):
                attempt_2 = results[1]['predictions'][test_idx]
            else:
                attempt_2 = attempt_1

            task_attempts.append({
                'attempt_1': attempt_1,
                'attempt_2': attempt_2,
            })

        submission[task_id] = task_attempts

    except Exception as e:
        print(f"Error on {task_id}: {e}")
        fallback = task['test'][0]['input']
        submission[task_id] = [{
            'attempt_1': fallback,
            'attempt_2': fallback,
        }]

# ── Validasi format ────────────────────────────────────────
sample     = load_json('data/sample_submission.json')
sample_id  = list(sample.keys())[0]
our_id     = list(submission.keys())[0]

print(f"\n=== FORMAT VALIDATION ===")
print(f"Sample: {sample[sample_id]}")
print(f"Ours  : {submission[our_id]}")

# Cek semua entry punya format yang benar
errors = 0
for tid, attempts in submission.items():
    if not isinstance(attempts, list):
        print(f"ERROR {tid}: bukan list"); errors += 1
        continue
    for i, att in enumerate(attempts):
        if 'attempt_1' not in att or 'attempt_2' not in att:
            print(f"ERROR {tid}[{i}]: missing attempt keys"); errors += 1
        if not isinstance(att['attempt_1'], list):
            print(f"ERROR {tid}[{i}]: attempt_1 bukan list"); errors += 1

if errors == 0:
    print("✅ Format valid! Semua entry OK")
else:
    print(f"❌ {errors} errors ditemukan")

# ── Save ───────────────────────────────────────────────────
output_path = 'outputs/submissions/submission.json'
with open(output_path, 'w') as f:
    json.dump(submission, f)

size_kb = os.path.getsize(output_path) / 1024
print(f"\n✅ Saved: {output_path}")
print(f"Size   : {size_kb:.1f} KB")
print(f"Tasks  : {len(submission)}")