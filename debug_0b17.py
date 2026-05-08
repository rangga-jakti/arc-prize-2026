# debug_0b17.py
import numpy as np
from src.utils.data_loader import load_training_data
from solver.rules.transformations import flip_diagonal
from solver.search.scorer import score_prediction

tasks = load_training_data()
task  = tasks['0b17323b']

print("=== Task 0b17323b ===")
for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    pred = np.array(flip_diagonal(pair['input']))
    s = score_prediction(pred.tolist(), pair['output'])
    print(f"Pair {i}: {inp.shape}->{out.shape} | flip_diag score={s:.3f}")
    print(f"  in_colors={sorted(np.unique(inp).tolist())}")
    print(f"  out_colors={sorted(np.unique(out).tolist())}")

    # Cari pixel yang berbeda
    diff = pred != out
    print(f"  Pixels wrong: {np.sum(diff)}")
    if np.sum(diff) < 15:
        for r,c in zip(*np.where(diff)):
            print(f"    [{r},{c}]: pred={pred[r,c]} actual={out[r,c]}")

print("\nTest:")
test_inp = np.array(task['test'][0]['input'])
test_sol = np.array(task['test_solutions'][0])
pred_test = np.array(flip_diagonal(task['test'][0]['input']))
s = score_prediction(pred_test.tolist(), task['test_solutions'][0])
print(f"  inp={test_inp.shape} sol={test_sol.shape}")
print(f"  flip_diag score on test={s:.3f}")
print(f"  in_colors={sorted(np.unique(test_inp).tolist())}")
print(f"  sol_colors={sorted(np.unique(test_sol).tolist())}")

# Cari pixel yang beda
diff = pred_test != test_sol
print(f"  Pixels wrong: {np.sum(diff)}")
if np.sum(diff) < 20:
    for r,c in zip(*np.where(diff)):
        print(f"    [{r},{c}]: pred={pred_test[r,c]} actual={test_sol[r,c]}")