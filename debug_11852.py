# debug_11852.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import load_training_data
from src.visualization.grid_viz import plot_grid, plot_task
from solver.rules.transformations import flip_diagonal
from solver.search.scorer import score_prediction

tasks = load_training_data()
task  = tasks['11852cab']

# Visual dulu
plot_task(task, task_id='11852cab', max_pairs=4)

# Analisis detail
print("=== Task 11852cab ===")
for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    diff = inp != out
    print(f"\nPair {i}: {inp.shape}")
    print(f"  in_colors : {sorted(np.unique(inp).tolist())}")
    print(f"  out_colors: {sorted(np.unique(out).tolist())}")
    print(f"  pixels changed: {np.sum(diff)}")
    if np.sum(diff) < 20:
        for r,c in zip(*np.where(diff)):
            print(f"    [{r},{c}]: {inp[r,c]}->{out[r,c]}")

# Test score
sol = task['test_solutions'][0]
print(f"\nTest sol colors: {sorted(np.unique(np.array(sol)).tolist())}")