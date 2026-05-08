# debug_045e.py
import numpy as np
from src.utils.data_loader import load_training_data
from solver.rules.object_solver import fill_object_interior, get_background_color

tasks = load_training_data()
task  = tasks['045e512c']

print("=== Task 045e512c ===")
for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    print(f"\nPair {i}: inp={inp.shape}")

    # Lihat warna unik
    print(f"  Input colors : {sorted(np.unique(inp).tolist())}")
    print(f"  Output colors: {sorted(np.unique(out).tolist())}")

    # Lihat pixel yang berubah
    changed = inp != out
    n_changed = np.sum(changed)
    print(f"  Pixels changed: {n_changed}")

    if n_changed > 0 and n_changed < 20:
        print("  Changed positions:")
        for r, c in zip(*np.where(changed)):
            print(f"    [{r},{c}]: {inp[r,c]} -> {out[r,c]}")

    # Coba berbagai fill
    bg = get_background_color(pair['input'])
    print(f"  Background: {bg}")