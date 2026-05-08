
# debug_diag2.py
import numpy as np
from src.utils.data_loader import load_training_data
from solver.rules.transformations import flip_diagonal

tasks = load_training_data()
task  = tasks['0b17323b']

for i, pair in enumerate(task['train']):
    inp  = np.array(pair['input'])
    out  = np.array(pair['output'])
    pred = np.array(flip_diagonal(pair['input']))

    print(f"\n=== Pair {i} ===")
    print(f"Posisi diagonal yang actual=2:")
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if out[r,c] == 2:
                print(f"  [{r},{c}]: inp={inp[r,c]} inp_T={inp[c,r]} r==c:{r==c}")

print("\n=== Pattern Analysis ===")
print("Apakah warna 2 muncul HANYA di r==c?")
for i, pair in enumerate(task['train']):
    out = np.array(pair['output'])
    positions = list(zip(*np.where(out == 2)))
    on_diag = all(r==c for r,c in positions)
    print(f"  Pair {i}: {len(positions)} pixel warna 2, semua di diagonal: {on_diag}")
    for r,c in positions:
        print(f"    [{r},{c}] r==c:{r==c}")