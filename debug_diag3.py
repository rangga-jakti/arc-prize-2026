# debug_diag3.py
import numpy as np
from src.utils.data_loader import load_training_data

tasks = load_training_data()
task  = tasks['0b17323b']

for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])

    print(f"\n=== Pair {i} — Posisi warna 2 ===")
    for r,c in zip(*np.where(out == 2)):
        print(f"\n  Posisi [{r},{c}]")
        # Lihat area 5x5 di sekitarnya di INPUT
        r1,r2 = max(0,r-3), min(inp.shape[0],r+4)
        c1,c2 = max(0,c-3), min(inp.shape[1],c+4)
        print(f"  Input sekitar [{r},{c}]:")
        for rr in range(r1,r2):
            row_str = ""
            for cc in range(c1,c2):
                marker = ">" if (rr==r and cc==c) else " "
                row_str += f"{marker}{inp[rr,cc]}"
            print(f"    {row_str}")

        # Cek apakah ada pola: crossing garis horizontal & vertikal
        row_vals = inp[r, :]  # baris r
        col_vals = inp[:, c]  # kolom c
        row_has_1 = 1 in row_vals
        col_has_1 = 1 in col_vals
        print(f"  Row {r} punya warna 1: {row_has_1}")
        print(f"  Col {c} punya warna 1: {col_has_1}")
        print(f"  CROSSING (row AND col punya 1): {row_has_1 and col_has_1}")