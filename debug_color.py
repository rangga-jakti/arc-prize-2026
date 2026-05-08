# debug_color.py
import numpy as np
from src.utils.data_loader import load_training_data
from solver.rules.color_solver import build_multi_recolor_transforms, infer_color_mapping
from solver.search.scorer import score_prediction

tasks = load_training_data()

# Cek 10 task WRONG_COLORS dari failure analysis
wrong_color_tasks = [
    '009d5c81', '00d62c1b', '00dbd492', '05a7bcf2', '05f2a901'
]

for tid in wrong_color_tasks:
    task = tasks[tid]
    train_pairs = task['train']
    solutions   = task.get('test_solutions', [])
    if not solutions:
        continue

    print(f"\n{'='*50}")
    print(f"Task: {tid}")
    print(f"Train pairs: {len(train_pairs)}")

    # Cek shape
    for i, pair in enumerate(train_pairs):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        print(f"  Pair {i}: inp={inp.shape} out={out.shape}")

        # Cek mapping
        mapping = infer_color_mapping(pair['input'], pair['output'])
        print(f"  Color mapping: {mapping}")

    # Cek apakah multi_recolor berhasil dibuat
    multi = build_multi_recolor_transforms(train_pairs)
    print(f"  Multi recolor transforms generated: {list(multi.keys())}")

    # Test hasilnya
    for name, fn in multi.items():
        pred  = fn(task['test'][0]['input'])
        score = score_prediction(pred, solutions[0])
        print(f"  [{score*100:.1f}%] {name}")