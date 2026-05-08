# debug_chain.py
from src.utils.data_loader import load_training_data
from solver.search.engine import search_single_transforms
from solver.search.chainer import search_chained_transforms
from solver.rules.transformations import TRANSFORM_REGISTRY
from solver.rules.color_solver import build_multi_recolor_transforms
from solver.search.engine import build_color_transforms

tasks = load_training_data()

# Cek 5 task partial match yang paling dekat perfect
partial_tasks = [
    '025d127b',  # score 0.88
    '045e512c',  # score 0.90
    '05f2a901',  # score 0.95
    '00d62c1b',  # score 0.92
]

for tid in partial_tasks:
    task = tasks[tid]
    singles = search_single_transforms(task)

    if not singles:
        continue

    best = singles[0]
    print(f"\nTask {tid}: best_single={best['transform']} score={best['score']:.3f}")

    if best['score'] == 1.0:
        print("  Already perfect!")
        continue

    # Coba chain dengan top 12
    train_pairs = task['train']
    base = {}
    base.update(TRANSFORM_REGISTRY)
    base.update(build_color_transforms(train_pairs))
    base.update(build_multi_recolor_transforms(train_pairs))

    chains = search_chained_transforms(task, base, top_singles=12)
    if chains:
        print(f"  Chain results ({len(chains)} found):")
        for c in chains[:3]:
            print(f"    [{c['score']*100:.1f}%] {c['transform']}")
    else:
        print("  No improvement from chaining")
        print(f"  Top 3 singles:")
        for s in singles[:3]:
            print(f"    [{s['score']*100:.1f}%] {s['transform']}")