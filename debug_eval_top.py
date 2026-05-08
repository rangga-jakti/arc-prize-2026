# debug_eval_top.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import load_evaluation_data
from src.visualization.grid_viz import plot_grid
from solver.rules.object_solver import fill_object_interior
from solver.search.scorer import score_prediction

tasks = load_evaluation_data()

# Analisis 3 task teratas
targets = ['135a2760', 'e376de54', '7b80bb43']

for tid in targets:
    task = tasks[tid]
    sol  = task['test_solutions'][0]

    # Cari transform terbaik
    from solver.search.engine import solve_task
    results = solve_task(task, top_k=1)
    best    = results[0]
    pred    = best['predictions'][0]

    pred_np = np.array(pred)
    sol_np  = np.array(sol)
    inp_np  = np.array(task['test'][0]['input'])

    score = score_prediction(pred, sol)
    diff  = pred_np != sol_np
    n_wrong = int(np.sum(diff))

    print(f"\n{'='*50}")
    print(f"Task: {tid} | Score: {score:.3f} | Wrong pixels: {n_wrong}")
    print(f"Transform: {best['transform']}")
    print(f"Pred colors : {sorted(np.unique(pred_np).tolist())}")
    print(f"Sol colors  : {sorted(np.unique(sol_np).tolist())}")
    print(f"Input colors: {sorted(np.unique(inp_np).tolist())}")

    # Lihat pixel yang salah
    wrong_positions = list(zip(*np.where(diff)))
    print(f"Sample wrong pixels (max 10):")
    for r,c in wrong_positions[:10]:
        print(f"  [{r:2d},{c:2d}]: pred={pred_np[r,c]} sol={sol_np[r,c]} inp={inp_np[r,c]}")

    # Visualisasi
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_grid(task['test'][0]['input'], axes[0], 'Test Input')
    plot_grid(pred, axes[1], f'Predicted ({score:.3f})')
    plot_grid(sol,  axes[2], 'Actual Solution')
    diff_vis = (diff.astype(int) * 2).tolist()
    plot_grid(diff_vis, axes[3], f'Diff ({n_wrong} pixels)')
    plt.suptitle(f'Task {tid}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'outputs/{tid}_debug.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Saved: outputs/{tid}_debug.png")