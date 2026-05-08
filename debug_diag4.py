# debug_diag4.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import load_training_data
from src.visualization.grid_viz import plot_grid

tasks = load_training_data()
task  = tasks['0b17323b']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    diff = (out == 2).astype(int) * 2  # highlight warna 2

    plot_grid(pair['input'],  axes[i][0], title=f'Pair {i} Input')
    plot_grid(pair['output'], axes[i][1], title=f'Pair {i} Output')
    plot_grid(diff.tolist(),  axes[i][2], title=f'Pair {i} Warna=2 positions')

    # Transpose input untuk compare
    plot_grid(np.array(pair['input']).T.tolist(), axes[i][3], title=f'Pair {i} Input.T')

plt.suptitle('Task 0b17323b - Full Analysis', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/debug_0b17323b.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved!")