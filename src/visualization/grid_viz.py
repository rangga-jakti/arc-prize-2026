
# src/visualization/grid_viz.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Warna resmi ARC (index 0-9)
ARC_COLORS = [
    '#000000',  # 0 - hitam
    '#0074D9',  # 1 - biru
    '#FF4136',  # 2 - merah
    '#2ECC40',  # 3 - hijau
    '#FFDC00',  # 4 - kuning
    '#AAAAAA',  # 5 - abu
    '#F012BE',  # 6 - magenta
    '#FF851B',  # 7 - orange
    '#7FDBFF',  # 8 - biru muda
    '#870C25',  # 9 - merah tua
]

ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM  = mcolors.BoundaryNorm(boundaries=range(11), ncolors=10)


def plot_grid(grid, ax, title=''):
    """Plot satu grid ARC dengan warna."""
    arr = np.array(grid)
    ax.imshow(arr, cmap=ARC_CMAP, norm=ARC_NORM)

    # Grid lines
    rows, cols = arr.shape
    ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # Tulis angka di tiap cell
    for r in range(rows):
        for c in range(cols):
            val = arr[r, c]
            color = 'white' if val in [0, 9] else 'black'
            ax.text(c, r, str(val), ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_title(title, fontsize=10, pad=4)


def plot_task(task, task_id='', max_pairs=4):
    """
    Visualisasi lengkap satu task ARC.
    Tampilkan train pairs (input->output) + test input.
    """
    train_pairs = task['train'][:max_pairs]
    test_inputs = task['test']
    n_train = len(train_pairs)
    n_test  = len(test_inputs)

    # Total kolom: input + output per train pair, + test input
    n_cols = 2 * n_train + n_test
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    col = 0
    for i, pair in enumerate(train_pairs):
        plot_grid(pair['input'],  axes[col], title=f'Train {i+1}\nInput')
        col += 1
        plot_grid(pair['output'], axes[col], title=f'Train {i+1}\nOutput')
        col += 1

    for i, test in enumerate(test_inputs):
        plot_grid(test['input'], axes[col], title=f'Test {i+1}\nInput (?)' )
        col += 1

    fig.suptitle(f'Task: {task_id}', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'outputs/{task_id}_visualization.png',
                bbox_inches='tight', dpi=120)
    plt.show()
    print(f'Saved: outputs/{task_id}_visualization.png')


def plot_multiple_tasks(tasks, n=6):
    """Lihat overview beberapa task sekaligus."""
    task_ids = list(tasks.keys())[:n]
    for tid in task_ids:
        plot_task(tasks[tid], task_id=tid)