# debug_visual.py
from src.utils.data_loader import load_training_data
from src.visualization.grid_viz import plot_task

tasks = load_training_data()

# Visualisasi 5 task yang gagal
target_tasks = [
    '009d5c81',
    '00d62c1b', 
    '00dbd492',
    '05f2a901',
    '007bbfb7',  # WRONG_POSITION
]

for tid in target_tasks:
    print(f"Visualisasi: {tid}")
    plot_task(tasks[tid], task_id=tid, max_pairs=3)