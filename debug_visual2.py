
# debug_visual2.py
from src.utils.data_loader import load_training_data
from src.visualization.grid_viz import plot_task

tasks = load_training_data()

for tid in ['025d127b', '045e512c', '05f2a901']:
    plot_task(tasks[tid], task_id=tid, max_pairs=3)