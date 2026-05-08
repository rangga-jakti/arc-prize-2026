# test_viz.py
from src.utils.data_loader import load_training_data
from src.visualization.grid_viz import plot_task, plot_multiple_tasks

tasks = load_training_data()

# Visualisasi task pertama (yang tadi kita lihat)
task_id = '00576224'
print(f'Visualisasi task: {task_id}')
plot_task(tasks[task_id], task_id=task_id)

# Visualisasi 3 task random lainnya
import random
random.seed(42)
sample_ids = random.sample(list(tasks.keys()), 3)
for tid in sample_ids:
    plot_task(tasks[tid], task_id=tid)