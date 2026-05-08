# debug_e376_viz.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import load_evaluation_data
from src.visualization.grid_viz import plot_task, plot_grid

tasks = load_evaluation_data()

# Visualisasi task ini lengkap
plot_task(tasks['e376de54'], task_id='e376de54', max_pairs=3)

# Juga visualisasi 135a2760 dan 7b80bb43
plot_task(tasks['135a2760'], task_id='135a2760', max_pairs=2)
plot_task(tasks['7b80bb43'], task_id='7b80bb43', max_pairs=2)