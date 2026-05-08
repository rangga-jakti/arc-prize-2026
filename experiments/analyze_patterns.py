# experiments/analyze_patterns.py
import numpy as np
import json
from src.utils.data_loader import load_training_data, load_evaluation_data

# Load kedua dataset
train_tasks = load_training_data()
eval_tasks  = load_evaluation_data()

print("=" * 50)
print("PATTERN FREQUENCY ANALYSIS")
print("=" * 50)

def analyze_dataset(tasks, name):
    print(f"\n=== {name} ({len(tasks)} tasks) ===")

    size_same     = 0
    size_bigger   = 0
    size_smaller  = 0
    color_same    = 0
    color_diff    = 0
    avg_colors    = []
    avg_pairs     = []

    for tid, task in tasks.items():
        pairs = task['train']
        avg_pairs.append(len(pairs))

        for pair in pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])

            if inp.shape == out.shape:
                size_same += 1
            elif out.size > inp.size:
                size_bigger += 1
            else:
                size_smaller += 1

            in_colors  = set(np.unique(inp).tolist())
            out_colors = set(np.unique(out).tolist())
            avg_colors.append(len(in_colors))

            if in_colors == out_colors:
                color_same += 1
            else:
                color_diff += 1

    total = size_same + size_bigger + size_smaller
    print(f"Grid size SAME    : {size_same/total*100:.1f}%")
    print(f"Grid size BIGGER  : {size_bigger/total*100:.1f}%")
    print(f"Grid size SMALLER : {size_smaller/total*100:.1f}%")
    print(f"Colors SAME       : {color_same/total*100:.1f}%")
    print(f"Colors DIFFERENT  : {color_diff/total*100:.1f}%")
    print(f"Avg colors/grid   : {sum(avg_colors)/len(avg_colors):.1f}")
    print(f"Avg train pairs   : {sum(avg_pairs)/len(avg_pairs):.1f}")

analyze_dataset(train_tasks, "TRAINING")
analyze_dataset(eval_tasks,  "EVALUATION")