# src/utils/data_loader.py
import json
import os
from src.config import TRAINING_DIR, EVALUATION_DIR, TEST_DIR

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_training_data():
    """
    Load training challenges + solutions, gabungkan jadi satu dict.
    Format output: { task_id: { 'train': [...], 'test': [...] } }
    """
    challenges = load_json(os.path.join(TRAINING_DIR, 'arc-agi_training_challenges.json'))
    solutions  = load_json(os.path.join(TRAINING_DIR, 'arc-agi_training_solutions.json'))

    tasks = {}
    for task_id, task_data in challenges.items():
        tasks[task_id] = {
            'train': task_data['train'],      # list of {input, output}
            'test' : task_data['test'],       # list of {input} only
            'test_solutions': solutions.get(task_id, [])  # list of grids
        }
    return tasks

def load_evaluation_data():
    challenges = load_json(os.path.join(EVALUATION_DIR, 'arc-agi_evaluation_challenges.json'))
    solutions  = load_json(os.path.join(EVALUATION_DIR, 'arc-agi_evaluation_solutions.json'))

    tasks = {}
    for task_id, task_data in challenges.items():
        tasks[task_id] = {
            'train': task_data['train'],
            'test' : task_data['test'],
            'test_solutions': solutions.get(task_id, [])
        }
    return tasks

def load_test_data():
    """Test data: tidak ada solutions (ini yang dikirim ke Kaggle)."""
    challenges = load_json(os.path.join(TEST_DIR, 'arc-agi_test_challenges.json'))

    tasks = {}
    for task_id, task_data in challenges.items():
        tasks[task_id] = {
            'train': task_data['train'],
            'test' : task_data['test'],
            'test_solutions': []  # unknown
        }
    return tasks

def get_task_summary(tasks):
    """Print ringkasan dataset."""
    print(f"Total tasks       : {len(tasks)}")

    train_pairs = [len(t['train']) for t in tasks.values()]
    print(f"Train pairs/task  : min={min(train_pairs)}, max={max(train_pairs)}, avg={sum(train_pairs)/len(train_pairs):.1f}")

    # Grid size info
    all_inputs = []
    for t in tasks.values():
        for pair in t['train']:
            all_inputs.append(pair['input'])

    heights = [len(g) for g in all_inputs]
    widths  = [len(g[0]) for g in all_inputs]
    print(f"Grid height       : min={min(heights)}, max={max(heights)}")
    print(f"Grid width        : min={min(widths)},  max={max(widths)}")