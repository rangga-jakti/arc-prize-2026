# test_phase1.py
from src.utils.data_loader import load_training_data, get_task_summary

print('=== TRAINING DATA ===')
tasks = load_training_data()
get_task_summary(tasks)

# Lihat 1 task sebagai contoh
task_id = list(tasks.keys())[0]
task = tasks[task_id]

print(f'\nContoh task ID : {task_id}')
print(f'Jumlah train pairs : {len(task["train"])}')

print(f'\nInput grid (pair 0):')
for row in task['train'][0]['input']:
    print(' ', row)

print(f'\nOutput grid (pair 0):')
for row in task['train'][0]['output']:
    print(' ', row)