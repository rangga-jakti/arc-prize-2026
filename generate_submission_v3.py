import json
from tqdm import tqdm
from solver.search.engine import search_single_transforms
with open('data/test/arc-agi_test_challenges.json') as f:
    tasks = json.load(f)
submission = {}
perfect = 0
fallback = 0
for task_id, task in tqdm(tasks.items()):
    try:
        results = search_single_transforms(task)
        if results and results[0]['score'] == 1.0:
            pred = results[0]['predictions'][0]
            perfect += 1
        else:
            pred = task['test'][0]['input']
            fallback += 1
    except:
        pred = task['test'][0]['input']
        fallback += 1
    submission[task_id] = [{'attempt_1': pred, 'attempt_2': pred}]
with open('outputs/submissions/submission_v3.json', 'w') as f:
    json.dump(submission, f)
print(f'Perfect (score=1.0): {perfect}/240')
print(f'Fallback: {fallback}/240')
print('Done!')
