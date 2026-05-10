import time

from src.utils.data_loader import (
    load_training_data
)

from solver.search.beam_search import (
    beam_search
)


tasks = load_training_data()

task_ids = list(tasks.keys())[:20]

total = 0
solved = 0

start = time.time()

for i, task_id in enumerate(task_ids):

    print(
        f'\n[{i+1}/{len(task_ids)}] {task_id}'
    )

    task = tasks[task_id]

    results = beam_search(

        task,

        beam_width=5,

        max_depth=3
    )

    best = results[0]

    score = best['score']

    total += score

    if score >= 0.999:

        solved += 1

    print(
        'BEST:',
        round(score, 4),
        best['program']
    )

elapsed = time.time() - start

print('\n====================')
print('BENCHMARK RESULTS')
print('====================')

print(
    'Average Score:',
    round(total / len(task_ids), 4)
)

print(
    'Solved:',
    solved,
    '/',
    len(task_ids)
)

print(
    'Time:',
    round(elapsed, 2),
    'sec'
)