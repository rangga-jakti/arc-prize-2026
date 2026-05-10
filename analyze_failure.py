from src.utils.data_loader import (
    load_training_data
)

from solver.search.beam_search import (
    beam_search
)

from visualize_task import (
    visualize_single_pair
)


tasks = load_training_data()

FAILED = [

    '017c7c7b',
    '0520fde7',
]

for tid in FAILED:

    print('\n===', tid, '===')

    task = tasks[tid]

    results = beam_search(

        task,

        beam_width=5,

        max_depth=3
    )

    best = results[0]

    print(
        'BEST:',
        best['score'],
        best['program']
    )

    pair = task['train'][0]

    pred = best['predictions'][0]

    visualize_single_pair(

        pair['input'],
        pred,
        pair['output'],

        save_path=f'fail_{tid}.png'
    )