# generate_submission.py

import json
from tqdm import tqdm

from src.utils.data_loader import (
    load_training_data
)

from solver.search.engine import (
    search_single_transforms
)


def stringify(grid):

    return "|".join(
        "".join(map(str, row))
        for row in grid
    )


def build_submission():

    tasks = load_training_data()

    submission = {}

    solved = 0

    print("\nGenerating submission...\n")

    for task_id, task in tqdm(tasks.items()):

        try:

            results = search_single_transforms(
                task
            )

            if not results:

                pred = task['test'][0]['input']

            else:

                pred = results[0][
                    'predictions'
                ][0]

            submission[task_id] = [{

                "attempt_1":
                    stringify(pred),

                "attempt_2":
                    stringify(pred),
            }]

            # Optional evaluation
            if 'test_solutions' in task:

                gt = task['test_solutions'][0]

                if pred == gt:
                    solved += 1

        except Exception as e:

            print(
                f"\n[ERROR] {task_id}:",
                e
            )

            fallback = task['test'][0]['input']

            submission[task_id] = [{

                "attempt_1":
                    stringify(fallback),

                "attempt_2":
                    stringify(fallback),
            }]

    with open(
        "submission.json",
        "w"
    ) as f:

        json.dump(
            submission,
            f
        )

    print("\nDONE!")
    print(
        "submission.json created"
    )

    print(
        f"\nSolved: {solved}"
    )


if __name__ == "__main__":

    build_submission()