# solver/dsl_search.py

import itertools

from solver.dsl import (
    DSL_OPERATIONS,
    execute_program
)

from solver.search.scorer import (
    score_prediction
)


def evaluate_program(
    program,
    train_pairs
):

    scores = []

    for pair in train_pairs:

        try:

            pred = execute_program(

                pair['input'],
                program
            )

            score = score_prediction(

                pred,
                pair['output']
            )

            scores.append(score)

        except Exception:

            scores.append(0.0)

    return sum(scores) / len(scores)


def search_programs(
    task,
    max_depth=2,
    top_k=10
):

    ops = list(
        DSL_OPERATIONS.keys()
    )

    results = []

    for depth in range(
        1,
        max_depth + 1
    ):

        print(
            f"Searching depth {depth}..."
        )

        programs = itertools.product(
            ops,
            repeat=depth
        )

        for prog in programs:

            score = evaluate_program(

                prog,
                task['train']
            )

            if score <= 0:
                continue

            predictions = []

            for test in task['test']:

                try:

                    pred = execute_program(

                        test['input'],
                        prog
                    )

                    predictions.append(pred)

                except Exception:

                    predictions.append(
                        test['input']
                    )

            results.append({

                'program'     : list(prog),

                'score'       : score,

                'predictions' : predictions,
            })

    results.sort(

        key=lambda x: x['score'],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00576224'

    task = tasks[task_id]

    results = search_programs(

        task,

        max_depth=2,

        top_k=10
    )

    print("\nTOP PROGRAMS\n")

    for r in results:

        print(

            f"{r['score']*100:.1f}%",

            r['program']
        )