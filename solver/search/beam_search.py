# solver/search/beam_search.py

from solver.search.heuristics import (
    suggest_operations
)

from solver.dsl import (
    execute_program,
    DSL_OPERATIONS
)

from solver.search.scorer import (
    score_prediction
)


def evaluate_program(
    task,
    program
):

    scores = []

    predictions = []

    try:

        for pair in task['train']:

            pred = execute_program(

                pair['input'],
                program
            )

            predictions.append(pred)

            score = score_prediction(
                pred,
                pair['output']
            )

            scores.append(score)

        avg_score = (
            sum(scores) / len(scores)
        )

        return {

            'program': program,

            'score': avg_score,

            'predictions': predictions,
        }

    except Exception:

        return {

            'program': program,

            'score': 0.0,

            'predictions': [],
        }


def beam_search(
    task,
    beam_width=20,
    max_depth=4
):

    # ============================================
    # HEURISTIC OPS
    # ============================================

    ops = suggest_operations(task)

    print("\nSuggested Ops:\n")

    for op in ops:

        print("-", op)

    beam = [

        {
            'program': [],
            'score': 0.0,
        }
    ]

    best_programs = []

    for depth in range(max_depth):

        print(
            f"\nDepth {depth+1}"
        )

        candidates = []

        for state in beam:

            for op in ops:

                new_program = (
                    state['program']
                    + [op]
                )

                result = evaluate_program(
                    task,
                    new_program
                )

                candidates.append(result)

        candidates.sort(
            key=lambda x: x['score'],
            reverse=True
        )

        beam = candidates[:beam_width]

        best_programs.extend(beam)

        best_programs.sort(
            key=lambda x: x['score'],
            reverse=True
        )

        best_programs = best_programs[:beam_width]

        print(
            "Best:",
            best_programs[0]['score'],
            best_programs[0]['program']
        )

    return best_programs


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00576224'

    task = tasks[task_id]

    results = beam_search(

        task,

        beam_width=10,

        max_depth=4
    )

    print("\nTOP PROGRAMS\n")

    for r in results[:10]:

        print(

            f"{r['score']*100:.1f}% "

            f"{r['program']}"
        )