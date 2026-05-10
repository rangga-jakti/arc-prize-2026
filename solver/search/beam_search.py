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

from solver.search.op_stats import (

    update_op_stats
)

# ============================================
# PROGRAM CACHE
# ============================================

PROGRAM_CACHE = {}


def evaluate_program(
    task,
    program
):

    key = tuple(program)

    if key in PROGRAM_CACHE:

        return PROGRAM_CACHE[key]

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

        result = {

            'program': program,

            'score': avg_score,

            'predictions': predictions,
        }

        PROGRAM_CACHE[key] = result

        return result

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

                # ============================================
                # LIMIT REPEATED OBJECT OPS
                # ============================================

                repeated_object_ops = [

                    'COPY_LARGEST_RIGHT',

                    'MOVE_UP',
                    'MOVE_DOWN',
                    'MOVE_LEFT',
                    'MOVE_RIGHT',
                ]

                if len(new_program) >= 2:

                    last = new_program[-1]

                    prev = new_program[-2]

                    if (
                        last == prev
                        and
                        last in repeated_object_ops
                    ):

                        continue

                # ============================================
                # SYMBOLIC ALGEBRA PRUNING
                # ============================================

                redundant_pairs = [

                    ['FLIP_H', 'FLIP_H'],
                    ['FLIP_V', 'FLIP_V'],

                    ['FLIP_H', 'FLIP_V'],
                    ['FLIP_V', 'FLIP_H'],

                    ['ROT90', 'ROT270'],
                    ['ROT270', 'ROT90'],

                    ['ROT180', 'ROT180'],

                    ['MOVE_UP', 'MOVE_DOWN'],
                    ['MOVE_DOWN', 'MOVE_UP'],

                    ['MOVE_LEFT', 'MOVE_RIGHT'],
                    ['MOVE_RIGHT', 'MOVE_LEFT'],
                ]

                skip_pair = False

                for pair in redundant_pairs:

                    if len(new_program) >= 2:

                        if new_program[-2:] == pair:

                            skip_pair = True

                if skip_pair:

                    continue

                # ============================================
                # GLOBAL ROTATION CANCELLATION
                # ============================================

                rot_score = 0

                for step in new_program:

                    if step == 'ROT90':

                        rot_score += 1

                    elif step == 'ROT270':

                        rot_score -= 1

                    elif step == 'ROT180':

                        rot_score += 2

                if (
                    len(new_program) >= 2
                    and
                    rot_score % 4 == 0
                    and
                    any(
                        s.startswith('ROT')
                        for s in new_program
                    )
                ):

                    continue

                # ============================================
                # SYMBOLIC ORDERING PRUNING
                # ============================================

                if len(new_program) >= 2:

                    prev = new_program[-2]

                    last = new_program[-1]

                    # SCALE <-> FLIP redundancy

                    if (

                        (
                            prev.startswith('SCALE')
                            and
                            last.startswith('FLIP')
                        )

                        or

                        (
                            prev.startswith('FLIP')
                            and
                            last.startswith('SCALE')
                        )
                    ):

                        continue

                    # FLIP <-> ROT redundancy

                    if (

                        (
                            prev.startswith('FLIP')
                            and
                            last.startswith('ROT')
                        )

                        or

                        (
                            prev.startswith('ROT')
                            and
                            last.startswith('FLIP')
                        )
                    ):

                        continue

                    # COMMUTATIVE MOVE NORMALIZATION

                    if (

                        (
                            prev in [
                                'MOVE_UP',
                                'MOVE_DOWN'
                            ]

                            and

                            last in [
                                'MOVE_LEFT',
                                'MOVE_RIGHT'
                            ]
                        )
                    ):

                        continue

                    # TILE -> FLIP redundant ordering

                    if (

                        prev.startswith('TILE')

                        and

                        last.startswith('FLIP')
                    ):

                        continue

                # ============================================
                # LIMIT TOTAL COPY OPS
                # ============================================

                copy_count = new_program.count(
                    'COPY_LARGEST_RIGHT'
                )

                if copy_count > 1:

                    continue

                # ============================================
                # EVALUATE PROGRAM
                # ============================================

                result = evaluate_program(
                    task,
                    new_program
                )

                candidates.append(result)

                # ============================================
                # LEARN OP SUCCESS
                # ============================================

                update_op_stats(

                    new_program,

                    result['score']
                )

        # ============================================
        # SEMANTIC DEDUPLICATION
        # ============================================

        semantic_best = {}

        for cand in candidates:

            sig = str(
                cand['predictions']
            )

            if sig not in semantic_best:

                semantic_best[sig] = cand

            else:

                prev = semantic_best[sig]

                if (

                    cand['score'] > prev['score']

                    or

                    (
                        cand['score'] == prev['score']
                        and
                        len(cand['program'])
                        <
                        len(prev['program'])
                    )
                ):

                    semantic_best[sig] = cand

        candidates = list(
            semantic_best.values()
        )

        candidates.sort(

            key=lambda x: (

                x['score'],

                -len(x['program'])
            ),

            reverse=True
        )

        beam = candidates[:beam_width]

        best_programs.extend(beam)

        best_programs.sort(

            key=lambda x: (

                x['score'],

                -len(x['program'])
            ),

            reverse=True
        )

        # ============================================
        # DOMINANCE PRUNING
        # ============================================

        pruned = []

        for cand in best_programs:

            dominated = False

            for other in pruned:

                if (

                    cand['score'] == other['score']

                    and

                    len(cand['program']) >

                    len(other['program'])
                ):

                    dominated = True

            if not dominated:

                pruned.append(cand)

        best_programs = pruned[:beam_width]

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

    task_id = '00d62c1b'

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