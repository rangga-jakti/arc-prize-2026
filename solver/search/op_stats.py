from collections import defaultdict


OP_STATS = defaultdict(list)


def update_op_stats(
    program,
    score
):

    for op in program:

        OP_STATS[op].append(
            score
        )


def get_op_score(op):

    vals = OP_STATS.get(op, [])

    if not vals:

        return 0.5

    return sum(vals) / len(vals)