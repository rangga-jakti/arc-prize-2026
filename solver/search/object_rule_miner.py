# solver/search/object_rule_miner.py

import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def infer_object_rules(inp, out):

    in_objs = detect_all_objects(
        inp,
        ignore_color=0
    )

    out_objs = detect_all_objects(
        out,
        ignore_color=0
    )

    rules = []

    # =====================================================
    # OBJECT COUNT CHANGE
    # =====================================================

    if len(out_objs) > len(in_objs):

        rules.append({

            'type': 'ADD_OBJECT',

            'count_added':
                len(out_objs) - len(in_objs)
        })

    elif len(out_objs) < len(in_objs):

        rules.append({

            'type': 'REMOVE_OBJECT',

            'count_removed':
                len(in_objs) - len(out_objs)
        })

    # =====================================================
    # COLOR INTRODUCTION
    # =====================================================

    in_colors = set(
        int(o['color'])
        for o in in_objs
    )

    out_colors = set(
        int(o['color'])
        for o in out_objs
    )

    new_colors = out_colors - in_colors

    if new_colors:

        rules.append({

            'type': 'NEW_COLOR',

            'colors': list(new_colors)
        })

    # =====================================================
    # LARGE OBJECT PRESERVED
    # =====================================================

    if in_objs and out_objs:

        largest_in = max(
            in_objs,
            key=lambda o: o['size']
        )

        largest_out = max(
            out_objs,
            key=lambda o: o['size']
        )

        if (
            largest_in['size']
            ==
            largest_out['size']
        ):

            rules.append({

                'type': 'MAIN_OBJECT_PRESERVED',

                'size':
                    int(largest_in['size'])
            })

    return rules


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00d62c1b'

    pair = tasks[task_id]['train'][0]

    rules = infer_object_rules(

        pair['input'],
        pair['output']
    )

    print("\nINFERRED RULES\n")

    print("="*50)

    for r in rules:

        print(r)