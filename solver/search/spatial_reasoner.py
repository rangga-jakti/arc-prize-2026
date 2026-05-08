# solver/search/spatial_reasoner.py

import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def object_center(obj):

    return (
        float(obj['center'][0]),
        float(obj['center'][1])
    )


def relative_position(a, b):

    ar, ac = object_center(a)
    br, bc = object_center(b)

    vertical = ""
    horizontal = ""

    if br < ar:
        vertical = "UP"
    elif br > ar:
        vertical = "DOWN"
    else:
        vertical = "SAME_ROW"

    if bc < ac:
        horizontal = "LEFT"
    elif bc > ac:
        horizontal = "RIGHT"
    else:
        horizontal = "SAME_COL"

    return f"{vertical}_{horizontal}"


def analyze_spatial_relations(grid):

    objects = detect_all_objects(
        grid,
        ignore_color=0
    )

    relations = []

    for i in range(len(objects)):

        for j in range(i + 1, len(objects)):

            a = objects[i]
            b = objects[j]

            relations.append({

                'obj_a_color': int(a['color']),
                'obj_b_color': int(b['color']),

                'obj_a_size': int(a['size']),
                'obj_b_size': int(b['size']),

                'relation': relative_position(
                    a,
                    b
                )
            })

    return relations


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00d62c1b'

    pair = tasks[task_id]['train'][0]['output']

    relations = analyze_spatial_relations(
        pair
    )

    print("\nSPATIAL RELATIONS\n")

    print("=" * 50)

    for r in relations[:20]:

        print(r)