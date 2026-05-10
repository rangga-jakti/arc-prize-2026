# solver/rules/copy_ops.py

import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def copy_largest_right(grid):

    g = np.array(grid)

    result = g.copy()

    objects = detect_all_objects(
        g,
        ignore_color=0
    )

    if not objects:

        return result.tolist()

    largest = objects[0]

    color = largest['color']

    cells = largest['cells']

    width = largest['width']

    for r, c in cells:

        nc = c + width + 1

        if (
            0 <= r < g.shape[0]
            and
            0 <= nc < g.shape[1]
        ):

            result[r, nc] = color

    return result.tolist()


if __name__ == "__main__":

    test = [

        [0,0,0,0,0,0],

        [0,2,2,0,0,0],

        [0,2,2,0,0,0],

        [0,0,0,0,0,0],
    ]

    out = copy_largest_right(
        test
    )

    print("\nRESULT\n")

    for row in out:

        print(row)