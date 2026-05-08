# solver/rules/object_ops.py

import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def move_object(
    grid,
    dr,
    dc
):

    g = np.array(grid)

    result = np.zeros_like(g)

    objects = detect_all_objects(
        g,
        ignore_color=0
    )

    for obj in objects:

        color = int(obj['color'])

        cells = obj['cells']

        for r, c in cells:

            nr = r + dr
            nc = c + dc

            if (
                0 <= nr < g.shape[0]
                and
                0 <= nc < g.shape[1]
            ):

                result[nr, nc] = color

    return result.tolist()


def move_up(grid):

    return move_object(
        grid,
        dr=-1,
        dc=0
    )


def move_down(grid):

    return move_object(
        grid,
        dr=1,
        dc=0
    )


def move_left(grid):

    return move_object(
        grid,
        dr=0,
        dc=-1
    )


def move_right(grid):

    return move_object(
        grid,
        dr=0,
        dc=1
    )


if __name__ == "__main__":

    test = [

        [0,0,0,0,0],

        [0,2,2,0,0],

        [0,2,2,0,0],

        [0,0,0,0,0],
    ]

    moved = move_right(test)

    print("\nRESULT\n")

    for row in moved:

        print(row)