import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def recolor_largest(
    grid,
    new_color=1
):

    grid = np.array(grid)

    objects = detect_all_objects(grid)

    if not objects:

        return grid.tolist()

    largest = objects[0]

    result = grid.copy()

    for r, c in largest['coords']:

        result[r, c] = new_color

    return result.tolist()


if __name__ == '__main__':

    test = [

        [0,0,0,0],
        [0,2,2,0],
        [0,2,2,0],
        [0,0,3,0],
    ]

    out = recolor_largest(
        test,
        new_color=9
    )

    print('\nRESULT\n')

    for row in out:

        print(row)