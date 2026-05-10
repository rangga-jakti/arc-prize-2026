import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def extract_largest_object(grid):

    grid = np.array(grid)

    objects = detect_all_objects(grid)

    if not objects:

        return grid.tolist()

    obj = objects[0]

    min_r = obj['min_r']
    max_r = obj['max_r']

    min_c = obj['min_c']
    max_c = obj['max_c']

    cropped = grid[
        min_r:max_r+1,
        min_c:max_c+1
    ]

    return cropped.tolist()


if __name__ == "__main__":

    test = [

        [0,0,0,0,0],
        [0,2,2,0,0],
        [0,2,2,0,0],
        [0,0,0,3,0],
    ]

    result = extract_largest_object(
        test
    )

    print('\nRESULT\n')

    for row in result:

        print(row)