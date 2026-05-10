import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def extract_compact_object(grid):

    grid = np.array(grid)

    objects = detect_all_objects(grid)

    if not objects:

        return grid.tolist()

    best_obj = None
    best_score = -1

    for obj in objects:

        h = obj['height']
        w = obj['width']

        area = h * w

        density = obj['size'] / area

        # penalize thin lines
        thin_penalty = 0

        if h == 1 or w == 1:

            thin_penalty = 0.5

        score = density - thin_penalty

        if score > best_score:

            best_score = score

            best_obj = obj

    min_r = best_obj['min_r']
    max_r = best_obj['max_r']

    min_c = best_obj['min_c']
    max_c = best_obj['max_c']

    cropped = grid[
        min_r:max_r+1,
        min_c:max_c+1
    ]

    return cropped.tolist()


if __name__ == "__main__":

    test = [

        [0,0,4,0,0],

        [0,0,4,0,0],

        [2,2,0,0,0],

        [2,2,0,0,0],
    ]

    result = extract_compact_object(
        test
    )

    print('\nRESULT\n')

    for row in result:

        print(row)