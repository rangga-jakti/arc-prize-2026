# solver/search/dynamic_generator.py

import numpy as np

from solver.search.primitive_miner import (
    analyze_task
)


def build_dynamic_transforms(task):

    analysis = analyze_task(task)

    transforms = {}

    # =========================================================
    # ROW ALTERNATING TILING
    # =========================================================

    row_alt_all = all(
        a['row_pattern'].get(
            'row_alternating',
            False
        )
        for a in analysis
    )

    if row_alt_all:

        rows = analysis[0]['row_pattern']['tile_rows']
        cols = analysis[0]['row_pattern']['tile_cols']

        def row_alternating_transform(grid):

            g = np.array(grid)

            row_tiles = []

            for r in range(rows):

                col_tiles = []

                for c in range(cols):

                    if r % 2 == 1:
                        tile = np.fliplr(g)
                    else:
                        tile = g

                    col_tiles.append(tile)

                row_tiles.append(
                    np.hstack(col_tiles)
                )

            return np.vstack(row_tiles).tolist()

        transforms[
            f'dynamic_row_alternating_{rows}x{cols}'
        ] = row_alternating_transform

    # =========================================================
    # NORMAL TILING
    # =========================================================

    normal_tiling = all(
        a['tiling_info'].get(
            'normal_tiling',
            False
        )
        for a in analysis
    )

    if normal_tiling:

        rows = analysis[0]['tiling_info']['tile_rows']
        cols = analysis[0]['tiling_info']['tile_cols']

        def normal_tile_transform(grid):

            return np.tile(
                np.array(grid),
                (rows, cols)
            ).tolist()

        transforms[
            f'dynamic_tile_{rows}x{cols}'
        ] = normal_tile_transform

    return transforms


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00576224'

    task = tasks[task_id]

    dynamic = build_dynamic_transforms(
        task
    )

    print("\nDYNAMIC TRANSFORMS\n")

    for name in dynamic:

        print(" -", name)

    # Test transform
    fn = list(dynamic.values())[0]

    pred = fn(
        task['test'][0]['input']
    )

    print("\nPREDICTION:\n")

    for row in pred:
        print(row)