# solver/search/heuristics.py

from solver.search.primitive_miner import (
    analyze_task
)


def suggest_operations(task):

    analysis = analyze_task(task)

    suggested = set()

    # =========================================================
    # SIZE ANALYSIS
    # =========================================================

    for a in analysis:

        size = a['size_info']

        if not size['same_shape']:

            scale_r = size['scale_row']
            scale_c = size['scale_col']

            # SCALE / TILE DETECTION
            if scale_r == 2 and scale_c == 2:

                suggested.update([

                    'TILE2',
                    'SCALE2',
                ])

            elif scale_r == 3 and scale_c == 3:

                suggested.update([

                    'TILE3',
                    'SCALE3',
                    'ROW_ALT_TILE3',
                ])

    # =========================================================
    # ROW ALTERNATING PATTERN
    # =========================================================

    for a in analysis:

        row_pattern = a.get(
            'row_pattern',
            {}
        )

        if row_pattern.get(
            'row_alternating',
            False
        ):

            suggested.add(
                'ROW_ALT_TILE3'
            )

    # =========================================================
    # TILING PATTERN
    # =========================================================

    for a in analysis:

        tiling = a.get(
            'tiling_info',
            {}
        )

        if tiling.get(
            'normal_tiling',
            False
        ):

            if (
                tiling['tile_rows'] == 2
            ):

                suggested.add(
                    'TILE2'
                )

            elif (
                tiling['tile_rows'] == 3
            ):

                suggested.add(
                    'TILE3'
                )

    # =========================================================
    # FALLBACK GEOMETRIC OPS
    # =========================================================

    suggested.update([

        'ROT90',
        'ROT180',
        'ROT270',

        'FLIP_H',
        'FLIP_V',
    ])

    return list(suggested)


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00576224'

    task = tasks[task_id]

    ops = suggest_operations(task)

    print("\nSUGGESTED OPS\n")

    for op in ops:

        print("-", op)