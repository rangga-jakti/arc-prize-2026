# solver/search/primitive_miner.py

import numpy as np


def detect_grid_size_change(inp, out):

    inp = np.array(inp)
    out = np.array(out)

    return {
        'input_shape'  : inp.shape,
        'output_shape' : out.shape,
        'same_shape'   : inp.shape == out.shape,
        'scale_row'    : out.shape[0] / inp.shape[0],
        'scale_col'    : out.shape[1] / inp.shape[1],
    }


def detect_color_mapping(inp, out):

    inp = np.array(inp)
    out = np.array(out)

    mapping = {}

    # Kalau ukuran beda,
    # kita belum bisa direct pixel mapping
    if inp.shape != out.shape:

        return {
            'type': 'shape_changed',
            'mapping': None
        }

    for color in np.unique(inp):

        mask = inp == color

        out_colors = out[mask]

        unique = np.unique(out_colors)

        if len(unique) == 1:
            mapping[int(color)] = int(unique[0])

    return {
        'type': 'direct_mapping',
        'mapping': mapping
    }


def detect_tiling_pattern(inp, out):

    inp = np.array(inp)
    out = np.array(out)

    in_h, in_w = inp.shape
    out_h, out_w = out.shape

    # Output harus kelipatan input
    if out_h % in_h != 0 or out_w % in_w != 0:

        return {
            'is_tiling': False
        }

    n_rows = out_h // in_h
    n_cols = out_w // in_w

    perfect_normal = True
    perfect_checker = True

    for r in range(n_rows):

        for c in range(n_cols):

            tile = out[
                r * in_h:(r + 1) * in_h,
                c * in_w:(c + 1) * in_w
            ]

            expected_normal = inp

            expected_checker = (
                np.fliplr(inp)
                if (r + c) % 2 == 1
                else inp
            )

            # Normal tiling
            if not np.array_equal(
                tile,
                expected_normal
            ):
                perfect_normal = False

            # Checkerboard tiling
            if not np.array_equal(
                tile,
                expected_checker
            ):
                perfect_checker = False

    return {
        'is_tiling'      : True,
        'tile_rows'      : n_rows,
        'tile_cols'      : n_cols,
        'normal_tiling'  : perfect_normal,
        'checker_tiling' : perfect_checker,
    }


def detect_row_alternating_pattern(inp, out):

    inp = np.array(inp)
    out = np.array(out)

    in_h, in_w = inp.shape
    out_h, out_w = out.shape

    # Harus kelipatan
    if out_h % in_h != 0 or out_w % in_w != 0:

        return {
            'row_alternating': False
        }

    n_rows = out_h // in_h
    n_cols = out_w // in_w

    perfect = True

    for r in range(n_rows):

        for c in range(n_cols):

            tile = out[
                r * in_h:(r + 1) * in_h,
                c * in_w:(c + 1) * in_w
            ]

            # Row ganjil -> flip horizontal
            if r % 2 == 1:
                expected = np.fliplr(inp)
            else:
                expected = inp

            if not np.array_equal(
                tile,
                expected
            ):
                perfect = False

    return {
        'row_alternating': perfect,
        'tile_rows'      : n_rows,
        'tile_cols'      : n_cols,
    }


def analyze_task(task):

    insights = []

    for i, pair in enumerate(task['train']):

        inp = pair['input']
        out = pair['output']

        size_info = detect_grid_size_change(
            inp,
            out
        )

        color_map = detect_color_mapping(
            inp,
            out
        )

        tiling_info = detect_tiling_pattern(
            inp,
            out
        )

        row_pattern = detect_row_alternating_pattern(
            inp,
            out
        )

        insights.append({

            'pair_index' : i,

            'size_info'  : size_info,

            'color_map'  : color_map,

            'tiling_info': tiling_info,

            'row_pattern': row_pattern,
        })

    return insights


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00576224'

    analysis = analyze_task(
        tasks[task_id]
    )

    print("\nTASK ANALYSIS\n")

    for a in analysis:

        print("=" * 40)

        print(f"PAIR {a['pair_index']}")

        print("\nSIZE:")
        print(a['size_info'])

        print("\nCOLOR MAP:")
        print(a['color_map'])

        print("\nTILING:")
        print(a['tiling_info'])

        print("\nROW PATTERN:")
        print(a['row_pattern'])