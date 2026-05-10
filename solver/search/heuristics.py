# solver/search/heuristics.py

from solver.search.primitive_miner import (
    analyze_task
)

from solver.search.op_stats import (

    get_op_score
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
		    'EXTRACT_LARGEST',
		    'EXTRACT_COMPACT',
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

        'MOVE_UP',
        'MOVE_DOWN',
        'MOVE_LEFT',
        'MOVE_RIGHT',

        'COPY_LARGEST_RIGHT',

	'EXTRACT_LARGEST',
        'EXTRACT_COMPACT',

        'RECOLOR_LARGEST',
    ])

    ranked = []

    # ============================================
    # PRIORITIZE OBJECT MOVEMENT
    # ============================================

    for a in analysis:

        size = a['size_info']

        if size['same_shape']:

            ranked.extend([

                'MOVE_RIGHT',
                'MOVE_LEFT',
                'MOVE_UP',
                'MOVE_DOWN',
            ])

    # ============================================
    # PRIORITIZE OBJECT COPYING
    # ============================================

    for a in analysis:

        try:

            inp_h = a['size_info']['input_shape'][0]

            out_h = a['size_info']['output_shape'][0]

            # kalau shape sama / mirip,
            # kemungkinan object manipulation

            if abs(inp_h - out_h) <= 2:

                ranked.extend([

                    'COPY_LARGEST_RIGHT',
		    'EXTRACT_LARGEST',
	 	    'EXTRACT_COMPACT',
                    'MOVE_RIGHT',
                    'MOVE_LEFT',
                    'MOVE_UP',
                    'MOVE_DOWN',

                    'RECOLOR_LARGEST',
                ])

        except:

            pass

    # ============================================
    # PRIORITIZE TILING
    # ============================================

    for a in analysis:

        tiling = a.get(
            'tiling_info',
            {}
        )

        if tiling.get(
            'is_tiling',
            False
        ):

            ranked.extend([

                'ROW_ALT_TILE3',
                'TILE3',
                'SCALE3',
            ])




    # ============================================
    # PRIORITIZE COLOR TRANSFORMS
    # ============================================

    for a in analysis:

        try:

            inp_colors = set(
                a['color_info']['input_colors']
            )

            out_colors = set(
                a['color_info']['output_colors']
            )

            if inp_colors != out_colors:

                ranked.extend([

                    'RECOLOR_LARGEST',
                ])

        except:

            pass

    # ============================================
    # FALLBACK
    # ============================================

    for op in suggested:

        if op not in ranked:

            ranked.append(op)

    # ============================================
    # REMOVE DUPLICATES
    # ============================================

    dedup = []

    for op in ranked:

        if op not in dedup:

            dedup.append(op)

    # ============================================
    # LEARNED OP PRIORITIZATION
    # ============================================

    dedup.sort(

        key=lambda op: get_op_score(op),

        reverse=True
    )

    return dedup


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