# solver/dsl.py

from solver.rules.transformations import (

    rotate_90,
    rotate_180,
    rotate_270,

    flip_horizontal,
    flip_vertical,

    tile_repeat,
    scale_up,

    tile_alternating_fliph_by_row,
)

from solver.rules.object_ops import (

    move_up,
    move_down,
    move_left,
    move_right,
)

from solver.rules.filter_ops import (

    extract_compact_object,
)

from solver.rules.extract_ops import (

    extract_largest_object,
)

from solver.rules.copy_ops import (

    copy_largest_right,
)

from solver.rules.color_ops import (

    recolor_largest,
)


DSL_OPERATIONS = {

    # ========================================================
    # ROTATIONS
    # ========================================================

    'ROT90':
        lambda g: rotate_90(g),

    'ROT180':
        lambda g: rotate_180(g),

    'ROT270':
        lambda g: rotate_270(g),

    # ========================================================
    # FLIPS
    # ========================================================

    'FLIP_H':
        lambda g: flip_horizontal(g),

    'FLIP_V':
        lambda g: flip_vertical(g),

    # ========================================================
    # TILING
    # ========================================================

    'TILE2':
        lambda g: tile_repeat(
            g,
            2,
            2
        ),

    'TILE3':
        lambda g: tile_repeat(
            g,
            3,
            3
        ),

    # ========================================================
    # SCALING
    # ========================================================

    'SCALE2':
        lambda g: scale_up(
            g,
            2
        ),

    'SCALE3':
        lambda g: scale_up(
            g,
            3
        ),

    # ========================================================
    # SPECIAL PATTERNS
    # ========================================================

    'ROW_ALT_TILE3':

        lambda g:

            tile_alternating_fliph_by_row(

                g,

                3,

                3
            ),

    # ========================================================
    # OBJECT OPS
    # ========================================================

    'MOVE_UP':
        lambda g: move_up(g),

    'MOVE_DOWN':
        lambda g: move_down(g),

    'MOVE_LEFT':
        lambda g: move_left(g),

    'MOVE_RIGHT':
        lambda g: move_right(g),

    # ========================================================
    # COPY OPS
    # ========================================================

    'COPY_LARGEST_RIGHT':
        lambda g: copy_largest_right(g),

    # ========================================================
    # COLOR OPS
    # ========================================================

    'RECOLOR_LARGEST':
        lambda g: recolor_largest(
            g,
            new_color=9
        ),

    # ========================================================
    # EXTRACTION OPS
    # ========================================================

    'EXTRACT_LARGEST':

        lambda g:

            extract_largest_object(g),

'EXTRACT_COMPACT':

    lambda g:

        extract_compact_object(g),

}


def execute_program(
    grid,
    program
):

    result = grid

    for op in program:

        if op not in DSL_OPERATIONS:

            raise ValueError(
                f"Unknown op: {op}"
            )

        result = DSL_OPERATIONS[op](
            result
        )

    return result


if __name__ == "__main__":

    test = [

        [0,0,0,0],

        [0,2,2,0],

        [0,2,2,0],

        [0,0,3,0],
    ]
    program = [

        'EXTRACT_COMPACT'
    ]

    result = execute_program(

        test,

        program
    )

    print("\nPROGRAM:")

    print(program)

    print("\nRESULT:\n")

    for row in result:

        print(row)