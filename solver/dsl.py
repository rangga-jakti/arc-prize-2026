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


DSL_OPERATIONS = {

    'ROT90':
        lambda g: rotate_90(g),

    'ROT180':
        lambda g: rotate_180(g),

    'ROT270':
        lambda g: rotate_270(g),

    'FLIP_H':
        lambda g: flip_horizontal(g),

    'FLIP_V':
        lambda g: flip_vertical(g),

    'TILE2':
        lambda g: tile_repeat(g, 2, 2),

    'TILE3':
        lambda g: tile_repeat(g, 3, 3),

    'SCALE2':
        lambda g: scale_up(g, 2),

    'SCALE3':
        lambda g: scale_up(g, 3),

    'ROW_ALT_TILE3':
        lambda g:
            tile_alternating_fliph_by_row(
                g,
                3,
                3
            ),
}


def execute_program(grid, program):

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

        [1, 2],
        [3, 4]
    ]

    program = [

        'ROW_ALT_TILE3'
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