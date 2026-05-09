# solver/rules/transformations.py

import numpy as np
from typing import Dict


def to_np(grid):
    return np.array(grid, dtype=int)


# ============================================================
# GEOMETRIC TRANSFORMATIONS
# ============================================================

def rotate_90(grid):
    return np.rot90(to_np(grid), k=1).tolist()


def rotate_180(grid):
    return np.rot90(to_np(grid), k=2).tolist()


def rotate_270(grid):
    return np.rot90(to_np(grid), k=3).tolist()


def flip_horizontal(grid):
    return np.fliplr(to_np(grid)).tolist()


def flip_vertical(grid):
    return np.flipud(to_np(grid)).tolist()


def flip_diagonal(grid):
    return to_np(grid).T.tolist()


def tile_repeat(grid, n_rows=3, n_cols=3):
    return np.tile(
        to_np(grid),
        (n_rows, n_cols)
    ).tolist()


def crop(grid, r1, c1, r2, c2):
    return to_np(grid)[r1:r2, c1:c2].tolist()


def scale_up(grid, factor=2):

    return np.kron(
        to_np(grid),
        np.ones((factor, factor), dtype=int)
    ).tolist()


def scale_down(grid, factor=2):

    g = to_np(grid)

    return g[::factor, ::factor].tolist()


# ============================================================
# COLOR TRANSFORMATIONS
# ============================================================

def recolor(grid, color_map: Dict[int, int]):

    g = to_np(grid).copy()

    result = g.copy()

    for old, new in color_map.items():

        result[g == old] = new

    return result.tolist()


def swap_colors(grid, color_a: int, color_b: int):

    return recolor(
        grid,
        {
            color_a: color_b,
            color_b: color_a
        }
    )


def fill_background(
    grid,
    new_bg: int,
    old_bg: int = 0
):

    return recolor(
        grid,
        {old_bg: new_bg}
    )


def invert_colors(grid, max_color=9):

    g = to_np(grid)

    return (max_color - g).tolist()


def most_common_color(grid, exclude=None):

    g = to_np(grid).flatten()

    if exclude is not None:
        g = g[g != exclude]

    values, counts = np.unique(
        g,
        return_counts=True
    )

    return int(values[np.argmax(counts)])


def least_common_color(grid, exclude=None):

    g = to_np(grid).flatten()

    if exclude is not None:
        g = g[g != exclude]

    values, counts = np.unique(
        g,
        return_counts=True
    )

    return int(values[np.argmin(counts)])


# ============================================================
# PATTERN TRANSFORMATIONS
# ============================================================

def apply_horizontal_symmetry(grid):

    g = to_np(grid)

    mid = g.shape[1] // 2

    result = g.copy()

    result[:, mid:] = np.fliplr(
        g[:, :mid + g.shape[1] % 2]
    )

    return result.tolist()


def apply_vertical_symmetry(grid):

    g = to_np(grid)

    mid = g.shape[0] // 2

    result = g.copy()

    result[mid:, :] = np.flipud(
        g[:mid + g.shape[0] % 2, :]
    )

    return result.tolist()


def fill_enclosed(
    grid,
    fill_color=1,
    background=0
):

    g = to_np(grid).copy()

    rows, cols = g.shape

    visited = np.zeros_like(
        g,
        dtype=bool
    )

    queue = []

    # Border flood fill
    for r in range(rows):

        for c in [0, cols - 1]:

            if (
                g[r, c] == background
                and
                not visited[r, c]
            ):

                queue.append((r, c))

                visited[r, c] = True

    for c in range(cols):

        for r in [0, rows - 1]:

            if (
                g[r, c] == background
                and
                not visited[r, c]
            ):

                queue.append((r, c))

                visited[r, c] = True

    while queue:

        r, c = queue.pop(0)

        for dr, dc in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]:

            nr, nc = r + dr, c + dc

            if (
                0 <= nr < rows
                and
                0 <= nc < cols
            ):

                if (
                    not visited[nr, nc]
                    and
                    g[nr, nc] == background
                ):

                    visited[nr, nc] = True

                    queue.append((nr, nc))

    result = g.copy()

    for r in range(rows):

        for c in range(cols):

            if (
                g[r, c] == background
                and
                not visited[r, c]
            ):

                result[r, c] = fill_color

    return result.tolist()


# ============================================================
# TILING VARIANTS
# ============================================================

def tile_alternating_fliph_by_row(
    grid,
    n_rows=3,
    n_cols=3
):

    g = np.array(grid)

    row_tiles = []

    for r in range(n_rows):

        if r % 2 == 0:

            tile_row = np.hstack(
                [g] * n_cols
            )

        else:

            tile_row = np.hstack(
                [np.fliplr(g)] * n_cols
            )

        row_tiles.append(tile_row)

    return np.vstack(row_tiles).tolist()


def tile_alternating_flipv_by_col(
    grid,
    n_rows=3,
    n_cols=3
):

    g = np.array(grid)

    row_tiles = []

    for r in range(n_rows):

        col_tiles = []

        for c in range(n_cols):

            if c % 2 == 0:

                col_tiles.append(g)

            else:

                col_tiles.append(
                    np.flipud(g)
                )

        row_tiles.append(
            np.hstack(col_tiles)
        )

    return np.vstack(row_tiles).tolist()


def tile_with_alternating_flip_v(
    grid,
    n_rows=3,
    n_cols=3
):

    g = np.array(grid)

    row_tiles = []

    for r in range(n_rows):

        col_tiles = []

        for c in range(n_cols):

            tile = (
                np.flipud(g)
                if r % 2 == 1
                else g
            )

            col_tiles.append(tile)

        row_tiles.append(
            np.hstack(col_tiles)
        )

    return np.vstack(row_tiles).tolist()


def tile_with_alternating_both(
    grid,
    n_rows=3,
    n_cols=3
):

    g = np.array(grid)

    row_tiles = []

    for r in range(n_rows):

        col_tiles = []

        for c in range(n_cols):

            tile = g.copy()

            if r % 2 == 1:
                tile = np.flipud(tile)

            if c % 2 == 1:
                tile = np.fliplr(tile)

            col_tiles.append(tile)

        row_tiles.append(
            np.hstack(col_tiles)
        )

    return np.vstack(row_tiles).tolist()


def tile_with_rotation(
    grid,
    n_rows=2,
    n_cols=2
):

    g = np.array(grid)

    row_tiles = []

    k = 0

    for r in range(n_rows):

        col_tiles = []

        for c in range(n_cols):

            col_tiles.append(
                np.rot90(g, k=k)
            )

            k += 1

        row_tiles.append(
            np.hstack(col_tiles)
        )

    return np.vstack(row_tiles).tolist()


# ============================================================
# REGISTRY
# ============================================================

TRANSFORM_REGISTRY = {

    'rotate_90'       : rotate_90,
    'rotate_180'      : rotate_180,
    'rotate_270'      : rotate_270,

    'flip_horizontal' : flip_horizontal,
    'flip_vertical'   : flip_vertical,
    'flip_diagonal'   : flip_diagonal,

    'tile_2x2' : lambda g:
        tile_repeat(g, 2, 2),

    'tile_3x3' : lambda g:
        tile_repeat(g, 3, 3),

    'scale_up_2' : lambda g:
        scale_up(g, 2),

    'scale_up_3' : lambda g:
        scale_up(g, 3),

    # fill_enclosed disabled - use build_interior_fill_transforms

    'invert_colors' : invert_colors,

    'tile_alt_fliph_row_3x3' : lambda g:
        tile_alternating_fliph_by_row(
            g,
            3,
            3
        ),

    'tile_alt_fliph_row_2x2' : lambda g:
        tile_alternating_fliph_by_row(
            g,
            2,
            2
        ),

    'tile_alt_flipv_col_3x3' : lambda g:
        tile_alternating_flipv_by_col(
            g,
            3,
            3
        ),

    'tile_alt_flipv_col_2x2' : lambda g:
        tile_alternating_flipv_by_col(
            g,
            2,
            2
        ),
}