# solver/rules/object_detector.py

import numpy as np

from typing import List
from typing import Tuple
from typing import Set


def get_connected_components(
    grid,
    ignore_color=0
):
    """
    Temukan semua connected components.
    """

    grid = np.array(grid)

    rows, cols = grid.shape

    visited = np.zeros_like(
        grid,
        dtype=bool
    )

    components = []

    def bfs(
        start_r,
        start_c,
        color
    ):

        component = set()

        queue = [

            (start_r, start_c)
        ]

        while queue:

            r, c = queue.pop(0)

            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= cols
            ):

                continue

            if (
                visited[r, c]
                or
                grid[r, c] != color
            ):

                continue

            visited[r, c] = True

            component.add((r, c))

            for dr, dc in [

                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1)
            ]:

                queue.append(

                    (
                        r + dr,
                        c + dc
                    )
                )

        return component

    for r in range(rows):

        for c in range(cols):

            if (

                not visited[r, c]

                and

                grid[r, c] != ignore_color
            ):

                comp = bfs(

                    r,
                    c,
                    grid[r, c]
                )

                if comp:

                    components.append(comp)

    return components


def get_object_info(
    grid,
    component
):
    """
    Extract object info.
    """

    grid = np.array(grid)

    coords = list(component)

    rows_c = [

        r for r, c in coords
    ]

    cols_c = [

        c for r, c in coords
    ]

    min_r, max_r = (
        min(rows_c),
        max(rows_c)
    )

    min_c, max_c = (
        min(cols_c),
        max(cols_c)
    )

    # ============================================
    # BBOX
    # ============================================

    bbox = np.zeros(

        (
            max_r - min_r + 1,
            max_c - min_c + 1
        ),

        dtype=int
    )

    for r, c in coords:

        bbox[
            r - min_r,
            c - min_c
        ] = grid[r, c]

    # ============================================
    # COLOR
    # ============================================

    colors = [

        grid[r, c]
        for r, c in coords
    ]

    color = max(
        set(colors),
        key=colors.count
    )

    # ============================================
    # OBJECT INFO
    # ============================================

    return {

        'coords': component,

        'cells': list(coords),

        'color': int(color),

        'size': len(component),

        'min_r': min_r,
        'max_r': max_r,

        'min_c': min_c,
        'max_c': max_c,

        'height': (
            max_r - min_r + 1
        ),

        'width': (
            max_c - min_c + 1
        ),

        'bbox': bbox.tolist(),

        'center': (

            (min_r + max_r) / 2,

            (min_c + max_c) / 2
        ),
    }


def detect_all_objects(
    grid,
    ignore_color=0
):
    """
    Detect semua object.
    """

    components = get_connected_components(

        grid,
        ignore_color
    )

    objects = [

        get_object_info(
            grid,
            comp
        )

        for comp in components
    ]

    objects.sort(

        key=lambda x: x['size'],

        reverse=True
    )

    return objects


def get_color_regions(grid):
    """
    Region per warna.
    """

    grid = np.array(grid)

    regions = {}

    for color in np.unique(grid):

        coords = set(

            zip(
                *np.where(grid == color)
            )
        )

        regions[int(color)] = coords

    return regions


if __name__ == "__main__":

    test_grid = [

        [0,0,0,0,0],

        [0,1,1,0,0],

        [0,1,0,0,0],

        [0,0,0,2,2],

        [0,0,0,2,2],
    ]

    objects = detect_all_objects(
        test_grid
    )

    print(
        f"\nDetected "
        f"{len(objects)} objects\n"
    )

    for i, obj in enumerate(objects):

        print(

            f"Object {i+1}: "

            f"color={obj['color']} "

            f"size={obj['size']} "

            f"bbox={obj['height']}x{obj['width']}"
        )

        print(
            "cells:",
            obj['cells']
        )

        print()