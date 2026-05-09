# visualize_task.py

import matplotlib.pyplot as plt
import numpy as np

from src.utils.data_loader import (
    load_training_data
)

from solver.search.beam_search import (
    beam_search
)

from solver.dsl import (
    execute_program
)


ARC_COLORS = {

    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def grid_to_rgb(grid):

    rgb = []

    for row in grid:

        rgb_row = []

        for val in row:

            hex_color = ARC_COLORS[
                int(val)
            ]

            rgb_row.append(hex_color)

        rgb.append(rgb_row)

    return rgb


def show_grid(
    ax,
    grid,
    title
):

    grid = np.array(grid)

    rgb = np.zeros(
        (grid.shape[0], grid.shape[1], 3)
    )

    for r in range(grid.shape[0]):

        for c in range(grid.shape[1]):

            color = ARC_COLORS[
                int(grid[r, c])
            ]

            color = color.lstrip('#')

            rgb[r, c] = tuple(

                int(color[i:i+2], 16)/255

                for i in (0, 2, 4)
            )

    ax.imshow(rgb)

    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":

    tasks = load_training_data()

    task_id = '00576224'

    task = tasks[task_id]

    pair = task['train'][0]

    results = beam_search(

        task,

        beam_width=10,

        max_depth=4
    )

    best = results[0]

    pred = execute_program(

        pair['input'],

        best['program']
    )

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(9,3)
    )

    show_grid(
        axs[0],
        pair['input'],
        "Input"
    )

    show_grid(
        axs[1],
        pred,
        "Prediction"
    )

    show_grid(
        axs[2],
        pair['output'],
        "Ground Truth"
    )

    plt.tight_layout()

    plt.savefig(
        "demo_result.png"
    )

    plt.show()

    print("\nSaved:")
    print("demo_result.png")