# solver/search/object_reasoner.py

import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def analyze_objects(grid):

    objects = detect_all_objects(
        grid,
        ignore_color=0
    )

    analysis = []

    for obj in objects:

        analysis.append({

            'color'  : obj['color'],

            'size'   : obj['size'],

            'height' : obj['height'],

            'width'  : obj['width'],

            'center' : obj['center'],

            'bbox'   : obj['bbox'].tolist(),
        })

    return analysis


def compare_input_output_objects(
    inp,
    out
):

    in_objs = analyze_objects(inp)
    out_objs = analyze_objects(out)

    return {

        'input_object_count'  : len(in_objs),

        'output_object_count' : len(out_objs),

        'input_objects'       : in_objs,

        'output_objects'      : out_objs,
    }


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    # Cari task object-based
    task_id = '00d62c1b'

    task = tasks[task_id]

    pair = task['train'][0]

    result = compare_input_output_objects(

        pair['input'],
        pair['output']
    )

    print("\nOBJECT ANALYSIS\n")

    print("="*50)

    print("\nINPUT OBJECTS:\n")

    for i, obj in enumerate(
        result['input_objects']
    ):

        print(f"Object {i+1}")
        print(obj)
        print()

    print("="*50)

    print("\nOUTPUT OBJECTS:\n")

    for i, obj in enumerate(
        result['output_objects']
    ):

        print(f"Object {i+1}")
        print(obj)
        print()