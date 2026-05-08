
# solver/search/object_transform_miner.py

import numpy as np

from solver.rules.object_detector import (
    detect_all_objects
)


def object_signature(obj):

    return {

        'color'  : int(obj['color']),

        'size'   : int(obj['size']),

        'height' : int(obj['height']),

        'width'  : int(obj['width']),
    }


def analyze_object_changes(inp, out):

    in_objs = detect_all_objects(
        inp,
        ignore_color=0
    )

    out_objs = detect_all_objects(
        out,
        ignore_color=0
    )

    input_sigs = [
        object_signature(o)
        for o in in_objs
    ]

    output_sigs = [
        object_signature(o)
        for o in out_objs
    ]

    added = []
    removed = []

    for o in output_sigs:

        if o not in input_sigs:
            added.append(o)

    for o in input_sigs:

        if o not in output_sigs:
            removed.append(o)

    return {

        'input_count'  : len(input_sigs),

        'output_count' : len(output_sigs),

        'added_objects': added,

        'removed_objects': removed,
    }


if __name__ == "__main__":

    from src.utils.data_loader import (
        load_training_data
    )

    tasks = load_training_data()

    task_id = '00d62c1b'

    pair = tasks[task_id]['train'][0]

    result = analyze_object_changes(

        pair['input'],
        pair['output']
    )

    print("\nOBJECT TRANSFORM ANALYSIS\n")

    print("="*50)

    for k, v in result.items():

        print(f"\n{k}:")
        print(v)