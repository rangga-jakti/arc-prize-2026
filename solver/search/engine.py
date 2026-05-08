
# solver/search/engine.py
import numpy as np
from solver.rules.transformations import TRANSFORM_REGISTRY
from solver.rules.object_detector import detect_all_objects, get_color_regions
from solver.search.scorer import score_against_all_pairs, is_perfect, score_prediction
from solver.rules.line_solver import LINE_TRANSFORMS
from solver.rules.object_solver import (
    build_interior_fill_transforms,
    build_recolor_by_size_transforms,
    gravity_down, gravity_up, gravity_left, gravity_right
)
from solver.rules.pattern_solver import PATTERN_TRANSFORMS 
from solver.rules.object_solver import (
    build_interior_fill_transforms,
    build_recolor_by_size_transforms,
    gravity_down, gravity_up, gravity_left, gravity_right,
    fill_interior_match_border,
    remove_smaller_objects,
    remove_larger_objects,
    build_keep_color_transforms
)
from solver.rules.rectangle_solver import (
    complete_rectangle_corners,
    complete_rectangle_corners_dominant,
    draw_rectangle_border
)

def build_color_transforms(train_pairs):
    transforms = {}
    all_input_colors  = set()
    all_output_colors = set()
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        all_input_colors.update(np.unique(inp).tolist())
        all_output_colors.update(np.unique(out).tolist())

    for src in all_input_colors:
        for tgt in all_output_colors:
            if src == tgt:
                continue
            name = f'recolor_{src}_to_{tgt}'
            def make_recolor(s, t):
                def fn(grid):
                    g = np.array(grid).copy()
                    result = g.copy()
                    result[g == s] = t
                    return result.tolist()
                return fn
            transforms[name] = make_recolor(src, tgt)
    return transforms


def build_tile_transforms(train_pairs):
    transforms = {}
    for nr in range(1, 5):
        for nc in range(1, 5):
            if nr == 1 and nc == 1:
                continue
            name = f'tile_{nr}x{nc}'
            def make_tile(r, c):
                def fn(grid):
                    return np.tile(np.array(grid), (r, c)).tolist()
                return fn
            transforms[name] = make_tile(nr, nc)
    return transforms


def search_single_transforms(task):
    """
    Hybrid search:
    - static transforms
    - dynamic generated transforms
    """

    train_pairs = task['train']
    test_inputs = task['test']

    # =====================================================
    # STATIC TRANSFORMS
    # =====================================================

    all_transforms = {}

    all_transforms.update(
        TRANSFORM_REGISTRY
    )

    all_transforms.update(
        build_color_transforms(train_pairs)
    )

    all_transforms.update(
        build_tile_transforms(train_pairs)
    )

    # =====================================================
    # DYNAMIC TRANSFORMS
    # =====================================================

    try:

        from solver.search.dynamic_generator import (
            build_dynamic_transforms
        )

        dynamic = build_dynamic_transforms(
            task
        )

        all_transforms.update(dynamic)

    except Exception as e:

        print(
            "[WARN] Dynamic generator failed:",
            e
        )

    # =====================================================
    # SEARCH
    # =====================================================

    results = []

    for name, fn in all_transforms.items():

        score = score_against_all_pairs(
            fn,
            train_pairs
        )

        if score <= 0:
            continue

        predictions = []

        for test in test_inputs:

            try:

                pred = fn(
                    test['input']
                )

                predictions.append(pred)

            except Exception:

                predictions.append(
                    test['input']
                )

        results.append({

            'transform' : name,

            'score'     : score,

            'predictions': predictions,
        })

    results.sort(
        key=lambda x: x['score'],
        reverse=True
    )

    return results

def solve_task(task, top_k=3):
    from solver.search.chainer import search_chained_transforms

    # Step 1: cari single transforms
    single_results = search_single_transforms(task)

    if not single_results:
        return [{
            'transform'  : 'identity',
            'score'      : 0.0,
            'predictions': [t['input'] for t in task['test']],
        }]

    # Step 2: kalau belum perfect, coba chain
    best_score = single_results[0]['score']
    if best_score < 1.0:
        # Kumpulkan base transforms untuk chaining
        from solver.rules.color_solver import (
            build_multi_recolor_transforms,
            build_geometric_plus_color
        )
        train_pairs = task['train']
        base = {}
        base.update(TRANSFORM_REGISTRY)
        base.update(build_color_transforms(train_pairs))
        base.update(build_multi_recolor_transforms(train_pairs))

        chain_results = search_chained_transforms(task, base, top_singles=8)

        # Gabungkan hasil single + chain
        all_results = single_results + chain_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]

    return single_results[:top_k]


def evaluate_solver(tasks, max_tasks=100):
    perfect  = 0
    partial  = 0
    failed   = 0
    total    = 0
    task_ids = list(tasks.keys())[:max_tasks]

    for tid in task_ids:
        task      = tasks[tid]
        solutions = task.get('test_solutions', [])
        if not solutions:
            continue

        results = solve_task(task, top_k=1)
        best    = results[0]

        for i, sol in enumerate(solutions):
            total += 1
            pred   = best['predictions'][i] if i < len(best['predictions']) else []
            s      = score_prediction(pred, sol)

            if s == 1.0:
                perfect += 1
            elif s > 0:
                partial += 1
            else:
                failed += 1

    print("=" * 45)
    print("       SOLVER EVALUATION RESULTS")
    print("=" * 45)
    print(f"Tasks evaluated : {len(task_ids)}")
    print(f"Perfect match   : {perfect}/{total} ({perfect/total*100:.1f}%)")
    print(f"Partial match   : {partial}/{total} ({partial/total*100:.1f}%)")
    print(f"Failed (0%)     : {failed}/{total}  ({failed/total*100:.1f}%)")
    print(f"Overall score   : {(perfect/total)*100:.2f}%")
    print("=" * 45)

    return {
        'perfect': perfect,
        'partial': partial,
        'failed' : failed,
        'total'  : total,
    }