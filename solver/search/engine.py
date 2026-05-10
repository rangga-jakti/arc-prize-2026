# solver/search/engine.py
import numpy as np
from solver.rules.transformations import TRANSFORM_REGISTRY
from solver.rules.object_detector import detect_all_objects, get_color_regions
from solver.search.scorer import score_against_all_pairs, is_perfect, score_prediction
from solver.rules.line_solver import LINE_TRANSFORMS
from solver.rules.object_solver import (
    build_interior_fill_transforms,
    build_recolor_by_size_transforms,
    gravity_down, gravity_up, gravity_left, gravity_right,
    fill_interior_match_border,
    remove_smaller_objects,
    remove_larger_objects,
    build_keep_color_transforms
)
from solver.rules.pattern_solver import PATTERN_TRANSFORMS
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
    train_pairs = task['train']
    test_inputs = task['test']
    all_transforms = {}
    all_transforms.update(TRANSFORM_REGISTRY)
    all_transforms.update(build_color_transforms(train_pairs))
    all_transforms.update(build_tile_transforms(train_pairs))
    # === FIX: tambah interior fill transforms ===
    try:
        interior_transforms = build_interior_fill_transforms(train_pairs)
        all_transforms.update(interior_transforms)
    except Exception as e:
        pass
    # === gravity transforms ===
    for name, fn in [
        ('gravity_down', gravity_down),
        ('gravity_up', gravity_up),
        ('gravity_left', gravity_left),
        ('gravity_right', gravity_right),
        ('fill_interior_match_border', fill_interior_match_border),
        ('remove_smaller_objects', remove_smaller_objects),
        ('remove_larger_objects', remove_larger_objects),
    ]:
        all_transforms[name] = fn
    # === keep color transforms ===
    try:
        all_transforms.update(build_keep_color_transforms(train_pairs))
    except Exception:
        pass
    # === recolor by size ===
    try:
        all_transforms.update(build_recolor_by_size_transforms(train_pairs))
    except Exception:
        pass
    # === pattern transforms ===
    all_transforms.update(PATTERN_TRANSFORMS)
    # === line transforms ===
    all_transforms.update(LINE_TRANSFORMS)
    # === rectangle transforms ===
    all_transforms['complete_rectangle_corners'] = complete_rectangle_corners
    all_transforms['complete_rectangle_corners_dominant'] = complete_rectangle_corners_dominant
    all_transforms['draw_rectangle_border'] = draw_rectangle_border
    # === dynamic transforms ===
    try:
        from solver.search.dynamic_generator import build_dynamic_transforms
        dynamic = build_dynamic_transforms(task)
        all_transforms.update(dynamic)
    except Exception as e:
        pass
    results = []
    for name, fn in all_transforms.items():
        score = score_against_all_pairs(fn, train_pairs)
        if score <= 0:
            continue
        predictions = []
        for test in test_inputs:
            try:
                pred = fn(test['input'])
                predictions.append(pred)
            except Exception:
                predictions.append(test['input'])
        results.append({
            'transform': name,
            'score': score,
            'predictions': predictions,
        })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results
def solve_task(task, top_k=3):
    from solver.search.chainer import search_chained_transforms
    single_results = search_single_transforms(task)
    if not single_results:
        return [{'transform': 'identity', 'score': 0.0, 'predictions': [t['input'] for t in task['test']]}]
    best_score = single_results[0]['score']
    if best_score < 1.0:
        from solver.rules.color_solver import build_multi_recolor_transforms, build_geometric_plus_color
        train_pairs = task['train']
        base = {}
        base.update(TRANSFORM_REGISTRY)
        base.update(build_color_transforms(train_pairs))
        try:
            base.update(build_multi_recolor_transforms(train_pairs))
        except Exception:
            pass
        chain_results = search_chained_transforms(task, base, top_singles=8)
        all_results = single_results + chain_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    return single_results[:top_k]
def evaluate_solver(tasks, max_tasks=100):
    perfect = 0
    partial = 0
    failed = 0
    total = 0
    task_ids = list(tasks.keys())[:max_tasks]
    for tid in task_ids:
        task = tasks[tid]
        solutions = task.get('test_solutions', [])
        if not solutions:
            continue
        results = solve_task(task, top_k=1)
        best = results[0]
        for i, sol in enumerate(solutions):
            total += 1
            pred = best['predictions'][i] if i < len(best['predictions']) else []
            s = score_prediction(pred, sol)
            if s == 1.0:
                perfect += 1
            elif s > 0:
                partial += 1
            else:
                failed += 1
    print(f"Perfect: {perfect}/{total} ({perfect/total*100:.1f}%)")
    print(f"Partial: {partial}/{total} ({partial/total*100:.1f}%)")
    print(f"Failed:  {failed}/{total} ({failed/total*100:.1f}%)")
    return {'perfect': perfect, 'partial': partial, 'failed': failed, 'total': total}
