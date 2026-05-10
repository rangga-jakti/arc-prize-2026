import json
notebook_code = '''
import json
import numpy as np
from collections import defaultdict
BASE = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
TEST_CHALLENGES = f'{BASE}/arc-agi_test_challenges.json'
with open(TEST_CHALLENGES) as f:
    tasks = json.load(f)
print(f"Loaded {len(tasks)} test tasks")
def get_bg(grid):
    g = np.array(grid).flatten()
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])
def fill_object_interior(grid, fill_color):
    g = np.array(grid).copy()
    bg = get_bg(grid)
    rows, cols = g.shape
    outside = np.zeros_like(g, dtype=bool)
    queue = []
    for r in range(rows):
        for c in [0, cols-1]:
            if g[r,c]==bg and not outside[r,c]:
                queue.append((r,c)); outside[r,c]=True
    for c in range(cols):
        for r in [0, rows-1]:
            if g[r,c]==bg and not outside[r,c]:
                queue.append((r,c)); outside[r,c]=True
    while queue:
        r,c = queue.pop(0)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc2 = r+dr, c+dc
            if 0<=nr<rows and 0<=nc2<cols and not outside[nr,nc2] and g[nr,nc2]==bg:
                outside[nr,nc2]=True; queue.append((nr,nc2))
    result = g.copy()
    for r in range(rows):
        for c in range(cols):
            if g[r,c]==bg and not outside[r,c]:
                result[r,c] = fill_color
    return result.tolist()
def build_interior_fill_transforms(train_pairs):
    transforms = {}
    bg = get_bg(train_pairs[0]['input'])
    fill_colors = set()
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if inp.shape != out.shape:
            continue
        changed = (inp == bg) & (out != bg)
        new_colors = np.unique(out[changed]).tolist()
        fill_colors.update([int(c) for c in new_colors])
    for fc in fill_colors:
        def make_fn(fill_c):
            def fn(grid): return fill_object_interior(grid, fill_c)
            return fn
        transforms[f'fill_interior_{fc}'] = make_fn(fc)
    return transforms
def score_transform(fn, train_pairs):
    scores = []
    for pair in train_pairs:
        try:
            pred = fn(pair['input'])
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            p = np.array(pred)
            if p.shape != out.shape:
                scores.append(0.0)
                continue
            scores.append(float(np.mean(p == out)))
        except:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0
def build_all_transforms(task):
    train_pairs = task['train']
    transforms = {}
    # Geometric
    transforms['rotate_90']  = lambda g: np.rot90(np.array(g), 1).tolist()
    transforms['rotate_180'] = lambda g: np.rot90(np.array(g), 2).tolist()
    transforms['rotate_270'] = lambda g: np.rot90(np.array(g), 3).tolist()
    transforms['flip_h']     = lambda g: np.fliplr(np.array(g)).tolist()
    transforms['flip_v']     = lambda g: np.flipud(np.array(g)).tolist()
    transforms['flip_diag']  = lambda g: np.array(g).T.tolist()
    transforms['scale_2x']   = lambda g: np.kron(np.array(g), np.ones((2,2),int)).tolist()
    transforms['scale_3x']   = lambda g: np.kron(np.array(g), np.ones((3,3),int)).tolist()
    # Tile
    for nr in range(1,5):
        for nc in range(1,5):
            if nr==1 and nc==1: continue
            def make_tile(r,c):
                return lambda g: np.tile(np.array(g),(r,c)).tolist()
            transforms[f'tile_{nr}x{nc}'] = make_tile(nr,nc)
    # Color
    all_in, all_out = set(), set()
    for p in train_pairs:
        all_in.update(np.unique(np.array(p['input'])).tolist())
        all_out.update(np.unique(np.array(p['output'])).tolist())
    for src in all_in:
        for tgt in all_out:
            if src==tgt: continue
            def make_rc(s,t):
                def fn(g):
                    arr = np.array(g).copy()
                    arr[arr==s] = t
                    return arr.tolist()
                return fn
            transforms[f'recolor_{src}_to_{tgt}'] = make_rc(src,tgt)
    # Interior fill (FIXED)
    try:
        transforms.update(build_interior_fill_transforms(train_pairs))
    except: pass
    # Gravity
    def gravity_down(grid):
        g = np.array(grid); bg = get_bg(grid)
        result = np.full_like(g, bg)
        for c in range(g.shape[1]):
            col = g[:,c]; non_bg = col[col!=bg]
            result[g.shape[0]-len(non_bg):,c] = non_bg
        return result.tolist()
    transforms['gravity_down'] = gravity_down
    def gravity_up(grid):
        g = np.array(grid); bg = get_bg(grid)
        result = np.full_like(g, bg)
        for c in range(g.shape[1]):
            col = g[:,c]; non_bg = col[col!=bg]
            result[:len(non_bg),c] = non_bg
        return result.tolist()
    transforms['gravity_up'] = gravity_up
    return transforms
def solve_task(task):
    transforms = build_all_transforms(task)
    best_score = -1
    best_pred = task['test'][0]['input']
    for name, fn in transforms.items():
        score = score_transform(fn, task['train'])
        if score > best_score:
            best_score = score
            try:
                best_pred = fn(task['test'][0]['input'])
            except:
                best_pred = task['test'][0]['input']
    return best_pred, best_score
submission = {}
solved = 0
for task_id, task in tasks.items():
    try:
        pred, score = solve_task(task)
        if score == 1.0:
            solved += 1
    except:
        pred = task['test'][0]['input']
    submission[task_id] = [{'attempt_1': pred, 'attempt_2': pred}]
print(f"Tasks with perfect training score: {solved}/240")
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)
print("submission.json saved!")
'''
nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": notebook_code.strip().splitlines(keepends=True)
        }
    ]
}
with open('temp_notebook2/notebooka723374446.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated!")
