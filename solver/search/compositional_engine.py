# solver/search/compositional_engine.py

import itertools
import numpy as np

from solver.rules.transformations import TRANSFORM_REGISTRY
from solver.search.scorer import score_prediction


def apply_chain(grid, chain):
    """
    Apply sequence of transforms ke grid.
    """
    result = grid

    for transform_name, transform_fn in chain:
        result = transform_fn(result)

    return result


def evaluate_chain(chain, train_pairs):
    """
    Score satu chain transform terhadap semua train pairs.
    """
    scores = []

    for pair in train_pairs:

        try:
            pred = apply_chain(pair['input'], chain)

            score = score_prediction(
                pred,
                pair['output']
            )

            scores.append(score)

        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores)


def search_compositional(task,
                         max_depth=2,
                         top_k=10):
    """
    Cari kombinasi transform terbaik.
    """

    train_pairs = task['train']
    test_inputs = task['test']

    transform_items = list(TRANSFORM_REGISTRY.items())

    results = []

    # depth = panjang chain
    for depth in range(1, max_depth + 1):

        print(f"Searching depth {depth}...")

        chains = itertools.product(
            transform_items,
            repeat=depth
        )

        for chain in chains:

            score = evaluate_chain(
                chain,
                train_pairs
            )

            if score <= 0:
                continue

            predictions = []

            for test in test_inputs:

                try:
                    pred = apply_chain(
                        test['input'],
                        chain
                    )

                    predictions.append(pred)

                except Exception:
                    predictions.append(
                        test['input']
                    )

            results.append({
                'chain'       : [x[0] for x in chain],
                'score'       : score,
                'predictions' : predictions,
            })

    results.sort(
        key=lambda x: x['score'],
        reverse=True
    )

    return results[:top_k]