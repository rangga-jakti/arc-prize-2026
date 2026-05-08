# solver/search/chainer.py
import numpy as np
from solver.search.scorer import score_against_all_pairs, score_prediction

def chain_two_transforms(fn1, fn2):
    """Buat fungsi baru: apply fn1 dulu, lalu fn2."""
    def chained(grid):
        return fn2(fn1(grid))
    return chained


def search_chained_transforms(task, base_transforms, top_singles=8):
    """
    Coba kombinasi 2 transformasi secara berurutan.
    Hanya chain top_singles transform terbaik untuk efisiensi.
    
    Strategy:
    - Ambil top N single transforms berdasarkan score
    - Coba semua kombinasi 2-chain dari top N itu
    - Return yang score-nya > single terbaik
    """
    train_pairs = task['train']
    test_inputs = task['test']

    # Score semua single transforms dulu
    single_scores = []
    for name, fn in base_transforms.items():
        score = score_against_all_pairs(fn, train_pairs)
        if score > 0:
            single_scores.append((score, name, fn))

    if not single_scores:
        return []

    single_scores.sort(key=lambda x: x[0], reverse=True)
    best_single_score = single_scores[0][0]

    # Kalau sudah perfect, ga perlu chain
    if best_single_score == 1.0:
        return []

    # Ambil top N untuk di-chain
    top_n = single_scores[:top_singles]

    results = []
    seen = set()

    for score1, name1, fn1 in top_n:
        for score2, name2, fn2 in top_n:
            if name1 == name2:
                continue

            chain_name = f"{name1}→{name2}"
            if chain_name in seen:
                continue
            seen.add(chain_name)

            chained_fn = chain_two_transforms(fn1, fn2)
            try:
                chain_score = score_against_all_pairs(chained_fn, train_pairs)
            except Exception:
                continue

            # Hanya simpan kalau lebih baik dari single terbaik
            if chain_score > best_single_score:
                predictions = []
                for test in test_inputs:
                    try:
                        pred = chained_fn(test['input'])
                        predictions.append(pred)
                    except Exception:
                        predictions.append(test['input'])

                results.append({
                    'transform'  : chain_name,
                    'score'      : chain_score,
                    'predictions': predictions,
                })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results