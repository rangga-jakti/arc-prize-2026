
# solver/search/scorer.py
import numpy as np

def score_prediction(predicted, expected):
    """
    Hitung seberapa cocok predicted vs expected grid.
    Return nilai 0.0 - 1.0 (1.0 = perfect match).
    """
    pred = np.array(predicted)
    exp  = np.array(expected)

    # Kalau shape beda, langsung 0
    if pred.shape != exp.shape:
        return 0.0

    matches = np.sum(pred == exp)
    total   = exp.size
    return float(matches / total)


def is_perfect(predicted, expected):
    return score_prediction(predicted, expected) == 1.0


def score_against_all_pairs(transform_fn, train_pairs):
    """
    Test satu transformasi terhadap semua training pairs.
    Return rata-rata score.
    """
    scores = []
    for pair in train_pairs:
        try:
            result = transform_fn(pair['input'])
            s = score_prediction(result, pair['output'])
            scores.append(s)
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0