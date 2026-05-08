# solver/rules/color_solver.py
import numpy as np
from itertools import permutations
from solver.search.scorer import score_prediction

def infer_color_mapping(inp, out):
    """
    Coba cari mapping warna dari input ke output secara otomatis.
    Return dict {old_color: new_color} atau None kalau ga ketemu.
    """
    inp = np.array(inp)
    out = np.array(out)

    if inp.shape != out.shape:
        return None

    mapping = {}
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            src = int(inp[r, c])
            tgt = int(out[r, c])
            if src in mapping:
                if mapping[src] != tgt:
                    return None  # inkonsisten
            else:
                mapping[src] = tgt
    return mapping


def apply_color_mapping(grid, mapping):
    """Apply color mapping dict ke grid."""
    g = np.array(grid).copy()
    result = g.copy()
    for src, tgt in mapping.items():
        result[g == src] = tgt
    return result.tolist()


def build_multi_recolor_transforms(train_pairs):
    """
    Cari consistent color mapping di semua training pairs.
    Kalau mapping konsisten di semua pairs → tambah sebagai transform.
    """
    transforms = {}

    if not train_pairs:
        return transforms

    # Cari mapping dari pair pertama
    first_pair = train_pairs[0]
    inp0 = np.array(first_pair['input'])
    out0 = np.array(first_pair['output'])

    if inp0.shape != out0.shape:
        return transforms

    candidate_mapping = infer_color_mapping(inp0, out0)
    if candidate_mapping is None:
        return transforms

    # Validasi mapping ini konsisten di semua pairs lain
    consistent = True
    for pair in train_pairs[1:]:
        m = infer_color_mapping(pair['input'], pair['output'])
        if m != candidate_mapping:
            consistent = False
            break

    if consistent and candidate_mapping:
        # Cek apakah ini bukan identity
        if any(k != v for k, v in candidate_mapping.items()):
            mapping_copy = dict(candidate_mapping)
            name = 'multi_recolor_' + '_'.join(
                f'{k}to{v}' for k, v in sorted(mapping_copy.items())
                if k != v
            )

            def make_fn(m):
                def fn(grid):
                    return apply_color_mapping(grid, m)
                return fn

            transforms[name] = make_fn(mapping_copy)

    return transforms


def build_geometric_plus_color(train_pairs, geom_transforms):
    """
    Coba kombinasi: geometric transform LALU color mapping.
    Ini handle task yang butuh rotate + recolor sekaligus.
    """
    import numpy as np
    transforms = {}

    for geom_name, geom_fn in geom_transforms.items():
        # Apply geom ke semua inputs
        try:
            transformed_pairs = []
            for pair in train_pairs:
                t_input = geom_fn(pair['input'])
                transformed_pairs.append({
                    'input' : t_input,
                    'output': pair['output']
                })

            # Cari color mapping setelah geom transform
            mapping = infer_color_mapping(
                transformed_pairs[0]['input'],
                transformed_pairs[0]['output']
            )

            if mapping is None:
                continue

            # Validasi konsisten
            ok = True
            for p in transformed_pairs[1:]:
                m = infer_color_mapping(p['input'], p['output'])
                if m != mapping:
                    ok = False
                    break

            if ok and any(k != v for k, v in mapping.items()):
                name = f'{geom_name}+recolor'
                m_copy = dict(mapping)
                g_fn   = geom_fn

                def make_combined(gf, mc):
                    def fn(grid):
                        return apply_color_mapping(gf(grid), mc)
                    return fn

                transforms[name] = make_combined(g_fn, m_copy)

        except Exception:
            continue

    return transforms