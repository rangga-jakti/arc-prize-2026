# solver/rules/pattern_solver.py
import numpy as np
from solver.rules.object_detector import detect_all_objects, get_color_regions


def get_background_color(grid):
    g = np.array(grid).flatten()
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


# ── SYMMETRY COMPLETION ──────────────────────────────────────

def complete_horizontal_symmetry(grid):
    """
    Jika grid hampir simetris horizontal, lengkapi sisi yang kurang.
    """
    g = np.array(grid).copy()
    rows, cols = g.shape
    bg = get_background_color(grid)
    result = g.copy()
    mid = cols // 2

    for r in range(rows):
        for c in range(mid):
            mirror_c = cols - 1 - c
            left  = g[r, c]
            right = g[r, mirror_c]
            if left != bg and right == bg:
                result[r, mirror_c] = left
            elif right != bg and left == bg:
                result[r, c] = right
    return result.tolist()


def complete_vertical_symmetry(grid):
    """Lengkapi simetri vertikal."""
    g = np.array(grid).copy()
    rows, cols = g.shape
    bg = get_background_color(grid)
    result = g.copy()
    mid = rows // 2

    for r in range(mid):
        mirror_r = rows - 1 - r
        for c in range(cols):
            top    = g[r, c]
            bottom = g[mirror_r, c]
            if top != bg and bottom == bg:
                result[mirror_r, c] = top
            elif bottom != bg and top == bg:
                result[r, c] = bottom
    return result.tolist()


def complete_diagonal_symmetry(grid):
    """Lengkapi simetri diagonal (transpose)."""
    g = np.array(grid).copy()
    rows, cols = g.shape
    if rows != cols:
        return grid
    bg = get_background_color(grid)
    result = g.copy()

    for r in range(rows):
        for c in range(cols):
            if g[r,c] != bg and g[c,r] == bg:
                result[c,r] = g[r,c]
            elif g[c,r] != bg and g[r,c] == bg:
                result[r,c] = g[c,r]
    return result.tolist()


def complete_point_symmetry(grid):
    """Lengkapi simetri rotasional 180 derajat."""
    g = np.array(grid).copy()
    rows, cols = g.shape
    bg = get_background_color(grid)
    result = g.copy()

    for r in range(rows):
        for c in range(cols):
            mirror_r = rows - 1 - r
            mirror_c = cols - 1 - c
            if g[r,c] != bg and g[mirror_r, mirror_c] == bg:
                result[mirror_r, mirror_c] = g[r,c]
            elif g[mirror_r, mirror_c] != bg and g[r,c] == bg:
                result[r,c] = g[mirror_r, mirror_c]
    return result.tolist()


# ── COLOR COUNTING / FREQUENCY ───────────────────────────────

def keep_most_common_color(grid):
    """Hapus semua warna kecuali yang paling banyak (non-bg)."""
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    flat = g.flatten()
    non_bg = flat[flat != bg]
    if len(non_bg) == 0:
        return grid
    values, counts = np.unique(non_bg, return_counts=True)
    dominant = values[np.argmax(counts)]
    result = np.full_like(g, bg)
    result[g == dominant] = dominant
    return result.tolist()


def keep_least_common_color(grid):
    """Pertahankan hanya warna yang paling jarang (non-bg)."""
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    flat = g.flatten()
    non_bg = flat[flat != bg]
    if len(non_bg) == 0:
        return grid
    values, counts = np.unique(non_bg, return_counts=True)
    rarest = values[np.argmin(counts)]
    result = np.full_like(g, bg)
    result[g == rarest] = rarest
    return result.tolist()


def color_objects_by_size(grid):
    """
    Warnai tiap objek berdasarkan size-rank-nya.
    Objek terbesar = warna 1, kedua = warna 2, dst.
    """
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    objects = detect_all_objects(grid, ignore_color=bg)
    result = np.full_like(g, bg)

    for rank, obj in enumerate(objects):  # sudah sorted by size desc
        color = rank + 1
        for r, c in obj['coords']:
            result[r, c] = color
    return result.tolist()


# ── OUTLINE / BORDER ─────────────────────────────────────────

def draw_outline(grid, outline_color=None):
    """
    Ganti interior objek dengan background, pertahankan hanya border.
    """
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    rows, cols = g.shape
    result = g.copy()

    for r in range(rows):
        for c in range(cols):
            if g[r,c] == bg:
                continue
            # Cek apakah semua tetangga non-bg (= interior)
            neighbors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols:
                    neighbors.append(g[nr,nc])
                else:
                    neighbors.append(bg)  # edge = bg
            if all(n != bg for n in neighbors):
                result[r,c] = bg  # hapus interior
    return result.tolist()


def fill_outline(grid):
    """Kebalikan draw_outline: isi interior objek."""
    from solver.rules.object_solver import fill_object_interior
    return fill_object_interior(grid)


# ── CROP TO CONTENT ──────────────────────────────────────────

def crop_to_content(grid, bg=None):
    """
    Crop grid ke bounding box konten non-background.
    """
    g = np.array(grid)
    if bg is None:
        bg = get_background_color(grid)
    mask = g != bg
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)

    if not np.any(rows_any):
        return grid

    r_min = int(np.argmax(rows_any))
    r_max = int(len(rows_any) - np.argmax(rows_any[::-1]) - 1)
    c_min = int(np.argmax(cols_any))
    c_max = int(len(cols_any) - np.argmax(cols_any[::-1]) - 1)

    return g[r_min:r_max+1, c_min:c_max+1].tolist()


# ── REGISTRY TAMBAHAN ────────────────────────────────────────

PATTERN_TRANSFORMS = {
    'complete_h_symmetry'   : complete_horizontal_symmetry,
    'complete_v_symmetry'   : complete_vertical_symmetry,
    'complete_d_symmetry'   : complete_diagonal_symmetry,
    'complete_point_sym'    : complete_point_symmetry,
    'keep_most_common'      : keep_most_common_color,
    'keep_least_common'     : keep_least_common_color,
    'color_by_size_rank'    : color_objects_by_size,
    'draw_outline'          : draw_outline,
    'fill_outline'          : fill_outline,
    'crop_to_content'       : crop_to_content,
}
