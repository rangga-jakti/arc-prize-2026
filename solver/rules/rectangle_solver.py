
# solver/rules/rectangle_solver.py
import numpy as np
from solver.rules.object_detector import detect_all_objects

def get_background_color(grid):
    g = np.array(grid).flatten()
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def complete_rectangle_corners(grid):
    """
    Temukan titik-titik berwarna yang membentuk rectangle (L-shape atau 3 corners),
    lalu isi corner ke-4 yang kosong dengan warna yang sama.
    Task 11852cab.
    """
    g      = np.array(grid).copy()
    bg     = get_background_color(grid)
    result = g.copy()
    rows, cols = g.shape

    # Kumpulkan semua warna non-bg
    colors = [int(c) for c in np.unique(g) if c != bg]

    for color in colors:
        # Posisi semua pixel warna ini
        positions = list(zip(*np.where(g == color)))
        if len(positions) < 2:
            continue

        rs = [r for r,c in positions]
        cs = [c for r,c in positions]

        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        # 4 pojok rectangle
        corners = [
            (min_r, min_c),
            (min_r, max_c),
            (max_r, min_c),
            (max_r, max_c),
        ]

        # Isi pojok yang kosong (bg)
        for r, c in corners:
            if 0 <= r < rows and 0 <= c < cols:
                if result[r, c] == bg:
                    result[r, c] = color

    return result.tolist()


def complete_rectangle_corners_dominant(grid):
    """
    Versi: hanya isi pojok dengan warna DOMINAN (paling banyak).
    """
    g   = np.array(grid).copy()
    bg  = get_background_color(grid)
    result = g.copy()
    rows, cols = g.shape

    colors = [int(c) for c in np.unique(g) if c != bg]
    if not colors:
        return grid

    # Hitung jumlah pixel tiap warna
    counts = {c: int(np.sum(g == c)) for c in colors}
    dominant = max(counts, key=counts.get)

    positions = list(zip(*np.where(g == dominant)))
    rs = [r for r,c in positions]
    cs = [c for r,c in positions]

    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    corners = [
        (min_r, min_c), (min_r, max_c),
        (max_r, min_c), (max_r, max_c),
    ]
    for r, c in corners:
        if 0 <= r < rows and 0 <= c < cols:
            if result[r, c] == bg:
                result[r, c] = dominant

    return result.tolist()


def draw_rectangle_border(grid):
    """
    Dari titik-titik non-bg, gambar border rectangle penuh.
    """
    g   = np.array(grid).copy()
    bg  = get_background_color(grid)
    result = g.copy()
    rows, cols = g.shape

    colors = [int(c) for c in np.unique(g) if c != bg]

    for color in colors:
        positions = list(zip(*np.where(g == color)))
        if len(positions) < 2:
            continue

        rs = [r for r,c in positions]
        cs = [c for r,c in positions]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        # Gambar 4 sisi rectangle
        for r in range(min_r, max_r+1):
            if result[r, min_c] == bg: result[r, min_c] = color
            if result[r, max_c] == bg: result[r, max_c] = color
        for c in range(min_c, max_c+1):
            if result[min_r, c] == bg: result[min_r, c] = color
            if result[max_r, c] == bg: result[max_r, c] = color

    return result.tolist()