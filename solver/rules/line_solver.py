# solver/rules/line_solver.py
import numpy as np
from solver.rules.object_detector import detect_all_objects

def get_bg(grid):
    g = np.array(grid).flatten()
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def complete_diagonal_lines(grid):
    """
    Temukan titik-titik yang membentuk diagonal line,
    isi gap yang kosong di antara mereka.
    Task e376de54.
    """
    g   = np.array(grid).copy()
    bg  = get_bg(grid)
    result = g.copy()
    rows, cols = g.shape

    colors = [int(c) for c in np.unique(g) if c != bg]

    for color in colors:
        positions = list(zip(*np.where(g == color)))
        if len(positions) < 2:
            continue

        # Coba semua pasangan titik, gambar garis diagonal
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                r1, c1 = positions[i]
                r2, c2 = positions[j]

                dr = r2 - r1
                dc = c2 - c1

                # Hanya proses jika diagonal (|dr| == |dc|)
                if abs(dr) != abs(dc) or dr == 0:
                    continue

                steps = abs(dr)
                sr = 1 if dr > 0 else -1
                sc = 1 if dc > 0 else -1

                # Isi semua titik di antara
                for k in range(steps + 1):
                    nr, nc = r1 + k*sr, c1 + k*sc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if result[nr, nc] == bg:
                            result[nr, nc] = color

    return result.tolist()


def complete_straight_lines(grid):
    """
    Temukan titik-titik yang membentuk garis lurus (horizontal/vertikal),
    isi gap di antara mereka.
    """
    g   = np.array(grid).copy()
    bg  = get_bg(grid)
    result = g.copy()
    rows, cols = g.shape

    colors = [int(c) for c in np.unique(g) if c != bg]

    for color in colors:
        positions = list(zip(*np.where(g == color)))
        if len(positions) < 2:
            continue

        rs = [r for r,c in positions]
        cs = [c for r,c in positions]

        # Horizontal line: semua row sama
        row_counts = {}
        for r in rs:
            row_counts[r] = row_counts.get(r, 0) + 1

        for r, cnt in row_counts.items():
            if cnt >= 2:
                row_cs = [c for rr,c in positions if rr == r]
                for c in range(min(row_cs), max(row_cs)+1):
                    if result[r, c] == bg:
                        result[r, c] = color

        # Vertical line: semua col sama
        col_counts = {}
        for c in cs:
            col_counts[c] = col_counts.get(c, 0) + 1

        for c, cnt in col_counts.items():
            if cnt >= 2:
                col_rs = [r for r,cc in positions if cc == c]
                for r in range(min(col_rs), max(col_rs)+1):
                    if result[r, c] == bg:
                        result[r, c] = color

    return result.tolist()


def complete_all_lines(grid):
    """Gabungan: complete diagonal + straight lines."""
    g1 = complete_straight_lines(grid)
    g2 = complete_diagonal_lines(g1)
    return g2


def remove_minority_color(grid):
    """
    Hapus warna yang paling sedikit pixelnya (non-bg),
    ganti dengan background.
    Task 7b80bb43: hapus objek merah (noise) dari background biru.
    """
    g   = np.array(grid).copy()
    bg  = get_bg(grid)
    result = g.copy()

    colors = [int(c) for c in np.unique(g) if c != bg]
    if not colors:
        return grid

    # Hitung pixel per warna
    counts = {c: int(np.sum(g == c)) for c in colors}

    # Hapus warna dengan pixel paling sedikit
    minority = min(counts, key=counts.get)
    result[g == minority] = bg

    return result.tolist()


def remove_second_minority_color(grid):
    """Hapus 2 warna paling sedikit."""
    g   = np.array(grid).copy()
    bg  = get_bg(grid)
    result = g.copy()

    colors = [int(c) for c in np.unique(g) if c != bg]
    if len(colors) < 2:
        return grid

    counts  = {c: int(np.sum(g == c)) for c in colors}
    sorted_c = sorted(counts, key=counts.get)

    for c in sorted_c[:2]:
        result[g == c] = bg

    return result.tolist()


def keep_dominant_color_per_region(grid):
    """
    Untuk setiap connected region, ganti semua pixel dengan
    warna yang paling dominan di region itu.
    Task 135a2760.
    """
    g   = np.array(grid).copy()
    bg  = get_bg(grid)
    result = g.copy()
    rows, cols = g.shape

    visited = np.zeros_like(g, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if visited[r,c]:
                continue

            color = int(g[r,c])
            # BFS untuk temukan region
            region = []
            queue  = [(r,c)]
            visited[r,c] = True

            while queue:
                cr, cc = queue.pop(0)
                region.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr, cc+dc
                    if 0<=nr<rows and 0<=nc<cols and not visited[nr,nc]:
                        if g[nr,nc] == color:
                            visited[nr,nc] = True
                            queue.append((nr,nc))

            # Tidak perlu ubah kalau region kecil
            if len(region) < 3:
                continue

    return result.tolist()


# Registry
LINE_TRANSFORMS = {
    'complete_diagonal_lines' : complete_diagonal_lines,
    'complete_straight_lines' : complete_straight_lines,
    'complete_all_lines'      : complete_all_lines,
    'remove_minority_color'   : remove_minority_color,
    'remove_second_minority'  : remove_second_minority_color,
}