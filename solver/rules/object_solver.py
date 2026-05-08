# solver/rules/object_solver.py
import numpy as np
from solver.rules.object_detector import detect_all_objects, get_color_regions

def get_background_color(grid):
    """Warna paling banyak = background."""
    g = np.array(grid).flatten()
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def fill_object_interior(grid, fill_color=None):
    """
    Temukan objek tertutup (enclosed), isi bagian dalamnya.
    Cocok untuk task 00d62c1b — kotak dengan interior kosong.
    """
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    rows, cols = g.shape

    # Flood fill dari luar untuk temukan "luar"
    outside = np.zeros_like(g, dtype=bool)
    queue = []

    for r in range(rows):
        for c in [0, cols-1]:
            if g[r,c] == bg and not outside[r,c]:
                queue.append((r,c))
                outside[r,c] = True
    for c in range(cols):
        for r in [0, rows-1]:
            if g[r,c] == bg and not outside[r,c]:
                queue.append((r,c))
                outside[r,c] = True

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols:
                if not outside[nr,nc] and g[nr,nc] == bg:
                    outside[nr,nc] = True
                    queue.append((nr,nc))

    result = g.copy()
    # Piksel background yang tidak terjangkau dari luar = interior
    for r in range(rows):
        for c in range(cols):
            if g[r,c] == bg and not outside[r,c]:
                if fill_color is None:
                    # Cari warna border terdekat
                    result[r,c] = _nearest_nonbg_color(g, r, c, bg)
                else:
                    result[r,c] = fill_color
    return result.tolist()


def _nearest_nonbg_color(grid, row, col, bg):
    """Cari warna non-bg terdekat dari posisi (row,col)."""
    rows, cols = grid.shape
    for dist in range(1, max(rows, cols)):
        for dr in range(-dist, dist+1):
            for dc in range(-dist, dist+1):
                if abs(dr) == dist or abs(dc) == dist:
                    nr, nc = row+dr, col+dc
                    if 0<=nr<rows and 0<=nc<cols:
                        if grid[nr,nc] != bg:
                            return int(grid[nr,nc])
    return bg


def build_interior_fill_transforms(train_pairs):
    """
    Detect task tipe fill-interior dan buat transform-nya.
    """
    transforms = {}
    bg = get_background_color(train_pairs[0]['input'])

    # Cari warna yang dipakai untuk fill di training output
    fill_colors = set()
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if inp.shape != out.shape:
            continue
        # Piksel yang berubah dari bg → warna lain
        changed = (inp == bg) & (out != bg)
        new_colors = np.unique(out[changed]).tolist()
        fill_colors.update([int(c) for c in new_colors])

    # Buat transform untuk tiap kemungkinan fill color
    for fc in fill_colors:
        name = f'fill_interior_{fc}'
        def make_fn(fill_c):
            def fn(grid):
                return fill_object_interior(grid, fill_color=fill_c)
            return fn
        transforms[name] = make_fn(fc)

    # Juga coba tanpa specify fill color (auto-detect)
    transforms['fill_interior_auto'] = fill_object_interior

    return transforms


def recolor_by_size(grid, size_color_map=None):
    """
    Warnai objek berdasarkan ukurannya.
    size_color_map: {size: color} atau None untuk auto-detect.
    """
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    objects = detect_all_objects(grid, ignore_color=bg)

    result = g.copy()
    for obj in objects:
        if size_color_map and obj['size'] in size_color_map:
            new_color = size_color_map[obj['size']]
            for r, c in obj['coords']:
                result[r, c] = new_color

    return result.tolist()


def build_recolor_by_size_transforms(train_pairs):
    """
    Detect apakah output mewarnai objek berdasarkan size.
    """
    transforms = {}

    # Kumpulkan mapping size→color dari training
    size_maps = []
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if inp.shape != out.shape:
            continue

        bg = get_background_color(pair['input'])
        objects = detect_all_objects(pair['input'], ignore_color=bg)

        size_map = {}
        consistent = True
        for obj in objects:
            # Cek warna objek di output
            out_colors = [int(out[r,c]) for r,c in obj['coords']
                         if out[r,c] != bg]
            if not out_colors:
                continue
            out_color = max(set(out_colors), key=out_colors.count)
            if obj['size'] in size_map and size_map[obj['size']] != out_color:
                consistent = False
                break
            size_map[obj['size']] = out_color

        if consistent and size_map:
            size_maps.append(size_map)

    # Cek konsistensi antar pairs
    if len(size_maps) >= 2 and size_maps[0] == size_maps[1]:
        final_map = size_maps[0]
        name = 'recolor_by_size'
        def make_fn(m):
            def fn(grid):
                return recolor_by_size(grid, size_color_map=m)
            return fn
        transforms[name] = make_fn(final_map)

    return transforms


def gravity_down(grid, bg=0):
    """Semua objek jatuh ke bawah (gravity)."""
    g = np.array(grid)
    result = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]
        non_bg = col[col != bg]
        result[g.shape[0]-len(non_bg):, c] = non_bg
    return result.tolist()


def gravity_up(grid, bg=0):
    g = np.array(grid)
    result = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]
        non_bg = col[col != bg]
        result[:len(non_bg), c] = non_bg
    return result.tolist()


def gravity_left(grid, bg=0):
    g = np.array(grid)
    result = np.full_like(g, bg)
    for r in range(g.shape[0]):
        row = g[r, :]
        non_bg = row[row != bg]
        result[r, :len(non_bg)] = non_bg
    return result.tolist()


def gravity_right(grid, bg=0):
    g = np.array(grid)
    result = np.full_like(g, bg)
    for r in range(g.shape[0]):
        row = g[r, :]
        non_bg = row[row != bg]
        result[r, g.shape[1]-len(non_bg):] = non_bg
    return result.tolist()

def fill_interior_match_border(grid):
    """
    Isi interior objek dengan warna border-nya masing-masing.
    Tiap enclosed region diisi dengan warna dinding yang mengurungnya.
    Cocok untuk task 045e512c.
    """
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    rows, cols = g.shape

    # Temukan semua pixel background yang enclosed
    outside = np.zeros_like(g, dtype=bool)
    queue = []

    for r in range(rows):
        for c in [0, cols-1]:
            if g[r,c] == bg and not outside[r,c]:
                queue.append((r,c))
                outside[r,c] = True
    for c in range(cols):
        for r in [0, rows-1]:
            if g[r,c] == bg and not outside[r,c]:
                queue.append((r,c))
                outside[r,c] = True

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols:
                if not outside[nr,nc] and g[nr,nc] == bg:
                    outside[nr,nc] = True
                    queue.append((nr,nc))

    result = g.copy()
    for r in range(rows):
        for c in range(cols):
            if g[r,c] == bg and not outside[r,c]:
                # Cari warna border terdekat pakai BFS
                fill_color = _find_border_color_bfs(g, r, c, bg, rows, cols)
                result[r,c] = fill_color

    return result.tolist()


def _find_border_color_bfs(grid, start_r, start_c, bg, rows, cols):
    """BFS dari titik interior, cari warna non-bg pertama yang ditemukan."""
    visited = set()
    queue = [(start_r, start_c)]
    visited.add((start_r, start_c))

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited:
                if grid[nr,nc] != bg:
                    return int(grid[nr,nc])
                visited.add((nr,nc))
                queue.append((nr,nc))
    return bg


def remove_smaller_objects(grid):
    """
    Hapus semua objek kecuali yang terbesar.
    Cocok untuk task yang butuh filter by size.
    """
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    objects = detect_all_objects(grid, ignore_color=bg)

    if not objects:
        return grid

    largest = objects[0]  # sudah sorted by size desc
    result = np.full_like(g, bg)
    for r, c in largest['coords']:
        result[r,c] = g[r,c]
    return result.tolist()


def remove_larger_objects(grid):
    """Hapus semua objek kecuali yang terkecil."""
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    objects = detect_all_objects(grid, ignore_color=bg)

    if not objects:
        return grid

    smallest = objects[-1]  # sorted desc, ambil terakhir
    result = np.full_like(g, bg)
    for r, c in smallest['coords']:
        result[r,c] = g[r,c]
    return result.tolist()


def keep_color(grid, target_color):
    """Hapus semua warna kecuali target_color."""
    g = np.array(grid).copy()
    bg = get_background_color(grid)
    result = np.full_like(g, bg)
    result[g == target_color] = target_color
    return result.tolist()


def build_keep_color_transforms(train_pairs):
    """Buat transform keep_color untuk tiap warna non-bg."""
    transforms = {}
    for pair in train_pairs:
        inp = np.array(pair['input'])
        bg  = get_background_color(pair['input'])
        colors = [int(c) for c in np.unique(inp) if c != bg]
        for color in colors:
            name = f'keep_color_{color}'
            def make_fn(col):
                def fn(grid):
                    return keep_color(grid, col)
                return fn
            transforms[name] = make_fn(color)
    return transforms