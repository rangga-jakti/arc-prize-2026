
# experiments/transformation_taxonomy.py
# Katalog lengkap semua jenis transformasi yang ditemukan di ARC dataset

TRANSFORMATION_CATEGORIES = {

    "GEOMETRIC": [
        "rotate_90",        # putar 90 derajat
        "rotate_180",
        "rotate_270",
        "flip_horizontal",  # mirror kiri-kanan
        "flip_vertical",    # mirror atas-bawah
        "flip_diagonal",    # transpose
        "tile_repeat",      # ulangi grid N x M kali  <-- contoh task tadi!
        "crop",             # ambil bagian tertentu
        "scale_up",         # perbesar grid
        "scale_down",       # perkecil grid
    ],

    "COLOR": [
        "recolor_single",   # ganti 1 warna jadi warna lain
        "recolor_map",      # ganti banyak warna sekaligus
        "swap_colors",      # tukar 2 warna
        "fill_background",  # isi background dengan warna tertentu
        "invert_colors",    # balik semua warna
        "color_by_size",    # warnai objek berdasarkan ukurannya
        "color_by_position",# warnai berdasarkan posisi
    ],

    "OBJECT": [
        "detect_objects",       # temukan connected components
        "move_object",          # geser objek
        "copy_object",          # duplikat objek
        "remove_object",        # hapus objek
        "align_objects",        # ratakan posisi objek
        "sort_objects",         # urutkan objek
        "count_objects",        # hitung jumlah objek
        "gravity",              # objek jatuh ke bawah/kiri/kanan
        "bounce",               # objek memantul
    ],

    "PATTERN": [
        "complete_pattern",     # lengkapi pola yang belum selesai
        "extend_pattern",       # lanjutkan pola
        "find_symmetry_axis",   # temukan sumbu simetri
        "apply_symmetry",       # terapkan simetri
        "repeat_motif",         # ulangi motif kecil
        "checkerboard",         # pola papan catur
        "stripe_pattern",       # pola garis
    ],

    "SPATIAL": [
        "inside_outside",       # bedakan dalam/luar bentuk
        "surround",             # objek mengelilingi objek lain
        "connect_dots",         # hubungkan titik-titik
        "draw_line",            # gambar garis
        "draw_rectangle",       # gambar kotak
        "fill_enclosed",        # isi area tertutup
        "distance_transform",   # transformasi berdasarkan jarak
    ],

    "LOGICAL": [
        "and_grids",            # AND dua grid
        "or_grids",             # OR dua grid
        "xor_grids",            # XOR dua grid
        "mask_apply",           # terapkan mask
        "conditional_fill",     # isi berdasarkan kondisi
    ],

    "COUNTING": [
        "count_color",          # hitung pixel per warna
        "size_filter",          # filter objek berdasarkan ukuran
        "frequency_map",        # map frekuensi kemunculan
        "majority_color",       # warna terbanyak
    ],
}

# Estimasi frekuensi di dataset ARC (berdasarkan riset komunitas)
FREQUENCY_ESTIMATE = {
    "GEOMETRIC" : "~35% tasks",
    "COLOR"     : "~25% tasks",
    "OBJECT"    : "~20% tasks",
    "PATTERN"   : "~10% tasks",
    "SPATIAL"   : "~5%  tasks",
    "LOGICAL"   : "~3%  tasks",
    "COUNTING"  : "~2%  tasks",
}

if __name__ == "__main__":
    total = sum(len(v) for v in TRANSFORMATION_CATEGORIES.values())
    print(f"Total transformation types: {total}")
    print()
    for cat, transforms in TRANSFORMATION_CATEGORIES.items():
        freq = FREQUENCY_ESTIMATE.get(cat, "?")
        print(f"[{cat}] ({freq}) — {len(transforms)} transforms:")
        for t in transforms:
            print(f"   • {t}")
        print()