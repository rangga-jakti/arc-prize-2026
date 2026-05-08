# debug_task.py
import numpy as np
from src.utils.data_loader import load_training_data

tasks = load_training_data()
task  = tasks['00576224']

print("=== TRAIN PAIRS ===")
for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    print(f"\nPair {i+1}:")
    print(f"Input ({inp.shape}):\n{inp}")
    print(f"Output ({out.shape}):\n{out}")

    # Cek rasio ukuran
    rh = out.shape[0] // inp.shape[0]
    rw = out.shape[1] // inp.shape[1]
    print(f"Scale factor: {rh}x{rw}")

    # Cek tiap tile di output
    print("Breakdown tiles:")
    for tr in range(rh):
        for tc in range(rw):
            tile = out[tr*inp.shape[0]:(tr+1)*inp.shape[0],
                       tc*inp.shape[1]:(tc+1)*inp.shape[1]]
            # Bandingkan dengan semua variasi input
            same     = np.array_equal(tile, inp)
            flip_h   = np.array_equal(tile, np.fliplr(inp))
            flip_v   = np.array_equal(tile, np.flipud(inp))
            rot90    = np.array_equal(tile, np.rot90(inp))
            rot180   = np.array_equal(tile, np.rot90(inp, 2))

            variant = "SAME" if same else \
                      "FLIP_H" if flip_h else \
                      "FLIP_V" if flip_v else \
                      "ROT90" if rot90 else \
                      "ROT180" if rot180 else "UNKNOWN"
            print(f"  tile[{tr},{tc}] = {variant}")