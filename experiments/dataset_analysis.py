# experiments/dataset_analysis.py
import numpy as np
from collections import Counter
from src.utils.data_loader import load_training_data

tasks = load_training_data()

size_changes   = []  # apakah ukuran grid berubah?
color_counts   = []  # berapa warna dipakai per task?
grid_sizes     = []  # ukuran grid input

for tid, task in tasks.items():
    for pair in task['train']:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Apakah ukuran berubah?
        size_changes.append(inp.shape != out.shape)

        # Berapa warna unik di input?
        color_counts.append(len(np.unique(inp)))

        grid_sizes.append(inp.shape)

size_change_pct = sum(size_changes) / len(size_changes) * 100
avg_colors      = sum(color_counts) / len(color_counts)

print("=" * 45)
print("      DATASET ANALYSIS - ARC Training")
print("=" * 45)
print(f"Total tasks          : {len(tasks)}")
print(f"Total train pairs    : {len(size_changes)}")
print()
print(f"Grid size BERUBAH    : {size_change_pct:.1f}% pairs")
print(f"Grid size SAMA       : {100-size_change_pct:.1f}% pairs")
print()
print(f"Rata2 warna/grid     : {avg_colors:.1f}")
print(f"Max warna/grid       : {max(color_counts)}")
print(f"Min warna/grid       : {min(color_counts)}")
print()

# Grid size terbanyak
size_counter = Counter(grid_sizes)
print("Top 10 ukuran grid input:")
for size, count in size_counter.most_common(10):
    print(f"   {size[0]}x{size[1]:2d}  →  {count} kali")

print()
print("Insight untuk solver:")
if size_change_pct > 50:
    print("  → Mayoritas task MENGUBAH ukuran grid")
    print("  → Prioritaskan: scaling, tiling, cropping")
else:
    print("  → Mayoritas task TIDAK mengubah ukuran grid")
    print("  → Prioritaskan: recoloring, object ops, pattern")