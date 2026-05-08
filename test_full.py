
# test_full.py
from src.utils.data_loader import load_training_data
from solver.search.engine import evaluate_solver
import time

tasks = load_training_data()

print("Evaluasi FULL 1000 tasks...")
print("(Ini butuh beberapa menit, sabar ya)\n")

start = time.time()
result = evaluate_solver(tasks, max_tasks=1000)
elapsed = time.time() - start

print(f"\nWaktu: {elapsed:.1f} detik")
print(f"Speed: {1000/elapsed:.1f} tasks/detik")