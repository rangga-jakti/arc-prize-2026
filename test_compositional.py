from src.utils.data_loader import load_training_data
from solver.search.compositional_engine import search_compositional

tasks = load_training_data()

task_id = '00576224'
task = tasks[task_id]

results = search_compositional(
    task,
    max_depth=2,
    top_k=10
)

print("\nTOP RESULTS\n")

for i, r in enumerate(results):

    print(
        f"#{i+1} "
        f"score={r['score']*100:.1f}% "
        f"chain={r['chain']}"
    )