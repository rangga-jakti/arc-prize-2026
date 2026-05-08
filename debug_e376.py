
# debug_e376.py
import numpy as np
from src.utils.data_loader import load_evaluation_data
from solver.search.scorer import score_prediction

tasks = load_evaluation_data()
task  = tasks['e376de54']
sol   = np.array(task['test_solutions'][0])
inp   = np.array(task['test'][0]['input'])

print("=== Task e376de54 ===")
print(f"Input shape : {inp.shape}")
print(f"Sol shape   : {sol.shape}")
print(f"Input colors : {sorted(np.unique(inp).tolist())}")
print(f"Sol colors   : {sorted(np.unique(sol).tolist())}")

# Lihat training pairs
for i, pair in enumerate(task['train']):
    p_inp = np.array(pair['input'])
    p_out = np.array(pair['output'])
    diff  = p_inp != p_out
    print(f"\nPair {i}: {np.sum(diff)} pixels changed")
    print(f"  In colors : {sorted(np.unique(p_inp).tolist())}")
    print(f"  Out colors: {sorted(np.unique(p_out).tolist())}")
    
    # Pixel yang berubah
    for r,c in list(zip(*np.where(diff)))[:5]:
        print(f"  [{r},{c}]: {p_inp[r,c]}→{p_out[r,c]}")

# Cek 3 pixel yang salah di prediction kita
from solver.search.engine import solve_task
results = solve_task(task, top_k=1)
pred = np.array(results[0]['predictions'][0])
diff = pred != sol

print(f"\n=== 3 Wrong pixels ===")
for r,c in zip(*np.where(diff)):
    print(f"[{r},{c}]: pred={pred[r,c]} sol={sol[r,c]} inp={inp[r,c]}")
    # Lihat area sekitar
    r1,r2 = max(0,r-2), min(sol.shape[0],r+3)
    c1,c2 = max(0,c-2), min(sol.shape[1],c+3)
    print(f"  Sol area:\n{sol[r1:r2,c1:c2]}")
    print(f"  Pred area:\n{pred[r1:r2,c1:c2]}")
    print(f"  Input area:\n{inp[r1:r2,c1:c2]}")