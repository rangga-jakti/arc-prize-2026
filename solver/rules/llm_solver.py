# solver/rules/llm_solver.py
import re
import time
from groq import Groq

def grid_to_str(grid):
    return '\n'.join(' '.join(map(str, row)) for row in grid)

def str_to_grid(text):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    grid = []
    for line in lines:
        nums = re.findall(r'\d+', line)
        if nums:
            grid.append([int(n) for n in nums])
    return grid if grid else None

def validate_grid(grid, expected_shape=None):
    if not grid or not isinstance(grid, list):
        return False
    if not all(isinstance(row, list) and len(row) > 0 for row in grid):
        return False
    if len(set(len(row) for row in grid)) != 1:
        return False
    if expected_shape:
        if len(grid) != expected_shape[0]: return False
        if len(grid[0]) != expected_shape[1]: return False
    return True
def make_arc_prompt(task):
    # Hitung expected output size
    test_inp = task['test'][0]['input']
    h = len(test_inp)
    w = len(test_inp[0])

    # Cek apakah output size berubah di training
    out_shapes = [(len(p['output']), len(p['output'][0])) for p in task['train']]
    in_shapes  = [(len(p['input']),  len(p['input'][0]))  for p in task['train']]

    if len(set(out_shapes)) == 1 and out_shapes[0] != in_shapes[0]:
        oh, ow = out_shapes[0]
        size_hint = f"Output must be exactly {oh} rows x {ow} cols."
    else:
        size_hint = f"Output must be exactly {h} rows x {w} cols."

    prompt = f"""Solve this ARC puzzle. Find the pattern from examples and apply to test.

STRICT OUTPUT FORMAT:
- Output ONLY digits separated by spaces
- Exactly one row per line
- {size_hint}
- NO explanation, NO markdown, NO extra text whatsoever

"""
    for i, pair in enumerate(task['train']):
        prompt += f"Example {i+1}:\nIN:\n{grid_to_str(pair['input'])}\nOUT:\n{grid_to_str(pair['output'])}\n\n"

    prompt += f"Test:\nIN:\n{grid_to_str(task['test'][0]['input'])}\nOUT:\n"
    return prompt
def make_arc_prompt(task):
    # Hitung expected output size
    test_inp = task['test'][0]['input']
    h = len(test_inp)
    w = len(test_inp[0])

    # Cek apakah output size berubah di training
    out_shapes = [(len(p['output']), len(p['output'][0])) for p in task['train']]
    in_shapes  = [(len(p['input']),  len(p['input'][0]))  for p in task['train']]

    if len(set(out_shapes)) == 1 and out_shapes[0] != in_shapes[0]:
        oh, ow = out_shapes[0]
        size_hint = f"Output must be exactly {oh} rows x {ow} cols."
    else:
        size_hint = f"Output must be exactly {h} rows x {w} cols."

    prompt = f"""Solve this ARC puzzle. Find the pattern from examples and apply to test.

STRICT OUTPUT FORMAT:
- Output ONLY digits separated by spaces
- Exactly one row per line
- {size_hint}
- NO explanation, NO markdown, NO extra text whatsoever

"""
    for i, pair in enumerate(task['train']):
        prompt += f"Example {i+1}:\nIN:\n{grid_to_str(pair['input'])}\nOUT:\n{grid_to_str(pair['output'])}\n\n"

    prompt += f"Test:\nIN:\n{grid_to_str(task['test'][0]['input'])}\nOUT:\n"
    return prompt


def llm_solve_task(task, client, model="llama-3.3-70b-versatile", max_retries=3):
    test_inp = task['test'][0]['input']
    expected_shape = (len(test_inp), len(test_inp[0]))

    # Cek apakah output shape berbeda dari input
    out_shapes = [(len(p['output']), len(p['output'][0])) for p in task['train']]
    if len(set(out_shapes)) == 1:
        expected_shape = out_shapes[0]

    prompt = make_arc_prompt(task)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()

            # Coba parse
            grid = str_to_grid(text)

            if grid and validate_grid(grid, expected_shape):
                return grid

            # Kalau shape salah, coba trim/fix
            if grid and validate_grid(grid):
                # Ambil baris yang sesuai expected
                if len(grid) >= expected_shape[0]:
                    trimmed = [row[:expected_shape[1]] for row in grid[:expected_shape[0]]]
                    if validate_grid(trimmed, expected_shape):
                        return trimmed

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 15 * (attempt + 1)
                print(f"  Rate limit, tunggu {wait}s...")
                time.sleep(wait)
            elif "401" in err:
                print(f"  Auth error — cek API key!")
                return None
            else:
                print(f"  Error: {e}")

        time.sleep(1)

    return None