import json
from tqdm import tqdm
from solver.search.engine import solve_task, search_single_transforms
def build_submission():
    with open("data/test/arc-agi_test_challenges.json") as f:
        tasks = json.load(f)
    submission = {}
    solved = 0
    print(f"\nGenerating submission for {len(tasks)} test tasks...\n")
    for task_id, task in tqdm(tasks.items()):
        try:
            results = search_single_transforms(task)
            if results:
                pred = results[0]["predictions"][0]
            else:
                pred = task["test"][0]["input"]
        except Exception as e:
            pred = task["test"][0]["input"]
        submission[task_id] = [{
            "attempt_1": pred,
            "attempt_2": pred,
        }]
    with open("outputs/submissions/submission_v2.json", "w") as f:
        json.dump(submission, f)
    print(f"\nDone! submission_v2.json created with {len(submission)} tasks")
    # verify format
    sample = list(submission.values())[0][0]
    print(f"Format check - attempt_1 type: {type(sample['attempt_1'])}")
    print(f"Format check - first row: {sample['attempt_1'][0] if sample['attempt_1'] else 'empty'}")
if __name__ == "__main__":
    build_submission()
