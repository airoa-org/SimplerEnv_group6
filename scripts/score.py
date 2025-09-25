import argparse
import csv
import json
import os
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Compute episode scores from results.json")
    parser.add_argument(
        "--result-path",
        required=True,
        help="Directory that contains results.json and where scores.csv will be written"
    )
    parser.add_argument(
        "--policy-setup",
        required=True,
        help="Policy setup string; must contain either 'widowx' or 'google_robot'"
    )
    return parser.parse_args()

# You can modify the following to customize your scoring config 
# 1) The "rubrics_score" dict MUST preserve order (use OrderedDict or rely on Python 3.7+ insertion order).
# 2) Weights MUST sum to 1.0.
widowx_rubrics_score = OrderedDict([
    ("is_src_obj_grasped", 0.3),
    ("consecutive_grasp", 0.3),
    ("src_on_target", 0.4),
])
google_robot_move_near_rubrics_score = OrderedDict([
    ("moved_correct_obj", 0.3),
    ("near_tgt_obj", 0.3),
    ("is_closest_to_tg", 0.4),
])
google_robot_put_in_rubrics_score = OrderedDict([
    ("is_drawer_open", 1.0),
])
SCORING_CONFIG = {
    "widowx": {
        "widowx_task2_stack_cube": widowx_rubrics_score,
        "widowx_task3_put_object_on_top": widowx_rubrics_score,
        "widowx_task4_put_object_in_basket": widowx_rubrics_score,
    },
    "google_robot": {
        "fractal_move_near_visual_matching": google_robot_move_near_rubrics_score,
        "fractal_move_near_variant_agg": google_robot_move_near_rubrics_score,
        "fractal_put_in_drawer_visual_matching": google_robot_put_in_rubrics_score,
        "fractal_put_in_drawer_variant_agg": google_robot_put_in_rubrics_score,
    }
}

def ensure_weights_sum_to_one(weights: OrderedDict):
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Scoring weights must sum to 1.0, got {total}")

def pick_policy(policy_setup: str) -> str:
    s = policy_setup.lower()
    if "widowx" in s:
        return "widowx"
    if "google_robot" in s:
        return "google_robot"
    return ValueError("`--policy-setup` must contain either 'widowx' or 'google_robot'.")

def main():
    args = parse_args()

    base_path = os.path.abspath(args.result_path)
    json_file = os.path.join(base_path, "results.json")
    csv_file = os.path.join(base_path, "scores.csv")

    policy = pick_policy(args.policy_setup)
    scoring_config = SCORING_CONFIG[policy]
    for rubrics_score in scoring_config.values():
        ensure_weights_sum_to_one(rubrics_score)

    # Load json file
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []

    score_list = []
    for task in data:
        task_name = task.get("task_name", "")
        if task_name not in scoring_config.keys():
            continue
        trials = task.get("episodes", [])
        for trial in trials:
            trial_id = trial.get("trial_id")
            episode_stats = trial.get("episode_stats", {})
            score = 0.0

            # Accumulate scores
            rubrics_score = scoring_config[task_name]
            for subtask in rubrics_score.keys():
                status = episode_stats.get(subtask, {}).get("status", False)
                if not status:
                    break
                score += rubrics_score[subtask]

            cur_score_data = {
                "task_name": task_name,
                "trial_id": trial_id,
                "score": score,
            }
            score_list.append(cur_score_data)

    # Export as csv file
    cols = ["task_name", "trial_id", "score"]
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(score_list)

    print(f"Scores saved to {csv_file}, total {len(score_list)} rows.")

if __name__ == "__main__":
    main()
