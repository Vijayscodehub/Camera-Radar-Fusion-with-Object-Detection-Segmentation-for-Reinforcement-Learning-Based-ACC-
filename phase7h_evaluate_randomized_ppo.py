import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from phase7e_acc_environment import ACTION_NAMES, AdaptiveCruiseControlEnv


CSV_COLUMNS = [
    "episode_number",
    "reset_seed",
    "lead_present",
    "initial_gap_m",
    "initial_ego_speed_mps",
    "initial_lead_speed_mps",
    "lead_acceleration_mps2",
    "episode_steps",
    "collision",
    "total_reward",
    "final_gap_m",
    "final_desired_gap_m",
    "final_gap_error_m",
    "action_changes",
    "brake_actions",
    "maintain_actions",
    "accelerate_actions",
]


def run_episode(
    model: PPO,
    max_steps: int,
    reset_seed: int,
    episode_number: int,
) -> dict:
    environment = AdaptiveCruiseControlEnv(
        max_steps=max_steps,
        randomize_lead_presence=True,
        randomize_lead_acceleration=True,
    )
    observation, info = environment.reset(seed=reset_seed)
    lead_present = environment.lead_present
    initial_gap_m = environment.gap_m if lead_present else None
    initial_ego_speed_mps = environment.ego_speed_mps
    initial_lead_speed_mps = environment.lead_speed_mps if lead_present else None
    lead_acceleration_mps2 = (
        environment.lead_acceleration_mps2 if lead_present else None
    )
    total_reward = 0.0
    action_counts = {action_name: 0 for action_name in ACTION_NAMES.values()}
    action_changes = 0
    previous_action: int | None = None
    collision = False

    for step_number in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        action_value = int(action)
        observation, reward, terminated, truncated, info = environment.step(action_value)
        total_reward += reward
        action_counts[ACTION_NAMES[action_value]] += 1
        if previous_action is not None and action_value != previous_action:
            action_changes += 1
        previous_action = action_value
        collision = collision or terminated
        if terminated or truncated:
            break

    final_gap_error_m = (
        abs(info["gap_m"] - info["desired_gap_m"]) if lead_present else None
    )
    return {
        "episode_number": episode_number,
        "reset_seed": reset_seed,
        "lead_present": lead_present,
        "initial_gap_m": initial_gap_m,
        "initial_ego_speed_mps": initial_ego_speed_mps,
        "initial_lead_speed_mps": initial_lead_speed_mps,
        "lead_acceleration_mps2": lead_acceleration_mps2,
        "episode_steps": step_number + 1,
        "collision": collision,
        "total_reward": round(total_reward, 3),
        "final_gap_m": info["gap_m"] if lead_present else None,
        "final_desired_gap_m": info["desired_gap_m"] if lead_present else None,
        "final_gap_error_m": round(final_gap_error_m, 3)
        if final_gap_error_m is not None
        else None,
        "action_changes": action_changes,
        "brake_actions": action_counts["BRAKE"],
        "maintain_actions": action_counts["MAINTAIN"],
        "accelerate_actions": action_counts["ACCELERATE"],
    }


def calculate_summary(rows: list[dict], arguments: argparse.Namespace) -> dict:
    lead_rows = [row for row in rows if row["lead_present"]]
    collisions = sum(row["collision"] for row in rows)
    lead_collisions = sum(row["collision"] for row in lead_rows)
    return {
        "model_directory": arguments.model_directory,
        "episodes": len(rows),
        "max_steps": arguments.max_steps,
        "seed": arguments.seed,
        "lead_episodes": len(lead_rows),
        "no_lead_episodes": len(rows) - len(lead_rows),
        "collision_count": collisions,
        "collision_rate": round(collisions / len(rows), 4),
        "lead_collision_rate": round(lead_collisions / len(lead_rows), 4)
        if lead_rows
        else None,
        "mean_total_reward": round(float(np.mean([row["total_reward"] for row in rows])), 3),
        "mean_final_gap_error_m": round(
            float(np.mean([row["final_gap_error_m"] for row in lead_rows])), 3
        )
        if lead_rows
        else None,
        "mean_action_changes": round(
            float(np.mean([row["action_changes"] for row in rows])), 3
        ),
        "mean_episode_steps": round(
            float(np.mean([row["episode_steps"] for row in rows])), 3
        ),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(arguments: argparse.Namespace) -> None:
    model_directory = Path(arguments.model_directory).resolve()
    model_path = model_directory / "ppo_acc_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved PPO model at {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    output_directory = model_directory / "randomized_evaluation"
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        run_episode(
            model,
            arguments.max_steps,
            arguments.seed + episode_number,
            episode_number,
        )
        for episode_number in range(arguments.episodes)
    ]
    summary = calculate_summary(rows, arguments)
    csv_path = output_directory / "randomized_episode_results.csv"
    summary_path = output_directory / "randomized_evaluation_summary.json"
    write_csv(rows, csv_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Randomized PPO evaluation complete")
    for name, value in summary.items():
        print(f"{name}: {value}")
    print(f"Saved episode CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PPO ACC model on held-out randomized episodes."
    )
    parser.add_argument(
        "--model-directory",
        default="outputs/ppo_approach_tuned",
        help="Directory containing ppo_acc_model.zip.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of held-out randomized episodes (default: 100).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum steps per episode (default: 150).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="First held-out reset seed (default: 1000).",
    )
    main(parser.parse_args())