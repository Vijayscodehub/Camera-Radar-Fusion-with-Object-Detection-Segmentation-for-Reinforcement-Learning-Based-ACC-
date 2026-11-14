import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from phase7e_acc_environment import (
    ACCELERATE,
    ACTION_NAMES,
    BRAKE,
    CLOSE_FOLLOWING_SAMPLE_14,
    MAINTAIN,
    NEUTRAL_FOLLOWING_SAMPLE_12,
    AdaptiveCruiseControlEnv,
)


BRAKE_DISTANCE_M = 12.0
MAINTAIN_DISTANCE_M = 30.0
BRAKE_TTC_S = 4.0
SCENARIOS = {
    "sample_012": NEUTRAL_FOLLOWING_SAMPLE_12,
    "sample_014": CLOSE_FOLLOWING_SAMPLE_14,
}
CSV_COLUMNS = [
    "step_number",
    "episode_time_s",
    "action",
    "action_name",
    "reward",
    "gap_m",
    "ego_speed_mps",
    "lead_speed_mps",
    "closing_speed_mps",
    "ttc_s",
    "desired_gap_m",
]
ACTION_COLORS = {
    "BRAKE": "tab:red",
    "MAINTAIN": "tab:orange",
    "ACCELERATE": "tab:green",
}


def rule_action(info: dict) -> int:
    gap_m = info["gap_m"]
    ttc_s = info["ttc_s"]
    if gap_m <= BRAKE_DISTANCE_M:
        return BRAKE
    if ttc_s is not None and ttc_s <= BRAKE_TTC_S:
        return BRAKE
    if gap_m <= MAINTAIN_DISTANCE_M:
        return MAINTAIN
    return ACCELERATE


def run_episode(scenario_name: str, max_steps: int) -> tuple[list[dict], bool]:
    environment = AdaptiveCruiseControlEnv(max_steps=max_steps)
    _, info = environment.reset(options=SCENARIOS[scenario_name])
    rows = []
    collision = False

    for step_number in range(max_steps):
        action = rule_action(info)
        _, reward, terminated, truncated, info = environment.step(action)
        rows.append(
            {
                "step_number": step_number,
                "episode_time_s": info["episode_time_s"],
                "action": action,
                "action_name": ACTION_NAMES[action],
                "reward": round(reward, 3),
                "gap_m": info["gap_m"],
                "ego_speed_mps": info["ego_speed_mps"],
                "lead_speed_mps": info["lead_speed_mps"],
                "closing_speed_mps": info["closing_speed_mps"],
                "ttc_s": info["ttc_s"],
                "desired_gap_m": info["desired_gap_m"],
            }
        )
        collision = collision or terminated
        if terminated or truncated:
            break

    return rows, collision


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def create_plot(rows: list[dict], scenario_name: str, output_path: Path) -> None:
    times_s = [row["episode_time_s"] for row in rows]
    gaps_m = [row["gap_m"] for row in rows]
    desired_gaps_m = [row["desired_gap_m"] for row in rows]
    ego_speeds_mps = [row["ego_speed_mps"] for row in rows]
    lead_speeds_mps = [row["lead_speed_mps"] for row in rows]
    rewards = [row["reward"] for row in rows]

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 9),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(times_s, gaps_m, label="Gap", color="tab:blue")
    axes[0].plot(times_s, desired_gaps_m, label="Desired gap", color="tab:gray", linestyle="--")
    axes[0].set_ylabel("Distance (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(times_s, ego_speeds_mps, label="Ego", color="tab:green")
    axes[1].plot(times_s, lead_speeds_mps, label="Lead", color="tab:purple")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(times_s, rewards, color="tab:blue", label="Reward")
    for time_s, row in zip(times_s, rows):
        axes[2].scatter(
            time_s,
            0,
            color=ACTION_COLORS[row["action_name"]],
            s=32,
            zorder=3,
        )
    for action_name, color in ACTION_COLORS.items():
        axes[2].scatter([], [], color=color, label=action_name)
    axes[2].set_xlabel("Episode time (s)")
    axes[2].set_ylabel("Reward")
    axes[2].legend(ncol=4)
    axes[2].grid(alpha=0.3)

    figure.suptitle(f"Rule Controller Baseline: {scenario_name}")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(scenario_names: list[str], max_steps: int) -> None:
    output_directory = Path("outputs/rl_baseline").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    for scenario_name in scenario_names:
        rows, collision = run_episode(scenario_name, max_steps)
        csv_path = output_directory / f"{scenario_name}_rule_episode.csv"
        plot_path = output_directory / f"{scenario_name}_rule_episode.png"
        write_csv(rows, csv_path)
        create_plot(rows, scenario_name, plot_path)

        actions = {action_name: 0 for action_name in ACTION_NAMES.values()}
        for row in rows:
            actions[row["action_name"]] += 1
        print(f"Scenario: {scenario_name}")
        print(f"Steps: {len(rows)}")
        print(f"Collision: {collision}")
        print(f"Total reward: {sum(row['reward'] for row in rows):.3f}")
        print(f"Actions: {actions}")
        print(f"Saved episode CSV: {csv_path}")
        print(f"Saved episode plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed ACC rule controller in full Gymnasium episodes."
    )
    parser.add_argument(
        "--scenario",
        choices=["sample_012", "sample_014", "all"],
        default="all",
        help="Seed scenario to evaluate (default: all).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum simulation steps per episode (default: 150).",
    )
    arguments = parser.parse_args()
    selected_scenarios = (
        list(SCENARIOS) if arguments.scenario == "all" else [arguments.scenario]
    )
    main(selected_scenarios, arguments.max_steps) #It runs the rule-based ACC controller in one or more scenarios, saves a CSV of the results, and saves a plot of distance, speed, and reward with colored dots for brake, maintain, or accelerate. instead of agent we have a rule based controller and after train we do the same evaluation as in phase7g_train_ppo.py