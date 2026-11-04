import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from phase7e_acc_environment import ACTION_NAMES


def create_observation(scene_state: dict) -> np.ndarray:
    lead_present = bool(scene_state["lead_present"])
    return np.array(
        [
            float(lead_present),
            float(scene_state["lead_distance_m"] or 0.0),
            float(scene_state["lead_relative_speed_mps"] or 0.0),
            float(scene_state["ego_speed_mps"]),
            float(scene_state["drivable_ahead"]),
        ],
        dtype=np.float32,
    )


def main(sample_index: int, model_directory: str) -> None:
    state_path = (
        Path("outputs/decision") / f"cam_front_sample_{sample_index:03d}_scene_state.json"
    )
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 6 state: {state_path}. "
            "Run the Phase 1-6 pipeline for this sample first."
        )

    model_path = Path(model_directory) / "ppo_acc_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PPO model: {model_path}")

    scene_state = json.loads(state_path.read_text())
    observation = create_observation(scene_state)
    model = PPO.load(str(model_path), device="cpu")
    action, _ = model.predict(observation, deterministic=True)
    action_value = int(action)
    result = {
        "sample_index": sample_index,
        "phase6_state_path": str(state_path),
        "ppo_model_path": str(model_path),
        "ppo_observation": observation.tolist(),
        "recommended_action": ACTION_NAMES[action_value],
        "conversion_assumption": (
            "Phase 6 lead_relative_speed_mps is used directly as the PPO "
            "simulator closing_speed_mps for this proof-of-concept inference. "
            "Its real radar-axis sign and physical meaning are not yet calibrated."
        ),
    }
    output_directory = Path("outputs/ppo_inference")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"cam_front_sample_{sample_index:03d}_ppo_action.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Phase 6 state: {state_path}")
    print(f"PPO observation: {result['ppo_observation']}")
    print(f"PPO recommended action: {result['recommended_action']}")
    print(f"Saved inference result: {output_path}")
    print("Caveat: raw radar velocity is an uncalibrated closing-speed assumption.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run saved PPO inference from a Phase 6 camera-radar scene state."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=308,
        help="Global nuScenes sample index with a saved Phase 6 state (default: 308).",
    )
    parser.add_argument(
        "--model-directory",
        default="outputs/ppo_approach_tuned2",
        help="Directory containing ppo_acc_model.zip.",
    )
    arguments = parser.parse_args()
    main(arguments.sample_index, arguments.model_directory)