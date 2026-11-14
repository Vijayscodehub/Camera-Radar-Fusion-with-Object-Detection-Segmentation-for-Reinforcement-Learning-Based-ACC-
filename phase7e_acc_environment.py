from typing import Any #any kind of value

import gymnasium as gym
import numpy as np
from gymnasium import spaces #describes what moves and measurements are allowed


BRAKE = 0
MAINTAIN = 1
ACCELERATE = 2
ACTION_NAMES = {
    BRAKE: "BRAKE",
    MAINTAIN: "MAINTAIN",
    ACCELERATE: "ACCELERATE",
}
ACTION_ACCELERATIONS_MPS2 = {
    BRAKE: -3.0,
    MAINTAIN: 0.0,
    ACCELERATE: 1.5,
}
NEUTRAL_FOLLOWING_SAMPLE_12 = {
    "scenario_name": "sample_012_neutral_following",
    "initial_gap_m": 36.75,
    "ego_speed_mps": 13.9,
    "lead_speed_mps": 13.9,
    "lead_acceleration_mps2": 0.0,
} #relaxed, same speed
CLOSE_FOLLOWING_SAMPLE_14 = {
    "scenario_name": "sample_014_close_following",
    "initial_gap_m": 12.18,
    "ego_speed_mps": 13.9,
    "lead_speed_mps": 10.62,
    "lead_acceleration_mps2": 0.0,
} #dangerous, we're faster and close


class AdaptiveCruiseControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        time_step_s: float = 0.2,
        max_steps: int = 150,
        #two switches for randomness: sometimes there's no lead car, and sometimes the lead car speeds up or slows down
        randomize_lead_presence: bool = False,
        randomize_lead_acceleration: bool = False,
    ) -> None:
        super().__init__()
        self.time_step_s = time_step_s
        self.max_steps = max_steps
        self.randomize_lead_presence = randomize_lead_presence
        self.randomize_lead_acceleration = randomize_lead_acceleration
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -30.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 100.0, 30.0, 40.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.lead_present = True
        self.gap_m = 0.0
        self.ego_speed_mps = 0.0
        self.lead_speed_mps = 0.0
        self.lead_acceleration_mps2 = 0.0
        self.step_count = 0
        self.previous_action: int | None = None
        self.scenario_name = "random"

    def _observation(self) -> np.ndarray:
        closing_speed_mps = (
            self.ego_speed_mps - self.lead_speed_mps if self.lead_present else 0.0
        )
        return np.array(
            [
                float(self.lead_present),
                self.gap_m,
                closing_speed_mps, #Closing speed = our speed − lead speed
                self.ego_speed_mps,
                1.0, #bias
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        closing_speed_mps = (
            self.ego_speed_mps - self.lead_speed_mps if self.lead_present else 0.0
        )
        ttc_s = self.gap_m / closing_speed_mps if closing_speed_mps > 0.05 else None
        desired_gap_m = max(8.0, 1.5 * self.ego_speed_mps) #but never less than 8 m
        return {
            "scenario_name": self.scenario_name,
            "gap_m": round(self.gap_m, 3),
            "ego_speed_mps": round(self.ego_speed_mps, 3),
            "lead_speed_mps": round(self.lead_speed_mps, 3),
            "closing_speed_mps": round(closing_speed_mps, 3),
            "ttc_s": round(ttc_s, 3) if ttc_s is not None else None,
            "desired_gap_m": round(desired_gap_m, 3),
            "episode_time_s": round(self.step_count * self.time_step_s, 3),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.lead_present = bool(
            options.get(
                "lead_present",
                self.np_random.random() < 0.9
                if self.randomize_lead_presence
                else True,
            )
        )
        self.gap_m = float(
            options.get(
                "initial_gap_m",
                self.np_random.uniform(8.0, 60.0) if self.lead_present else 0.0,
            )
        )
        self.ego_speed_mps = float(
            options.get("ego_speed_mps", self.np_random.uniform(8.0, 18.0))
        )
        self.lead_speed_mps = float(
            options.get("lead_speed_mps", self.np_random.uniform(5.0, 18.0))
        )
        self.lead_acceleration_mps2 = float(
            options.get(
                "lead_acceleration_mps2",
                self.np_random.uniform(-1.5, 0.5)
                if self.randomize_lead_acceleration and self.lead_present
                else 0.0,
            )
        )
        self.step_count = 0
        self.previous_action = None
        self.scenario_name = str(options.get("scenario_name", "random"))
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.ego_speed_mps = max(
            0.0,
            self.ego_speed_mps
            + ACTION_ACCELERATIONS_MPS2[action] * self.time_step_s,
        )
        if self.lead_present:
            self.lead_speed_mps = max(
                0.0,
                self.lead_speed_mps
                + self.lead_acceleration_mps2 * self.time_step_s,
            )
            self.gap_m += (
                self.lead_speed_mps - self.ego_speed_mps
            ) * self.time_step_s

        self.step_count += 1
        collision = self.lead_present and self.gap_m <= 0.0
        if collision:
            self.gap_m = 0.0

        info = self._info()
        reward = self._reward(collision, info, action)
        terminated = collision
        truncated = self.step_count >= self.max_steps
        self.previous_action = action
        return self._observation(), reward, terminated, truncated, info

    def _reward(self, collision: bool, info: dict[str, Any], action: int) -> float:
        if collision:
            return -100.0

        action_change_penalty = (
            0.05
            if self.previous_action is not None and self.previous_action != action
            else 0.0
        )
        progress_reward = 0.2 * min(self.ego_speed_mps / 20.0, 1.0)
        if not self.lead_present:
            return progress_reward - action_change_penalty

        following_error = abs(self.gap_m - info["desired_gap_m"]) / info["desired_gap_m"]
        following_reward = 1.0 - min(following_error, 1.0)
        excessive_gap_penalty = (
            0.5
            * min(
                (self.gap_m - info["desired_gap_m"]) / info["desired_gap_m"],
                1.0,
            )
            if self.gap_m > info["desired_gap_m"]
            else 0.0
        )
        approach_reward = (
            0.5
            * min(max(info["closing_speed_mps"], 0.0) / 1.5, 1.0)
            if self.gap_m > 1.1 * info["desired_gap_m"]
            else 0.0
        )
        reward = (
            following_reward
            + progress_reward
            - excessive_gap_penalty
            + approach_reward
        )
        if self.gap_m < 8.0:
            reward -= 5.0
        if info["ttc_s"] is not None and info["ttc_s"] < 2.0:
            reward -= 10.0
        return reward - action_change_penalty