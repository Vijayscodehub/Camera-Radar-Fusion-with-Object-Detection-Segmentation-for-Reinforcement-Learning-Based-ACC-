# Phase 7: RL Extension

The RL extension has two different parts. They work together, but they have different jobs.

## Part 1: Camera-Radar Replay

This part uses the recorded nuScenes camera images and radar point clouds.

```text
Camera image + radar point cloud
-> object detection + road segmentation + radar processing
-> camera-radar fusion
-> lead vehicle, distance, relative speed, and road information
-> recommended action
```

For one frame, the perception pipeline can create a small driving state such as:

```text
Lead vehicle: bus
Lead distance: 31.88 m
Relative speed: 1.84 m/s
Ego speed: 13.9 m/s
Road ahead: yes
```

This can be repeated for consecutive frames in a nuScenes scene. It demonstrates that the project can understand camera-radar data over time and recommend `BRAKE`, `MAINTAIN`, or `ACCELERATE` at each frame.

### Important Limitation

nuScenes is a recording. If a controller recommends `BRAKE` in one frame, the next recorded frame does not change. It still shows what the original vehicle did when the data was collected.

```text
Controller says BRAKE
-> next nuScenes frame is unchanged
```

Therefore, recorded camera-radar data is useful for perception and replay, but not for directly training an RL agent whose actions must affect the next state.

## Replay Data Split

Use complete, non-overlapping nuScenes scenes for replay development, validation, and final evaluation. This prevents nearby frames from the same scene appearing in more than one role.

| Role | Scene | CAM_FRONT timestamp range | Camera-radar keyframes | Purpose |
| --- | --- | --- | ---: | --- |
| Development | `scene-0061` | `1532402932662460` to `1532402946512460` | 27 | Build and debug continuous camera-radar replay. |
| Validation | `scene-1077` | `1542800368512460` to `1542800387662460` | 38 | Tune replay filtering and check behavior while developing the simulator and PPO. |
| Final test | `scene-1094` | `1542800848512460` to `1542800866362460` | 35 | Run once after the design is frozen and report the final replay result. |

The source intervals contain many camera sweep images, but replay processing uses only the listed camera-radar keyframes. Each keyframe has both `CAM_FRONT` and `RADAR_FRONT` data, so it can run through the existing Phase 1-6 pipeline.

The validation and final-test scenes come from the same recording day, so this is a practical small-split protocol for `v1.0-mini`, not a claim of full real-world generalization. Do not use the final-test sequence to adjust thresholds, rewards, or model settings.

### Replay Processing Results

The development sequence is processed one keyframe at a time through the existing Phases 1-6 pipeline. This produces a replay table with the selected lead class, raw radar distance, raw radar velocity, TTC, and Phase 6 rule action for every frame.

```text
manifest
-> Phase 1-6 per keyframe
-> replay CSV and Markdown table
-> raw distance / velocity / TTC plot
-> temporal lead-tracking CSV
```

The first development replay confirms that the pipeline produces measurements continuously, but it also shows camera class flicker and changing radar associations. Therefore:

```text
Use replay measurements as realistic example ranges.
Do not treat the raw nearest-object value in every frame as one perfect ground-truth trajectory.
```

The temporal tracker keeps the vehicle box with the greatest overlap from the preceding frame. It makes association changes visible in the output rather than hiding them. A future tracker can add motion prediction and radar gating, but the initial Gymnasium environment does not depend on a perfect real-data track because it maintains its own ego and lead state.

## Part 2: Gymnasium ACC Simulation

This part is a small driving simulation for training RL. It does not need a 3D car, steering, CARLA, or a realistic game world.

The simulator contains only:

```text
one straight lane
one ego vehicle (our vehicle)
one lead vehicle (the vehicle in front)
their speeds
the distance between them
```

The RL observation uses the same kind of numerical information produced by the camera-radar pipeline:

```python
[
    lead_present,
    lead_distance_m,
    relative_speed_mps,
    ego_speed_mps,
    drivable_ahead,
]
```

The RL agent chooses one action:

```text
BRAKE
MAINTAIN
ACCELERATE
```

Gymnasium then moves the simulated vehicles forward for a short time step. The ego vehicle responds to the action, so the distance and relative speed change. This creates a new observation for the next action.

```text
RL chooses BRAKE
-> ego vehicle slows down
-> distance to the lead vehicle changes
-> RL receives the new distance and speed
```

The agent receives a reward for safe, smooth forward driving and penalties for getting too close or colliding. After many simulated episodes, PPO learns which action gives the best long-term reward.

### How Real Frames Set Up Gymnasium

The project follows the original connection between perception and RL: a real Phase 6 state can initialize one realistic Gymnasium scenario.

```text
Real camera + radar frame
-> Phase 1-6 state: distance, ego speed, road information
-> starting state for one simulated ego-and-lead-vehicle episode
```

For example, a fused lead vehicle at `31.88 m` can create this starting situation:

```text
[ego vehicle] -------- 31.88 m -------- [lead vehicle]
```

This does not mean PPO trains only on that one frame. During training, Gymnasium starts many related but different episodes by varying the gap, ego speed, lead speed, and lead behavior. Real replay frames have two uses:

```text
Direct use: start or demonstrate one realistic simulator episode.
Indirect use: define realistic value ranges for randomized PPO training episodes.
```

The simulator defines closing speed explicitly as:

$$
v_{closing} = v_{ego} - v_{lead}
$$

A positive $v_{closing}$ means the ego vehicle is getting closer to the lead vehicle. The current radar output is a compensated radar-axis velocity, so it must not be treated as this exact closing speed until its sign and meaning are verified. The first Gymnasium version will keep its own unambiguous ego and lead speeds.

## Step-by-Step Connection

### 1. Process One Real Frame

Start with a nuScenes front-camera image and front-radar point cloud:

```text
camera image + radar point cloud
-> existing Phases 1-6
-> driving state
```

For example, sample `095` provides a fused lead bus at `31.88 m` with an ego speed assumption of `13.9 m/s`.

### 2. Start a Simulation Scenario

Use that state as one realistic starting example for Gymnasium:

```text
[ego vehicle] -------- 31.88 m -------- [lead vehicle]
```

The simulator stores the gap, ego speed, and lead speed. The real dataset helps choose realistic distances and speeds, but the simulator controls what happens next.

### 3. Give the State to RL

RL does not receive the raw image or radar `.pcd` file. It receives the compact numerical state:

```text
lead exists, distance, relative speed, ego speed, road available
```

### 4. RL Chooses an Action

For example:

```text
Current gap: 31.88 m
-> PPO chooses ACCELERATE
```

### 5. Gymnasium Updates the World

The simulation applies the action for a short time, such as $0.2\,s$:

```text
ACCELERATE
-> ego speed increases
-> ego vehicle moves forward
-> gap may become smaller
```

The lead vehicle also moves forward at its own simulated speed.

### 6. RL Receives the Next State and Reward

Gymnasium calculates the new gap and speeds, then gives RL the next state and a reward:

```text
safe distance and smooth driving -> positive reward
unsafe following distance -> negative reward
collision -> very large negative reward
```

This repeats until the scenario safely ends, times out, or reaches a collision.

### 7. Train With Many Different Scenarios

Do not train only from sample `095`. The simulation should start with many realistic but varied situations:

```text
close lead vehicle
safe lead vehicle
slow lead vehicle
lead vehicle braking
no lead vehicle
```

This prevents PPO from memorizing one distance or speed.

## Final Connection

```text
Real camera + radar
-> Phases 1-6 produce numerical state
-> trained PPO reads that same state format
-> PPO recommends BRAKE / MAINTAIN / ACCELERATE
```

The dataset teaches the project how to understand the road. Gymnasium gives PPO a place where its actions have consequences and it can learn safely.

## Rule-Controller Baseline

Before PPO, a fixed controller tests the complete simulation loop:

```text
read simulated gap and TTC
-> choose BRAKE, MAINTAIN, or ACCELERATE
-> Gymnasium updates speed and gap
-> record reward and the next simulated state
```

The controller uses the simulator's explicit closing speed and TTC rather than raw radar velocity:

```text
gap <= 12 m or positive TTC <= 4 s -> BRAKE
gap <= 30 m                         -> MAINTAIN
otherwise                           -> ACCELERATE
```

The rule evaluator saves one CSV and plot per scenario under `outputs/rl_baseline/`. With 150 steps, it completed both seeded scenarios without a collision:

| Scenario | Initial gap | Final gap | Total reward | Result |
| --- | ---: | ---: | ---: | --- |
| `sample_012` neutral following | 36.75 m | 18.27 m | 121.980 | No collision |
| `sample_014` close following | 12.18 m | 31.62 m | 77.310 | No collision |

The plots show a coarse threshold controller that alternates between braking, holding, and accelerating near its thresholds. This is acceptable for a baseline because it verifies action-dependent dynamics and rewards. PPO will be evaluated against it and should be encouraged to make smoother choices through the action-change penalty, which now applies only when the selected action changes.

## First PPO Training Run

Stable-Baselines3 PPO is trained on randomized versions of the small ACC simulator, not directly on recorded nuScenes frames. Each training reset varies the initial gap, ego speed, lead speed, and, when a lead is present, its acceleration. The model still receives the same five-value observation used by the fixed scenarios.

```text
randomized simulator reset
-> PPO chooses an action
-> simulator returns next state and reward
-> repeat for many episodes
-> evaluate the saved model on samples 12 and 14
```

The first reproducible CPU run used 50,000 requested PPO timesteps, four parallel environments, and seed `7`. PPO completed both fixed tests without collision, so the training and evaluation connection works. It is not yet better than the rule controller:

| Scenario | PPO final gap | PPO total reward | Rule total reward | Current interpretation |
| --- | ---: | ---: | ---: | --- |
| `sample_012` neutral following | 82.11 m | 17.286 | 121.980 | PPO slows too much and lets the lead move far ahead. |
| `sample_014` close following | 62.76 m | 24.505 | 77.310 | PPO avoids collision but creates a much larger than desired gap. |

This is a valid first RL result, not a final controller. The next PPO iteration tunes the following-distance and forward-progress reward balance, then retrains and compares against this saved baseline using the same fixed scenarios. Do not use the final replay test scene to tune it.

## Reward-Tuned PPO Result

The first PPO model was safe but stayed too far behind. Reward tuning kept the collision and too-close penalties, then added two bounded signals:

```text
gap above desired distance -> penalty
gap above 110% of desired distance and safely closing -> approach reward
```

The approach reward is zero at normal or close following distances, so it does not reward acceleration when the ego vehicle needs to protect its headway. The final model is saved separately under `outputs/ppo_approach_tuned/`, preserving earlier PPO experiments for comparison.

| Scenario | Controller | Collision | Final gap | Final desired gap | Total reward |
| --- | --- | --- | ---: | ---: | ---: |
| `sample_012` neutral following | Rule | No | 18.27 m | 16.80 m | 120.211 |
| `sample_012` neutral following | PPO | No | 19.29 m | 20.85 m | 158.447 |
| `sample_014` close following | Rule | No | 31.62 m | 17.70 m | 49.451 |
| `sample_014` close following | PPO | No | 14.94 m | 15.45 m | 145.552 |

For these two fixed, real-data-seeded examples, PPO now maintains a gap closer to the simulator's desired gap than the threshold rule controller and remains collision-free. It still alternates frequently between `BRAKE` and `ACCELERATE` because the action space has only three discrete choices. More realistic smoothness would require continuous acceleration actions or a stronger action-change penalty, which should be a later extension rather than changing this validated baseline.


-------------just understading- ignore 
1. Real nuScenes camera + radar
   -> Phases 1-6 extract a lead distance and speed-like information.
   -> We used real examples such as samples 12 and 14.

2. Gymnasium simulator
   -> starts with realistic distance/speed values
   -> contains ego vehicle and lead vehicle
   -> updates the gap after every action

3. Rule controller test
   -> verifies simulator logic before RL
   -> BRAKE, MAINTAIN, ACCELERATE have sensible results

4. PPO training next
   -> PPO sees the same numerical state:
      gap, closing speed, ego speed, lead present, road available
   -> PPO chooses one action
   -> simulator gives reward or penalty
   -> PPO repeats this many times and learns