# Phase 7 Final Results

## Scope

This project is a CPU-friendly camera-radar perception and adaptive-cruise-control proof of concept. It is not a real-world autonomous-driving system.

The final design was frozen before the final replay scene was processed:

```text
Camera + radar replay -> perception and fusion measurements
Gymnasium simulator -> PPO action-feedback evaluation
One real Phase 6 state -> saved PPO action recommendation
```

## Replay Split

| Role | Scene | Keyframes | Status |
| --- | --- | ---: | --- |
| Development | `scene-0061` | 27 | Completed during development. |
| Validation | `scene-1077` | 38 | Completed without changing the PPO model. |
| Final test | `scene-1094` | 35 | Completed once after the design was frozen. |

## Camera-Radar Replay Results

| Scene | Lead selected | Radar-matched detections | Lead-distance range | BRAKE | MAINTAIN | ACCELERATE |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `scene-0061` | 27 / 27 | 77 | 4.85 to 56.13 m | 9 | 9 | 9 |
| `scene-1077` | 36 / 38 | 74 | 15.15 to 84.09 m | 31 | 0 | 7 |
| `scene-1094` | 29 / 35 | 88 | 4.83 to 68.61 m | 11 | 5 | 19 |

The replay scripts successfully run detection, road segmentation, radar processing, fusion, scene-state creation, and the rule decision on held-out real nuScenes frames. The selected lead and radar values can vary sharply between adjacent frames because this initial implementation has no motion-predicted, radar-gated multi-object tracker. The replay measurements should therefore be read as raw fused observations, not as a perfect physical trajectory of one vehicle.

## Fixed Simulator Comparison

The rule controller and PPO use the same action-responsive Gymnasium environment. Both scenarios use realistic seed values drawn from replay samples.

| Scenario | Controller | Collision | Final gap | Desired gap | Total reward |
| --- | --- | --- | ---: | ---: | ---: |
| `sample_012` neutral following | Rule | No | 18.27 m | 16.80 m | 120.211 |
| `sample_012` neutral following | PPO | No | 19.29 m | 20.85 m | 158.447 |
| `sample_014` close following | Rule | No | 31.62 m | 17.70 m | 49.451 |
| `sample_014` close following | PPO | No | 14.94 m | 15.45 m | 145.552 |

On these two fixed scenarios, the reward-tuned PPO model ends nearer to the simulator's desired following distance than the simple threshold controller, without collision.

## Randomized PPO Evaluation

The saved PPO model in `outputs/ppo_approach_tuned2/` was evaluated, without further training, on 100 held-out randomized simulator episodes using reset seeds `1000` through `1099`.

| Metric | Result |
| --- | ---: |
| Episodes | 100 |
| Episodes with a lead vehicle | 86 |
| Episodes with no lead vehicle | 14 |
| Collisions | 5 |
| Overall collision rate | 5.0% |
| Lead-episode collision rate | 5.81% |
| Mean total reward | 69.576 |
| Mean final gap error for lead episodes | 4.602 m |
| Mean action changes per episode | 67.5 |

This is the main safety limitation of the PPO extension. The model performs well on the two fixed demonstrations but is not robust enough across randomized scenarios to support any real-world safety claim.

## End-to-End Perception-to-PPO Example

Validation replay sample `308` from `scene-1077` produced a Phase 6 state with a selected lead at `20.04 m`. The bridge script supplied the saved PPO model with:

```text
[lead present=1, distance=20.04 m, assumed closing speed=13.58 m/s,
 ego speed=13.9 m/s, drivable ahead=1]
```

The PPO model recommended `BRAKE`. This demonstrates the software connection:

```text
Real camera + radar frame
-> Phase 1-6 numerical state
-> PPO observation
-> BRAKE / MAINTAIN / ACCELERATE recommendation
```

For this proof-of-concept bridge, the Phase 6 radar relative velocity is directly used as the simulator closing speed. Its sign convention and physical meaning have not been calibrated, so this is an inference demonstration rather than deployable vehicle control.

## Known Limitations

- Raw replay lead association can flicker between objects and can report missing leads.
- Radar velocity is not calibrated to the simulator's closing-speed convention.
- The simulator is one-dimensional, uses one lead vehicle, and has no steering or road geometry.
- PPO has only three discrete actions, producing frequent brake/accelerate switching.
- PPO collided in 5 of 100 randomized held-out simulator episodes.
- No result represents real-vehicle testing or real-time autonomous control.
