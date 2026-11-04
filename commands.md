# Commands Reference

Run every command from the project root after activating the virtual environment:

```bash
cd "/home/qtg6528/AI/Camera-Radar Project"
source .venv/bin/activate
```

All sample indices are global nuScenes indices. For example, `95` is written as `095` in output filenames.

## Phase 0: Dataset Inspection

These Phase 0 utility scripts use the fixed first nuScenes sample (`sample[0]`). They do not accept command-line inputs.

```bash
python phase0_dataset_check.py
python phase0_visualize_annotations.py
python phase0_visualize_radar.py
python phase0_project_radar_to_camera.py
```

`print_radar_pcd.py` is a one-off debugging script. It uses one hard-coded radar file path:

```bash
python print_radar_pcd.py
```

## Phases 1-6: One Sample

The following scripts accept the same optional input:

```text
--sample-index NUMBER
```

If omitted, the default is sample `95`.

### Phase 1: Camera Object Detection

```bash
python phase1_object_detection.py --sample-index 95
```

Output:

```text
outputs/detections/cam_front_sample_095_yolo11n.png
outputs/detections/cam_front_sample_095_yolo11n.json
```

### Phase 2: Semantic Segmentation

```bash
python phase2_semantic_segmentation.py --sample-index 95
```

Requires the local SegFormer model files under `models/segformer-b0-cityscapes/`.

### Phase 3: Drivable Area

Run Phase 2 for the same sample first.

```bash
python phase3_drivable_area.py --sample-index 95
```

### Phase 4: Radar Processing

```bash
python phase4_radar_processing.py --sample-index 95
```

### Phase 5A: Radar-to-Camera Projection

Run Phase 4 for the same sample first.

```bash
python phase5a_radar_to_camera_projection.py --sample-index 95
```

### Phase 5B: Camera-Radar Association

Run Phases 1, 4, and 5A for the same sample first.

```bash
python phase5b_camera_radar_association.py --sample-index 95
```

### Phase 6A: Scene State

Run Phases 3 and 5B for the same sample first.

```bash
python phase6_scene_state.py --sample-index 95
```

### Phase 6B: Rule-Based Decision

Run Phase 6A for the same sample first.

```bash
python phase6_rule_based_decision.py --sample-index 95
```

## Full Phase 1-6 Pipeline

This runner executes Phases 1-6 in the required order and creates an evaluation table.

```bash
python phase6_multi_sample_evaluation.py --sample-indices 0 50 95
```

Use any number of sample indices:

```bash
python phase6_multi_sample_evaluation.py --sample-indices 11 12 13
```

If outputs already exist and only the summary table needs rebuilding, skip model inference:

```bash
python phase6_multi_sample_evaluation.py --sample-indices 0 50 95 --skip-pipeline
```

Outputs:

```text
outputs/evaluation/phase1_to_6_evaluation.csv
outputs/evaluation/phase1_to_6_evaluation.md
```

## Phase 7A: Create a Camera-Radar Replay Manifest

### Index and Frame-Count Mode

Create a manifest beginning at a global sample index:

```bash
python phase7a_prepare_replay_sequence.py \
  --start-sample-index 9 \
  --frame-count 10
```

### Camera Timestamp-Range Mode

Provide both timestamps to select all continuous `CAM_FRONT` keyframes in one scene. This is the preferred mode for the agreed replay split.

Development replay:

```bash
python phase7a_prepare_replay_sequence.py \
  --start-camera-timestamp 1532402932662460 \
  --end-camera-timestamp 1532402946512460
```

Validation replay:

```bash
python phase7a_prepare_replay_sequence.py \
  --start-camera-timestamp 1542800368512460 \
  --end-camera-timestamp 1542800387662460
```

Final replay test:

```bash
python phase7a_prepare_replay_sequence.py \
  --start-camera-timestamp 1542800848512460 \
  --end-camera-timestamp 1542800866362460
```

A manifest is saved under `outputs/replay/`.

## Phase 7B: Run the Pipeline Over a Replay Manifest

Start with the first three frames as a CPU-friendly smoke test:

```bash
python phase7b_run_replay_pipeline.py \
  outputs/replay/scene-0061_camera_1532402932662460_to_1532402946512460_manifest.json \
  --max-frames 3
```

Process every frame in a manifest:

```bash
python phase7b_run_replay_pipeline.py \
  outputs/replay/scene-0061_camera_1532402932662460_to_1532402946512460_manifest.json
```

Rebuild only the replay summary files after all individual outputs already exist:

```bash
python phase7b_run_replay_pipeline.py \
  outputs/replay/scene-0061_camera_1532402932662460_to_1532402946512460_manifest.json \
  --skip-pipeline
```

Replay results are saved under `outputs/replay/` as CSV and Markdown files matching the manifest name.

## Phase 7C: Visualize Raw Replay Measurements

Create a distance, radar-velocity, TTC, and rule-action plot from a completed replay CSV:

```bash
python phase7c_visualize_replay.py \
  outputs/replay/scene-0061_camera_1532402932662460_to_1532402946512460_results.csv
```

The PNG is saved beside the replay CSV.

## Phase 7D: Track a Replay Lead Vehicle

Track the most image-overlapping radar-matched vehicle box between consecutive replay frames:

```bash
python phase7d_track_replay_lead.py \
  outputs/replay/scene-0061_camera_1532402932662460_to_1532402946512460_manifest.json
```

The tracked-lead CSV is saved under `outputs/replay/`. It exposes tracking changes and lost associations; it does not replace the raw Phase 6 measurements.

## Phase 7E: Gymnasium ACC Environment

The Gymnasium environment is imported by the baseline and later PPO scripts. It models action-responsive ego and lead vehicles with a $0.2\,s$ time step. Use this focused syntax check after changing the environment:

```bash
python -m py_compile phase7e_acc_environment.py
```

## Phase 7F: Rule-Controller Baseline

Run the Phase 6-style simulation-specific controller over both real-data-seeded scenarios:

```bash
python phase7f_evaluate_rule_controller.py
```

Run either scenario separately:

```bash
python phase7f_evaluate_rule_controller.py --scenario sample_012
python phase7f_evaluate_rule_controller.py --scenario sample_014
```

The runner writes a per-step CSV and plot for each episode under `outputs/rl_baseline/`. The 150-step baseline completed without a collision in both scenarios. It is a safety and reward sanity check before PPO training, not an optimal controller.

## Phase 7G: PPO Training and Evaluation

PPO uses `stable-baselines3==2.9.0`, installed in the project virtual environment. Train on randomized simulator episodes, then evaluate the learned policy on the two fixed real-data-seeded scenarios:

```bash
python phase7g_train_ppo.py --timesteps 50000 --num-envs 4
```

This saves the model, training configuration, per-step evaluation CSVs, and plots under `outputs/ppo/`.

The final reward-tuned PPO experiment is stored separately from the initial conservative model:

```bash
python phase7g_train_ppo.py \
  --timesteps 50000 \
  --num-envs 4 \
  --output-directory outputs/ppo_approach_tuned
```

Evaluate that saved model without retraining:

```bash
python phase7g_train_ppo.py \
  --skip-training \
  --scenario all \
  --output-directory outputs/ppo_approach_tuned
```

To evaluate an existing saved model without training it again:

```bash
python phase7g_train_ppo.py --skip-training --scenario all
```

A short end-to-end smoke test is available for code changes. It verifies training, saving, and evaluation but is not a meaningful final training run:

```bash
python phase7g_train_ppo.py --timesteps 2048 --num-envs 2
```

## Phase 7H: Randomized PPO Evaluation

Evaluate the final saved PPO model on 100 held-out randomized simulator episodes. This does not retrain or change the model.

```bash
python phase7h_evaluate_randomized_ppo.py \
  --model-directory outputs/ppo_approach_tuned2 \
  --episodes 100 \
  --seed 1000
```

Outputs:

```text
outputs/ppo_approach_tuned2/randomized_evaluation/randomized_episode_results.csv
outputs/ppo_approach_tuned2/randomized_evaluation/randomized_evaluation_summary.json
```

## Phase 7I: Perception-to-PPO Inference

Run PPO on the saved Phase 6 state for validation scene `1077`, replay frame `23` / global sample `308`:

```bash
python phase7i_perception_to_ppo.py \
  --sample-index 308 \
  --model-directory outputs/ppo_approach_tuned2
```

The JSON inference result is saved under `outputs/ppo_inference/`. It is an end-to-end proof-of-concept bridge; it does not calibrate raw radar velocity for vehicle control.

## Final Evaluation Summary

The frozen evaluation results and limitations are consolidated in `phase7_final_results.md`.

## Check a Script's Inputs

For scripts with argument support, display the available command-line options with:

```bash
python SCRIPT_NAME.py --help
```

For example:

```bash
python phase7b_run_replay_pipeline.py --help
```
