"""
resume_run.py
=============

Resume an interrupted results_v2 experiment and complete evaluation.

This script is designed for the exact folder structure:

D:\FINKI\ai-driven\
    run.py
    env.py
    datasets\time_series_60min_singleindex.csv
    results_v2\
        checkpoints\
        configuration\
        evaluation\
        models\
        training\

What it does
------------
1. Reads the original experiment configuration.
2. Detects completed final models and skips them.
3. Detects the latest checkpoint for an incomplete algorithm/seed.
4. Continues only the missing timesteps.
5. Saves the missing final model.
6. Evaluates every final PPO/SAC/TD3 model and the rule-based baseline.
7. Recreates all evaluation CSV files.

Important
---------
The original run.py stored episode-reward history only in memory and intended to
write it after every algorithm completed. Because CMD was closed, the exact
episode-by-episode history for already completed models cannot be recovered.
This script therefore creates:
    training/resume_training_history.csv
for the resumed segment only, and a clear recovery note.
The final models and evaluation results remain valid.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

import run as pipeline


MODEL_CLASSES = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume interrupted renewable-aware DRL training and evaluation."
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results_v2"),
        help="Existing interrupted results folder.",
    )
    parser.add_argument(
        "--data",
        default=str(
            Path(__file__).resolve().parent
            / "datasets"
            / "time_series_60min_singleindex.csv"
        ),
        help="Path to the OPSD hourly CSV file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Do not resume training; evaluate only if all final models exist.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_experiment_config(output_dir: Path) -> Dict:
    config_path = output_dir / "configuration" / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {config_path}"
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def final_model_path(models_dir: Path, algorithm: str, seed: int) -> Path:
    return models_dir / f"{algorithm}_seed{seed}_final.zip"


def find_latest_checkpoint(
    checkpoints_dir: Path,
    algorithm: str,
    seed: int,
    target_timesteps: int,
) -> Optional[Tuple[Path, int]]:
    pattern = re.compile(
        rf"^{re.escape(algorithm)}_seed{seed}_step(\d+)\.zip$",
        flags=re.IGNORECASE,
    )
    found: List[Tuple[Path, int]] = []

    for path in checkpoints_dir.glob(f"{algorithm}_seed{seed}_step*.zip"):
        match = pattern.match(path.name)
        if match:
            step = int(match.group(1))
            if step <= target_timesteps:
                found.append((path, step))

    if not found:
        return None

    return max(found, key=lambda item: item[1])


def make_training_environment(
    split_data,
    normalization,
    system_ids,
    seed: int,
):
    training_frames = {
        system_id: split_data[system_id]["train"]
        for system_id in system_ids
    }
    base_env = pipeline.RenewableDemandResponseEnv(
        data_by_system=training_frames,
        normalization=normalization,
        all_system_ids=system_ids,
        balanced_sampling=True,
        fixed_system=None,
        deterministic_starts=False,
        seed=seed,
    )
    return Monitor(base_env)


def load_model(
    algorithm: str,
    path: Path,
    env=None,
    device: str = "cpu",
):
    cls = MODEL_CLASSES[algorithm]
    return cls.load(str(path), env=env, device=device)


def resume_missing_model(
    algorithm: str,
    seed: int,
    target_timesteps: int,
    checkpoint_steps: List[int],
    checkpoints_dir: Path,
    models_dir: Path,
    split_data,
    normalization,
    system_ids,
    device: str,
) -> Tuple[object, Optional[pd.DataFrame], Dict]:
    destination = final_model_path(models_dir, algorithm, seed)

    if destination.exists():
        print(f"[SKIP] Final model exists: {destination.name}")
        model = load_model(algorithm, destination, env=None, device=device)
        return model, None, {
            "method": algorithm,
            "seed": seed,
            "status": "already_complete",
            "starting_timestep": target_timesteps,
            "additional_timesteps": 0,
            "training_seconds": 0.0,
            "source_model": str(destination),
            "final_model": str(destination),
        }

    latest = find_latest_checkpoint(
        checkpoints_dir, algorithm, seed, target_timesteps
    )
    if latest is None:
        raise RuntimeError(
            f"No final model and no checkpoint found for {algorithm}, seed {seed}. "
            "Do not restart blindly. Check the checkpoints folder."
        )

    checkpoint_path, checkpoint_timestep = latest
    remaining = target_timesteps - checkpoint_timestep

    if remaining <= 0:
        model = load_model(algorithm, checkpoint_path, env=None, device=device)
        model.save(str(destination.with_suffix("")))
        return model, None, {
            "method": algorithm,
            "seed": seed,
            "status": "promoted_final_checkpoint",
            "starting_timestep": checkpoint_timestep,
            "additional_timesteps": 0,
            "training_seconds": 0.0,
            "source_model": str(checkpoint_path),
            "final_model": str(destination),
        }

    print(
        f"[RESUME] {algorithm}, seed {seed}: "
        f"{checkpoint_timestep:,} -> {target_timesteps:,} timesteps "
        f"({remaining:,} remaining)"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_training_environment(
        split_data=split_data,
        normalization=normalization,
        system_ids=system_ids,
        seed=seed,
    )

    model = load_model(
        algorithm,
        checkpoint_path,
        env=env,
        device=device,
    )

    future_steps = [
        step for step in checkpoint_steps
        if checkpoint_timestep < step <= target_timesteps
    ]

    callback = pipeline.EpisodeCheckpointCallback(
        checkpoint_steps=future_steps,
        checkpoint_dir=checkpoints_dir,
        algorithm_name=algorithm,
        seed=seed,
    )

    start = time.perf_counter()
    model.learn(
        total_timesteps=remaining,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=False,
    )
    elapsed = time.perf_counter() - start

    model.save(str(destination.with_suffix("")))
    env.close()

    history = pd.DataFrame(
        {
            "method": algorithm,
            "seed": seed,
            "resume_episode": np.arange(
                1, len(callback.episode_rewards) + 1
            ),
            "estimated_global_episode": (
                checkpoint_timestep // pipeline.EPISODE_LENGTH
                + np.arange(1, len(callback.episode_rewards) + 1)
            ),
            "reward": callback.episode_rewards,
            "episode_length": callback.episode_lengths,
            "history_scope": "resumed_segment_only",
        }
    )

    print(
        f"[DONE] {algorithm}, seed {seed} completed in "
        f"{elapsed / 60.0:.2f} minutes"
    )

    return model, history, {
        "method": algorithm,
        "seed": seed,
        "status": "resumed_and_completed",
        "starting_timestep": checkpoint_timestep,
        "additional_timesteps": remaining,
        "training_seconds": elapsed,
        "source_model": str(checkpoint_path),
        "final_model": str(destination),
    }


def evaluate_all(
    output_dir: Path,
    config: Dict,
    split_data,
    normalization,
    system_ids,
    device: str,
) -> None:
    models_dir = output_dir / "models"
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    algorithms = list(config.get("algorithms", pipeline.ALGORITHMS))
    seeds = [int(seed) for seed in config.get("seeds", pipeline.SEEDS)]
    evaluation_episodes = int(
        config.get(
            "evaluation_episodes_per_system",
            pipeline.PROFILE_EVALUATION_EPISODES[
                config.get("profile", "full")
            ],
        )
    )

    episode_frames: List[pd.DataFrame] = []
    trace_frames: List[pd.DataFrame] = []

    print("\n" + "=" * 80)
    print("EVALUATING ALL COMPLETED MODELS")
    print("=" * 80)

    for algorithm in algorithms:
        for seed in seeds:
            path = final_model_path(models_dir, algorithm, seed)
            if not path.exists():
                raise FileNotFoundError(
                    f"Final model is still missing: {path}"
                )

            print(f"Evaluating {algorithm}, seed {seed}...")
            model = load_model(
                algorithm, path, env=None, device=device
            )

            for system_id in system_ids:
                env = pipeline.RenewableDemandResponseEnv(
                    data_by_system={
                        system_id: split_data[system_id]["test"]
                    },
                    normalization=normalization,
                    all_system_ids=system_ids,
                    balanced_sampling=False,
                    fixed_system=system_id,
                    deterministic_starts=True,
                    seed=seed,
                )

                metrics, traces = pipeline.evaluate_policy(
                    model=model,
                    environment=env,
                    episodes=evaluation_episodes,
                    method_name=algorithm,
                )
                metrics["seed"] = seed
                traces["seed"] = seed
                episode_frames.append(metrics)
                trace_frames.append(traces)
                env.close()

    baseline_seed = seeds[0]
    print("Evaluating RuleBased baseline...")

    for system_id in system_ids:
        env = pipeline.RenewableDemandResponseEnv(
            data_by_system={
                system_id: split_data[system_id]["test"]
            },
            normalization=normalization,
            all_system_ids=system_ids,
            balanced_sampling=False,
            fixed_system=system_id,
            deterministic_starts=True,
            seed=baseline_seed,
        )

        metrics, traces = pipeline.evaluate_policy(
            model=None,
            environment=env,
            episodes=evaluation_episodes,
            method_name="RuleBased",
        )
        metrics["seed"] = baseline_seed
        traces["seed"] = baseline_seed
        episode_frames.append(metrics)
        trace_frames.append(traces)
        env.close()

    episode_results = pd.concat(episode_frames, ignore_index=True)
    evaluation_traces = pd.concat(trace_frames, ignore_index=True)

    episode_results.to_csv(
        evaluation_dir / "episode_level_results.csv",
        index=False,
    )

    pipeline.aggregate_results(
        episode_results,
        ["method", "electricity_system", "geographic_region"],
    ).to_csv(
        evaluation_dir / "electricity_system_results.csv",
        index=False,
    )

    pipeline.aggregate_results(
        episode_results,
        ["method", "geographic_region"],
    ).to_csv(
        evaluation_dir / "geographic_region_results.csv",
        index=False,
    )

    pipeline.aggregate_results(
        episode_results,
        ["method"],
    ).to_csv(
        evaluation_dir / "overall_results.csv",
        index=False,
    )

    evaluation_traces.to_csv(
        evaluation_dir / "evaluation_traces.csv.gz",
        index=False,
        compression="gzip",
    )

    print("\nEvaluation files created successfully:")
    for name in (
        "episode_level_results.csv",
        "electricity_system_results.csv",
        "geographic_region_results.csv",
        "overall_results.csv",
        "evaluation_traces.csv.gz",
    ):
        print(f"  - {evaluation_dir / name}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    data_path = Path(args.data).resolve()
    device = resolve_device(args.device)

    config = load_experiment_config(output_dir)

    algorithms = list(config.get("algorithms", pipeline.ALGORITHMS))
    seeds = [int(seed) for seed in config.get("seeds", pipeline.SEEDS)]
    target_timesteps = int(config["timesteps_per_run"])
    checkpoint_episodes = [
        int(value) for value in config["checkpoints"]
    ]
    checkpoint_steps = [
        episode * int(config["episode_length"])
        for episode in checkpoint_episodes
    ]

    models_dir = output_dir / "models"
    checkpoints_dir = output_dir / "checkpoints"
    training_dir = output_dir / "training"

    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INTERRUPTED EXPERIMENT RECOVERY")
    print("=" * 80)
    print(f"Output folder: {output_dir}")
    print(f"Dataset: {data_path}")
    print(f"Device: {device}")
    print(f"Target timesteps per run: {target_timesteps:,}")
    print(f"Algorithms: {algorithms}")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    print("\nReloading and preparing the dataset...")
    prepared, eligible, excluded = pipeline.discover_and_prepare_data(data_path)
    split_data, split_manifest = pipeline.chronological_split(prepared)
    system_ids = sorted(split_data)

    # Use the exact normalization already saved for the interrupted run.
    normalization_path = (
        output_dir / "configuration" / "normalization.json"
    )
    if not normalization_path.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {normalization_path}"
        )
    normalization = json.loads(
        normalization_path.read_text(encoding="utf-8")
    )

    recovery_rows: List[Dict] = []
    resume_histories: List[pd.DataFrame] = []

    if not args.evaluation_only:
        for algorithm in algorithms:
            for seed in seeds:
                _, history, recovery = resume_missing_model(
                    algorithm=algorithm,
                    seed=seed,
                    target_timesteps=target_timesteps,
                    checkpoint_steps=checkpoint_steps,
                    checkpoints_dir=checkpoints_dir,
                    models_dir=models_dir,
                    split_data=split_data,
                    normalization=normalization,
                    system_ids=system_ids,
                    device=device,
                )
                recovery_rows.append(recovery)
                if history is not None and not history.empty:
                    resume_histories.append(history)

    pd.DataFrame(recovery_rows).to_csv(
        training_dir / "recovery_manifest.csv",
        index=False,
    )

    if resume_histories:
        pd.concat(
            resume_histories, ignore_index=True
        ).to_csv(
            training_dir / "resume_training_history.csv",
            index=False,
        )

    note = (
        "The original run.py retained episode-level training history in memory "
        "and wrote it only after all runs completed. Because the terminal was "
        "closed, exact histories for already completed models were not persisted "
        "and cannot be reconstructed from model ZIP files. "
        "resume_training_history.csv contains only newly resumed episodes. "
        "All final models and deterministic evaluation outputs remain usable.\n"
    )
    (training_dir / "RECOVERY_NOTE.txt").write_text(
        note, encoding="utf-8"
    )

    evaluate_all(
        output_dir=output_dir,
        config=config,
        split_data=split_data,
        normalization=normalization,
        system_ids=system_ids,
        device=device,
    )

    print("\n" + "=" * 80)
    print("RECOVERY COMPLETED")
    print("=" * 80)
    print("The interrupted model was resumed from its latest checkpoint.")
    print("All final models were evaluated.")
    print("Do not run the original run.py again for this results_v2 folder.")


if __name__ == "__main__":
    main()
