
"""
run.py
======

Final CSV-only training and evaluation pipeline for renewable-aware demand
response using the OPSD hourly time-series dataset.

The script:

- reads D:\\FINKI\\ai-driven\\datasets\\time_series_60min_singleindex.csv
- automatically detects every eligible electricity system
- requires actual load, solar generation, wind generation, and sufficient data
- groups eligible systems into European geographic regions
- balances pooled training across geographic regions and electricity systems
- trains PPO, SAC, and TD3 with seeds 11, 25, and 125
- uses 168-hour weekly episodes
- runs 1,500 episodes per algorithm and seed
- evaluates each policy separately on every eligible electricity system
- evaluates a renewable-aware rule-based baseline
- saves models, checkpoints, CSV tables, JSON metadata, and compressed traces
- creates no charts and does not import matplotlib

Install dependencies
--------------------
pip install pandas numpy gymnasium stable-baselines3 torch

Run full experiment
-------------------
python run.py

Optional pilot
--------------
python run.py --profile pilot

Optional explicit dataset path
------------------------------
python run.py --data "D:\\FINKI\\ai-driven\\datasets\\time_series_60min_singleindex.csv"

The original dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIR / "datasets" / "time_series_60min_singleindex.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_v2"

ALGORITHMS = ("PPO", "SAC", "TD3")
SEEDS = (11, 25, 125)

EPISODE_LENGTH = 168
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

MIN_USABLE_ROWS = 8_000
MAX_MISSING_SHARE = 0.20
MIN_CONTIGUOUS_HOURS = EPISODE_LENGTH * 8

FLEXIBLE_LOAD_FRACTION = 0.10
MAX_HOURLY_SHIFT_FRACTION = 0.04
MAX_BACKLOG_FRACTION = 0.15
MAX_DEFERRAL_HOURS = 24
ACTION_SATURATION_THRESHOLD = 0.95

PROFILE_EPISODES = {
    "pilot": 30,
    "quick": 300,
    "full": 1500,
}

PROFILE_CHECKPOINTS = {
    "pilot": (10, 20, 30),
    "quick": (60, 150, 300),
    "full": (300, 750, 1500),
}

PROFILE_EVALUATION_EPISODES = {
    "pilot": 2,
    "quick": 6,
    "full": 12,
}

REWARD_WEIGHTS = {
    "renewable_alignment": 0.30,
    "peak_reduction": 0.20,
    "ramp_reduction": 0.15,
    "action_magnitude_penalty": 0.08,
    "action_change_penalty": 0.07,
    "backlog_penalty": 0.10,
    "aged_backlog_penalty": 0.05,
    "energy_balance_penalty": 0.05,
}


# =============================================================================
# GEOGRAPHIC REGION MAPPING
# =============================================================================

NORTHERN = {
    "DK", "DK_1", "DK_2", "SE", "SE_1", "SE_2", "SE_3", "SE_4",
    "NO", "NO_1", "NO_2", "NO_3", "NO_4", "NO_5", "FI", "EE", "LV", "LT",
}

WESTERN = {
    "DE", "DE_AT_LU", "DE_LU", "FR", "BE", "NL", "LU",
    "GB", "GB_GBN", "GB_NIR", "IE", "CH",
}

CENTRAL = {
    "AT", "CZ", "PL", "SK", "HU",
}

SOUTHERN = {
    "ES", "PT", "IT", "IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD",
    "IT_SICI", "IT_SARD", "GR", "MT", "CY",
}

SOUTHEASTERN = {
    "RO", "BG", "HR", "SI", "RS", "BA", "ME", "MK", "AL", "XK",
}


def geographic_group(system_id: str) -> str:
    if system_id in NORTHERN:
        return "Northern Europe"
    if system_id in WESTERN:
        return "Western Europe"
    if system_id in CENTRAL:
        return "Central Europe"
    if system_id in SOUTHERN:
        return "Southern Europe"
    if system_id in SOUTHEASTERN:
        return "Southeastern Europe"

    prefix = system_id.split("_")[0]

    if prefix in {"DK", "SE", "NO", "FI", "EE", "LV", "LT"}:
        return "Northern Europe"
    if prefix in {"DE", "FR", "BE", "NL", "LU", "GB", "IE", "CH"}:
        return "Western Europe"
    if prefix in {"AT", "CZ", "PL", "SK", "HU"}:
        return "Central Europe"
    if prefix in {"ES", "PT", "IT", "GR", "MT", "CY"}:
        return "Southern Europe"
    if prefix in {"RO", "BG", "HR", "SI", "RS", "BA", "ME", "MK", "AL", "XK"}:
        return "Southeastern Europe"

    return "Other Europe"


# =============================================================================
# CONFIGURATION RECORD
# =============================================================================

@dataclass
class RunConfig:
    profile: str
    data_file: str
    output_dir: str
    algorithms: Tuple[str, ...]
    seeds: Tuple[int, ...]
    episode_length: int
    episodes_per_run: int
    timesteps_per_run: int
    checkpoints: Tuple[int, ...]
    evaluation_episodes_per_system: int
    train_fraction: float
    validation_fraction: float
    min_usable_rows: int
    min_contiguous_hours: int
    flexible_load_fraction: float
    max_hourly_shift_fraction: float
    max_backlog_fraction: float
    max_deferral_hours: int
    reward_weights: Dict[str, float]


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def discover_system_ids(columns: Sequence[str]) -> List[str]:
    """
    Detect electricity-system prefixes from actual-load columns.
    """
    suffixes = (
        "_load_actual_entsoe_transparency",
        "_load_actual",
    )

    systems = set()

    for column in columns:
        for suffix in suffixes:
            if column.endswith(suffix):
                systems.add(column[: -len(suffix)])

    return sorted(systems)


def identify_columns(columns: Sequence[str], system_id: str) -> Dict[str, Optional[str]]:
    return {
        "load": first_existing(
            columns,
            [
                f"{system_id}_load_actual_entsoe_transparency",
                f"{system_id}_load_actual",
            ],
        ),
        "forecast": first_existing(
            columns,
            [
                f"{system_id}_load_forecast_entsoe_transparency",
                f"{system_id}_load_forecast",
            ],
        ),
        "solar": first_existing(
            columns,
            [
                f"{system_id}_solar_generation_actual",
            ],
        ),
        "wind_total": first_existing(
            columns,
            [
                f"{system_id}_wind_generation_actual",
            ],
        ),
        "wind_onshore": first_existing(
            columns,
            [
                f"{system_id}_wind_onshore_generation_actual",
            ],
        ),
        "wind_offshore": first_existing(
            columns,
            [
                f"{system_id}_wind_offshore_generation_actual",
            ],
        ),
        "price": first_existing(
            columns,
            [
                f"{system_id}_price_day_ahead",
            ],
        ),
    }


def longest_hourly_contiguous_run(timestamps: pd.Series) -> int:
    if timestamps.empty:
        return 0

    ordered = timestamps.sort_values().reset_index(drop=True)
    diffs = ordered.diff().dt.total_seconds().div(3600)

    longest = 1
    current = 1

    for value in diffs.iloc[1:]:
        if np.isfinite(value) and abs(value - 1.0) < 1e-9:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return int(longest)


def prepare_single_system(
    raw: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame["timestamp"] = raw["utc_timestamp"]
    frame["load"] = pd.to_numeric(raw[mapping["load"]], errors="coerce")
    frame["solar"] = pd.to_numeric(raw[mapping["solar"]], errors="coerce")

    if mapping["wind_total"] is not None:
        frame["wind"] = pd.to_numeric(raw[mapping["wind_total"]], errors="coerce")
    else:
        wind_parts = []
        for key in ("wind_onshore", "wind_offshore"):
            column = mapping[key]
            if column is not None:
                wind_parts.append(pd.to_numeric(raw[column], errors="coerce"))

        if not wind_parts:
            frame["wind"] = np.nan
        else:
            frame["wind"] = pd.concat(wind_parts, axis=1).sum(axis=1, min_count=1)

    if mapping["forecast"] is not None:
        frame["load_forecast"] = pd.to_numeric(
            raw[mapping["forecast"]], errors="coerce"
        )
    else:
        frame["load_forecast"] = frame["load"].shift(24)

    if mapping["price"] is not None:
        frame["price"] = pd.to_numeric(raw[mapping["price"]], errors="coerce")
    else:
        frame["price"] = np.nan

    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.set_index("timestamp").asfreq("h")

    core_columns = ["load", "solar", "wind"]

    for column in core_columns:
        frame[column] = frame[column].interpolate(
            method="time",
            limit=3,
            limit_direction="both",
        )

    frame["load_forecast"] = frame["load_forecast"].interpolate(
        method="time",
        limit=24,
        limit_direction="both",
    )
    frame["load_forecast"] = frame["load_forecast"].fillna(frame["load"])

    frame = frame.reset_index()
    frame = frame.dropna(subset=core_columns)

    frame = frame[
        (frame["load"] > 0)
        & (frame["solar"] >= 0)
        & (frame["wind"] >= 0)
    ].copy()

    frame["renewable"] = frame["solar"] + frame["wind"]
    frame["net_load"] = frame["load"] - frame["renewable"]
    frame["renewable_penetration"] = (
        frame["renewable"] / frame["load"].replace(0, np.nan)
    ).clip(lower=0.0, upper=3.0)

    frame["load_ramp"] = frame["load"].diff().abs().fillna(0.0)
    frame["net_load_ramp"] = frame["net_load"].diff().abs().fillna(0.0)

    hour = frame["timestamp"].dt.hour
    weekday = frame["timestamp"].dt.dayofweek

    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * weekday / 7)

    return frame.reset_index(drop=True)


def discover_and_prepare_data(
    data_file: Path,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    if not data_file.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{data_file}\n\n"
            "Expected default location:\n"
            f"{DEFAULT_DATA_FILE}\n"
        )

    print(f"Reading dataset:\n{data_file}")
    raw = pd.read_csv(data_file, low_memory=False)

    if "utc_timestamp" not in raw.columns:
        raise ValueError("Required column 'utc_timestamp' was not found.")

    raw["utc_timestamp"] = pd.to_datetime(
        raw["utc_timestamp"],
        utc=True,
        errors="coerce",
    )
    raw = raw.dropna(subset=["utc_timestamp"]).sort_values("utc_timestamp")

    columns = list(raw.columns)
    system_ids = discover_system_ids(columns)

    if not system_ids:
        raise ValueError("No OPSD electricity-system prefixes were detected.")

    print(f"Detected {len(system_ids)} candidate electricity systems.")

    prepared: Dict[str, pd.DataFrame] = {}
    eligible_rows: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []

    for system_id in system_ids:
        mapping = identify_columns(columns, system_id)
        reasons: List[str] = []

        if mapping["load"] is None:
            reasons.append("missing actual load")
        if mapping["solar"] is None:
            reasons.append("missing actual solar")

        has_wind = any(
            mapping[key] is not None
            for key in ("wind_total", "wind_onshore", "wind_offshore")
        )
        if not has_wind:
            reasons.append("missing actual wind")

        if reasons:
            excluded_rows.append(
                {
                    "electricity_system": system_id,
                    "geographic_region": geographic_group(system_id),
                    "reason": "; ".join(reasons),
                }
            )
            continue

        frame = prepare_single_system(raw, mapping)

        if len(frame) < MIN_USABLE_ROWS:
            reasons.append(
                f"insufficient usable rows ({len(frame)} < {MIN_USABLE_ROWS})"
            )

        contiguous = longest_hourly_contiguous_run(frame["timestamp"])
        if contiguous < MIN_CONTIGUOUS_HOURS:
            reasons.append(
                f"insufficient continuous hourly coverage "
                f"({contiguous} < {MIN_CONTIGUOUS_HOURS})"
            )

        if reasons:
            excluded_rows.append(
                {
                    "electricity_system": system_id,
                    "geographic_region": geographic_group(system_id),
                    "reason": "; ".join(reasons),
                }
            )
            continue

        prepared[system_id] = frame

        eligible_rows.append(
            {
                "electricity_system": system_id,
                "geographic_region": geographic_group(system_id),
                "usable_rows": len(frame),
                "longest_contiguous_hours": contiguous,
                "start_timestamp": str(frame["timestamp"].min()),
                "end_timestamp": str(frame["timestamp"].max()),
                "load_column": mapping["load"],
                "solar_column": mapping["solar"],
                "wind_total_column": mapping["wind_total"],
                "wind_onshore_column": mapping["wind_onshore"],
                "wind_offshore_column": mapping["wind_offshore"],
                "forecast_column": mapping["forecast"],
                "price_column": mapping["price"],
            }
        )

        print(
            f"  Included {system_id:12s} | "
            f"{geographic_group(system_id):22s} | "
            f"{len(frame):,} rows"
        )

    eligible = pd.DataFrame(eligible_rows)
    excluded = pd.DataFrame(excluded_rows)

    if len(prepared) < 3:
        raise ValueError(
            "Fewer than three eligible electricity systems were found. "
            "Inspect excluded_regions.csv and the OPSD columns."
        )

    print(f"\nEligible systems: {len(prepared)}")
    print(f"Excluded systems: {len(excluded)}")

    return prepared, eligible, excluded


# =============================================================================
# SPLITTING AND NORMALIZATION
# =============================================================================

def chronological_split(
    prepared: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
    split_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    rows: List[Dict[str, object]] = []

    for system_id, frame in prepared.items():
        n = len(frame)
        train_end = int(n * TRAIN_FRACTION)
        validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

        parts = {
            "train": frame.iloc[:train_end].reset_index(drop=True),
            "validation": frame.iloc[train_end:validation_end].reset_index(drop=True),
            "test": frame.iloc[validation_end:].reset_index(drop=True),
        }

        valid = True
        for split_name, split_frame in parts.items():
            if len(split_frame) < EPISODE_LENGTH:
                valid = False
                break

        if not valid:
            continue

        split_data[system_id] = parts

        for split_name, split_frame in parts.items():
            rows.append(
                {
                    "electricity_system": system_id,
                    "geographic_region": geographic_group(system_id),
                    "split": split_name,
                    "rows": len(split_frame),
                    "start_timestamp": str(split_frame["timestamp"].min()),
                    "end_timestamp": str(split_frame["timestamp"].max()),
                }
            )

    if len(split_data) < 3:
        raise ValueError("Insufficient systems remain after chronological splitting.")

    return split_data, pd.DataFrame(rows)


def compute_normalization(
    split_data: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Dict[str, float]]:
    combined = pd.concat(
        [parts["train"] for parts in split_data.values()],
        ignore_index=True,
    )

    features = [
        "load",
        "load_forecast",
        "solar",
        "wind",
        "renewable",
        "net_load",
        "renewable_penetration",
        "load_ramp",
    ]

    stats: Dict[str, Dict[str, float]] = {}

    for feature in features:
        values = (
            combined[feature]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        mean = float(values.mean())
        std = float(values.std())

        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        stats[feature] = {
            "mean": mean,
            "std": std,
        }

    return stats


# =============================================================================
# BALANCED POOLED ENVIRONMENT
# =============================================================================

class RenewableDemandResponseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_by_system: Dict[str, pd.DataFrame],
        normalization: Dict[str, Dict[str, float]],
        all_system_ids: Sequence[str],
        balanced_sampling: bool,
        fixed_system: Optional[str] = None,
        deterministic_starts: bool = False,
        seed: int = 11,
    ) -> None:
        super().__init__()

        self.data_by_system = data_by_system
        self.normalization = normalization
        self.all_system_ids = list(all_system_ids)
        self.available_systems = list(data_by_system.keys())
        self.balanced_sampling = balanced_sampling
        self.fixed_system = fixed_system
        self.deterministic_starts = deterministic_starts
        self.rng = np.random.default_rng(seed)

        self.region_to_systems: Dict[str, List[str]] = {}
        for system_id in self.available_systems:
            group = geographic_group(system_id)
            self.region_to_systems.setdefault(group, []).append(system_id)

        self.available_regions = sorted(self.region_to_systems)

        # 8 normalized energy variables
        # 4 cyclical time variables
        # 4 internal control variables
        # N one-hot electricity-system indicators
        observation_size = 16 + len(self.all_system_ids)

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_system = self.available_systems[0]
        self.current_frame: Optional[pd.DataFrame] = None
        self.start_index = 0
        self.local_step = 0

        self.backlog = 0.0
        self.backlog_ages: List[Tuple[float, int]] = []
        self.previous_action = 0.0
        self.previous_controlled_load = 0.0
        self.cumulative_original_energy = 0.0
        self.cumulative_controlled_energy = 0.0
        self.trace: List[Dict[str, object]] = []
        self._deterministic_counter = 0

    def _z(self, feature: str, value: float) -> float:
        stat = self.normalization[feature]
        return float((value - stat["mean"]) / stat["std"])

    def _choose_system(self) -> str:
        if self.fixed_system is not None:
            return self.fixed_system

        if not self.balanced_sampling:
            return str(self.rng.choice(self.available_systems))

        chosen_region = str(self.rng.choice(self.available_regions))
        return str(self.rng.choice(self.region_to_systems[chosen_region]))

    def _choose_start(self, frame: pd.DataFrame) -> int:
        max_start = len(frame) - EPISODE_LENGTH

        if max_start <= 0:
            return 0

        if self.deterministic_starts:
            start = (
                self._deterministic_counter * EPISODE_LENGTH
            ) % (max_start + 1)
            self._deterministic_counter += 1
            return int(start)

        return int(self.rng.integers(0, max_start + 1))

    def _aged_backlog(self) -> float:
        return float(
            sum(amount for amount, age in self.backlog_ages if age >= MAX_DEFERRAL_HOURS)
        )

    def _increment_backlog_ages(self) -> None:
        self.backlog_ages = [
            (amount, age + 1)
            for amount, age in self.backlog_ages
            if amount > 1e-9
        ]

    def _add_backlog(self, amount: float) -> None:
        if amount > 0:
            self.backlog_ages.append((float(amount), 0))

    def _restore_backlog(self, amount: float) -> float:
        remaining = max(0.0, float(amount))
        restored = 0.0
        updated: List[Tuple[float, int]] = []

        for backlog_amount, age in sorted(
            self.backlog_ages,
            key=lambda item: item[1],
            reverse=True,
        ):
            if remaining <= 1e-12:
                updated.append((backlog_amount, age))
                continue

            used = min(backlog_amount, remaining)
            backlog_amount -= used
            remaining -= used
            restored += used

            if backlog_amount > 1e-9:
                updated.append((backlog_amount, age))

        self.backlog_ages = updated
        self.backlog = max(0.0, self.backlog - restored)

        return restored

    def _observation(self) -> np.ndarray:
        assert self.current_frame is not None

        row = self.current_frame.iloc[self.start_index + self.local_step]
        load = max(float(row["load"]), 1.0)

        max_backlog = MAX_BACKLOG_FRACTION * load
        backlog_ratio = np.clip(self.backlog / max(max_backlog, 1.0), 0.0, 2.0)
        aged_ratio = np.clip(
            self._aged_backlog() / max(max_backlog, 1.0),
            0.0,
            2.0,
        )
        energy_balance_ratio = np.clip(
            (
                self.cumulative_controlled_energy
                - self.cumulative_original_energy
            )
            / max(self.cumulative_original_energy, load),
            -2.0,
            2.0,
        )

        one_hot = np.zeros(len(self.all_system_ids), dtype=np.float32)
        one_hot[self.all_system_ids.index(self.current_system)] = 1.0

        core = np.array(
            [
                self._z("load", float(row["load"])),
                self._z("load_forecast", float(row["load_forecast"])),
                self._z("solar", float(row["solar"])),
                self._z("wind", float(row["wind"])),
                self._z("renewable", float(row["renewable"])),
                self._z("net_load", float(row["net_load"])),
                self._z(
                    "renewable_penetration",
                    float(row["renewable_penetration"]),
                ),
                self._z("load_ramp", float(row["load_ramp"])),
                float(row["hour_sin"]),
                float(row["hour_cos"]),
                float(row["dow_sin"]),
                float(row["dow_cos"]),
                float(backlog_ratio),
                float(aged_ratio),
                float(self.previous_action),
                float(energy_balance_ratio),
            ],
            dtype=np.float32,
        )

        return np.concatenate([core, one_hot]).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_system = self._choose_system()
        self.current_frame = self.data_by_system[self.current_system]
        self.start_index = self._choose_start(self.current_frame)
        self.local_step = 0

        self.backlog = 0.0
        self.backlog_ages = []
        self.previous_action = 0.0
        self.cumulative_original_energy = 0.0
        self.cumulative_controlled_energy = 0.0
        self.trace = []

        first_row = self.current_frame.iloc[self.start_index]
        self.previous_controlled_load = float(first_row["load"])

        return self._observation(), {
            "electricity_system": self.current_system,
            "geographic_region": geographic_group(self.current_system),
            "start_index": self.start_index,
        }

    def step(self, action):
        assert self.current_frame is not None

        row = self.current_frame.iloc[self.start_index + self.local_step]

        original_load = float(row["load"])
        renewable = float(row["renewable"])
        original_net_load = original_load - renewable

        action_value = float(
            np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)
        )

        self._increment_backlog_ages()

        max_hourly_shift = MAX_HOURLY_SHIFT_FRACTION * original_load
        max_backlog = MAX_BACKLOG_FRACTION * original_load
        flexible_limit = FLEXIBLE_LOAD_FRACTION * original_load

        aged_backlog = self._aged_backlog()
        forced_restore = min(
            aged_backlog,
            max_hourly_shift,
        )

        actual_shift = 0.0
        forced_restoration_MW = 0.0

        if forced_restore > 0:
            restored = self._restore_backlog(forced_restore)
            actual_shift += restored
            forced_restoration_MW = restored

        available_shift_capacity = max(0.0, max_hourly_shift - abs(actual_shift))
        requested_shift = action_value * available_shift_capacity

        if requested_shift < 0:
            deferrable = min(
                abs(requested_shift),
                flexible_limit,
                max(0.0, max_backlog - self.backlog),
            )
            actual_shift -= deferrable
            self.backlog += deferrable
            self._add_backlog(deferrable)
        elif requested_shift > 0:
            restored = self._restore_backlog(requested_shift)
            actual_shift += restored

        controlled_load = max(0.0, original_load + actual_shift)
        controlled_net_load = controlled_load - renewable

        self.cumulative_original_energy += original_load
        self.cumulative_controlled_energy += controlled_load

        scale = max(original_load, 1.0)

        renewable_signal = np.clip(
            (renewable - original_load) / scale,
            -1.0,
            1.0,
        )

        alignment_score = np.clip(
            action_value * renewable_signal,
            -1.0,
            1.0,
        )

        peak_score = np.clip(
            (
                abs(original_net_load)
                - abs(controlled_net_load)
            )
            / scale,
            -1.0,
            1.0,
        )

        original_ramp = abs(original_load - self.previous_controlled_load)
        controlled_ramp = abs(controlled_load - self.previous_controlled_load)

        ramp_score = np.clip(
            (original_ramp - controlled_ramp) / scale,
            -1.0,
            1.0,
        )

        action_magnitude_penalty = abs(action_value)
        action_change_penalty = abs(action_value - self.previous_action)
        backlog_penalty = self.backlog / max(max_backlog, 1.0)
        aged_backlog_penalty = self._aged_backlog() / max(max_backlog, 1.0)

        energy_balance_error = abs(
            self.cumulative_controlled_energy
            - self.cumulative_original_energy
            + self.backlog
        ) / max(self.cumulative_original_energy, scale)

        reward = (
            REWARD_WEIGHTS["renewable_alignment"] * alignment_score
            + REWARD_WEIGHTS["peak_reduction"] * peak_score
            + REWARD_WEIGHTS["ramp_reduction"] * ramp_score
            - REWARD_WEIGHTS["action_magnitude_penalty"] * action_magnitude_penalty
            - REWARD_WEIGHTS["action_change_penalty"] * action_change_penalty
            - REWARD_WEIGHTS["backlog_penalty"] * backlog_penalty
            - REWARD_WEIGHTS["aged_backlog_penalty"] * aged_backlog_penalty
            - REWARD_WEIGHTS["energy_balance_penalty"] * energy_balance_error
        )

        constraint_violation = int(
            self.backlog > max_backlog + 1e-6
            or abs(actual_shift) > max_hourly_shift + 1e-6
        )

        self.trace.append(
            {
                "timestamp": row["timestamp"],
                "electricity_system": self.current_system,
                "geographic_region": geographic_group(self.current_system),
                "original_load_MW": original_load,
                "controlled_load_MW": controlled_load,
                "renewable_MW": renewable,
                "solar_MW": float(row["solar"]),
                "wind_MW": float(row["wind"]),
                "original_net_load_MW": original_net_load,
                "controlled_net_load_MW": controlled_net_load,
                "action": action_value,
                "actual_shift_MW": actual_shift,
                "forced_restoration_MW": forced_restoration_MW,
                "backlog_MWh": self.backlog,
                "aged_backlog_MWh": self._aged_backlog(),
                "energy_balance_error_fraction": energy_balance_error,
                "constraint_violation": constraint_violation,
                "reward": float(reward),
            }
        )

        self.previous_action = action_value
        self.previous_controlled_load = controlled_load
        self.local_step += 1

        terminated = self.local_step >= EPISODE_LENGTH
        truncated = False

        if terminated:
            terminal_backlog_ratio = self.backlog / max(max_backlog, 1.0)
            terminal_energy_error = abs(
                self.cumulative_controlled_energy
                - self.cumulative_original_energy
                + self.backlog
            ) / max(self.cumulative_original_energy, scale)

            reward -= 0.50 * terminal_backlog_ratio
            reward -= 0.25 * terminal_energy_error

        if terminated:
            next_observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )
        else:
            next_observation = self._observation()

        info = {
            "electricity_system": self.current_system,
            "geographic_region": geographic_group(self.current_system),
            "original_load_MW": original_load,
            "controlled_load_MW": controlled_load,
            "renewable_MW": renewable,
            "backlog_MWh": self.backlog,
            "aged_backlog_MWh": self._aged_backlog(),
            "actual_shift_MW": actual_shift,
            "constraint_violation": constraint_violation,
        }

        return next_observation, float(reward), terminated, truncated, info


# =============================================================================
# RULE-BASED BASELINE
# =============================================================================

def rule_based_action(observation: np.ndarray) -> np.ndarray:
    net_load_z = float(observation[5])
    renewable_penetration_z = float(observation[6])
    backlog_ratio = float(observation[12])
    aged_backlog_ratio = float(observation[13])
    previous_action = float(observation[14])

    if aged_backlog_ratio > 0.05:
        action = 0.80
    elif backlog_ratio > 0.35 and renewable_penetration_z > -0.10:
        action = 0.55
    elif renewable_penetration_z > 0.60:
        action = 0.45
    elif net_load_z > 0.75 and backlog_ratio < 0.70:
        action = -0.45
    elif net_load_z < -0.35 and backlog_ratio > 0.05:
        action = 0.40
    else:
        action = 0.0

    # Limit abrupt control changes.
    action = np.clip(action, previous_action - 0.35, previous_action + 0.35)

    return np.array([action], dtype=np.float32)


# =============================================================================
# TRAINING CALLBACK AND MODELS
# =============================================================================

class EpisodeCheckpointCallback(BaseCallback):
    def __init__(
        self,
        checkpoint_steps: Sequence[int],
        checkpoint_dir: Path,
        algorithm_name: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.checkpoint_steps = sorted(set(int(step) for step in checkpoint_steps))
        self.checkpoint_dir = checkpoint_dir
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.saved_steps: set[int] = set()

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self.current_reward += float(np.asarray(rewards).reshape(-1)[0])
            self.current_length += 1

        if dones is not None and bool(np.asarray(dones).reshape(-1)[0]):
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.current_reward = 0.0
            self.current_length = 0

        for checkpoint_step in self.checkpoint_steps:
            if (
                self.num_timesteps >= checkpoint_step
                and checkpoint_step not in self.saved_steps
            ):
                path = self.checkpoint_dir / (
                    f"{self.algorithm_name}_seed{self.seed}_"
                    f"step{checkpoint_step}"
                )
                self.model.save(path)
                self.saved_steps.add(checkpoint_step)

        return True


def build_model(
    algorithm_name: str,
    environment: gym.Env,
    seed: int,
    device: str,
):
    common = {
        "policy": "MlpPolicy",
        "env": environment,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "seed": seed,
        "device": device,
        "verbose": 0,
        "policy_kwargs": {
            "net_arch": [128, 128],
            "activation_fn": torch.nn.ReLU,
        },
    }

    if algorithm_name == "PPO":
        return PPO(
            **common,
            n_steps=1024,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    if algorithm_name == "SAC":
        return SAC(
            **common,
            buffer_size=150_000,
            learning_starts=2_000,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )

    if algorithm_name == "TD3":
        return TD3(
            **common,
            buffer_size=150_000,
            learning_starts=2_000,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm_name}")


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def safe_percentage(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return 100.0 * numerator / abs(denominator)


def trace_metrics(trace: pd.DataFrame) -> Dict[str, float]:
    original = trace["original_load_MW"].to_numpy(float)
    controlled = trace["controlled_load_MW"].to_numpy(float)
    renewable = trace["renewable_MW"].to_numpy(float)
    actions = trace["action"].to_numpy(float)
    backlog = trace["backlog_MWh"].to_numpy(float)

    original_net = original - renewable
    controlled_net = controlled - renewable

    original_peak = float(np.max(original))
    controlled_peak = float(np.max(controlled))

    original_par = original_peak / max(float(np.mean(original)), 1e-12)
    controlled_par = controlled_peak / max(float(np.mean(controlled)), 1e-12)

    original_ramps = np.abs(np.diff(original_net))
    controlled_ramps = np.abs(np.diff(controlled_net))

    original_alignment = float(
        np.mean(np.minimum(original, renewable))
        / max(np.mean(original), 1e-12)
    )
    controlled_alignment = float(
        np.mean(np.minimum(controlled, renewable))
        / max(np.mean(controlled), 1e-12)
    )

    cumulative_original = float(np.sum(original))
    cumulative_controlled = float(np.sum(controlled))
    final_backlog = float(backlog[-1])

    energy_balance_error = abs(
        cumulative_controlled
        - cumulative_original
        + final_backlog
    )

    action_changes = np.abs(np.diff(actions))
    saturation_rate = float(
        100.0 * np.mean(np.abs(actions) >= ACTION_SATURATION_THRESHOLD)
    )

    positive_backlog = backlog[backlog > 1e-9]
    mean_backlog = float(np.mean(positive_backlog)) if len(positive_backlog) else 0.0

    return {
        "mean_reward": float(trace["reward"].mean()),
        "episode_reward": float(trace["reward"].sum()),
        "peak_load_MW": controlled_peak,
        "peak_reduction_pct": safe_percentage(
            original_peak - controlled_peak,
            original_peak,
        ),
        "peak_to_average_ratio": controlled_par,
        "par_improvement_pct": safe_percentage(
            original_par - controlled_par,
            original_par,
        ),
        "mean_absolute_net_load_MW": float(np.mean(np.abs(controlled_net))),
        "mean_net_load_ramp_MW": float(np.mean(controlled_ramps)),
        "ramp_reduction_pct": safe_percentage(
            float(np.mean(original_ramps))
            - float(np.mean(controlled_ramps)),
            float(np.mean(original_ramps)),
        ),
        "renewable_alignment": controlled_alignment,
        "renewable_alignment_improvement_pct": safe_percentage(
            controlled_alignment - original_alignment,
            original_alignment,
        ),
        "total_shifted_energy_MWh": float(
            np.sum(np.abs(trace["actual_shift_MW"].to_numpy(float)))
        ),
        "mean_absolute_action": float(np.mean(np.abs(actions))),
        "mean_action_change": float(np.mean(action_changes))
        if len(action_changes)
        else 0.0,
        "action_variance": float(np.var(actions)),
        "action_saturation_rate_pct": saturation_rate,
        "mean_backlog_MWh": mean_backlog,
        "maximum_backlog_MWh": float(np.max(backlog)),
        "final_backlog_MWh": final_backlog,
        "unresolved_demand_MWh": final_backlog,
        "energy_balance_error_MWh": energy_balance_error,
        "energy_balance_error_pct": safe_percentage(
            energy_balance_error,
            cumulative_original,
        ),
        "constraint_violation_rate_pct": float(
            100.0 * trace["constraint_violation"].mean()
        ),
        "forced_restoration_MWh": float(
            trace["forced_restoration_MW"].sum()
        ),
    }


def evaluate_policy(
    model,
    environment: RenewableDemandResponseEnv,
    episodes: int,
    method_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: List[Dict[str, object]] = []
    traces: List[pd.DataFrame] = []

    for episode in range(episodes):
        observation, info = environment.reset()
        done = False
        inference_times: List[float] = []

        while not done:
            start = time.perf_counter()

            if method_name == "RuleBased":
                action = rule_based_action(observation)
            else:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

            inference_times.append(
                (time.perf_counter() - start) * 1000.0
            )

            observation, reward, terminated, truncated, info = (
                environment.step(action)
            )
            done = terminated or truncated

        trace = pd.DataFrame(environment.trace)
        result = trace_metrics(trace)
        result.update(
            {
                "method": method_name,
                "electricity_system": environment.current_system,
                "geographic_region": geographic_group(
                    environment.current_system
                ),
                "evaluation_episode": episode,
                "inference_latency_ms": float(np.mean(inference_times)),
            }
        )
        metrics_rows.append(result)

        trace["method"] = method_name
        trace["evaluation_episode"] = episode
        traces.append(trace)

    return pd.DataFrame(metrics_rows), pd.concat(traces, ignore_index=True)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def aggregate_results(
    episode_results: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    excluded = {
        "method",
        "electricity_system",
        "geographic_region",
        "evaluation_episode",
        "seed",
    }

    numeric_metrics = [
        column
        for column in episode_results.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(episode_results[column])
    ]

    result = (
        episode_results.groupby(list(group_columns))[numeric_metrics]
        .agg(["mean", "std", "median"])
    )

    result.columns = [
        f"{metric}_{stat}"
        for metric, stat in result.columns
    ]

    return result.reset_index()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate renewable-aware PPO, SAC, and TD3 "
            "across all eligible OPSD electricity systems."
        )
    )

    parser.add_argument(
        "--profile",
        choices=("pilot", "quick", "full"),
        default="full",
        help="full is the default final experiment",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help="Path to time_series_60min_singleindex.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch device",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    episodes_per_run = PROFILE_EPISODES[args.profile]
    timesteps_per_run = episodes_per_run * EPISODE_LENGTH
    checkpoint_episodes = PROFILE_CHECKPOINTS[args.profile]
    checkpoint_steps = [
        episode * EPISODE_LENGTH
        for episode in checkpoint_episodes
    ]
    evaluation_episodes = PROFILE_EVALUATION_EPISODES[args.profile]

    output_dir = Path(args.output).resolve()

    configuration_dir = output_dir / "configuration"
    training_dir = output_dir / "training"
    evaluation_dir = output_dir / "evaluation"
    models_dir = output_dir / "models"
    checkpoints_dir = output_dir / "checkpoints"

    for directory in (
        output_dir,
        configuration_dir,
        training_dir,
        evaluation_dir,
        models_dir,
        checkpoints_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        profile=args.profile,
        data_file=str(Path(args.data).resolve()),
        output_dir=str(output_dir),
        algorithms=tuple(args.algorithms),
        seeds=tuple(args.seeds),
        episode_length=EPISODE_LENGTH,
        episodes_per_run=episodes_per_run,
        timesteps_per_run=timesteps_per_run,
        checkpoints=tuple(checkpoint_episodes),
        evaluation_episodes_per_system=evaluation_episodes,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        min_usable_rows=MIN_USABLE_ROWS,
        min_contiguous_hours=MIN_CONTIGUOUS_HOURS,
        flexible_load_fraction=FLEXIBLE_LOAD_FRACTION,
        max_hourly_shift_fraction=MAX_HOURLY_SHIFT_FRACTION,
        max_backlog_fraction=MAX_BACKLOG_FRACTION,
        max_deferral_hours=MAX_DEFERRAL_HOURS,
        reward_weights=REWARD_WEIGHTS,
    )

    (configuration_dir / "experiment_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    print("=" * 86)
    print("FINAL MULTI-REGION RENEWABLE-AWARE DRL PIPELINE")
    print("=" * 86)
    print(f"Profile: {args.profile}")
    print(f"Dataset: {Path(args.data).resolve()}")
    print(f"Output: {output_dir}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per training run: {episodes_per_run:,}")
    print(f"Timesteps per training run: {timesteps_per_run:,}")
    print(f"Evaluation episodes per electricity system: {evaluation_episodes}")
    print("=" * 86)

    total_start = time.perf_counter()

    prepared, eligible, excluded = discover_and_prepare_data(
        Path(args.data)
    )

    split_data, split_manifest = chronological_split(prepared)
    final_system_ids = sorted(split_data)

    eligible = eligible[
        eligible["electricity_system"].isin(final_system_ids)
    ].reset_index(drop=True)

    eligible.to_csv(
        configuration_dir / "eligible_regions.csv",
        index=False,
    )
    excluded.to_csv(
        configuration_dir / "excluded_regions.csv",
        index=False,
    )
    split_manifest.to_csv(
        configuration_dir / "chronological_splits.csv",
        index=False,
    )

    region_mapping = eligible[
        ["electricity_system", "geographic_region"]
    ].drop_duplicates()

    region_mapping.to_csv(
        configuration_dir / "geographic_region_mapping.csv",
        index=False,
    )

    normalization = compute_normalization(split_data)
    (configuration_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2),
        encoding="utf-8",
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nEligible electricity systems used: {len(final_system_ids)}")
    print(
        eligible.groupby("geographic_region")["electricity_system"]
        .count()
        .to_string()
    )
    print(f"\nDevice: {device}")

    training_frames = {
        system_id: split_data[system_id]["train"]
        for system_id in final_system_ids
    }

    training_history_frames: List[pd.DataFrame] = []
    runtime_rows: List[Dict[str, object]] = []
    checkpoint_rows: List[Dict[str, object]] = []
    episode_result_frames: List[pd.DataFrame] = []
    trace_frames: List[pd.DataFrame] = []

    for algorithm_name in args.algorithms:
        for seed in args.seeds:
            print(
                f"\nTraining {algorithm_name}, seed {seed}: "
                f"{timesteps_per_run:,} timesteps"
            )

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            base_environment = RenewableDemandResponseEnv(
                data_by_system=training_frames,
                normalization=normalization,
                all_system_ids=final_system_ids,
                balanced_sampling=True,
                fixed_system=None,
                deterministic_starts=False,
                seed=seed,
            )
            monitored_environment = Monitor(base_environment)

            model = build_model(
                algorithm_name=algorithm_name,
                environment=monitored_environment,
                seed=seed,
                device=device,
            )

            callback = EpisodeCheckpointCallback(
                checkpoint_steps=checkpoint_steps,
                checkpoint_dir=checkpoints_dir,
                algorithm_name=algorithm_name,
                seed=seed,
            )

            start = time.perf_counter()

            model.learn(
                total_timesteps=timesteps_per_run,
                callback=callback,
                progress_bar=False,
            )

            training_seconds = time.perf_counter() - start

            final_model_path = (
                models_dir
                / f"{algorithm_name}_seed{seed}_final"
            )
            model.save(final_model_path)

            history = pd.DataFrame(
                {
                    "method": algorithm_name,
                    "seed": seed,
                    "episode": np.arange(
                        1,
                        len(callback.episode_rewards) + 1,
                    ),
                    "reward": callback.episode_rewards,
                    "episode_length": callback.episode_lengths,
                }
            )
            training_history_frames.append(history)

            runtime_rows.append(
                {
                    "method": algorithm_name,
                    "seed": seed,
                    "training_seconds": training_seconds,
                    "training_minutes": training_seconds / 60.0,
                    "timesteps": timesteps_per_run,
                    "timesteps_per_second": (
                        timesteps_per_run
                        / max(training_seconds, 1e-12)
                    ),
                    "device": device,
                }
            )

            for checkpoint_episode, checkpoint_step in zip(
                checkpoint_episodes,
                checkpoint_steps,
            ):
                checkpoint_rows.append(
                    {
                        "method": algorithm_name,
                        "seed": seed,
                        "checkpoint_episode": checkpoint_episode,
                        "checkpoint_timestep": checkpoint_step,
                        "model_path": str(
                            checkpoints_dir
                            / (
                                f"{algorithm_name}_seed{seed}_"
                                f"step{checkpoint_step}.zip"
                            )
                        ),
                    }
                )

            print(
                f"  Completed in {training_seconds / 60.0:.2f} minutes"
            )

            for system_id in final_system_ids:
                test_environment = RenewableDemandResponseEnv(
                    data_by_system={
                        system_id: split_data[system_id]["test"]
                    },
                    normalization=normalization,
                    all_system_ids=final_system_ids,
                    balanced_sampling=False,
                    fixed_system=system_id,
                    deterministic_starts=True,
                    seed=seed,
                )

                metrics, traces = evaluate_policy(
                    model=model,
                    environment=test_environment,
                    episodes=evaluation_episodes,
                    method_name=algorithm_name,
                )

                metrics["seed"] = seed
                traces["seed"] = seed

                episode_result_frames.append(metrics)
                trace_frames.append(traces)

            monitored_environment.close()

    baseline_seed = args.seeds[0]

    print("\nEvaluating rule-based baseline...")

    for system_id in final_system_ids:
        baseline_environment = RenewableDemandResponseEnv(
            data_by_system={
                system_id: split_data[system_id]["test"]
            },
            normalization=normalization,
            all_system_ids=final_system_ids,
            balanced_sampling=False,
            fixed_system=system_id,
            deterministic_starts=True,
            seed=baseline_seed,
        )

        metrics, traces = evaluate_policy(
            model=None,
            environment=baseline_environment,
            episodes=evaluation_episodes,
            method_name="RuleBased",
        )

        metrics["seed"] = baseline_seed
        traces["seed"] = baseline_seed

        episode_result_frames.append(metrics)
        trace_frames.append(traces)

    training_history = pd.concat(
        training_history_frames,
        ignore_index=True,
    )
    training_runtime = pd.DataFrame(runtime_rows)
    checkpoint_manifest = pd.DataFrame(checkpoint_rows)
    episode_results = pd.concat(
        episode_result_frames,
        ignore_index=True,
    )
    evaluation_traces = pd.concat(
        trace_frames,
        ignore_index=True,
    )

    training_history.to_csv(
        training_dir / "training_history.csv",
        index=False,
    )
    training_runtime.to_csv(
        training_dir / "training_runtime.csv",
        index=False,
    )
    checkpoint_manifest.to_csv(
        training_dir / "checkpoint_manifest.csv",
        index=False,
    )

    episode_results.to_csv(
        evaluation_dir / "episode_level_results.csv",
        index=False,
    )

    electricity_system_results = aggregate_results(
        episode_results,
        ["method", "electricity_system", "geographic_region"],
    )
    electricity_system_results.to_csv(
        evaluation_dir / "electricity_system_results.csv",
        index=False,
    )

    geographic_region_results = aggregate_results(
        episode_results,
        ["method", "geographic_region"],
    )
    geographic_region_results.to_csv(
        evaluation_dir / "geographic_region_results.csv",
        index=False,
    )

    overall_results = aggregate_results(
        episode_results,
        ["method"],
    )
    overall_results.to_csv(
        evaluation_dir / "overall_results.csv",
        index=False,
    )

    evaluation_traces.to_csv(
        evaluation_dir / "evaluation_traces.csv.gz",
        index=False,
        compression="gzip",
    )

    total_seconds = time.perf_counter() - total_start

    metadata = {
        "completed": True,
        "total_runtime_seconds": total_seconds,
        "total_runtime_minutes": total_seconds / 60.0,
        "eligible_electricity_systems": len(final_system_ids),
        "electricity_system_ids": final_system_ids,
        "device": device,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 86)
    print("FULL PIPELINE COMPLETED")
    print("=" * 86)
    print(f"Runtime: {total_seconds / 60.0:.2f} minutes")
    print(f"Eligible electricity systems: {len(final_system_ids)}")
    print(f"Results folder: {output_dir}")
    print("\nMain output files:")
    print(
        f"  {configuration_dir / 'eligible_regions.csv'}"
    )
    print(
        f"  {configuration_dir / 'excluded_regions.csv'}"
    )
    print(
        f"  {training_dir / 'training_history.csv'}"
    )
    print(
        f"  {evaluation_dir / 'episode_level_results.csv'}"
    )
    print(
        f"  {evaluation_dir / 'electricity_system_results.csv'}"
    )
    print(
        f"  {evaluation_dir / 'geographic_region_results.csv'}"
    )
    print(
        f"  {evaluation_dir / 'overall_results.csv'}"
    )
    print(
        f"  {evaluation_dir / 'evaluation_traces.csv.gz'}"
    )
    print("=" * 86)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment stopped by the user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
