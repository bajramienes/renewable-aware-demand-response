"""
env.py
======
Physically constrained renewable-aware demand-response environment.

Design principles:
- identical physical constraints for DRL and the rule-based baseline;
- per-electricity-system normalization;
- energy-neutral load shifting within every 168-hour episode;
- zero terminal backlog by construction;
- no reward for action direction alone;
- reward based on target tracking, peak/ramp control, smoothness, and feasibility;
- no charts and no file I/O.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

EPISODE_LENGTH = 168
FLEXIBLE_LOAD_FRACTION = 0.10
MAX_HOURLY_SHIFT_FRACTION = 0.02
MAX_BACKLOG_FRACTION = 0.10
MAX_DEFERRAL_HOURS = 24
RESTORATION_WINDOW_HOURS = 24
ACTION_SATURATION_THRESHOLD = 0.95

REWARD_WEIGHTS = {
    "target_tracking": 0.42,
    "peak_control": 0.20,
    "ramp_control": 0.18,
    "backlog_penalty": 0.10,
    "action_change_penalty": 0.05,
    "action_magnitude_penalty": 0.02,
    "saturation_penalty": 0.03,
}


def compute_per_system_normalization(
    split_data: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute training-only statistics separately for every system."""
    features = [
        "load", "load_forecast", "solar", "wind", "renewable",
        "net_load", "renewable_penetration", "load_ramp",
    ]
    result: Dict[str, Dict[str, Dict[str, float]]] = {}

    for system_id, parts in split_data.items():
        train = parts["train"]
        result[system_id] = {}
        for feature in features:
            values = train[feature].replace([np.inf, -np.inf], np.nan).dropna()
            mean = float(values.mean())
            std = float(values.std())
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            result[system_id][feature] = {"mean": mean, "std": std}

    return result


class RenewableDemandResponseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_by_system: Dict[str, pd.DataFrame],
        normalization: Dict[str, Dict[str, Dict[str, float]]],
        all_system_ids: Sequence[str],
        geographic_group_fn: Callable[[str], str],
        balanced_sampling: bool,
        fixed_system: Optional[str] = None,
        deterministic_starts: bool = False,
        seed: int = 11,
    ) -> None:
        super().__init__()
        self.data_by_system = data_by_system
        self.normalization = normalization
        self.all_system_ids = list(all_system_ids)
        self.geographic_group_fn = geographic_group_fn
        self.available_systems = list(data_by_system)
        self.balanced_sampling = balanced_sampling
        self.fixed_system = fixed_system
        self.deterministic_starts = deterministic_starts
        self.rng = np.random.default_rng(seed)

        self.region_to_systems: Dict[str, List[str]] = {}
        for system_id in self.available_systems:
            region = self.geographic_group_fn(system_id)
            self.region_to_systems.setdefault(region, []).append(system_id)
        self.available_regions = sorted(self.region_to_systems)

        # 8 energy features + 4 cyclical time features + 6 control features
        # + one-hot system identity.
        observation_size = 18 + len(self.all_system_ids)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(observation_size,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_system = self.available_systems[0]
        self.current_frame: Optional[pd.DataFrame] = None
        self.episode_frame: Optional[pd.DataFrame] = None
        self.start_index = 0
        self.local_step = 0
        self.backlog = 0.0
        self.previous_action = 0.0
        self.previous_original_net = 0.0
        self.previous_controlled_net = 0.0
        self.cumulative_original_energy = 0.0
        self.cumulative_controlled_energy = 0.0
        self.episode_mean_load = 1.0
        self.episode_peak_load = 1.0
        self.episode_mean_renewable_penetration = 0.0
        self.fixed_backlog_capacity = 1.0
        self.future_restore_capacity = np.zeros(EPISODE_LENGTH + 1)
        self.trace: List[Dict[str, object]] = []
        self._deterministic_counter = 0

    def _z(self, feature: str, value: float) -> float:
        stat = self.normalization[self.current_system][feature]
        return float(np.clip((value - stat["mean"]) / stat["std"], -8.0, 8.0))

    def _choose_system(self) -> str:
        if self.fixed_system is not None:
            return self.fixed_system
        if not self.balanced_sampling:
            return str(self.rng.choice(self.available_systems))
        region = str(self.rng.choice(self.available_regions))
        return str(self.rng.choice(self.region_to_systems[region]))

    def _choose_start(self, frame: pd.DataFrame) -> int:
        max_start = len(frame) - EPISODE_LENGTH
        if max_start <= 0:
            return 0
        if self.deterministic_starts:
            start = (self._deterministic_counter * EPISODE_LENGTH) % (max_start + 1)
            self._deterministic_counter += 1
            return int(start)
        return int(self.rng.integers(0, max_start + 1))

    def _remaining_hours(self) -> int:
        return EPISODE_LENGTH - self.local_step

    def _observation(self) -> np.ndarray:
        assert self.episode_frame is not None
        row = self.episode_frame.iloc[self.local_step]
        load = max(float(row["load"]), 1.0)
        max_shift = MAX_HOURLY_SHIFT_FRACTION * load
        backlog_ratio = self.backlog / max(self.fixed_backlog_capacity, 1.0)
        remaining_ratio = self._remaining_hours() / EPISODE_LENGTH
        peak_ratio = load / max(self.episode_peak_load, 1.0)
        renewable_centered = (
            float(row["renewable_penetration"])
            - self.episode_mean_renewable_penetration
        )

        one_hot = np.zeros(len(self.all_system_ids), dtype=np.float32)
        one_hot[self.all_system_ids.index(self.current_system)] = 1.0

        core = np.array([
            self._z("load", float(row["load"])),
            self._z("load_forecast", float(row["load_forecast"])),
            self._z("solar", float(row["solar"])),
            self._z("wind", float(row["wind"])),
            self._z("renewable", float(row["renewable"])),
            self._z("net_load", float(row["net_load"])),
            self._z("renewable_penetration", float(row["renewable_penetration"])),
            self._z("load_ramp", float(row["load_ramp"])),
            float(row["hour_sin"]), float(row["hour_cos"]),
            float(row["dow_sin"]), float(row["dow_cos"]),
            float(np.clip(backlog_ratio, 0.0, 1.0)),
            float(self.previous_action),
            float(remaining_ratio),
            float(np.clip(peak_ratio, 0.0, 2.0)),
            float(np.clip(renewable_centered, -2.0, 2.0)),
            float(np.clip(max_shift / max(self.episode_mean_load, 1.0), 0.0, 1.0)),
        ], dtype=np.float32)
        return np.concatenate([core, one_hot]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_system = self._choose_system()
        self.current_frame = self.data_by_system[self.current_system]
        self.start_index = self._choose_start(self.current_frame)
        self.episode_frame = self.current_frame.iloc[
            self.start_index:self.start_index + EPISODE_LENGTH
        ].reset_index(drop=True)

        self.local_step = 0
        self.backlog = 0.0
        self.previous_action = 0.0
        self.cumulative_original_energy = 0.0
        self.cumulative_controlled_energy = 0.0
        self.trace = []

        self.episode_mean_load = max(float(self.episode_frame["load"].mean()), 1.0)
        self.episode_peak_load = max(float(self.episode_frame["load"].max()), 1.0)
        self.episode_mean_renewable_penetration = float(
            self.episode_frame["renewable_penetration"].mean()
        )
        self.fixed_backlog_capacity = (
            MAX_BACKLOG_FRACTION * self.episode_mean_load
        )

        hourly_restore = (
            MAX_HOURLY_SHIFT_FRACTION
            * self.episode_frame["load"].to_numpy(float)
        )
        self.future_restore_capacity = np.zeros(EPISODE_LENGTH + 1)
        for idx in range(EPISODE_LENGTH - 1, -1, -1):
            self.future_restore_capacity[idx] = (
                self.future_restore_capacity[idx + 1] + hourly_restore[idx]
            )

        first = self.episode_frame.iloc[0]
        self.previous_original_net = float(first["net_load"])
        self.previous_controlled_net = float(first["net_load"])

        return self._observation(), {
            "electricity_system": self.current_system,
            "geographic_region": self.geographic_group_fn(self.current_system),
            "start_index": self.start_index,
        }

    def step(self, action):
        assert self.episode_frame is not None
        row = self.episode_frame.iloc[self.local_step]

        original_load = float(row["load"])
        renewable = float(row["renewable"])
        original_net = original_load - renewable
        raw_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))

        max_shift = MAX_HOURLY_SHIFT_FRACTION * original_load
        flexible_limit = FLEXIBLE_LOAD_FRACTION * original_load
        requested_shift = raw_action * min(max_shift, flexible_limit)

        remaining_after_this_step = EPISODE_LENGTH - self.local_step - 1
        future_capacity = self.future_restore_capacity[self.local_step + 1]

        # Required restoration makes terminal backlog exactly zero by construction.
        minimum_restore_now = max(0.0, self.backlog - future_capacity)
        if remaining_after_this_step < RESTORATION_WINDOW_HOURS:
            minimum_restore_now = max(
                minimum_restore_now,
                self.backlog / max(remaining_after_this_step + 1, 1),
            )

        actual_shift = 0.0
        forced_restoration = 0.0

        if minimum_restore_now > 0.0:
            forced_restoration = min(self.backlog, max_shift, minimum_restore_now)
            actual_shift += forced_restoration
            self.backlog -= forced_restoration

        remaining_shift_capacity = max(0.0, max_shift - abs(actual_shift))

        if requested_shift < 0.0 and remaining_after_this_step >= RESTORATION_WINDOW_HOURS:
            # New deferral is capped by fixed backlog capacity and future ability to restore.
            maximum_new_backlog = min(
                self.fixed_backlog_capacity,
                future_capacity,
            )
            deferrable = min(
                -requested_shift,
                remaining_shift_capacity,
                max(0.0, maximum_new_backlog - self.backlog),
            )
            actual_shift -= deferrable
            self.backlog += deferrable
        elif requested_shift > 0.0:
            restored = min(requested_shift, remaining_shift_capacity, self.backlog)
            actual_shift += restored
            self.backlog -= restored

        # Last step closes any tiny numerical residue.
        if self.local_step == EPISODE_LENGTH - 1 and self.backlog > 1e-9:
            residue = min(self.backlog, max(0.0, max_shift - abs(actual_shift)))
            actual_shift += residue
            forced_restoration += residue
            self.backlog -= residue

        self.backlog = max(0.0, float(self.backlog))
        controlled_load = max(0.0, original_load + actual_shift)
        controlled_net = controlled_load - renewable

        self.cumulative_original_energy += original_load
        self.cumulative_controlled_energy += controlled_load

        scale = max(self.episode_mean_load, 1.0)
        renewable_signal = np.clip(
            float(row["renewable_penetration"])
            - self.episode_mean_renewable_penetration,
            -1.0, 1.0,
        )
        peak_signal = np.clip(
            (original_load - self.episode_mean_load)
            / max(self.episode_peak_load - self.episode_mean_load, 1.0),
            -1.0, 1.0,
        )

        # Desired load is lower during peaks and higher during renewable-rich hours.
        desired_shift = max_shift * np.clip(
            0.75 * renewable_signal - 0.85 * peak_signal,
            -1.0, 1.0,
        )
        target_load = original_load + desired_shift
        target_error = (controlled_load - target_load) / max(max_shift, 1.0)

        original_ramp = abs(original_net - self.previous_original_net)
        controlled_ramp = abs(controlled_net - self.previous_controlled_net)
        ramp_improvement = np.clip(
            (original_ramp - controlled_ramp) / scale,
            -1.0, 1.0,
        )
        peak_improvement = np.clip(
            (original_load - controlled_load) / scale
            if original_load > self.episode_mean_load else 0.0,
            -1.0, 1.0,
        )
        backlog_ratio = self.backlog / max(self.fixed_backlog_capacity, 1.0)
        action_change = raw_action - self.previous_action
        saturation_penalty = max(0.0, abs(raw_action) - 0.80) ** 2

        reward = (
            -REWARD_WEIGHTS["target_tracking"] * target_error ** 2
            + REWARD_WEIGHTS["peak_control"] * peak_improvement
            + REWARD_WEIGHTS["ramp_control"] * ramp_improvement
            - REWARD_WEIGHTS["backlog_penalty"] * backlog_ratio ** 2
            - REWARD_WEIGHTS["action_change_penalty"] * action_change ** 2
            - REWARD_WEIGHTS["action_magnitude_penalty"] * raw_action ** 2
            - REWARD_WEIGHTS["saturation_penalty"] * saturation_penalty
        )

        # The projection above makes these impossible except for floating error.
        constraint_violation = int(
            self.backlog > self.fixed_backlog_capacity + 1e-6
            or abs(actual_shift) > max_shift + 1e-6
            or self.backlog > future_capacity + 1e-6
        )

        energy_balance_error = abs(
            self.cumulative_controlled_energy
            - self.cumulative_original_energy
            + self.backlog
        ) / max(self.cumulative_original_energy, scale)

        self.trace.append({
            "timestamp": row["timestamp"],
            "electricity_system": self.current_system,
            "geographic_region": self.geographic_group_fn(self.current_system),
            "original_load_MW": original_load,
            "controlled_load_MW": controlled_load,
            "renewable_MW": renewable,
            "solar_MW": float(row["solar"]),
            "wind_MW": float(row["wind"]),
            "original_net_load_MW": original_net,
            "controlled_net_load_MW": controlled_net,
            "target_load_MW": target_load,
            "action": raw_action,
            "actual_shift_MW": actual_shift,
            "forced_restoration_MW": forced_restoration,
            "backlog_MWh": self.backlog,
            "aged_backlog_MWh": 0.0,
            "energy_balance_error_fraction": energy_balance_error,
            "constraint_violation": constraint_violation,
            "reward": float(reward),
        })

        self.previous_action = raw_action
        self.previous_original_net = original_net
        self.previous_controlled_net = controlled_net
        self.local_step += 1

        terminated = self.local_step >= EPISODE_LENGTH
        truncated = False
        if terminated:
            # Numerical safety. This should already be zero.
            if self.backlog < 1e-6:
                self.backlog = 0.0
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_observation = self._observation()

        info = {
            "electricity_system": self.current_system,
            "geographic_region": self.geographic_group_fn(self.current_system),
            "original_load_MW": original_load,
            "controlled_load_MW": controlled_load,
            "renewable_MW": renewable,
            "backlog_MWh": self.backlog,
            "aged_backlog_MWh": 0.0,
            "actual_shift_MW": actual_shift,
            "constraint_violation": constraint_violation,
        }
        return next_observation, float(reward), terminated, truncated, info


def rule_based_action(observation: np.ndarray) -> np.ndarray:
    """Simple non-learning reference using the same observation and action bounds."""
    backlog_ratio = float(observation[12])
    previous_action = float(observation[13])
    remaining_ratio = float(observation[14])
    peak_ratio = float(observation[15])
    renewable_centered = float(observation[16])

    # Conservative threshold controller, intentionally not optimized by learning.
    if remaining_ratio < RESTORATION_WINDOW_HOURS / EPISODE_LENGTH and backlog_ratio > 0.01:
        desired = 0.45
    elif backlog_ratio > 0.55:
        desired = 0.30
    elif renewable_centered > 0.35 and backlog_ratio > 0.05:
        desired = 0.25
    elif peak_ratio > 0.92 and backlog_ratio < 0.45:
        desired = -0.25
    else:
        desired = 0.0

    action = np.clip(desired, previous_action - 0.25, previous_action + 0.25)
    return np.array([action], dtype=np.float32)
