"""Structured and safety action projections for multi-zone HVAC control."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TopologyActionProjectionWrapper(gym.ActionWrapper):
    """Project flat per-zone setpoints into a topology-consistent action.

    The wrapped action keeps the same shape and bounds as the policy action.
    For an action ordered as ``[heat_0, cool_0, heat_1, cool_1, ...]``, heating
    and cooling setpoints are projected separately with:

        x = (I + lambda * L)^(-1) x_raw

    where ``L`` is the weighted zone-adjacency Laplacian. This is a deterministic
    action generator layer.
    """

    def __init__(
        self,
        env: gym.Env,
        zone_edges: Sequence[tuple[int, int, float]],
        num_zones: int,
        smoothing_lambda: float = 0.10,
        normalize_edge_weights: bool = True,
    ) -> None:
        super().__init__(env)

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("TopologyActionProjectionWrapper requires a continuous Box action space.")

        expected_action_dim = 2 * num_zones
        if self.action_space.shape != (expected_action_dim,):
            raise ValueError(
                f"Expected action shape {(expected_action_dim,)}, got {self.action_space.shape}."
            )

        if smoothing_lambda < 0:
            raise ValueError("smoothing_lambda must be non-negative.")

        self.num_zones = num_zones
        self.smoothing_lambda = float(smoothing_lambda)
        self.normalize_edge_weights = normalize_edge_weights
        self.heating_indices = np.arange(0, expected_action_dim, 2)
        self.cooling_indices = np.arange(1, expected_action_dim, 2)
        self.projection_matrix = self._build_projection_matrix(zone_edges)

        self.last_raw_action: np.ndarray | None = None
        self.last_structured_action: np.ndarray | None = None
        self.last_projection_l2: float = 0.0
        self.last_projection_max_abs: float = 0.0

    def _build_projection_matrix(
        self, zone_edges: Sequence[tuple[int, int, float]]
    ) -> np.ndarray:
        adjacency = np.zeros((self.num_zones, self.num_zones), dtype=np.float64)

        max_weight = 1.0
        if self.normalize_edge_weights and zone_edges:
            max_weight = max(float(weight) for _, _, weight in zone_edges)
            max_weight = max(max_weight, 1.0)

        for source, target, weight in zone_edges:
            source = int(source)
            target = int(target)
            if not 0 <= source < self.num_zones or not 0 <= target < self.num_zones:
                raise ValueError(f"Invalid zone edge ({source}, {target}) for {self.num_zones} zones.")
            if source == target:
                continue

            edge_weight = float(weight) / max_weight
            adjacency[source, target] += edge_weight
            adjacency[target, source] += edge_weight

        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency
        system_matrix = np.eye(self.num_zones, dtype=np.float64) + self.smoothing_lambda * laplacian
        identity = np.eye(self.num_zones, dtype=np.float64)
        return np.linalg.solve(system_matrix, identity).astype(np.float32)

    def _project_single_action(self, action: np.ndarray) -> np.ndarray:
        structured = action.astype(np.float32, copy=True)
        structured[self.heating_indices] = self.projection_matrix @ structured[self.heating_indices]
        structured[self.cooling_indices] = self.projection_matrix @ structured[self.cooling_indices]
        return np.clip(structured, self.action_space.low, self.action_space.high).astype(np.float32)

    def structured_action(self, action: np.ndarray) -> np.ndarray:
        """Return the topology-projected action without stepping the environment."""
        action_array = np.asarray(action, dtype=np.float32)

        if action_array.ndim == 1:
            return self._project_single_action(action_array)

        if action_array.ndim == 2:
            return np.stack([self._project_single_action(row) for row in action_array], axis=0)

        raise ValueError(f"Unsupported action shape {action_array.shape}.")

    def action(self, action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(action, dtype=np.float32)
        structured = self.structured_action(raw_action)

        delta = structured - raw_action
        self.last_raw_action = raw_action.copy()
        self.last_structured_action = structured.copy()
        self.last_projection_l2 = float(np.linalg.norm(delta))
        self.last_projection_max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0

        return structured


class RateConstrainedSafetyProjectionWrapper(gym.ActionWrapper):
    """Apply a rate-constrained quadratic safety projection to setpoints.

    The layer solves the separable QP

        min_a ||a - a_ref||^2 + rho * ||a - a_prev||^2
        s.t.  action_low <= a <= action_high
              |a - a_prev| <= r

    with a closed-form projection. When wrapping ``NormalizeAction``, the rate
    limit can be specified in degrees Celsius per control step and is converted
    to the normalized action scale automatically.
    """

    def __init__(
        self,
        env: gym.Env,
        num_zones: int,
        rate_limit_degc_per_step: float = 0.20,
        smoothing_rho: float = 0.05,
    ) -> None:
        super().__init__(env)

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("RateConstrainedSafetyProjectionWrapper requires a continuous Box action space.")

        expected_action_dim = 2 * num_zones
        if self.action_space.shape != (expected_action_dim,):
            raise ValueError(
                f"Expected action shape {(expected_action_dim,)}, got {self.action_space.shape}."
            )

        if rate_limit_degc_per_step <= 0:
            raise ValueError("rate_limit_degc_per_step must be positive.")
        if smoothing_rho < 0:
            raise ValueError("smoothing_rho must be non-negative.")

        self.num_zones = num_zones
        self.rate_limit_degc_per_step = float(rate_limit_degc_per_step)
        self.smoothing_rho = float(smoothing_rho)
        self.rate_limit = self._rate_limit_in_action_space()
        self.previous_safe_action: np.ndarray | None = None
        self.last_reference_action: np.ndarray | None = None
        self.last_safe_action: np.ndarray | None = None
        self.last_safety_projection_l2: float = 0.0
        self.last_safety_projection_max_abs: float = 0.0

    def reset(self, **kwargs):
        self.previous_safe_action = None
        return self.env.reset(**kwargs)

    def _rate_limit_in_action_space(self) -> np.ndarray:
        rate_limit = np.full(self.action_space.shape, self.rate_limit_degc_per_step, dtype=np.float32)

        real_space = getattr(self.env, "real_space", None)
        if isinstance(real_space, spaces.Box) and real_space.shape == self.action_space.shape:
            action_span = self.action_space.high - self.action_space.low
            real_span = real_space.high - real_space.low
            rate_limit = self.rate_limit_degc_per_step * action_span / real_span

        return rate_limit.astype(np.float32)

    def _project_single_action(self, action: np.ndarray) -> np.ndarray:
        reference = np.clip(action.astype(np.float32, copy=True), self.action_space.low, self.action_space.high)
        if self.previous_safe_action is None:
            self.previous_safe_action = reference.copy()
            return reference

        lower = np.maximum(self.action_space.low, self.previous_safe_action - self.rate_limit)
        upper = np.minimum(self.action_space.high, self.previous_safe_action + self.rate_limit)

        qp_unconstrained = (reference + self.smoothing_rho * self.previous_safe_action) / (1.0 + self.smoothing_rho)
        safe_action = np.clip(qp_unconstrained, lower, upper).astype(np.float32)
        self.previous_safe_action = safe_action.copy()
        return safe_action

    def safe_action(self, action: np.ndarray) -> np.ndarray:
        """Return the safety-projected action without stepping the environment."""
        action_array = np.asarray(action, dtype=np.float32)

        if action_array.ndim == 1:
            return self._project_single_action(action_array)

        if action_array.ndim == 2:
            return np.stack([self._project_single_action(row) for row in action_array], axis=0)

        raise ValueError(f"Unsupported action shape {action_array.shape}.")

    def action(self, action: np.ndarray) -> np.ndarray:
        reference = np.asarray(action, dtype=np.float32)
        safe_action = self.safe_action(reference)

        delta = safe_action - reference
        self.last_reference_action = reference.copy()
        self.last_safe_action = safe_action.copy()
        self.last_safety_projection_l2 = float(np.linalg.norm(delta))
        self.last_safety_projection_max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0

        return safe_action
