"""Structured action projection for multi-zone HVAC setpoint control."""

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