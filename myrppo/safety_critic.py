"""Safety critic for constrained HVAC policy optimization."""

from __future__ import annotations

from collections.abc import Sequence

import torch as th
from torch import nn


class CostCritic(nn.Module):
    """Estimate future cumulative thermal-comfort violation cost V_c(s)."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation_fn: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, states: th.Tensor) -> th.Tensor:
        if states.ndim > 2:
            states = th.flatten(states, start_dim=1)
        return self.net(states).squeeze(-1)
