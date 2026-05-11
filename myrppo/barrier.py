"""Neural barrier model and losses for safety-constrained PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch as th
import torch.nn.functional as F
from torch import nn


class BarrierNet(nn.Module):
    """Map a state to a scalar barrier value h(s).

    By convention, h(s) <= 0 means safe and h(s) > 0 means unsafe.
    """

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

        self.net = nn.Sequential(*layers)

    def forward(self, states: th.Tensor) -> th.Tensor:
        if states.ndim > 2:
            states = th.flatten(states, start_dim=1)
        return self.net(states).squeeze(-1)


@dataclass
class BarrierLosses:
    safe: th.Tensor
    unsafe: th.Tensor
    invariant: th.Tensor
    total: th.Tensor
    invariant_per_sample: th.Tensor


def _masked_mean(values: th.Tensor, mask: th.Tensor) -> th.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.new_zeros(())


def compute_barrier_losses(
    barrier_net: BarrierNet,
    states: th.Tensor,
    next_states: th.Tensor,
    safe_mask: th.Tensor,
    lambda_param: float,
) -> BarrierLosses:
    """Compute safe, unsafe and transition-invariance barrier losses."""
    h_states = barrier_net(states)
    h_next_states = barrier_net(next_states)
    unsafe_mask = th.logical_not(safe_mask)

    safe_loss = _masked_mean(F.relu(h_states), safe_mask)
    unsafe_loss = _masked_mean(F.relu(-h_states), unsafe_mask)
    invariant_per_sample = F.relu(h_next_states - (1.0 - lambda_param) * h_states)
    invariant_loss = invariant_per_sample.mean()
    total_loss = safe_loss + unsafe_loss + invariant_loss

    return BarrierLosses(
        safe=safe_loss,
        unsafe=unsafe_loss,
        invariant=invariant_loss,
        total=total_loss,
        invariant_per_sample=invariant_per_sample,
    )
