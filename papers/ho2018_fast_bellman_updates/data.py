r"""Instances of Section 5: random problems and the inventory problem.

Each instance is a pair $(\bar p, z)$ of a nominal transition distribution
and the vector $z = r_{s,a}\mathbf 1 + \gamma v$ whose worst-case
expectation gives the robust Q-value.
"""

from __future__ import annotations

import math

import torch

HOLDING_COST = 0.1
PURCHASE_COST = 1.0
SALE_PRICE = 1.6


def random_instance(
    seed: int, num_states: int, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""One random instance of Section 5.

    The nominal distribution is drawn from $U[0, 1]^S$ and normalized.
    Rewards are zero and the value function is drawn from $U[0, 1]^S$, so
    $z = v$.

    Args:
        seed: Seed of the run.
        num_states: Number of states $S$.
        dtype: Floating-point type.

    Returns:
        Nominal distribution and $z$, both of shape `(num_states,)` on CPU.
    """
    generator = torch.Generator().manual_seed(seed * 100_003 + num_states)
    weights = torch.rand(num_states, generator=generator, dtype=dtype)
    values = torch.rand(num_states, generator=generator, dtype=dtype)
    return weights / weights.sum(), values


def demand_distribution(num_states: int, dtype: torch.dtype) -> torch.Tensor:
    r"""Demand on $\{0, \dots, S - 1\}$ rounded from $N(S/2, S/5)$.

    The probability of demand $k$ is the normal mass of $[k - 1/2, k + 1/2)$,
    with the mass below $-1/2$ added to $k = 0$ and the mass above
    $S - 3/2$ added to $k = S - 1$; demand beyond the inventory cap has the
    same effect as demand equal to it.

    Args:
        num_states: Number of states $S$.
        dtype: Floating-point type.

    Returns:
        Tensor of shape `(num_states,)` summing to one.
    """
    mean, deviation = num_states / 2.0, num_states / 5.0
    edges = torch.arange(num_states + 1, dtype=dtype) - 0.5
    cdf = 0.5 * (1.0 + torch.erf((edges - mean) / (deviation * math.sqrt(2.0))))
    cdf[0], cdf[-1] = 0.0, 1.0
    return cdf[1:] - cdf[:-1]


def inventory_instances(
    num_states: int, discount: float, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""All order quantities in the initial state of the inventory problem.

    The state is the inventory level $s \in \{0, \dots, S - 1\}$, starting
    at $s = 0$. Ordering $a$ units costs $1.0a$; with demand $D$ the
    inventory sells $\min(a, D)$ at price 1.6, holds $(a - D)_+$ at cost 0.1
    and moves to state $(a - D)_+$. The value function is $v(s) = s$, as in
    Section 5.

    Args:
        num_states: Number of states $S$.
        discount: Discount factor $\gamma$.
        dtype: Floating-point type.

    Returns:
        Nominal distributions and vectors $z$, both of shape
        `(num_states, num_states)`, one row per order quantity $a$.
    """
    demand = demand_distribution(num_states, dtype)
    orders = torch.arange(num_states, dtype=dtype).unsqueeze(-1)
    levels = torch.arange(num_states, dtype=dtype)
    next_state = torch.clamp(orders - levels, min=0.0)
    sales = torch.minimum(orders, levels)
    reward = (
        SALE_PRICE * (demand * sales).sum(-1)
        - PURCHASE_COST * orders.squeeze(-1)
        - HOLDING_COST * (demand * next_state).sum(-1)
    )
    index = next_state.long()
    nominal = torch.zeros(num_states, num_states, dtype=dtype)
    nominal.scatter_add_(1, index, demand.expand(num_states, -1).contiguous())
    z = reward.unsqueeze(-1) + discount * levels
    return nominal, z
