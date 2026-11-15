r"""Robust Q-values under an $L_1$ ball, with Optora and with HiGHS.

$q(\xi) = \min\{p^\top z : p \in \Delta^S, \|p - \bar p\|_1 \le \xi\}$ is the
worst-case response of nature in an $s,a$-rectangular robust MDP
(Eq. 5 of the paper).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np
import optora
import torch
from scipy import sparse
from scipy.optimize import linprog

_HIGHS_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}


def optora_q(
    nominal: torch.Tensor, z: torch.Tensor, budgets: torch.Tensor
) -> torch.Tensor:
    r"""Robust Q-values of many instances and budgets in one Optora call.

    Uses $q(\xi) = -\sup\{\mathbb E_p[-z] : \mathrm{TV}(p, \bar p) \le \xi/2\}$,
    since total variation is half the $L_1$ distance.

    Args:
        nominal: Nominal distributions of shape `(b, S)`.
        z: Vectors $z$ of shape `(b, S)`.
        budgets: Budgets $\xi \in [0, 2]$ of shape `(k,)`.

    Returns:
        Tensor of shape `(b, k)`.
    """
    batch, states = nominal.shape
    shape = (batch, budgets.numel(), states)
    ambiguity_set = optora.TotalVariationAmbiguitySet(
        nominal.unsqueeze(-2).expand(shape),
        radius=(budgets / 2.0).expand(batch, -1),
    )
    worst: torch.Tensor = ambiguity_set.worst_case_expectation(
        -z.unsqueeze(-2).expand(shape)
    )
    return -worst


def optora_single_q(nominal: torch.Tensor, z: torch.Tensor, budget: float) -> float:
    """One robust Q-value with Optora, building the set from the data.

    This is the unit of work timed for claim c2.

    Args:
        nominal: Nominal distribution of shape `(S,)`.
        z: Vector $z$ of shape `(S,)`.
        budget: Budget $\\xi$.

    Returns:
        The value $q(\\xi)$.
    """
    ambiguity_set = optora.TotalVariationAmbiguitySet(nominal, radius=budget / 2.0)
    value: torch.Tensor = ambiguity_set.worst_case_expectation(-z)
    return -float(value)


def lp_q(nominal: np.ndarray, z: np.ndarray, budget: float) -> float:
    r"""One robust Q-value from the linear program (6) of the paper, by HiGHS.

    Variables are $p$ and $l \ge |p - \bar p|$; the program is
    $\min z^\top p$ subject to $p - \bar p \le l$, $\bar p - p \le l$,
    $\mathbf 1^\top l \le \xi$, $\mathbf 1^\top p = 1$, $p, l \ge 0$.

    Args:
        nominal: Nominal distribution of shape `(S,)`.
        z: Vector $z$ of shape `(S,)`.
        budget: Budget $\xi$.

    Returns:
        The optimal value.

    Raises:
        RuntimeError: If HiGHS does not report an optimal solution.
    """
    states = nominal.shape[0]
    identity = sparse.identity(states, format="csr")
    zero_row = sparse.csr_matrix((1, states))
    one_row = sparse.csr_matrix(np.ones((1, states)))
    upper = sparse.vstack(
        [
            sparse.hstack([identity, -identity]),
            sparse.hstack([-identity, -identity]),
            sparse.hstack([zero_row, one_row]),
        ],
        format="csr",
    )
    result = linprog(
        np.concatenate([z, np.zeros(states)]),
        A_ub=upper,
        b_ub=np.concatenate([nominal, -nominal, [budget]]),
        A_eq=sparse.hstack([one_row, zero_row], format="csr"),
        b_eq=[1.0],
        bounds=(0.0, None),
        method="highs-ds",
        options=_HIGHS_OPTIONS,
    )
    if result.status != 0:
        raise RuntimeError(f"HiGHS did not solve the program: {result.message}")
    return float(result.fun)


def median_seconds(call: Callable[[], object], repeats: int) -> float:
    """Median wall-clock time of `call` after one untimed warm-up call.

    Args:
        call: Function to time.
        repeats: Number of timed calls.

    Returns:
        The median duration in seconds.
    """
    call()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)
