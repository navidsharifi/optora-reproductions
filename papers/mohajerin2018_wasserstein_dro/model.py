r"""Wasserstein distributionally robust mean-CVaR portfolios of Section 7.

The paper minimizes, over portfolios $x$ on the simplex $\mathbb X$ and an
auxiliary $\tau \in \mathbb R$, the worst-case expectation of the
piecewise affine loss

$$
\ell(x, \tau, \xi) = \max_{k \in \{1, 2\}} a_k \langle x, \xi \rangle + b_k \tau,
\qquad a = \left(-1, -1 - \tfrac{\rho}{\alpha}\right),
\quad b = \left(\rho, \rho\left(1 - \tfrac{1}{\alpha}\right)\right),
$$

over the 1-Wasserstein ball of radius $\varepsilon$ around the empirical
distribution, with the 1-norm as transport cost and support
$\Xi = \mathbb R^m$ (Eq. 50 and Section 7.2). Its expectation under $P$ is
the mean-CVaR objective $\mathbb E[-\langle x, \xi\rangle] + \rho\,
\mathrm{CVaR}_\alpha(-\langle x, \xi\rangle)$ once minimized over $\tau$.

Optora works on a finite support. The worst case over $\Xi = \mathbb R^m$
equals $\frac1N \sum_i \ell(x, \tau, \hat\xi_i) + \varepsilon \max_k |a_k|
\|x\|_\infty$ (Corollary 5.1), the supremum being approached by moving mass
of order $\varepsilon / D$ to distance $D$ along one coordinate. The support
used here is the $N$ samples and the $2m$ points $\pm D e_j$, with zero
nominal mass on the latter, which reproduces that value up to $O(1/D)$.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import optora
import torch
from scipy import sparse
from scipy.optimize import linprog

ALPHA = 0.2
RHO = 10.0
SLOPES = (-1.0, -1.0 - RHO / ALPHA)
INTERCEPTS = (RHO, RHO * (1.0 - 1.0 / ALPHA))

_HIGHS_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}


def portfolio_loss(
    points: torch.Tensor, x: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Mean-CVaR loss of every support point.

    Args:
        points: Returns of shape `(..., n, m)`.
        x: Portfolios of shape `(..., m)`, broadcastable against the batch
            of `points`.
        tau: Auxiliary variables of shape `(...)`.

    Returns:
        Loss of shape `(..., n)`.
    """
    returns = (points @ x.unsqueeze(-1)).squeeze(-1)
    slopes = returns.new_tensor(SLOPES)
    intercepts = returns.new_tensor(INTERCEPTS)
    pieces = slopes * returns.unsqueeze(-1) + intercepts * tau[..., None, None]
    return pieces.amax(dim=-1)


def finite_support(
    samples: torch.Tensor, far: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Support, nominal distribution and 1-norm cost of every simulation.

    Args:
        samples: Training datasets of shape `(s, N, m)`.
        far: Distance $D$ of the points $\pm D e_j$.

    Returns:
        Points of shape `(s, N + 2m, m)`, the nominal distribution of shape
        `(N + 2m,)` with mass $1/N$ on each sample, and the cost of shape
        `(s, N + 2m, N + 2m)`.
    """
    simulations, num_samples, num_assets = samples.shape
    eye = far * torch.eye(num_assets, dtype=samples.dtype, device=samples.device)
    directions = torch.cat([eye, -eye]).expand(simulations, -1, -1)
    points = torch.cat([samples, directions], dim=1)
    nominal = torch.zeros(points.shape[1], dtype=samples.dtype, device=samples.device)
    nominal[:num_samples] = 1.0 / num_samples
    return points, nominal, torch.cdist(points, points, p=1)


def simplex_basis(num_assets: int, dtype: torch.dtype) -> torch.Tensor:
    r"""Orthonormal basis of the hyperplane $\{x : \mathbf 1^\top x = 0\}$.

    Args:
        num_assets: Dimension $m$.
        dtype: Floating-point type.

    Returns:
        Matrix $U$ of shape `(m, m - 1)` with $U^\top U = I$ and
        $\mathbf 1^\top U = 0$.
    """
    centred = torch.eye(num_assets, dtype=torch.float64) - 1.0 / num_assets
    basis: torch.Tensor = torch.linalg.svd(centred).U[:, : num_assets - 1]
    return basis.to(dtype)


def robust_portfolios(
    samples: torch.Tensor,
    radii: torch.Tensor,
    far: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Solve every simulation at every radius by the ellipsoid method.

    Optora's `WassersteinAmbiguitySet` gives the worst-case expectation and,
    by automatic differentiation, a subgradient in $(x, \tau)$. The
    portfolio is written $x = e_m + U y$ with $U$ from `simplex_basis`, so
    the variables $(y, \tau)$ have dimension $n = m$ and the only remaining
    constraints are $x \ge 0$. Each step makes a central cut through the
    centre of the current ellipsoid, with the objective subgradient when
    the centre is feasible and the gradient of $-x_j$ for the most violated
    $j$ otherwise (Bland, Goldfarb and Todd, Operations Research 29(6),
    1981, Section 3). The volume shrinks by at least $e^{-1/(2(n + 1))}$ per
    step, and the output is the feasible centre of lowest value.

    The first ellipsoid is centred at the portfolio fully invested in asset
    $m$ with $\tau = 0$, away from the equally weighted portfolio that claim
    c1 tests. Its semi-axes are $2$ along $y$ and $\sqrt 2\, \tau_{\max}$
    along $\tau$ with $\tau_{\max} = \max_{i,j} \lvert\hat\xi_{ij}\rvert$, so it
    contains the simplex times $[-\tau_{\max}, \tau_{\max}]$. That interval
    holds an optimal $\tau$. By Corollary 5.1 the radius term does not depend
    on $\tau$, so an optimal $\tau$ is an empirical $(1 - \alpha)$-quantile of
    $-\langle x, \hat\xi_i\rangle$ (Rockafellar and Uryasev, Journal of Risk
    2(3), 2000, Theorem 1), and $\lvert\langle x, \hat\xi_i\rangle\rvert \le
    \tau_{\max}$ on the simplex.

    Args:
        samples: Training datasets of shape `(s, N, m)`.
        radii: Wasserstein radii of shape `(r,)`.
        far: Distance $D$ of the far support points.
        iterations: Number of ellipsoid steps.

    Returns:
        Portfolios $\hat x_N(\varepsilon)$ of shape `(s, r, m)`, the
        auxiliary $\hat\tau$ of shape `(s, r)`, and the certificate
        $\hat J_N(\varepsilon)$, the worst-case expectation at
        $(\hat x, \hat\tau)$, of shape `(s, r)`.
    """
    dtype, device = samples.dtype, samples.device
    points, nominal, cost = finite_support(samples, far)
    ambiguity_set = optora.WassersteinAmbiguitySet(nominal, cost.unsqueeze(1), radii)
    points = points.unsqueeze(1)
    num_assets = samples.shape[-1]
    dimension = num_assets
    basis = simplex_basis(num_assets, dtype).to(device)
    corner = torch.zeros(num_assets, dtype=dtype, device=device)
    corner[-1] = 1.0
    batch = (samples.shape[0], radii.numel())
    centre = torch.zeros(*batch, dimension, dtype=dtype, device=device)
    tau_radius = samples.abs().amax(dim=(-2, -1))
    axes = torch.full((*batch, dimension), 4.0, dtype=dtype, device=device)
    axes[..., -1] = 2.0 * tau_radius.unsqueeze(-1) ** 2
    shape = torch.diag_embed(axes)
    best_value = torch.full(batch, torch.inf, dtype=dtype, device=device)
    best_centre = centre.clone()
    expand = dimension**2 / (dimension**2 - 1.0)
    for _ in range(iterations):
        y = centre[..., :-1].requires_grad_()
        tau = centre[..., -1].requires_grad_()
        x = corner + y @ basis.T
        value = ambiguity_set.worst_case_expectation(portfolio_loss(points, x, tau))
        grad_y, grad_tau = torch.autograd.grad(value.sum(), (y, tau))
        value, x = value.detach(), x.detach()
        feasible = x.amin(dim=-1) >= 0.0
        violated = torch.nn.functional.one_hot(x.argmin(dim=-1), num_assets).to(dtype)
        cut = torch.where(
            feasible.unsqueeze(-1),
            torch.cat([grad_y, grad_tau.unsqueeze(-1)], dim=-1),
            torch.cat(
                [-violated @ basis, torch.zeros_like(grad_tau).unsqueeze(-1)], -1
            ),
        )
        improved = feasible & (value < best_value)
        best_value = torch.where(improved, value, best_value)
        best_centre = torch.where(improved.unsqueeze(-1), centre, best_centre)
        direction = (shape @ cut.unsqueeze(-1)).squeeze(-1)
        curvature = (cut * direction).sum(dim=-1, keepdim=True)
        moving = curvature > 0.0
        direction = direction / torch.sqrt(torch.where(moving, curvature, 1.0))
        centre = centre.detach() - direction / (dimension + 1)
        outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
        updated = expand * (shape - 2.0 / (dimension + 1) * outer)
        updated = 0.5 * (updated + updated.transpose(-1, -2))
        shape = torch.where(moving.unsqueeze(-1), updated, shape)
    x_hat = corner + best_centre[..., :-1] @ basis.T
    return x_hat, best_centre[..., -1], best_value


def out_of_sample(
    x: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    r"""Mean-CVaR objective $J(x)$ under the true normal distribution.

    For normal returns with mean $\mu_x = \mu^\top x$ and standard deviation
    $\sigma_x = \sqrt{x^\top \Sigma x}$,
    $J(x) = -(1 + \rho)\mu_x + \frac{\rho}{\alpha}\varphi(\Phi^{-1}(\alpha))
    \sigma_x$, the closed form the paper uses (Section 7.2.A).

    Args:
        x: Portfolios of shape `(..., m)`.
        mean: Mean return of shape `(m,)`.
        covariance: Covariance of shape `(m, m)`.

    Returns:
        Tensor of shape `(...)`.
    """
    mean, covariance = mean.to(x), covariance.to(x)
    density = NormalDist().pdf(NormalDist().inv_cdf(ALPHA))
    deviation = torch.sqrt(((x @ covariance) * x).sum(dim=-1))
    return -(1.0 + RHO) * (x @ mean) + RHO / ALPHA * density * deviation


def closed_form_worst_case(
    samples: torch.Tensor, x: torch.Tensor, tau: torch.Tensor, radius: float
) -> torch.Tensor:
    r"""Worst case over $\Xi = \mathbb R^m$ from Corollary 5.1.

    Args:
        samples: Training datasets of shape `(..., N, m)`.
        x: Portfolios of shape `(..., m)`.
        tau: Auxiliary variables of shape `(...)`.
        radius: Wasserstein radius $\varepsilon$.

    Returns:
        $\frac1N \sum_i \ell(x, \tau, \hat\xi_i) + \varepsilon \max_k |a_k|
        \|x\|_\infty$, of shape `(...)`.
    """
    steepest = max(abs(slope) for slope in SLOPES)
    empirical = portfolio_loss(samples, x, tau).mean(dim=-1)
    return empirical + radius * steepest * x.abs().amax(dim=-1)


def lp_portfolio(samples: np.ndarray, radius: float) -> tuple[np.ndarray, float, float]:
    r"""Reference solution of the convex reformulation, by HiGHS.

    Theorem 4.2 with $\Xi = \mathbb R^m$ turns the problem into the linear
    program over $(x, \tau, \lambda, s)$

    $$
    \min \lambda\varepsilon + \frac1N \sum_i s_i \quad\text{s.t.}\quad
    s_i \ge a_k \langle x, \hat\xi_i\rangle + b_k\tau,\;
    \lambda \ge |a_k| x_j,\; x \in \mathbb X,\; \lambda \ge 0,
    $$

    since $\|a_k x\|_\infty \le \lambda$ is linear for $x \ge 0$.

    Args:
        samples: One training dataset of shape `(N, m)`.
        radius: Wasserstein radius $\varepsilon$.

    Returns:
        The optimal portfolio of shape `(m,)`, $\tau$ and the optimal value.

    Raises:
        RuntimeError: If HiGHS does not report an optimal solution.
    """
    num_samples, num_assets = samples.shape
    slopes, intercepts = np.array(SLOPES), np.array(INTERCEPTS)
    pieces = len(SLOPES)
    epigraph = sparse.hstack(
        [
            sparse.csr_matrix(np.concatenate([a * samples for a in slopes])),
            sparse.csr_matrix(np.repeat(intercepts, num_samples)[:, None]),
            sparse.csr_matrix((pieces * num_samples, 1)),
            -sparse.vstack([sparse.identity(num_samples)] * pieces),
        ]
    )
    norm = sparse.hstack(
        [
            sparse.vstack([abs(a) * sparse.identity(num_assets) for a in slopes]),
            sparse.csr_matrix((pieces * num_assets, 1)),
            -np.ones((pieces * num_assets, 1)),
            sparse.csr_matrix((pieces * num_assets, num_samples)),
        ]
    )
    objective = np.concatenate(
        [np.zeros(num_assets + 1), [radius], np.full(num_samples, 1.0 / num_samples)]
    )
    budget = np.concatenate([np.ones(num_assets), np.zeros(num_samples + 2)])
    bounds = [(0.0, None)] * num_assets + [(None, None), (0.0, None)]
    result = linprog(
        objective,
        A_ub=sparse.vstack([epigraph, norm], format="csr"),
        b_ub=np.zeros((num_samples + num_assets) * pieces),
        A_eq=budget[None, :],
        b_eq=[1.0],
        bounds=bounds + [(None, None)] * num_samples,
        method="highs-ds",
        options=_HIGHS_OPTIONS,
    )
    if result.status != 0:
        raise RuntimeError(f"HiGHS did not solve the program: {result.message}")
    return result.x[:num_assets], float(result.x[num_assets]), float(result.fun)
