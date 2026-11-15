# Variance-based regularization with convex objectives

> Status. `in progress`

## Reference

John Duchi and Hongseok Namkoong. Variance-based regularization with
convex objectives. Journal of Machine Learning Research, 20(68):1-55,
2019. arXiv:1610.02581. An earlier version appeared at NeurIPS 2017.

Citation checked against the JMLR page on 2026-11-15. The authors' public
code (`hsnamkoong/robustopt`) holds projection routines only, not the
simulation reproduced here.

## Claim

We reproduce the simulation of Section 5.2. The paper minimizes the
robustly regularized risk

$$
R_n(\theta, \mathcal P_n) = \sup_{P} \left\{ \mathbb E_P[\ell(\theta; X)] :
D_\phi(P \,\|\, \hat P_n) \le \rho / n \right\}, \qquad \phi(t) = \tfrac12 (t - 1)^2,
$$

and compares its minimizer $\hat\theta^{\rm rob}$ with the empirical risk
minimizer $\hat\theta^{\rm erm}$ through the population risk
$R(\theta) = \tfrac12 \|\theta - v\|_2^2$.

| Claim id | Statement |
| --- | --- |
| c1 | The robust solution has lower population risk than ERM in every replication, for every sample size and noise level |
| c2 | For $B = 1$ and $n = 1000$ the robust solution recovers $v$, so its risk is zero up to the numerical precision of the solver |
| c3 | For $B = 1$ and $n = 100$, the one setting where the robust solution is not $v$, its risk is about 0.18 times the ERM risk |

The location, metric, paper value and tolerance of each claim are fixed
in `claims.py`, committed before the full run.

## Setup

| Item | Value |
| --- | --- |
| Loss | $\ell(\theta; x) = \tfrac12 \lVert\theta - v\rVert_2^2 + x^\top(\theta - v)$ with $d = 50$ |
| Data | $X \sim \mathrm{Uni}(\{-B, B\}^d)$, $B \in \{0.01, 0.1, 1, 10\}$, $n \in \{10^2, 5\cdot10^2, 10^3, 5\cdot10^3, 10^4\}$ |
| Domain | $\Theta = \{\theta : \lVert\theta\rVert_2 \le r\}$, $r = 10$, $v = \tfrac{r}{2\sqrt d}\mathbf 1$ |
| Radius | $\rho = \log(2/\delta) + d \log(2nDL)$ (Eq. 17), $\delta = 0.05$, $D = 2r$, $L = 3r + \sqrt d B$ |
| Optora components | `optora.ChiSquareAmbiguitySet` with radius $2\rho/n$, since Optora's chi-square divergence $\sum_i (q_i - p_i)^2 / p_i$ equals $2 D_\phi$ |
| Robust solver | Two phases, batched over replications. First the solver of Section 5.1, projected gradient descent with a backtracking (Armijo) line search. Then a projected subgradient method with step $2/(k+1)$ and $k$-weighted averaging (Lacoste-Julien, Schmidt and Bach 2012), valid because $R_n$ is 1-strongly convex. Each replication keeps the point with the lower robust objective. Both phases take values and gradients from Optora only |
| ERM solver | Closed form $\hat\theta^{\rm erm} = \Pi_\Theta(v - \bar X_n)$, the exact minimizer of the empirical risk over $\Theta$ |
| Replications | 1,200 per $(n, B)$ as in the paper, split into 3 seeds of 400 so each claim has a spread over seeds |
| Precision | float64 |
| Hardware | RTX 3050 Laptop 6 GB (full run), CPU (smoke run) |

Known discrepancy, recorded before any run. Every ERM entry of Table 1
with $B \ge 0.1$ equals $dB^2/(12n)$ to about 1%. The setup as stated gives
$\mathbb E[R(\hat\theta^{\rm erm})] = \tfrac12 \mathbb E\lVert\bar X_n\rVert_2^2 = dB^2/(2n)$,
six times larger. We run the setup as stated. Claims c1 and c3 do not
depend on this scale and c2 sits at the numerical floor.

Solver change, made during the independent solver check (workflow step 6)
and before the full run. At $n = 100$ the radius $2\rho/n \approx 12$ lets
the chi-square ball hold every distribution on a set of about 8 tied top
losses, so the maximizer is not unique there and $R_n$ is not
differentiable. Section 5.1 states that $R_n$ is differentiable whenever
the losses have nonzero variance, but that does not hold in this regime.
The Section 5.1 solver stalls at such points: on 100 replications at
$n = 100$, $B = 1$ it gives a robust to ERM risk ratio of 0.519, while the
exact minimizer from an independent conic solver (CVXPY with Clarabel)
gives 0.006. The paper reports 0.182. Optora's worst-case value at the
conic optimum equals the conic objective to $10^{-6}$, and its gradient
matches finite differences, so the stall is a property of the solver, not
of Optora. With the user's approval the main result uses the two-phase
solver above, which matches the conic minimizer to within 1% in mean
risk. The descent phase alone is still recorded for every cell, so the
stall is reported next to it.

The subgradient phase runs only for replications whose descent point is
farther than $10^{-10}$ from $v$, and the table in the findings reports
that share per cell. A descent point at $v$ is kept as it is. This saves
hours of GPU time and is checked independently. In the solver check, the
descent phase ended at $v$ in all 400 replications at $n = 1000$ and
$n = 5000$ (and in none at $n = 100$), and the conic solver puts the
minimizer at $v$ for every replication it solved at $n = 500$ (20
replications) and $n = 5000$ (3 replications, risk below $10^{-18}$).

## Run

```bash
python -m papers.duchi2019_variance_regularization.run --config papers/duchi2019_variance_regularization/configs/smoke.toml
python -m papers.duchi2019_variance_regularization.run --config papers/duchi2019_variance_regularization/configs/full.toml
```

## Result

See the results table in the root `README.md`, generated from
`results/duchi2019_variance_regularization.<claim_id>.json`.

## Findings

Pending the full run.
