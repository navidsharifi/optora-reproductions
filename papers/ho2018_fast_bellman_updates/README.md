# Fast Bellman updates for robust MDPs

> Status. `partially reproduced`

## Reference

Chin Pang Ho, Marek Petrik and Wolfram Wiesemann. Fast Bellman updates for
robust MDPs. Proceedings of the 35th International Conference on Machine
Learning, PMLR 80:1979-1988, 2018.

Citation checked against the PMLR page on 2026-11-14. The authors' C++
implementation is linked from the paper; this reproduction does not use
it.

## Claim

The paper computes the robust Q-value of one state-action pair under an
$s,a$-rectangular $L_1$ ambiguity set,

$$
q(\xi) = \min_{p \in \Delta^S} \left\{ p^\top z : \|p - \bar p\|_1 \le \xi \right\},
$$

with a homotopy method, and compares its runtime with Gurobi. In Optora
this is one call of `TotalVariationAmbiguitySet`, because total variation
is half the $L_1$ distance:

$$
q(\xi) = -\sup_{p} \left\{ \mathbb E_p[-z] : \mathrm{TV}(p, \bar p) \le \xi / 2 \right\}.
$$

| Claim id | Statement |
| --- | --- |
| c1 | Optora returns the exact optimal value of the linear program, as the paper's methods do |
| c2 | Computing one $q(\xi)$ with Optora is about three orders of magnitude faster than with a general linear programming solver |

The location, metric, paper value and tolerance of each claim are fixed
in `claims.py`, committed before the full run.

## Setup

| Item | Value |
| --- | --- |
| Random domain | $\bar p$ drawn from $U[0, 1]^S$ and normalized, $z \sim U[0, 1]^S$ (rewards zero, values uniform), as in Section 5 |
| Inventory domain | Holding cost 0.1, purchase cost 1.0, sale price 1.6, no backlog, inventory capped at $S - 1$, demand rounded from $N(S/2, S/5)$, value function $v(s) = s$, initial state 0, every order quantity $a$ as one pair $(0, a)$, discount 0.9 (our choice; the paper does not state one) |
| Grid | $S \in \{50, 100, \dots, 400\}$, $\xi \in \{0, 0.25, \dots, 2\}$, 5 runs, as in Section 5.1 |
| Optora components | `optora.TotalVariationAmbiguitySet` with radius $\xi / 2$, batched over instances and budgets |
| Reference solver | HiGHS through `scipy.optimize.linprog`, on the linear program (6) of the paper |
| Timing (c2) | One $q(\xi)$ at $S = 400$ in the random domain, on CPU. Each timed call builds its problem from $(\bar p, z, \xi)$ and solves it, for both solvers. Median of repeated calls per budget, averaged over the nine budgets and the five runs, as the paper averages over $\kappa$ |
| Precision | float64 |
| Hardware | The laptop CPU recorded in each result's environment (Intel64 family 6 model 154), with the machine otherwise idle during the full run |
| Differences from the paper | HiGHS instead of Gurobi 7.5, Python and PyTorch instead of C++, other hardware. Only the unweighted $L_1$ norm is reproduced: a weighted $L_1$ ball is not a $\varphi$-divergence ball and no shipped Optora set covers it. The $s$-rectangular bisection of Section 5.2 couples the actions of a state and is not one ambiguity-set call, so it is out of scope |

## Run

```bash
python -m papers.ho2018_fast_bellman_updates.run --config papers/ho2018_fast_bellman_updates/configs/smoke.toml
python -m papers.ho2018_fast_bellman_updates.run --config papers/ho2018_fast_bellman_updates/configs/full.toml
```

## Result

See the results table in the root `README.md`, generated from
`results/ho2018_fast_bellman_updates.<claim_id>.json`.

## Findings

Full run on 2026-11-15 with Optora 0.2.0, float64, on the laptop CPU with
nothing else heavy running. It took about three minutes. The complete table
is in `runs/ho2018_fast_bellman_updates/full_table.md`.

| Claim | Paper | This run (mean over 5 runs, sample std) | Within tolerance |
| --- | --- | --- | --- |
| c1 | 0 | 1.8e-13 (std 0) | yes |
| c2 | 3.0 | 1.83 (std 0.009) | no |

Claim c1 reproduces. Over every size, budget and run, Optora's $q(\xi)$
matches the HiGHS linear program to $6.7\cdot10^{-16}$ in the random domain
and $1.9\cdot10^{-13}$ in the inventory domain. The inventory error grows
with $S$ only because the values $z$ grow with $S$ (to a few hundred), so it
stays within a few units in the last place.

Claim c2 does not reproduce. One $q(\xi)$ at $S = 400$ takes 7.4 ms with
HiGHS and 0.109 ms with Optora, a speedup of about 68 rather than about
1,000.

| $S$ | HiGHS ms | Optora ms | Speedup |
| --- | --- | --- | --- |
| 50 | 2.20 | 0.092 | 24 |
| 200 | 4.00 | 0.101 | 40 |
| 400 | 7.42 | 0.109 | 68 |

The HiGHS time grows with $S$ while the Optora time barely moves, so a
single Optora call is bound by the fixed cost of about fifteen small eager
PyTorch operations, not by the $O(S \log S)$ sort. The paper times a C++
implementation against Gurobi, so its ratio measures a different pair of
programs, and the gap here is about per-call overhead rather than the
algorithm.

A post hoc measurement, made after the run and not part of any claim,
shows what batching does. Evaluating 400 instances at all nine budgets in
one call costs 22.7 ms on the same CPU, about 6 $\mu$s per $q(\xi)$, which
is roughly 1,200 times faster than one HiGHS solve at $S = 400$. This is
how a robust value iteration would call Optora, with every state-action
pair in one batch, but the pre-registered claim times a single call and
stays outside its tolerance.

Optora. No mismatch or failure. The per-call overhead was profiled earlier
(about 15 eager operations of 3-5 $\mu$s each and one sort) and is normal
for eager PyTorch, so it is recorded here rather than filed.
