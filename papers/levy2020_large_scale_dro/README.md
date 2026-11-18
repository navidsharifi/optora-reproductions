# Large-scale methods for distributionally robust optimization

> Status. `in progress`

## Reference

Daniel Levy, Yair Carmon, John C. Duchi and Aaron Sidford. Large-scale
methods for distributionally robust optimization. Advances in Neural
Information Processing Systems 33 (NeurIPS 2020), pages 8847-8860.
arXiv:2010.05893.

Citation checked against the NeurIPS proceedings record on 2026-11-17.
Section and table numbers below follow arXiv v2. The authors' code is at
https://github.com/daniellevy/fast-dro (commit dc75246), with the
hyperparameters and seeds of every run in `hyperparameters/`.

## Claim

We reproduce the Digits experiment of Section 7 and Appendix F. A linear
classifier $[x, b]$ on fixed 500-dimensional features is trained with the
regularized log loss

$$
\ell([x, b]; (z, y)) = \log \sum_{c=1}^{C} \exp\left(\langle x_c - x_y, z\rangle + b_c - b_y\right)
    + \frac{\mu}{2} \sum_{c=1}^{C} \lVert x_c \rVert_2^2,
$$

$\mu = 10^{-2}$, under three robust objectives over the training
distribution $P_0$. They are CVaR at level $\alpha = 0.02$, the
$\chi^2$ ball $\sup\{\mathbb E_Q \ell : D_{\chi^2}(Q \,\|\, P_0) \le \rho\}$ with
$\rho = 1$, and the penalized objective
$\sup_Q \{\mathbb E_Q \ell - \lambda D_{\chi^2}(Q \,\|\, P_0)\}$ with $\lambda = 0.05$, where
$D_{\chi^2}(Q \,\|\, P) = \tfrac12 \mathbb E_P[(dQ/dP - 1)^2]$ (Eqs. 2-4). Every
minibatch method takes stochastic gradient steps on the plug-in robust loss
of a batch of $n$ samples (Section 3).

| Claim id | Statement |
| --- | --- |
| c1 | For CVaR, the best minibatch size reaches 2% of the optimal training objective in about 9 times fewer epochs than the full-batch method (Table 5) |
| c2 | The same speed-up is about 9.5 for the constrained $\chi^2$ objective (Table 5) |
| c3 | The same speed-up is about 16.2 for the penalized $\chi^2$ objective (Table 5) |
| c4 | Compared with ERM, DRO lowers the test log loss of the worst subgroup by 17.5% to 27% (Section 7, Appendix F.5.1) |

The location, metric, paper value and tolerance of each claim are fixed
in `claims.py`, committed before the full run. The tolerance of c1-c3 is
half the paper value, because an epoch count depends on the optimum found
across all runs and on the step-size grid. The value of c4 is the middle of
the stated range and its tolerance is half the range.

## Setup

| Item | Value |
| --- | --- |
| Data | The authors' released LeNet features of the Digits data (Dropbox link in the fast-dro README, 110 MB, SHA-256 `c4aa57a2...908c`). The training set holds the 60,000 MNIST training images and 600 typed digits drawn from 2,160 with `numpy` seed 124, as the authors' `subsample_arrays` does. The test set holds the 10,000 MNIST test images and 8,000 class-balanced typed digits (Appendix F.1) |
| Subgroups | The 20 pairs of population (MNIST, typed) and class (Appendix F.1) |
| Model | `torch.nn.Linear(500, 10)` with bias, the weight decay applied to the weight only |
| Optimizer | SGD with Nesterov momentum 0.9 and a constant step, iterate averaging of Shamir and Zhang (2013) with parameter 3 on minibatches, none on the full batch (Appendix F.2). Step sizes from the authors' files `mnist_final_cvar.yaml`, `mnist_final_cs.yaml`, `mnist_final_lam.yaml` and `mnist_erm.yaml` |
| Budget | 300 epochs per minibatch run, 30,000 for the full batch (Appendix F.4) |
| Seeds | 1, 10, 100, 1000 and 1337 for every minibatch and ERM run, seed 1 for each full-batch run (Table 5 ran one full-batch seed) |
| Robust objectives | `optora.CVaRAmbiguitySet(alpha=0.02)`, `optora.ChiSquareAmbiguitySet(radius=2)` and `optora.PenalizedChiSquareDRO(penalty=0.025)`, each on the uniform distribution of the batch (Optora 0.4.0, which added the penalized objective for this reproduction, optora#116). Optora's $\chi^2$ divergence is $\sum_i (q_i - p_i)^2 / p_i$, twice the paper's, so $\rho = 1$ is radius 2 and $\lambda = 0.05$ is penalty 0.025 |
| ERM baseline | Batch 500 with $\mu = 10^{-4}$ (step 0.05) and $\mu = 10^{-3}$ (step 0.02), the two values chosen in Appendix F.2. c4 compares each objective at batch 500 with the better of the two for the same seed |
| Evaluation | The training objective of the averaged iterate after every epoch, every 15 epochs for the full batch as in `mnist_full_batch.yaml`, and the per-group test log loss after the last epoch |
| Precision | float32, as the authors' code |
| Hardware | RTX 3050 Laptop 6 GB |

Choices recorded before any full run.

- ImageNet is not reproduced. Its ResNet-50 features alone take about 10 GB in float32 and no claim needs them.
- Batch size 10 is not run. Table 5 reports no epoch count for it, and its speed matters for no claim.
- The full batch is treated as one batch of the whole training set, the case `batch_size = -1` of the authors' code.
- The paper's $\chi^2$ objectives are evaluated exactly by Optora rather than by the authors' bisection with tolerance $10^{-4}$.
- The forward and backward pass of every step is captured in a CUDA graph and replayed, and the optimizer update runs eagerly (optora#117). Replay gives the same objective values as eager training bit for bit in our checks, so this changes the wall clock only.
- Wall-clock time per epoch is recorded next to the results. It is not a claim, since the paper timed four CPU cores.

## Run

```bash
python -m papers.levy2020_large_scale_dro.run --config papers/levy2020_large_scale_dro/configs/smoke.toml
python -m papers.levy2020_large_scale_dro.run --config papers/levy2020_large_scale_dro/configs/full.toml
```

## Result

See the results table in the root `README.md`, generated from
`results/levy2020_large_scale_dro.<claim_id>.json`.

## Findings

Not run yet.
