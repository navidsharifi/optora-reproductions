# Large-scale methods for distributionally robust optimization

> Status. `partially reproduced`

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
| Evaluation | The training objective and the per-group test log loss of the averaged iterate after every epoch, every 15 epochs for the full batch as in `mnist_full_batch.yaml`. c4 reads the group losses after the last epoch |
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

Full run on 2026-11-18 with Optora 0.4.0, float32, on the RTX 3050. The
58 runs took 3 hours 17 minutes. The complete table is in
`runs/levy2020_large_scale_dro/full_table.md` and the training curves are
in `full_curves.json` next to it.

| Claim | Paper | This run (mean over 5 seeds, sample std) | Within tolerance |
| --- | --- | --- | --- |
| c1 | 9.0 | 8.73 (std 0.06) | yes |
| c2 | 9.5 | 9.48 (std 0.28) | yes |
| c3 | 16.2 | 11.62 (std 0.13) | yes |
| c4 | 0.2225 | 0.149 (std 0.001) | no |

Three of the four claims reproduce. The optimal training objectives found
across all runs are 0.4032 for CVaR, 0.1802 for the $\chi^2$ ball and
0.1303 for the penalized objective, each reached by the full batch. The
epochs to 2% of these values follow Table 5 closely.

| Objective | Source | $n = 50$ | $n = 500$ | $n = 5000$ | Full batch | Speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| CVaR | Table 5 | 189 ± 3 | 115 ± 1 | 193 ± 4 | 1035 | 9.0 |
| CVaR | ours | 194.8 ± 3.6 | 116.8 ± 0.8 | 195.4 ± 4.9 | 1020 | 8.7 |
| $\chi^2$ ball | Table 5 | - | 74 ± 1 | 60 ± 3 | 570 | 9.5 |
| $\chi^2$ ball | ours | not reached | 74.4 ± 0.9 | 60.2 ± 1.8 | 570 | 9.5 |
| Penalized $\chi^2$ | Table 5 | 107 ± 1 | 104 ± 1 | 131 ± 5 | 1680 | 16.2 |
| Penalized $\chi^2$ | ours | 107.6 ± 1.1 | 104.6 ± 1.1 | 134.8 ± 5.8 | 1215 | 11.6 |

Every minibatch count is within about one standard deviation of the
paper. With $n = 50$ the $\chi^2$ ball ends at 0.2060 in every seed,
above the target of 0.1838, so it never reaches it within 300 epochs.
Table 5 has no entry there either. This is the bias floor of small batches
that Figures 3 and 4 show, since the plug-in robust loss of a batch is a
biased estimate of the robust loss of the population (Section 3,
Proposition 1).

The one count that does not match is the full batch on the penalized
objective, 1215 epochs against 1680, and it alone moves c3 from 16.2 to
11.6. The settings are those of the authors' `mnist_full_batch.yaml`, and
the minibatch counts for the same objective agree with the paper. Our
full-batch curve is flat near the optimum and reaches 1%, 1.5% and 2% of
it at epochs 1680, 1380 and 1215. A paper optimum lower by about 1% would
explain the full-batch count, but it would also raise our minibatch counts
to 165, 165 and 193, far from Table 5. So the gap is specific to the full
batch, and we did not find its cause. c3 still lies within its tolerance.

c4 does not reproduce under the comparison we fixed before the run. In
every run the worst group is the typed digit 9, as Appendix F.4 says. The
worst-group test loss at $n = 500$ is 0.348 to 0.350 for CVaR and 0.337 to
0.340 for both $\chi^2$ objectives. ERM reaches 0.401 to 0.403 with
$\mu = 10^{-3}$ and 0.451 to 0.456 with $\mu = 10^{-4}$.

| Objective | Reduction against ERM, $\mu = 10^{-3}$ | Reduction against ERM, $\mu = 10^{-4}$ |
| --- | --- | --- |
| CVaR | 12.9% to 13.4% | 22.7% to 23.4% |
| $\chi^2$ ball | 15.5% to 16.0% | 25.2% to 25.7% |
| Penalized $\chi^2$ | 15.6% to 16.0% | 25.2% to 25.6% |

The pre-registered metric uses the better ERM, the left column, and gives
0.149. The paper states its range of 17.5% to 27% in Appendix F.5.1
without naming the ERM run or the epoch, and it reads the range from
Figure 3, which plots both ERM runs. Our two columns together span 13% to
26%, close to that range, but this reading was not fixed in advance, so
it is not counted.

The paper also notes that the minibatch methods generalize best well
before they converge (Appendix F.5.1). To see whether its range belongs to
an earlier epoch, we reran the full configuration on 2026-11-21 with the
per-group test loss recorded after every epoch, for the DRO and the ERM
runs alike (issue #8). The rerun gives the same per-seed values of c1 to
c4 bit for bit, so the records in `results/` are unchanged. The table
below applies the pre-registered comparison at each epoch, with the DRO
runs at $n = 500$ and the better ERM both read at that epoch. Losses are
means over the five seeds, and c4 is the mean with its sample standard
deviation.

| Epoch | CVaR | $\chi^2$ ball | Penalized $\chi^2$ | ERM, $\mu = 10^{-4}$ | ERM, $\mu = 10^{-3}$ | c4 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.584 | 0.634 | 0.620 | 1.974 | 0.975 | 0.370 (0.038) |
| 5 | 0.392 | 0.433 | 0.419 | 0.865 | 0.673 | 0.384 (0.010) |
| 10 | 0.354 | 0.384 | 0.376 | 0.605 | 0.556 | 0.332 (0.013) |
| 20 | 0.345 | 0.354 | 0.350 | 0.484 | 0.462 | 0.243 (0.011) |
| 50 | 0.344 | 0.343 | 0.343 | 0.446 | 0.415 | 0.172 (0.004) |
| 100 | 0.347 | 0.340 | 0.342 | 0.446 | 0.405 | 0.153 (0.004) |
| 300 | 0.349 | 0.338 | 0.339 | 0.454 | 0.402 | 0.149 (0.001) |

The mean c4 lies inside the stated range of 17.5% to 27% from epoch 17 to
epoch 47, and it peaks at 0.394 at epoch 4. The ERM run with
$\mu = 10^{-3}$ is the better baseline at every epoch and in every seed.
The large early values say more about speed than about robustness. In the
first ten epochs ERM has not yet fit the typed digit 9, whose test loss
starts near 1 for $\mu = 10^{-3}$ and near 2 for $\mu = 10^{-4}$, while
the DRO objectives weight the samples of highest loss more from the first
step.
After about epoch 50 every run changes little. The lowest worst-group
loss of each run over all epochs, which is what early stopping on this
metric would select, gives about 16% against the better ERM, close to
the final value. So the paper's range matches our curves read between
epochs 17 and 47, but a reduction above 17.5% that persists to
convergence does not appear in our runs. The pre-registered c4 stays
not reproduced. The complete table is in
`runs/levy2020_large_scale_dro/full_table.md`, and the losses of all 20
groups at every evaluation are in `full_curves.json`. The files of the
first run are kept next to them with the suffix `_2026-11-18`.

The average test metrics agree with Appendix F.5.1 in accuracy but not in
loss. DRO reaches a test accuracy of 0.970 to 0.973 against 0.969 to 0.972
for ERM, and a test log loss of 0.086 to 0.094 against 0.096 to 0.100,
where the paper reports a slightly worse average loss. The test set holds
8,000 typed digits out of 18,000 against 600 of 60,600 in training
(Appendix F.1), so the average test loss already weights the minority
population heavily.

Apart from the first three CVaR runs at $n = 50$, a training step costs
1.0 to 2.7 ms at every batch size, so the time per epoch follows the
number of steps. It is about 2.4 s at $n = 50$, 0.19 to 0.24 s at
$n = 500$ and 0.03 s at $n = 5000$, and one full-batch step takes 5.4 to
6.5 ms. The same work does not always take the same time on this laptop.
The first two CVaR seeds at $n = 50$ ran at 0.50 s per epoch, or 0.41 ms
per step, and the last two at 2.2 s, so these times describe this machine
only.

Optora. Two problems found before the run were fixed upstream and released
in 0.4.0. The penalized $\chi^2$ objective was missing (optora#116, now
`PenalizedChiSquareDRO`), and the minibatch step was bound by kernel
launches with no tested way to capture it (optora#117, now covered by CUDA
graph tests and described in `docs/training.md`). In the full run the new
objective gives the epoch counts of Table 5 at every batch size, and no
run failed. The captured forward and backward pass alone takes about
0.14 ms in the optora#117 benchmark, so most of the 2 ms step lies outside
the captured pass, in the eager Nesterov update, the iterate averaging and
the batch selection of this reproduction. We did not profile it further,
and nothing in the run points to a problem in the library.
