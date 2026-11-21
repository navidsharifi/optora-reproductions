# Certifying some distributional robustness with principled adversarial training

> Status. `reproduced`

## Reference

Aman Sinha, Hongseok Namkoong and John Duchi. Certifying some
distributional robustness with principled adversarial training.
International Conference on Learning Representations (ICLR 2018).
OpenReview `Hk6kPgZA-`, arXiv:1710.10571.

Citation checked in plan step 2b. Section, figure and equation numbers
below follow arXiv v4 of 1 May 2018, the last version with the three
authors of the ICLR paper, since v5 adds Riccardo Volpi. The authors' code
is at https://github.com/duchi-lab/certifiable-distributional-robustness
(commit b6e21a4).

## Claim

For a loss $\ell(\theta; z)$ and the transport cost
$c(z, z_0) = \lVert x - x_0 \rVert_2^2$ on the inputs, with labels left
unchanged, WRM minimizes the Lagrangian relaxation of the worst case over a
Wasserstein ball,

$$
\min_{\theta} \; \mathbb E_{\hat P_n}\big[\phi_\gamma(\theta; Z)\big], \qquad
\phi_\gamma(\theta; z_0) = \sup_{z} \big\{\ell(\theta; z) - \gamma\, c(z, z_0)\big\}
$$

(Eqs. 2b and 7). For a smooth loss and a large enough $\gamma$ the inner
problem is strongly concave, so gradient ascent finds its maximizer, the
transport map $T_\gamma(\theta; z_0)$ (Eq. 9). Theorem 3 turns the trained
model into a certificate. For every $\rho$ the worst-case population loss
over the ball of radius $\rho$ is at most
$\gamma\rho + \mathbb E_{\hat P_n}[\phi_\gamma(\theta; Z)]$ up to a
statistical error term, and the bound is tightest at the robustness level
$\hat\rho_n = \mathbb E_{\hat P_n}[c(T_\gamma(\theta; Z), Z)]$ that the
model reaches on the training set (Eq. 14). Section 4 checks the
certificate on synthetic data and on MNIST, and compares WRM on MNIST with
ERM and with adversarial training by FGM, IFGM and PGM.

| Claim id | Statement |
| --- | --- |
| c1 | On the synthetic data the certificate is almost tight at $\hat\rho_n$, 0.240 against a test worst-case loss of 0.234 (Figure 2a) |
| c2 | All five MNIST models reach at least 99% clean test accuracy (Section 4.2) |
| c3 | On MNIST the certificate bounds the test worst-case loss, 0.707 against 0.329 at $\hat\rho_n$ (Figure 2b) |
| c4 | Under an $\ell_2$ PGM attack at the training radius, WRM errs on 2.3% of the test set and ERM on 10.8% (Figure 3a) |
| c5 | Under an $\ell_\infty$ PGM attack with $\epsilon = 0.1$, WRM errs on 14.7% against 25.2% for the best of FGM, IFGM and PGM (Figure 3b) |
| c6 | At the training penalty, the test robustness level $\hat\rho_{\mathrm{test}}$ of ERM is more than two orders of magnitude above that of WRM, 161 against 0.58 (Figure 4a) |

The paper reports these numbers only as curves. Every paper value was read
from the vector data of the figures in arXiv v4, converted with
`pdftocairo -svg` and mapped through the axis ticks. The readings agree
with values stated elsewhere. The two certificates have the slopes
$\gamma = 2$ and $\gamma = 1/2.6$, the vertical bar of Figure 4(a) sits at
$C_2 / \gamma = 23.9$, and the bar of Figure 3(a) at
$\sqrt{\hat\rho_n} / C_2 = 0.049$, close to the 0.050 that the
$\hat\rho_n = 0.209$ of Figure 2(b) gives. The location, metric, paper value and tolerance of each claim
are fixed in `claims.py`, committed before the full run. The tolerances
cover the seed-to-seed spread that a single run in the paper hides, and
the tolerance of c6 is half an order of magnitude.

## Setup

| Item | Value |
| --- | --- |
| Optora components | None in the experiment. The inner maximization is the continuous adversary that Optora 0.4.0 does not have, written as `prototypes.continuous_wasserstein.transport_map` for a later Optora issue. Its tests use `optora.solvers.GradientDescent` as an oracle for the exact maximizer |
| MNIST | The `mnist.npz` that the authors' `keras.datasets.mnist.load_data` reads, checked against the SHA-256 digest Keras pins for it. 60,000 training and 10,000 test images with pixels in $[0, 1]$, for which $C_2 = \mathbb E\lVert X \rVert_2 = 9.21$ and $C_\infty = 1.00$ as in footnote 3 |
| Synthetic data | $X \sim \mathcal N(0, I_2)$ with norms in $(\sqrt 2 / 1.3, 1.3\sqrt 2)$ rejected and the label $\mathrm{sign}(\lVert X \rVert_2 - \sqrt 2)$ (Section 4.1), 10,000 training and 10,000 test points |
| MNIST model | The authors' `cnn_model(activation="elu")`, convolutions with 64, 128 and 128 filters of size 8, 6 and 5, ELU, a linear output, 710,218 parameters, Glorot-uniform initialization as in Keras |
| MNIST training | Adam with learning rate $10^{-3}$, batch size 256, 25 epochs, training labels clipped to $[0.1/9, 0.9]$, loss at the perturbed inputs only, all as in `train_mnist_models.py`. The training set is shuffled once per seed and visited in that order every epoch, with the last batch shifted back to be full, as the authors' `batch_indices` does. The five models of a seed share their initial weights and this order |
| WRM | $\gamma = 1/2.6$, the authors' `eps` $= 1.3 = 1/(2\gamma)$, which the paper rounds to $0.04\, C_2$. The transport map takes 16 ascent steps of size $1/\sqrt t$ on $\ell / (2\gamma) - \tfrac12 \lVert x - x_0 \rVert_2^2$, the first step and the 15 iterations of the authors' `wrm` |
| FGM, IFGM, PGM | $\ell_2$ balls of radius $\epsilon = \sqrt{\hat\rho_n(\theta_{\mathrm{WRM}})}$, measured on the training set for each seed (Eq. 19), and 15 steps for IFGM and PGM |
| Synthetic WRM | The network with hidden layers of 4 and 2 ELU units, $\gamma = 2$, 15 ascent steps of size $1/\sqrt t$ (footnote 1), Adam with learning rate $10^{-2}$, batch size 256 and 25 epochs over a new permutation each epoch |
| Certificate | $\gamma \hat\rho_n + \mathbb E_{\hat P_n}[\phi_\gamma] = \mathbb E_{\hat P_n}[\ell(\theta; T_\gamma(\theta; Z))]$ on the training set with the training labels (Eq. 14) |
| Test worst case | The mean loss at $T_{\gamma_{\mathrm{adv}}}(\theta; Z)$ on the test set against the mean transport cost $\hat\rho_{\mathrm{test}}$ (Eq. 20), for 24 penalties with $C_2 / \gamma_{\mathrm{adv}}$ from 1 to 100 on MNIST and $\gamma_{\mathrm{adv}}$ from 50 to 0.5 on the synthetic data, interpolated linearly at $\hat\rho_n$ |
| Attacks | PGM on the 10,000 test images with their one-hot labels, at 14 $\ell_2$ radii up to $0.26\, C_2$ and 11 $\ell_\infty$ radii up to $0.2\, C_\infty$ (Figure 3), and at the radii of c4 and c5 |
| Seeds | 0 to 4. The paper reports one run, with seed 12345 in the code |
| Precision | float32, with TensorFloat-32 in the cuDNN convolutions as PyTorch does by default, and deterministic kernels |
| Hardware | RTX 3050 Laptop 6 GB |

Choices recorded before any full run.

- Only Figures 2 to 4 are reproduced. The cart-pole experiment of Table 1
  has no public code, and the paper does not give the ranges of its
  discretized states, the learning rates or the penalty of the
  perturbation. The experiments with other penalties and larger budgets in
  Appendices A.3 to A.5 support no claim here.
- The paper does not give the step sizes of IFGM and PGM. IFGM takes 15
  steps of length $\epsilon/15$ along the normalized gradient, the basic
  iterative method of Kurakin et al. (2017) with a step small enough that
  it never leaves the ball. PGM follows Eq. (18) with the step
  $2.5\epsilon/15$ of Madry et al. (2018), the projection after each step
  and no random start. The attacks of Figure 3 use the same PGM. No
  adversary clips the pixels to $[0, 1]$, since the authors' WRM does not.
- On MNIST the certificate uses the smoothed training labels and the test
  worst case the one-hot test labels. At $\rho = 0$ the two curves of
  Figure 2(b) start at 0.63 and 0.16. The cross-entropy against the
  smoothed labels cannot fall below their entropy of 0.545, so the first
  value fits them, while the second fits one-hot labels on a model that
  puts mass 0.9 on the true class.
- The paper gives neither the sample size nor the optimizer and epochs of
  the synthetic experiment. The values above were chosen in a trial on
  seeds 0 to 2 with three settings, which also printed the c1 gap. All
  three settings gave gaps between -0.002 and 0.005, so the choice did not
  favour the claim, but c1 is not a blind test.
- A pilot of two epochs on one seed measured the cost of every stage and
  checked that every claim can be computed. Its numbers are not results.
  With TensorFloat-32 disabled and the deterministic kernels that every
  reproduction here requests, a WRM epoch took 108 s, and the full run
  would have taken about 17 hours. TensorFloat-32 in the convolutions,
  PyTorch's default, brings it to 24 s and keeps the run bitwise
  reproducible, while cuDNN autotuning in float32 gave 36 s but chooses its
  kernels by timing. The run therefore uses TensorFloat-32, which rounds
  the inputs of each convolution to a 10-bit mantissa, while the authors
  trained in float32. A WRM step takes 101 ms and an ERM step 8.5 ms, so a
  seed takes about 47 minutes and the full run about 4 hours.

## Run

```bash
python -m papers.sinha2018_certifiable_robustness.run --config papers/sinha2018_certifiable_robustness/configs/smoke.toml
python -m papers.sinha2018_certifiable_robustness.run --config papers/sinha2018_certifiable_robustness/configs/full.toml
```

The full run saves every finished seed under
`runs/sinha2018_certifiable_robustness/full/`. If it stops, the same
command with `--resume` continues at the first missing seed.

## Result

See the results table in the root `README.md`, generated from
`results/sinha2018_certifiable_robustness.<claim_id>.json`.

## Findings

Full run on 2026-11-20 and 2026-11-21 on the RTX 3050, with the settings
above and Optora 0.4.0 installed. The five seeds took 5.5 hours. The
per-seed table is in `runs/sinha2018_certifiable_robustness/full_table.md`
and the complete curves of every seed are in `full/seed_<s>.json` next to
it.

| Claim | Paper | This run, mean over 5 seeds (sample std) | Within tolerance |
| --- | --- | --- | --- |
| c1 | 0.0056 | -0.0009 (0.0030) | yes |
| c2 | 0.990 | 0.9895 (0.0009) | yes |
| c3 | 0.378 | 0.374 (0.004) | yes |
| c4 | 0.085 | 0.092 (0.014) | yes |
| c5 | 0.105 | 0.094 (0.014) | yes |
| c6 | 2.44 | 2.63 (0.07) | yes |

All six claims reproduce. The table below gives the quantities behind
them as paper / ours, with our means over the five seeds and the range in
brackets. Errors are in percent.

| Model | Clean error | $\ell_2$ PGM error at $\sqrt{\hat\rho_n}$ | $\ell_\infty$ PGM error at 0.1 | $\hat\rho_{\mathrm{test}}$ at $\gamma$ |
| --- | --- | --- | --- | --- |
| ERM | 0.98 / 1.00 [0.88, 1.10] | 10.8 / 11.3 [9.3, 12.8] | 92.2 / 98.0 [97.4, 98.5] | 161 / 241 [190, 273] |
| FGM | 0.84 / 0.87 [0.82, 0.94] | 2.96 / 2.55 [2.23, 2.67] | 26.9 / 30.1 [27.9, 32.7] | 2.26 / 1.38 [1.31, 1.52] |
| IFGM | 0.77 / 0.87 [0.80, 0.96] | 3.75 / 2.46 [2.32, 2.62] | 63.2 / 28.1 [26.3, 29.7] | 23.4 / 1.24 [1.16, 1.32] |
| PGM | 0.84 / 0.86 [0.77, 0.96] | 2.71 / 2.41 [2.25, 2.54] | 25.2 / 26.6 [25.0, 28.7] | 1.58 / 1.17 [1.06, 1.27] |
| WRM | 0.87 / 0.95 [0.85, 1.16] | 2.31 / 2.19 [1.89, 2.39] | 14.7 / 17.2 [16.7, 18.0] | 0.582 / 0.564 [0.518, 0.593] |

The certificate behaves as in Figure 2. On the synthetic data WRM reaches
$\hat\rho_n = 0.068$ (0.062 in the paper), where the certificate is 0.250
and the test worst-case loss 0.250 (0.240 and 0.234). The gap is negative
in three seeds, by at most 0.004. Theorem 3 bounds the population worst
case by the certificate only up to the statistical term
$\epsilon_n(t)$, so on 10,000 test points a gap of this size agrees with
a bound that is tight at $\hat\rho_n$. On MNIST $\hat\rho_n$ is 0.192,
between 0.166 and 0.203 over the seeds (0.209 in the paper), the
certificate 0.695 and the test worst case 0.321 (0.707 and 0.329). Part
of this gap of 0.37 comes from the labels. The certificate uses the
smoothed training labels, and the cross-entropy against them is never
below their entropy of 0.545, while the test loss uses one-hot labels.

WRM has the lowest $\ell_2$ error at the training radius in every seed,
and the lowest $\ell_\infty$ error by 7.5 to 10.8 points over the best
other model, which is PGM in every seed as in Figure 3(b). Against
$\ell_\infty$ attacks our models are a little weaker than in the paper,
17.2% against 14.7% for WRM and 98.0% against 92.2% for ERM, and the
differences between models are what c5 measures.

IFGM is the one model that departs from the paper. In Figures 3 and 4 it
is the weakest of the adversarially trained models, with an $\ell_\infty$
error of 63.2% and $\hat\rho_{\mathrm{test}} = 23.4$, while ours lies
between FGM and PGM on every measure. The paper does not give the IFGM
step size, and ours, $\epsilon/15$ for 15 steps, never leaves the ball,
so it trains much like PGM with a shorter step. We cannot tell which step
the authors used. No claim depends on IFGM alone, since c5 takes the
minimum over FGM, IFGM and PGM.

c6 comes out 0.19 above the paper because our ERM models move further
under the test adversary, $\hat\rho_{\mathrm{test}}$ between 190 and 273
against 161, while WRM matches at 0.564 against 0.582. The ratio is above
two orders of magnitude in every seed, from $10^{2.50}$ to $10^{2.68}$.

c2 holds on average but not in every seed. In seeds 3 and 4 all five
models reach 99%. In seeds 0 and 2 ERM stops at 98.90%, and in seed 1 WRM
stops at 98.84%. The paper reports one run, with clean errors between
0.77% and 0.98%, and ours lie between 0.77% and 1.16%.

A seed took between 61 and 72 minutes, longer than the 47 minutes
estimated from the pilot. Training took 31 to 42 minutes of each seed and
the rest is the attack and penalty sweeps over the test set, whose cost
the pilot did not time separately. Epoch times of the same model varied
between seeds by up to 65%, 24.0 s to 39.6 s for WRM, although every seed
does the same work. The GPU stayed near its power limit of 45 W at about
70 degrees Celsius.

Optora. The experiment uses no Optora component, because Optora 0.4.0 has
no continuous Wasserstein adversary. The prototype
`prototypes.continuous_wasserstein.transport_map` worked unchanged in
every run. What the paper needed from it is a batched inner ascent on
$\ell(\theta; x) - \gamma \lVert x - x_0 \rVert_2^2$ for any
differentiable loss on a batch of inputs, with the step schedule
$1/\sqrt t$, a penalty that can differ per example, detached outputs and
no gradient left on the model, together with the transport cost that
gives $\hat\rho_n$ and the certificate of Eq. (14). Optora's
`GradientDescent` takes a fixed step and solves one problem at a time, so
the prototype uses it only as an oracle in its tests. This is the input
to the Optora issue of plan step 3e and to P13 in step 3g.
