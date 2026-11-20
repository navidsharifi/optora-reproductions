# Certifying some distributional robustness with principled adversarial training

> Status. `in progress`

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

Not run yet.
