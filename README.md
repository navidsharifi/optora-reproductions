# Optora reproductions

[![CI](https://github.com/navidsharifi/optora-reproductions/actions/workflows/ci.yml/badge.svg)](https://github.com/navidsharifi/optora-reproductions/actions/workflows/ci.yml)
[![Optora DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23850922.svg)](https://doi.org/10.5281/zenodo.23850922)

Reproductions of published distributionally robust optimization (DRO)
papers with [Optora](https://github.com/navidsharifi/optora), a PyTorch
library for DRO ([documentation](https://navidsharifi.github.io/optora/)).

Each reproduction installs the released `optora` package from PyPI and uses
its public API only. A claim counts as reproduced when the mean over seeds
lies within a tolerance that was committed before the full run. When a
result disagrees with the paper, we trace the cause, and problems in the
library are reported on the Optora issue tracker.

## Results

<!-- results-table:start -->
| Paper | Claim | Metric | Paper value | Ours, mean (std) | Seeds | Tolerance | Within tolerance | Device | optora |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [duchi2019_variance_regularization](papers/duchi2019_variance_regularization/README.md) | c1 | fraction of replications with robust risk below ERM risk | 1 | 1 (0) | 3 | 0 | yes | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 0.2.0 |
| [duchi2019_variance_regularization](papers/duchi2019_variance_regularization/README.md) | c2 | mean risk of the robust solution | 6.32e-18 | 0 (0) | 3 | 1e-08 | yes | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 0.2.0 |
| [duchi2019_variance_regularization](papers/duchi2019_variance_regularization/README.md) | c3 | mean robust risk divided by mean ERM risk | 0.182 | 0.005475 (0.001256) | 3 | 0.05 | no | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 0.2.0 |
| [ho2018_fast_bellman_updates](papers/ho2018_fast_bellman_updates/README.md) | c1 | max absolute error of q(xi) against the HiGHS linear program | 0 | 1.847e-13 (0) | 5 | 1e-09 | yes | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | 0.2.0 |
| [ho2018_fast_bellman_updates](papers/ho2018_fast_bellman_updates/README.md) | c2 | log10 of LP time over Optora time for one q(xi), S = 400 | 3 | 1.831 (0.008877) | 5 | 1 | no | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | 0.2.0 |
| [mohajerin2018_wasserstein_dro](papers/mohajerin2018_wasserstein_dro/README.md) | c1 | max_i \|x_N(eps)_i - 1/m\| at eps = 1, N = 300 | 0 | 1.621e-09 (3.644e-09) | 200 | 0.001 | yes | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 0.3.0 |
| [mohajerin2018_wasserstein_dro](papers/mohajerin2018_wasserstein_dro/README.md) | c2 | indicator of J(x_N) <= J_N(eps) at eps = 1, N = 300 | 1 | 1 (0) | 200 | 0.02 | yes | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 0.3.0 |
| [mohajerin2018_wasserstein_dro](papers/mohajerin2018_wasserstein_dro/README.md) | c3 | J at eps = 1e-4 minus J at the radius of lowest mean J, N = 30 | 0.2 | 0.154 (0.1184) | 200 | 0.1 | yes | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 0.3.0 |
<!-- results-table:end -->

The table is generated from `results/` by `python -m tools.results_table`
and checked in CI.

## Running a reproduction

Python 3.11 or newer is required. For a GPU, install PyTorch from the
[PyTorch index](https://pytorch.org/get-started/locally/) before the
package itself.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m papers.<paper_id>.run --config papers/<paper_id>/configs/full.toml
python -m tools.results_table
```

Every paper also has a `smoke.toml` that runs in seconds on CPU. The test
suite runs all of them.

```bash
pytest && ruff check . && ruff format --check . && mypy
```

## Layout

| Path | Content |
| --- | --- |
| `papers/<paper_id>/` | One reproduction, with its report in `README.md` |
| `papers/_template/` | Starting point, copied by `python -m tools.new_paper <paper_id>` |
| `results/` | One JSON record per reproduced claim |
| `notebooks/` | One generated Google Colab notebook with every paper's report and recorded results ([open in Colab](https://colab.research.google.com/github/navidsharifi/optora-reproductions/blob/main/notebooks/reproductions.ipynb)) |
| `optora_reproductions/` | Shared helpers for settings, seeding and result records |
| `prototypes/` | Components Optora does not ship yet, kept here until upstreamed |
| `tools/` | Results table generator and paper scaffolder |

## License

MIT, see `LICENSE`.
