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
No claims have been reproduced yet.
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
| `optora_reproductions/` | Shared helpers for settings, seeding and result records |
| `prototypes/` | Components Optora does not ship yet, kept here until upstreamed |
| `tools/` | Results table generator and paper scaffolder |

## License

MIT, see `LICENSE`.
