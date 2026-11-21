# Results

One JSON record per reproduced claim, named `<paper_id>.<claim_id>.json`.
These files are the single source of truth for every number in the
repository. The table in the root `README.md` is generated from them with

```bash
python -m tools.results_table          # rewrite the table
python -m tools.results_table --check  # fail if the table is stale (CI)
```

Each record holds the paper id, the claim id, the metric, the paper value,
our mean and standard deviation over seeds, the tolerance, the status, the
seeds, the hardware, and the `torch` and `optora` versions used. The
schema is defined once in `optora_reproductions/results.py`.

## Curve records

`curves/<paper_id>.json` holds the series behind the figures of a paper
report, for example a metric at every epoch or at every radius of a sweep.
A claim record has one value per seed and no place for them. The run of a
paper writes its curve record next to its claim records, and

```bash
python -m tools.figures          # rewrite papers/<paper_id>/figures/*.svg
python -m tools.figures --check  # fail if a figure is stale (CI)
```

draws the figures from both. Each figure stores a digest of the records
it was drawn from, which is what the check compares.

The curve records of `duchi2019_variance_regularization`,
`ho2018_fast_bellman_updates` and `mohajerin2018_wasserstein_dro` were
read from the summary tables of their full runs, because those runs
predate the curve records. Their values therefore have the three or four
significant digits of the tables until the next full run replaces them.

Logs, checkpoints and per-run curves go to the git-ignored `runs/` folder,
not here.
