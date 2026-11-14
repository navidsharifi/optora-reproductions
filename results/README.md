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

Logs, checkpoints and figures go to the git-ignored `runs/` folder, not
here.
