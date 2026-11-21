# Papers

One folder per reproduced paper. Folder names follow
`<firstauthor><year>_<topic>` in lowercase, for example
`sagawa2020_group_dro`, so they sort by author and stay valid Python
package names.

Create a new folder with

```bash
python -m tools.new_paper <firstauthor><year>_<topic>
```

which copies `_template/`. The copy runs unchanged, because each paper
reads its identifier from its own package name. Every folder has the same
layout.

| File | Purpose |
| --- | --- |
| `README.md` | Reproduction report with the verified citation, the claims and our findings |
| `claims.py` | Location, metric, paper value and tolerance of every claim, committed before the full run |
| `config.py` | Frozen dataclass holding every setting of the experiment |
| `configs/full.toml` | Settings taken from the paper |
| `configs/smoke.toml` | A run of a few seconds on CPU, used by CI |
| `data.py` | Download with checksum, or a synthetic generator |
| `run.py` | Entry point that runs every seed and writes `results/<id>.<claim_id>.json` and the curve record `results/curves/<id>.json` |
| `figures.py` | The figures of the report, each a function of the curve record and the claim records |
| `figures/` | The figures as SVG, one file per colour scheme, written by `python -m tools.figures` |
| `tests/test_smoke.py` | Runs `run.py` on the smoke config |

Add a `model.py` for models and per-sample losses once the paper has them.

Run a paper with

```bash
python -m papers.<id>.run --config papers/<id>/configs/full.toml
python -m tools.results_table
python -m tools.figures
python -m tools.notebooks
```

## Figures

A report shows its main results as figures. Every figure is drawn from the
recorded data, so it is declared in code and never edited by hand.

1. `run.py` returns the series to plot from `curve_record` and writes them
   with `write_curves`.
2. `figures.py` lists one `FigureSpec` per figure in `FIGURES`. The style
   and the two colour schemes come from `optora_reproductions/figures.py`.
3. `python -m tools.figures` writes `figures/<name>-light.svg` and
   `figures/<name>-dark.svg`.
4. The README embeds both files, so that GitHub shows the one that matches
   the reader's colour scheme.

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/<name>-dark.svg">
  <img alt="What the figure shows, in one or two sentences" src="figures/<name>-light.svg">
</picture>
```

CI runs `python -m tools.figures --check`, which fails when a figure is
older than the records, missing, not embedded in the README, or not
declared in `figures.py`. The Colab notebook and the Optora site load the
same files from the `main` branch.

Optora is used only through its public API. If a reproduction needs
anything else, open an issue on the Optora repository instead of working
around it here.
