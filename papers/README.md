# Papers

One folder per reproduced paper. Folder names follow
`<firstauthor><year>_<topic>` in lowercase, for example
`sagawa2020_group_dro`, so they sort by author and stay valid Python
package names.

Create a new folder with

```bash
python -m tools.new_paper <firstauthor><year>_<topic>
```

which copies `_template/`. Every folder has the same layout.

| File | Purpose |
| --- | --- |
| `README.md` | Reproduction report with the verified citation, the claim, the tolerance and our result |
| `config.py` | Frozen dataclass holding every setting of the experiment |
| `configs/full.toml` | Settings taken from the paper |
| `configs/smoke.toml` | A run of a few seconds on CPU, used by CI |
| `data.py` | Download with checksum, or a synthetic generator |
| `model.py` | Models and per-sample losses |
| `run.py` | Entry point that trains, evaluates and writes `results/<id>.json` |
| `tests/test_smoke.py` | Runs `run.py` on the smoke config |

Run a paper with

```bash
python -m papers.<id>.run --config papers/<id>/configs/full.toml
```

Optora is used only through its public API. If a reproduction needs
anything else, open an issue on the Optora repository instead of working
around it here.
