# <Paper title>

> Status. `not started` | `in progress` | `reproduced` | `partially reproduced` | `not reproduced` | `blocked (optora#<n>)`

## Reference

<Authors>. <Title>. <Venue>, <year>. <DOI or arXiv id>.

Citation checked against the published version on <YYYY-MM-DD>.
Authors' code, if public, is at <URL> (commit <short hash>).

## Claim

The claim we reproduce, with its location in the paper (for example
Table 2, row 3). One claim per row.

| Claim id | Location | Metric | Paper value | Tolerance |
| --- | --- | --- | --- | --- |
| c1 | Table _ | _ | _ | _ |

The tolerance is fixed in this file before the full run.

## Setup

| Item | Value |
| --- | --- |
| Optora components | `optora.<Name>`, ... |
| Data | Source, size, split, preprocessing |
| Model | Architecture and parameter count |
| Seeds | e.g. 0-4 |
| Hardware | e.g. RTX 3050 Laptop 6 GB |
| Differences from the paper | Anything scaled down or changed, and why |

## Run

```bash
python -m papers.<id>.run --config papers/<id>/configs/smoke.toml
python -m papers.<id>.run --config papers/<id>/configs/full.toml
```

## Result

Generated from `results/<id>.json`.

| Claim id | Paper value | Ours (mean +- std over seeds) | Within tolerance |
| --- | --- | --- | --- |
| c1 | _ | _ | _ |

## Findings

What matched, what did not, and the most likely reason. Link every Optora
issue this paper produced.
