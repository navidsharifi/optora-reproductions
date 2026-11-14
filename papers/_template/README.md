# <Paper title>

> Status. `not started` | `in progress` | `reproduced` | `partially reproduced` | `not reproduced` | `blocked (optora#<n>)`

## Reference

<Authors>. <Title>. <Venue>, <year>. <DOI or arXiv id>.

Citation checked against the published version on <YYYY-MM-DD>.
Authors' code, if public, is at <URL> (commit <short hash>).

## Claim

What each claim says and why we chose it. The location, metric, paper
value and tolerance of every claim are defined once in `claims.py`, which
is committed before the full run, and they appear in the root results
table.

| Claim id | Statement |
| --- | --- |
| c1 | _ |

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

See the results table in the root `README.md`, generated from
`results/<id>.<claim_id>.json`.

## Findings

What matched, what did not, and the most likely reason. Link every Optora
issue this paper produced.
