r"""Template for a paper reproduction.

Copy it with `python -m tools.new_paper <paper_id>` and replace the
experiment. The template reproduces one exact identity so that its smoke
test exercises the whole pipeline from data to result record. For an
empirical distribution $p$ and a chi-square ball of radius $\rho$,

$$
\sup_{q:\, D_{\chi^2}(q \,\|\, p) \le \rho} \mathbb{E}_q[\ell]
    = \mathbb{E}_p[\ell] + \sqrt{\rho \, \mathrm{Var}_p(\ell)}
$$

whenever $\rho \le \mathrm{Var}_p(\ell) / (\mathbb{E}_p[\ell] - \min \ell)^2$,
the regime in which the maximizer keeps positive mass on every scenario
(Optora `tests/dro/test_equivalences.py`).
"""

PAPER_ID: str = __name__.rpartition(".")[2]
