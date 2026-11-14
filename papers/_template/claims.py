"""Claims reproduced from the paper.

Commit this file before the full run. Its paper values and tolerances are
the pre-registered acceptance criteria, and the README results table is
generated from the records that `run.py` builds from them.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Optora tests/dro/test_equivalences.py",
        metric="absolute error of the chi-square worst case vs mean + sqrt(rho Var)",
        paper_value=0.0,
        tolerance=1e-10,
    ),
)
