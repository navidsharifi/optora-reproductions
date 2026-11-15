"""Claims reproduced from Ho, Petrik and Wiesemann (ICML 2018), Section 5.1.

Pre-registered before any full run. The paper values and tolerances below
are the acceptance criteria approved in plan step 2c on 2026-11-15.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Section 5 (the methods are exact)",
        metric="max absolute error of q(xi) against the HiGHS linear program",
        paper_value=0.0,
        tolerance=1e-9,
    ),
    Claim(
        claim_id="c2",
        location="Section 5.1 and Figure 4 (plain L1, about 1,000 times faster)",
        metric="log10 of LP time over Optora time for one q(xi), S = 400",
        paper_value=3.0,
        tolerance=1.0,
    ),
)
