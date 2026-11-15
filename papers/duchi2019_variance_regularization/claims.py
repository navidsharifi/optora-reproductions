"""Claims reproduced from Duchi and Namkoong (2019), Section 5.2.

Pre-registered on 2026-11-15, before any full run. The paper values and
tolerances below are the acceptance criteria approved in plan step 2c.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Section 5.2 and Figure 2",
        metric="fraction of replications with robust risk below ERM risk",
        paper_value=1.0,
        tolerance=0.0,
    ),
    Claim(
        claim_id="c2",
        location="Table 1, B = 1, n = 1000",
        metric="mean risk of the robust solution",
        paper_value=6.32e-18,
        tolerance=1e-8,
    ),
    Claim(
        claim_id="c3",
        location="Table 1, B = 1, n = 100",
        metric="mean robust risk divided by mean ERM risk",
        paper_value=0.182,
        tolerance=0.05,
    ),
)
