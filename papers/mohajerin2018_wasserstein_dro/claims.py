"""Claims reproduced from Mohajerin Esfahani and Kuhn (2018), Section 7.2.

Pre-registered before any full run. The paper values and tolerances below
are the acceptance criteria approved in plan step 2c on 2026-11-15. Each
seed is one simulation, so the mean over seeds is the mean over the 200
simulations of the paper.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Proposition 7.2 and Figure 4 (equal weights at large radius)",
        metric="max_i |x_N(eps)_i - 1/m| at eps = 1, N = 300",
        paper_value=0.0,
        tolerance=1e-3,
    ),
    Claim(
        claim_id="c2",
        location="Figure 5(b) (reliability reaches one)",
        metric="indicator of J(x_N) <= J_N(eps) at eps = 1, N = 300",
        paper_value=1.0,
        tolerance=0.02,
    ),
    Claim(
        claim_id="c3",
        location="Figure 5(a) and Section 7.2.A (robustness helps at N = 30)",
        metric="J at eps = 1e-4 minus J at the radius of lowest mean J, N = 30",
        paper_value=0.2,
        tolerance=0.1,
    ),
)
