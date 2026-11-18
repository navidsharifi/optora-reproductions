"""Claims reproduced from Levy, Carmon, Duchi and Sidford (2020), Digits experiment.

Pre-registered before any full run. The paper values and tolerances below
are the acceptance criteria approved in plan step 2c on 2026-11-17. Each
seed is one training run per configuration, with the five seeds of the
authors' hyperparameter files.

The training objective of a run is the robust loss over the whole training
set plus the squared-norm regularizer, evaluated at the averaged iterate
after every epoch. For each robust objective, "opt" is the lowest value over
all runs of that objective, and a run reaches it when its objective is
within 2% of opt. A speed-up is the number of epochs the full-batch run
needs divided by the fewest epochs any minibatch size (50, 500, 5000) needs
for the same seed. A run that never gets within 2% of opt counts as needing
one epoch more than its budget, so its speed-up is a lower bound.

Claim c4 takes the worst of the 20 subgroups by mean test log loss, at the
averaged iterate after the last epoch. For each seed and robust objective it
compares the run with batch size 500 against the better of the two ERM runs
of that seed, and averages the relative reduction over the three objectives.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Table 5, CVaR with alpha = 0.02 (speed-up over full batch)",
        metric="full-batch epochs to 2% of opt / best minibatch epochs, CVaR",
        paper_value=9.0,
        tolerance=4.5,
    ),
    Claim(
        claim_id="c2",
        location="Table 5, constrained chi-square with rho = 1 (speed-up)",
        metric="full-batch epochs to 2% of opt / best minibatch epochs, chi-square",
        paper_value=9.5,
        tolerance=4.75,
    ),
    Claim(
        claim_id="c3",
        location="Table 5, penalized chi-square with lambda = 0.05 (speed-up)",
        metric="full-batch epochs to 2% of opt / best minibatch epochs, penalty",
        paper_value=16.2,
        tolerance=8.1,
    ),
    Claim(
        claim_id="c4",
        location="Section 7 and Appendix F.5.1 (17.5% to 27% lower worst-group loss)",
        metric="1 - worst-group test loss of DRO / of the better ERM, mean of 3 losses",
        paper_value=0.2225,
        tolerance=0.0475,
    ),
)
