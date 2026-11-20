"""Claims reproduced from Sagawa, Koh, Hashimoto and Liang (2020), Waterbirds.

Pre-registered before any full run. The paper values and tolerances below
are the acceptance criteria approved in plan step 2c on 2026-11-19. Every
value comes from the Waterbirds rows of Table 1, which reports one model per
cell, and accuracies are fractions in `[0, 1]`.

A model is evaluated once, after its last epoch, in evaluation mode. The
accuracy of a group is the fraction of its examples classified correctly,
and the worst-group accuracy is the smallest of the four. The average
accuracy weights the group accuracies by the group proportions of the
training set, as Appendix C.1 does for the test set, since the validation
and test sets are balanced across backgrounds.

Claims c1, c2, c3 and c5 use the seeds of the 300-epoch runs and c4 the
seeds of the early-stopping runs. Claim c5 pairs, for each seed, the
strong-penalty runs with the early-stopping runs of the same seed and
averages the two differences.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Table 1, Waterbirds, standard regularization (train rows)",
        metric="min over ERM and group DRO of the worst-group training accuracy",
        paper_value=1.0,
        tolerance=0.01,
    ),
    Claim(
        claim_id="c2",
        location="Table 1, Waterbirds, standard regularization, ERM (97.3 vs 60.0)",
        metric="ERM average test accuracy minus worst-group test accuracy",
        paper_value=0.373,
        tolerance=0.15,
    ),
    Claim(
        claim_id="c3",
        location="Table 1, Waterbirds, strong L2 penalty lambda = 1 (84.6 vs 21.3)",
        metric="group DRO minus ERM worst-group test accuracy, strong L2 penalty",
        paper_value=0.633,
        tolerance=0.2,
    ),
    Claim(
        claim_id="c4",
        location="Table 1, Waterbirds, early stopping after 1 epoch (86.0 vs 6.7)",
        metric="group DRO minus ERM worst-group test accuracy, early stopping",
        paper_value=0.793,
        tolerance=0.2,
    ),
    Claim(
        claim_id="c5",
        location="Table 1, Waterbirds, strong L2 and early stopping, average test",
        metric="ERM minus group DRO average test accuracy, mean of the two regimes",
        paper_value=-0.0015,
        tolerance=0.03,
    ),
)
