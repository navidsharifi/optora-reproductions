"""Reproduction of Duchi and Namkoong (2019), Section 5.2.

The simulation compares the empirical risk minimizer with the minimizer of
the chi-square robustly regularized risk on a quadratic loss with a pure
noise linear perturbation. See `README.md` for the setup and the claims.
"""

PAPER_ID: str = __name__.rpartition(".")[2]
