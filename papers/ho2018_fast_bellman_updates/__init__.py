"""Reproduction of Ho, Petrik and Wiesemann (ICML 2018), Section 5.1.

Robust Bellman updates under an $s,a$-rectangular $L_1$ ambiguity set,
computed exactly by Optora's total-variation set and checked against a
linear programming solver. See `README.md` for the setup and the claims.
"""

PAPER_ID: str = __name__.rpartition(".")[2]
