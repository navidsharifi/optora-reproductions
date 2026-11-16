"""Reproduction of Mohajerin Esfahani and Kuhn (2018), Section 7.2.

Wasserstein distributionally robust mean-CVaR portfolios, with the worst
case over the continuous support computed by Optora on a finite support and
checked against the paper's linear programming reformulation. See
`README.md` for the setup and the claims.
"""

PAPER_ID: str = __name__.rpartition(".")[2]
