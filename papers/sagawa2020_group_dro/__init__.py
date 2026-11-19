r"""Reproduction of Sagawa, Koh, Hashimoto and Liang (2020), Waterbirds.

Group DRO minimizes the worst expected loss over $m$ known groups,

$$
\hat\theta_{\mathrm{DRO}} = \arg\min_{\theta \in \Theta} \max_{g \in \mathcal G}
    \mathbb{E}_{(x, y) \sim \hat P_g}\big[\ell(\theta; (x, y))\big],
$$

(Eq. 4), and Section 3 shows that on an overparameterized ResNet-50 it helps
the worst group only together with a strong $\ell_2$ penalty or early
stopping. The experiment is the Waterbirds part of Table 1.
"""

PAPER_ID: str = __name__.rpartition(".")[2]
