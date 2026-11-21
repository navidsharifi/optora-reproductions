r"""Reproduction of Sinha, Namkoong and Duchi (ICLR 2018).

WRM trains a model against the Lagrangian relaxation of a Wasserstein ball
around the data,

$$
\min_{\theta} \; \mathbb E_{\hat P_n}\big[\phi_\gamma(\theta; Z)\big], \qquad
\phi_\gamma(\theta; z_0) = \sup_{z} \big\{\ell(\theta; z) - \gamma\, c(z, z_0)\big\}
$$

(Eqs. 2b and 7), with $c$ the squared Euclidean distance between inputs.
Its value certifies the worst-case loss over the ball of the robustness
level it reaches (Theorem 3). The experiments are the synthetic data of
Section 4.1 and MNIST in Section 4.2, Figures 2 to 4.
"""

PAPER_ID: str = __name__.rpartition(".")[2]
