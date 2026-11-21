r"""Continuous Wasserstein adversary in its Lagrangian form.

For a loss $\ell(\theta; z)$ and the transport cost
$c(z, z_0) = \lVert z - z_0 \rVert_2^2$, Sinha, Namkoong and Duchi (2018)
replace the Wasserstein ball by the penalty

$$
\phi_\gamma(\theta; z_0) = \sup_{z} \big\{ \ell(\theta; z) - \gamma\, c(z, z_0) \big\},
$$

the robust surrogate of their Eq. (2b), whose empirical mean is the dual
objective of the ball by their Proposition 1. When $z \mapsto \ell(\theta; z)$
has $L$-Lipschitz gradients and $\gamma > L$, the inner problem is strongly
concave and gradient ascent finds its maximizer $T_\gamma(\theta; z_0)$, the
transport (Monge) map of their Eq. (9).

The prototype is shaped by two reproductions, Sinha et al. (2018) and Bai,
He, Jiang and Obloj (2023), before it is proposed to Optora.
"""

from .transport import squared_distance, transport_map

__all__ = ["squared_distance", "transport_map"]
