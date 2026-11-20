r"""Claims reproduced from Sinha, Namkoong and Duchi (ICLR 2018).

Pre-registered before any full run. The paper values and tolerances below
are the acceptance criteria approved in plan step 2c on 2026-11-20. The
paper reports these quantities only as curves, so every paper value was read
from the vector data of Figures 2, 3 and 4 in arXiv:1710.10571v4. The axes
were checked against values stated elsewhere. The certificates have the
slopes $\gamma = 2$ and $\gamma = 1/2.6$ of the code, and the vertical bar
of Figure 4(a) sits at $C_2 / \gamma = 23.9$.

The certificate of a model at the robustness level $\hat\rho_n$ it reaches
on the training set is $\mathbb E_{\hat P_n}[\phi_\gamma(\theta; Z)] +
\gamma \hat\rho_n = \mathbb E_{\hat P_n}[\ell(\theta; T_\gamma(\theta; Z))]$
(Eqs. 11 and 14). The test worst case at the same level is the red curve of
Figure 2, interpolated at $\rho = \hat\rho_n$. On MNIST the training loss
uses the smoothed labels the model was trained with and the test loss uses
the one-hot labels, which the intercepts of Figure 2(b), 0.63 and 0.16,
imply. Errors are fractions in $[0, 1]$, and every attack radius is in the
units of the pixels, which lie in $[0, 1]$.
"""

from optora_reproductions.results import Claim

CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="c1",
        location="Figure 2(a), synthetic data, at rho_hat_n(theta_WRM) = 0.062",
        metric="certificate minus test worst-case loss at rho_hat_n, synthetic WRM",
        paper_value=0.0056,
        tolerance=0.05,
    ),
    Claim(
        claim_id="c2",
        location="Section 4.2 and Figure 3(a) at eps_adv = 0 (99.0% to 99.2%)",
        metric="min over ERM, FGM, IFGM, PGM and WRM of the clean test accuracy",
        paper_value=0.990,
        tolerance=0.005,
    ),
    Claim(
        claim_id="c3",
        location="Figure 2(b), MNIST, at rho_hat_n(theta_WRM) = 0.209",
        metric="certificate minus test worst-case loss at rho_hat_n, MNIST WRM",
        paper_value=0.378,
        tolerance=0.15,
    ),
    Claim(
        claim_id="c4",
        location="Figure 3(a), l2 PGM at eps_adv = sqrt(rho_hat_n) (10.8% vs 2.3%)",
        metric="ERM minus WRM test error, l2 PGM attack at the training radius",
        paper_value=0.085,
        tolerance=0.05,
    ),
    Claim(
        claim_id="c5",
        location="Figure 3(b), linf PGM attack at eps_adv = 0.1 C_inf (25.2% vs 14.7%)",
        metric="min over FGM, IFGM and PGM minus WRM test error, linf PGM attack",
        paper_value=0.105,
        tolerance=0.07,
    ),
    Claim(
        claim_id="c6",
        location="Figure 4(a), at gamma_adv = gamma, C_2 / gamma = 23.9 (161 vs 0.58)",
        metric="log10 of rho_hat_test(ERM) / rho_hat_test(WRM) at the training gamma",
        paper_value=2.44,
        tolerance=0.5,
    ),
)
