r"""Bernacchi et al. (2013) C3 photosynthesis model.

The model represents net leaf CO2 assimilation limited by Rubisco
carboxylation, RuBP regeneration, and triose-phosphate utilization. Figures
F2-F7 reproduce the steady-state and diurnal responses described in the paper.

https://doi.org/10.1111/pce.12118

|             |                                                                   |
| ----------- | ----------------------------------------------------------------- |
| doi         | 10.1111/pce.12118                                                 |
| main author | Carl J. Bernacchi                                                 |
| paper title | Modelling C3 photosynthesis from the chloroplast to the ecosystem |
| published   | April 2013                                                        |
| journal     | Plant, Cell & Environment                                         |
| organism    | C3 leaf                                                           |
| Ported by   | Tanvir Hassan ( )                                                 |
"""

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]


def electron_transport(
    PPFD: npt.ArrayLike, Jmax: float, alpha: float, theta: float
) -> Array:
    absorbed = alpha * np.asarray(PPFD, dtype=float)
    discriminant = (absorbed + Jmax) ** 2 - 4 * theta * absorbed * Jmax
    return (absorbed + Jmax - np.sqrt(np.maximum(discriminant, 0))) / (2 * theta)


def get_bernacchi_2013(
    Ci: npt.ArrayLike,
    PPFD: npt.ArrayLike,
    Vcmax: float,
    Jmax: float,
    TPU: float,
    Rd: float,
    Gamma_star: float,
    Kc: float,
    Ko: float,
    O: float,
    alpha: float,
    theta: float,
) -> tuple[Array, Array, Array, Array]:
    Ci, PPFD = np.broadcast_arrays(
        np.asarray(Ci, dtype=float),
        np.asarray(PPFD, dtype=float),
    )
    Ci_safe = np.maximum(Ci, 1.0)

    J = electron_transport(PPFD, Jmax, alpha, theta)

    Wc = Vcmax * Ci_safe / (Ci_safe + Kc * (1 + O / Ko))
    rubisco = (1 - Gamma_star / Ci_safe) * Wc - Rd

    rubp = J * (Ci_safe - Gamma_star) / (4 * Ci_safe + 8 * Gamma_star) - Rd

    tpu = np.full_like(Ci_safe, 3 * TPU - Rd)
    assimilation = np.minimum.reduce([rubisco, rubp, tpu])

    return rubisco, rubp, tpu, assimilation
