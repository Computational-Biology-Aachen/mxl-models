r"""Bernacchi et al. (2013) C3 photosynthesis model.

The model represents net leaf CO2 assimilation limited by Rubisco
carboxylation, RuBP regeneration, and triose-phosphate utilization, with the
temperature responses the paper uses: Arrhenius functions (Eqn 10, Bernacchi
et al. 2001) for Vcmax, Rd, Gamma*, Kc and Ko, June et al. (2004) for Jmax
(Eqn 14) and Harley et al. (1992) for TPU (Eqn 11). Net assimilation is
Eqn 2, (1 - Gamma*/C) min(wc, wj, wp) - Rd, so below Gamma* it follows the
Rubisco limitation.

The paper gives no values for the Jmax and TPU temperature constants nor for
the light-response parameters; those, and Jmax at 25 degC, are fitted to its
Figs 2-4 (digitized). With Eqn 6 as printed, Fig 2 needs J = 168 at
PPFD 1500, above the caption's Jmax of 160: the figures imply Jmax25 = 180
(= 160 * 4.5/4). Figs 3 and 4 reproduce at Ci = 247.4 rather than the
caption's 270, as their Rubisco-limited rate (19.8) lies below Fig 2's at 270
(21.6). TPU (10.62, its value at 25 degC without deactivation) is fitted
too: the figures' TPU-limited rate of 29.55 implies 10.22 at 25 degC, not
the caption's 10.5. Figures F2-F4 reproduce the CO2, leaf-temperature and PPFD responses.

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

R = 8.314  # J K^-1 mol^-1
T25 = 298.15  # K

# Activation energies (J mol^-1), Bernacchi et al. (2001)
DHA_VCMAX = 65330.0
DHA_RD = 46390.0
DHA_GAMMA_STAR = 37830.0
DHA_KC = 79430.0
DHA_KO = 36380.0

# June et al. (2004) Jmax response (degC), fitted to Fig 3
JMAX_TOPT = 56.4
JMAX_OMEGA = 30.6

# Harley et al. (1992) TPU response, fitted to Figs 2-4
TPU_DHA = 50630.0  # J mol^-1
TPU_DS = 840.7  # J K^-1 mol^-1
TPU_DHD = 258480.0  # J mol^-1

# Ci (umol mol^-1) at which Figs 3 and 4 reproduce
CI_FIG34 = 247.4


def arrhenius(p25: float, dHa: float, Tk: Array) -> Array:
    r"""Parameter at leaf temperature Tk (K) from its 25 degC value (Eqn 10)."""
    return p25 * np.exp(dHa * (Tk - T25) / (R * Tk * T25))


def june_jmax(Jmax25: float, T_leaf: Array, Topt: float, omega: float) -> Array:
    r"""Jmax at leaf temperature T_leaf (degC), normalized to 25 degC (Eqn 14)."""
    return Jmax25 * np.exp(((25 - Topt) / omega) ** 2 - ((T_leaf - Topt) / omega) ** 2)


def harley_tpu(TPU25: float, Tk: Array, dHa: float, dS: float, dHd: float) -> Array:
    r"""TPU at leaf temperature Tk (K) with high-temperature deactivation (Eqn 11).

    The scaling constant c is set by TPU25, the rate at 25 degC without
    deactivation, so the deactivation term already lowers TPU at 25 degC.
    """
    return arrhenius(TPU25, dHa, Tk) / (1 + np.exp((dS * Tk - dHd) / (R * Tk)))


def electron_transport(
    PPFD: npt.ArrayLike,
    Jmax: npt.ArrayLike,
    alpha: float,
    beta: float,
    phi: float,
    theta: float,
) -> Array:
    r"""Potential electron transport rate J from the non-rectangular hyperbola (Eqn 8)."""
    absorbed = np.asarray(PPFD, dtype=float) * alpha * phi * beta
    Jmax = np.asarray(Jmax, dtype=float)
    discriminant = (absorbed + Jmax) ** 2 - 4 * theta * absorbed * Jmax
    return (absorbed + Jmax - np.sqrt(np.maximum(discriminant, 0))) / (2 * theta)


def get_bernacchi_2013(
    Ci: npt.ArrayLike,
    PPFD: npt.ArrayLike,
    T_leaf: npt.ArrayLike = 25.0,
    Vcmax: float = 98.0,
    Jmax: float = 180.0,
    TPU: float = 10.62,
    Rd: float = 1.1,
    Gamma_star: float = 42.75,
    Kc: float = 404.9,
    Ko: float = 278.4,
    O: float = 210.0,
    alpha: float = 0.85,
    beta: float = 0.5,
    phi: float = 0.82,
    theta: float = 0.85,
) -> tuple[Array, Array, Array, Array]:
    r"""Rubisco-, RuBP- and TPU-limited rates and net assimilation (their minimum).

    Vcmax, Jmax, TPU, Rd, Gamma_star, Kc and Ko are their values at 25 degC,
    scaled to T_leaf (TPU: without deactivation). Defaults reproduce the
    paper's Figs 2-4.

    Units:

    - Ci: intercellular CO2 mole fraction, umol mol^-1
    - PPFD: incident photon flux density, umol m^-2 s^-1
    - T_leaf: leaf temperature, degC
    - Vcmax, Jmax, TPU, Rd: umol m^-2 s^-1
    - Gamma_star, Kc: umol mol^-1
    - Ko, O: mmol mol^-1
    - alpha, beta, phi, theta: dimensionless
    - returned rates: umol m^-2 s^-1
    """
    Ci, PPFD, T_leaf = np.broadcast_arrays(
        np.asarray(Ci, dtype=float),
        np.asarray(PPFD, dtype=float),
        np.asarray(T_leaf, dtype=float),
    )
    Ci_safe = np.maximum(Ci, 1.0)
    Tk = T_leaf + 273.15

    Vcmax_T = arrhenius(Vcmax, DHA_VCMAX, Tk)
    Rd_T = arrhenius(Rd, DHA_RD, Tk)
    Gamma_star_T = arrhenius(Gamma_star, DHA_GAMMA_STAR, Tk)
    Kc_T = arrhenius(Kc, DHA_KC, Tk)
    Ko_T = arrhenius(Ko, DHA_KO, Tk)
    Jmax_T = june_jmax(Jmax, T_leaf, JMAX_TOPT, JMAX_OMEGA)
    TPU_T = harley_tpu(TPU, Tk, TPU_DHA, TPU_DS, TPU_DHD)

    J = electron_transport(PPFD, Jmax_T, alpha, beta, phi, theta)
    co2_factor = 1 - Gamma_star_T / Ci_safe

    wc = Vcmax_T * Ci_safe / (Ci_safe + Kc_T * (1 + O / Ko_T))
    wj = J * Ci_safe / (4.5 * Ci_safe + 10.5 * Gamma_star_T)

    rubisco = co2_factor * wc - Rd_T
    rubp = co2_factor * wj - Rd_T
    tpu = 3 * TPU_T - Rd_T

    # Eqn 2 with wp of Eqn 7; TPU cannot limit at or below Gamma*
    wp = 3 * TPU_T / np.maximum(co2_factor, 1e-12)
    assimilation = co2_factor * np.minimum.reduce([wc, wj, wp]) - Rd_T

    return rubisco, rubp, tpu, assimilation
