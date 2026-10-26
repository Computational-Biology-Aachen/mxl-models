r"""Salvatori 2022 model

|             |                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------- |
| doi         | 10.3389/fpls.2021.787877                                                                 |
| main author | Nicole Salvatori                                                                         |
| paper title | A System Dynamics Approach to Model Photosynthesis at Leaf Level Under Fluctuating Light |
| published   | January 2022                                                                             |
| journal     | Frontiers Plant Science                                                                  |
| organism    | soybean, leaf, c3 photosynthesis                                                         |
| Ported by   | Joshua Ebeling ( @pmfjosh )                                                              |

The Salvatori 2022 model is a soybean leaf C3 photosynthesis model, developed
to investigate the effect of fluctuating light on two soybean varients: Eiko
(WT) and MinnGold (chlorophyll deficient mutant). The goal was to investigate
the role of the chlorophyll content to adjust to light fluctuations. It is a
simple, small model, only including the important processes for the goal of the
study. As RuBisCO activation is known to be the main limitation in dark-light
transition, and stomal conductance is the same in both varients, RuBisCO is
included but stomatal conductance not. CBB is also not included as the focus
lies on fluctuations, which makes CBB irrelevant. Only light harvesting around
PSII is included for simplicity, but represented as ETR, that produces NADPH.
The NADP+ to NADPH ratio determines the delta pH in the model, which activates
RuBisCO. In case of excess energy, energy can be dissipated through NPQ, here
only qE, as it is the fastest responding NPQ mechanism and the rest are
irrelevant for the study. CEF is inlcuded only as a regulator of qE, as it is
described to be only relevant in stress conditions when delta pH generation,
without NADPH production is necessary.

The model is easy to reproduce, but their figure timescale is in minutes, even
though simulation results differe strongly when performed over the same
timescale. If reduced to seconds the simulations are 1:1. So either they used
different units for their simulations without a parameter change or put the
wrong labels on their figures.
"""

from typing import Literal

import numpy as np
from mxlpy import Model, Parameter, Variable
from mxlpy.units import dimensionless, mumol, ppfd, second, sqm

# Units from Table 1
_umol_m2 = mumol / sqm
_per_s = 1 / second
_umol_m2_s = _umol_m2 / second


def _energy_input(
    e_psii: float, alpha: float, c_in: float, par: float, ec_psii: float
) -> float:
    return alpha * c_in * par * (1 - e_psii / ec_psii)


def _etr_out(e_psii: float, nadp: float, v_etr: float) -> float:
    return v_etr * e_psii * nadp


def _energy_dissipation(
    e_psii: float, p_npq: float, q: float, v_d: float, qc: float
) -> float:
    return v_d * e_psii * p_npq * (1 - q / qc)


def _npq_func(q: float, v_npq: float) -> float:
    return v_npq * q


def _activation_p_npq(
    p_npq: float,
    e_psii: float,
    nadp: float,
    alpha: float,
    c_in: float,
    par: float,
    ec_psii: float,
    v_etr: float,
    c_y: float,
    v_p: float,
) -> float:
    CEF = alpha * c_in * par * (1 - e_psii / ec_psii) - v_etr * e_psii * nadp
    if c_y < CEF:
        return v_p * (1 - p_npq)
    return 0.0


def _etr_in(nadp: float, e_psii: float, v_etr: float, eta_nadp: float) -> float:
    # FIXME: the flux is in umol m^-2 s^-1 (Table 1), but NADP+ and NADPH are
    # dimensionless, so this reduction of NADP+ is dimensionally inconsistent
    return v_etr * e_psii * nadp * eta_nadp


def _a_in(nadph: float, r: float, v_c: float, eta_nadph: float) -> float:
    return v_c * r * nadph * eta_nadph


def _rubisco_activation(r: float, p_h: float, v_r: float, d: float) -> float:
    return v_r * (1 - r) * np.minimum(d, p_h)


def _p_h_func(nadph: float, nadp: float) -> float:
    return nadph / nadp


def _etr_func(e_psii: float, nadp: float, v_etr: float) -> float:
    return v_etr * e_psii * nadp


def _a_func(nadph: float, r: float, v_c: float) -> float:
    return v_c * r * nadph


def get_salvatori2022(load: Literal["Minn", "Eiko"] = "Eiko") -> Model:
    r"""Get Salvatori model"""
    m = Model()

    if load == "Minn":
        m.add_parameters(
            {
                "PAR": Parameter(
                    0, unit=ppfd, display_name="PPFD"
                ),  # Photosynthetically active radiation
                "alpha": Parameter(0.54, unit=dimensionless),  # Absorption coefficient
                "c_in": Parameter(0.25, unit=dimensionless),  # Energy input coefficient
                "Ec_PSII": Parameter(
                    9.98, unit=_umol_m2
                ),  # PSII energy carrying capacity
                "v_ETR": Parameter(11.56, unit=_per_s),  # Velocity of ETR
                "v_d": Parameter(7.00, unit=_per_s),  # Velocity of energy dissipation
                "Qc": Parameter(
                    0.03, unit=_umol_m2
                ),  # PSII-zeax complex energy carrying capacity
                "v_NPQ": Parameter(53.87, unit=_per_s),  # Velocity of NPQ
                "v_p": Parameter(
                    0.01, unit=_per_s
                ),  # Maximum velocity of NPQ-related proteins activation
                "v_C": Parameter(
                    13.04, unit=_per_s
                ),  # Maximum velocity of Calvin Cycle reactions
                "eta_NADPH": Parameter(4.10, unit=dimensionless),  # Efficiency of NADPH
                "eta_NADP": Parameter(0.75, unit=dimensionless),  # Efficiency of NADP+
                "v_R": Parameter(
                    14e-4, unit=_per_s
                ),  # Maximum velocity of Rubisco activation
                "d": Parameter(3.69, unit=dimensionless),  # Maximum pH balance value
                "c_y": Parameter(
                    0, unit=dimensionless
                ),  # Minimum necessary cyclic electron flow
            }
        )

    else:
        m.add_parameters(
            {
                "PAR": Parameter(
                    0, unit=ppfd, display_name="PPFD"
                ),  # Photosynthetically active radiation
                "alpha": Parameter(0.78, unit=dimensionless),  # Absorption coefficient
                "c_in": Parameter(0.23, unit=dimensionless),  # Energy input coefficient
                "Ec_PSII": Parameter(
                    157.56, unit=_umol_m2
                ),  # PSII energy carrying capacity
                "v_ETR": Parameter(0.78, unit=_per_s),  # Velocity of ETR
                "v_d": Parameter(0.08, unit=_per_s),  # Velocity of energy dissipation
                "Qc": Parameter(
                    0.07, unit=_umol_m2
                ),  # PSII-zeax complex energy carrying capacity
                "v_NPQ": Parameter(70.58, unit=_per_s),  # Velocity of NPQ
                "v_p": Parameter(
                    0.07, unit=_per_s
                ),  # Maximum velocity of NPQ-related proteins activation
                "v_C": Parameter(
                    11.75, unit=_per_s
                ),  # Maximum velocity of Calvin Cycle reactions
                "eta_NADPH": Parameter(5.07, unit=dimensionless),  # Efficiency of NADPH
                "eta_NADP": Parameter(0.89, unit=dimensionless),  # Efficiency of NADP+
                "v_R": Parameter(
                    8.9e-4, unit=_per_s
                ),  # Maximum velocity of Rubisco activation
                "d": Parameter(8.40, unit=dimensionless),  # Maximum pH balance value
                "c_y": Parameter(
                    -4, unit=dimensionless
                ),  # Minimum necessary cyclic electron flow
            }
        )

    m.add_variables(
        {
            "E_PSII": Variable(0, unit=_umol_m2),
            "Q": Variable(0, unit=_umol_m2),
            "P_NPQ": Variable(0, unit=dimensionless),
            "NADP": Variable(5, unit=dimensionless),
            "NADPH": Variable(5, unit=dimensionless),
            "R": Variable(0.001, unit=dimensionless),
        }
    )

    m.add_derived("pH", _p_h_func, args=["NADPH", "NADP"], unit=dimensionless)
    m.add_derived("ETR", _etr_func, args=["E_PSII", "NADP", "v_ETR"], unit=_umol_m2_s)
    m.add_derived("A", _a_func, args=["NADPH", "R", "v_C"], unit=_umol_m2_s)

    m.add_reaction(
        "Energy_input",
        _energy_input,
        stoichiometry={"E_PSII": 1},
        args=["E_PSII", "alpha", "c_in", "PAR", "Ec_PSII"],
        unit=_umol_m2_s,
    )
    m.add_reaction(
        "ETR_out",
        _etr_out,
        stoichiometry={"E_PSII": -1},
        args=["E_PSII", "NADP", "v_ETR"],
        unit=_umol_m2_s,
    )
    m.add_reaction(
        "ETR_in",
        _etr_in,
        stoichiometry={"NADP": -1, "NADPH": 1},
        args=["NADP", "E_PSII", "v_ETR", "eta_NADP"],
        unit=_umol_m2_s,
    )
    m.add_reaction(
        "energy_dissipation",
        _energy_dissipation,
        stoichiometry={"E_PSII": -1, "Q": 1},
        args=["E_PSII", "P_NPQ", "Q", "v_d", "Qc"],
        unit=_umol_m2_s,
    )
    m.add_reaction(
        "NPQ", _npq_func, stoichiometry={"Q": -1}, args=["Q", "v_NPQ"], unit=_umol_m2_s
    )
    m.add_reaction(
        "NPQ_activation",
        _activation_p_npq,
        stoichiometry={"P_NPQ": 1},
        args=[
            "P_NPQ",
            "E_PSII",
            "NADP",
            "alpha",
            "c_in",
            "PAR",
            "Ec_PSII",
            "v_ETR",
            "c_y",
            "v_p",
        ],
        unit=_per_s,
    )
    m.add_reaction(
        "Carbon_assimilation",
        _a_in,
        stoichiometry={"NADPH": -1, "NADP": 1},
        args=["NADPH", "R", "v_C", "eta_NADPH"],
        unit=_umol_m2_s,
    )
    m.add_reaction(
        "RuBisCO_activation",
        _rubisco_activation,
        stoichiometry={"R": 1},
        args=["R", "pH", "v_R", "d"],
        unit=_per_s,
    )

    return m
