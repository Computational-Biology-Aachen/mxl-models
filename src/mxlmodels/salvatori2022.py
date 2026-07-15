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
from mxlpy import Model


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
                "PAR": 0,  # Photosynthetically active radiation
                "alpha": 0.54,  # Absorption coefficient
                "c_in": 0.25,  # Energy input coefficient
                "Ec_PSII": 9.98,  # PSII energy carrying capacity
                "v_ETR": 11.56,  # Velocity of ETR
                "v_d": 7.00,  # Velocity of energy dissipation
                "Qc": 0.03,  # PSII-zeax complex energy carrying capacity
                "v_NPQ": 53.87,  # Velocity of NPQ
                "v_p": 0.01,  # Maximum velocity of NPQ-related proteins activation
                "v_C": 13.04,  # Maximum velocity of Calvin Cycle reactions
                "eta_NADPH": 4.10,  # Efficiency of NADPH
                "eta_NADP": 0.75,  # Efficiency of NADP+
                "v_R": 14e-4,  # Maximum velocity of Rubisco activation
                "d": 3.69,  # Maximum pH balance value
                "c_y": 0,  # Minimum necessary cyclic electron flow
            }
        )

    else:
        m.add_parameters(
            {
                "PAR": 0,  # Photosynthetically active radiation
                "alpha": 0.78,  # Absorption coefficient
                "c_in": 0.23,  # Energy input coefficient
                "Ec_PSII": 157.56,  # PSII energy carrying capacity
                "v_ETR": 0.78,  # Velocity of ETR
                "v_d": 0.08,  # Velocity of energy dissipation
                "Qc": 0.07,  # PSII-zeax complex energy carrying capacity
                "v_NPQ": 70.58,  # Velocity of NPQ
                "v_p": 0.07,  # Maximum velocity of NPQ-related proteins activation
                "v_C": 11.75,  # Maximum velocity of Calvin Cycle reactions
                "eta_NADPH": 5.07,  # Efficiency of NADPH
                "eta_NADP": 0.89,  # Efficiency of NADP+
                "v_R": 8.9e-4,  # Maximum velocity of Rubisco activation
                "d": 8.40,  # Maximum pH balance value
                "c_y": -4,  # Minimum necessary cyclic electron flow
            }
        )

    m.add_variables(
        {"E_PSII": 0, "Q": 0, "P_NPQ": 0, "NADP": 5, "NADPH": 5, "R": 0.001}
    )

    m.add_derived("pH", _p_h_func, args=["NADPH", "NADP"])
    m.add_derived("ETR", _etr_func, args=["E_PSII", "NADP", "v_ETR"])
    m.add_derived("A", _a_func, args=["NADPH", "R", "v_C"])

    m.add_reaction(
        "Energy_input",
        _energy_input,
        stoichiometry={"E_PSII": 1},
        args=["E_PSII", "alpha", "c_in", "PAR", "Ec_PSII"],
    )
    m.add_reaction(
        "ETR_out",
        _etr_out,
        stoichiometry={"E_PSII": -1},
        args=["E_PSII", "NADP", "v_ETR"],
    )
    m.add_reaction(
        "ETR_in",
        _etr_in,
        stoichiometry={"NADP": -1, "NADPH": 1},
        args=["NADP", "E_PSII", "v_ETR", "eta_NADP"],
    )
    m.add_reaction(
        "energy_dissipation",
        _energy_dissipation,
        stoichiometry={"E_PSII": -1, "Q": 1},
        args=["E_PSII", "P_NPQ", "Q", "v_d", "Qc"],
    )
    m.add_reaction("NPQ", _npq_func, stoichiometry={"Q": -1}, args=["Q", "v_NPQ"])
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
    )
    m.add_reaction(
        "Carbon_assimilation",
        _a_in,
        stoichiometry={"NADPH": -1, "NADP": 1},
        args=["NADPH", "R", "v_C", "eta_NADPH"],
    )
    m.add_reaction(
        "RuBisCO_activation",
        _rubisco_activation,
        stoichiometry={"R": 1},
        args=["R", "pH", "v_R", "d"],
    )

    return m
