"""Salvatori 2022 model

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

import numpy as np
from mxlpy import Model


def energy_input(E_PSII, alpha, c_in, PAR, Ec_PSII):
    return alpha * c_in * PAR * (1 - E_PSII / Ec_PSII)


def ETR_out(E_PSII, NADP, v_ETR):
    return v_ETR * E_PSII * NADP


def energy_dissipation(E_PSII, P_NPQ, Q, v_d, Qc):
    return v_d * E_PSII * P_NPQ * (1 - Q / Qc)


def NPQ_func(Q, v_NPQ):
    return v_NPQ * Q


def activation_P_NPQ(P_NPQ, E_PSII, NADP, alpha, c_in, PAR, Ec_PSII, v_ETR, c_y, v_p):
    CEF = alpha * c_in * PAR * (1 - E_PSII / Ec_PSII) - v_ETR * E_PSII * NADP
    if c_y < CEF:
        return v_p * (1 - P_NPQ)
    return 0.0


def ETR_in(NADP, E_PSII, v_ETR, eta_NADP):
    return v_ETR * E_PSII * NADP * eta_NADP


def A_in(NADPH, R, v_C, eta_NADPH):
    return v_C * R * NADPH * eta_NADPH


def Rubisco_activation(R, pH, v_R, d):
    return v_R * (1 - R) * np.minimum(d, pH)


def pH_func(NADPH, NADP):
    return NADPH / NADP


def ETR_func(E_PSII, NADP, v_ETR):
    return v_ETR * E_PSII * NADP


def A_func(NADPH, R, v_C):
    return v_C * R * NADPH


def get_salvatori2022(load="Eiko") -> Model:
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

    m.add_derived("pH", pH_func, args=["NADPH", "NADP"])
    m.add_derived("ETR", ETR_func, args=["E_PSII", "NADP", "v_ETR"])
    m.add_derived("A", A_func, args=["NADPH", "R", "v_C"])

    m.add_reaction(
        "Energy_input",
        energy_input,
        stoichiometry={"E_PSII": 1},
        args=["E_PSII", "alpha", "c_in", "PAR", "Ec_PSII"],
    )
    m.add_reaction(
        "ETR_out",
        ETR_out,
        stoichiometry={"E_PSII": -1},
        args=["E_PSII", "NADP", "v_ETR"],
    )
    m.add_reaction(
        "ETR_in",
        ETR_in,
        stoichiometry={"NADP": -1, "NADPH": 1},
        args=["NADP", "E_PSII", "v_ETR", "eta_NADP"],
    )
    m.add_reaction(
        "energy_dissipation",
        energy_dissipation,
        stoichiometry={"E_PSII": -1, "Q": 1},
        args=["E_PSII", "P_NPQ", "Q", "v_d", "Qc"],
    )
    m.add_reaction("NPQ", NPQ_func, stoichiometry={"Q": -1}, args=["Q", "v_NPQ"])
    m.add_reaction(
        "NPQ_activation",
        activation_P_NPQ,
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
        A_in,
        stoichiometry={"NADPH": -1, "NADP": 1},
        args=["NADPH", "R", "v_C", "eta_NADPH"],
    )
    m.add_reaction(
        "RuBisCO_activation",
        Rubisco_activation,
        stoichiometry={"R": 1},
        args=["R", "pH", "v_R", "d"],
    )

    return m
