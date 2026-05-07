from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from mxlpy import Model


############SUPPORT FUNCTION##############

def chlorophyll_fluo_lifetime(kappa_QV, QV,
                              kappa_QA, QA,
                              kappa_QZ, QZ,
                              kappa_QL, QL,
                              kappa_qZ, Z,
                              k_qI, PSIId,
                              delta):
    """
    Compute fluorescence lifetime from concentration time series.
    """
    tau  = 1.0 / (delta
                  + kappa_QV * QV
                  + kappa_QA * QA
                  + kappa_QZ * QZ
                  + kappa_QL * QL
                  + kappa_qZ * Z
                  + k_qI * PSIId)
    return tau



############DERIVED##############
def P_free(Ptot, Xtot, V, A, Z):
        return Ptot - Xtot + V + A + Z

def moiety(Ptot, Xtot, V, A, Z):
        return Ptot - Xtot + V + A + Z

############RATES##############

def mass_action_2s(k, s1, s2):
      return k*s1*s2

def mass_action_1s(k, s):
      return k*s

def mass_action_light_dark_1s(ppfd, k_light, k_dark, s):
    if ppfd ==0:
        return k_dark*s
    return k_light*s

def mass_action_light_dark_2s(ppfd, k_light, k_dark, s1, s2):
    if ppfd ==0:
        return k_dark*s1*s2
    return k_light*s1*s2
# --- VDE enzyme activation ---

def v_VDE(t, E):
        li = I(t)
        E_eq = [gamma, 1.0][li]
        return k["vde"][li] * (E_eq - E)

############MODELS##############

def get_lam2026() -> Model:
    m = Model()
    m.add_parameters({
        "k_L_VA": 2.47,
        "k_D_VA": 0.0014,
        "k_L_AZ": 0.5,
        "k_D_AZ": 0, # guess, no documentation
        "k_AV": 1.12,
        "k_ZA": 0.07,
        "k_PV_f": 2.18,
        "k_PV_b": 9.43,
        "k_PA_f": 130,
        "k_PA_b": 254,
        "k_PZ_f": 130,
        "k_PZ_b": 126,
        "k_L_QV_f": 0.027,
        "k_QV_b": 0.066,
        "k_L_QA_f": 0.66,
        "k_QA_b": 8.57,
        "k_L_QZ_f": 0.56,
        "k_QZ_b": 1.22,
        "k_L_QL_f": 0.056,
        "k_QL_b": 3.68,
        "k_D_QX_f": 0, # for all QX complex the rate in the dark is set to 0
        "k_L_damage":0.0222,
        "k_D_damage": 0.0161,
        "k_D_VDE": 0.24,
        "k_L_VDE": 0.28,
        "k_AV_aba1": 0.006,
        "k_ZA_aba1": 0.038,
        "k_PV_f_lut2": 1.43,
        "k_PV_b_lut2": 13.1,
        "k_PA_f_lut2": 34.4,
        "k_PA_b_lut2": 294,
        "k_PZ_f_lut2": 74.1,
        "k_PZ_b_lut2": 168,
        "V_tot_npq1": 49.8,
        "V_tot_lut2": 71.2,
        "V_tot_npq4": 40.6,
        "V_tot_aba1": 10.7,
        "V_tot_WT": 35.9,
        "P_tot": 45.4,
        "P_tot_lut2": 49.9,
        "kappa_QV": 0.040,
        "kappa_QA": 0.174,
        "kappa_QZ": 0.177,
        "kappa_QL": 0.262,
        "kappa_qZ": 0.030,
        "kappa_QI": 3.86,
        "kappa_QI_double_mut": 7.05,
        "ppfd":0,
    })

        # Variables and initial conditions
    m.add_variables(
               {
                      "V": 1,
                      "A": 0,
                      "Z": 0,
                      "PV": 1,
                      "PA": 0,
                      "PZ": 0,
                      "QV": 1,
                      "QA": 0,
                      "QZ": 0,
                      "PL": 0,
                      "qI": 0,
                      "PSIId": 0,
                      "VDE": 0,
               }
        )

    m.add_reaction("VA",    mass_action_light_dark_2s,    stoichiometry={"V": -1, "A":  1}, args=["ppfd","k_L_VA", "k_D_VA", "VDE", "V"])
    m.add_reaction("AV",    mass_action_1s,    stoichiometry={"A": -1, "V":  1}, args=["k_AV","A"])
    m.add_reaction("AZ",    mass_action_light_dark_1s,    stoichiometry={"A": -1, "Z":  1}, args=["ppfd", "k_L_AZ", "k_D_AZ","VDE", "A"])
    m.add_reaction("ZA",    mass_action_1s,    stoichiometry={"Z": -1, "A":  1}, args=["k_ZA","Z"])

    m.add_reaction("PVf",   mass_action_2s,   stoichiometry={"V": -1, "PV":  1}, args=["k_PV_f", "V", "P"])
    m.add_reaction("PVb",   mass_action_1s,   stoichiometry={"PV": -1, "V":  1}, args=["k_PV_b", "PV"])
    m.add_reaction("PAf",   mass_action_2s,   stoichiometry={"A": -1, "PA":  1}, args=["k_PA_f", "A", "P"])
    m.add_reaction("PAb",   mass_action_1s,   stoichiometry={"PA": -1, "A":  1}, args=["k_PA_b", "PA"])
    m.add_reaction("PZf",   mass_action_2s,   stoichiometry={"Z": -1, "PZ":  1}, args=["k_PZ_f", "Z", "P"])
    m.add_reaction("PZb",   mass_action_1s,   stoichiometry={"PZ": -1, "Z":  1}, args=["k_PZ_b", "PZ"])

    m.add_reaction("QVf",   mass_action_light_dark_1s,   stoichiometry={"PV": -1, "QV":  1}, args=["ppfd", "k_L_QV_f", "k_D_QV_f", "PV"])
    m.add_reaction("QVb",   mass_action_1s,   stoichiometry={"QV": -1, "PV":  1}, args=["k_QV_b", "QV"])
    m.add_reaction("QAf",   mass_action_light_dark_1s,   stoichiometry={"PA": -1, "QA":  1}, args=["ppfd", "k_L_QA_f", "k_D_QA_f", "PA"])
    m.add_reaction("QAb",   mass_action_1s,   stoichiometry={"QA": -1, "PA":  1}, args=["k_QA_b", "QA"])
    m.add_reaction("QZf",   mass_action_light_dark_1s,   stoichiometry={"PZ": -1, "QZ":  1}, args=["ppfd", "k_L_QZ_f", "k_D_QZ_f", "PZ"])
    m.add_reaction("QZb",   mass_action_1s,   stoichiometry={"QZ": -1, "PZ":  1}, args=["k_QA_b", "QZ"])

    m.add_reaction("QLf",   mass_action_light_dark_1s,   stoichiometry={"PL": -1, "QL":  1}, args=["ppfd", "k_L_QL_f", "k_D_QL_f", "PL"])
    m.add_reaction("QLb",   mass_action_1s,   stoichiometry={"QL": -1, "PL":  1}, args=["k_QL_b", "QL"])

    m.add_reaction("damage", mass_action_light_dark_2s, stoichiometry={"PSIId": 1},
                    args=["k_L_damage", "k_D_damage","tau_Fluo", "PSII_active"])

    m.add_reaction("VDE",   v_VDE,   stoichiometry={"VDE": 1}, args=["t", "VDE"])

    return m
