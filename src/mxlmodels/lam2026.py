from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from mxlpy import Model


############SUPPORT FUNCTION##############

def mass_action_2s(k, s1, s2):
      return k*s1*s2

def mass_action_1s(k, s):
      return k*s

def mass_action_light_dark(ppfd, k_light, k_dark, s):
    if ppfd ==0:
        return k_dark*s
    else:
        return k_light*s

def make_light_function(DeltaT):
    """
    Return I(t) -> int {0, 1}  (0 = dark, 1 = light).

    Parameters
    ----------
    DeltaT : float or array-like
        If scalar  : half-period of a symmetric L/D square wave (period = 2*DeltaT).
        If sequence: cumulative switch times, e.g. [0, 5, 5] for L5-D5.
    """
    DeltaT = np.asarray(DeltaT, dtype=float)
    if DeltaT.ndim == 0 or DeltaT.size == 1:
        period = float(DeltaT.flat[0])
        def pfd(t):
            # light on in first half of each period (matches MATLAB sign convention)
            phase = t % period
            return 1 if phase < 0.5 * period else 0
    else:
        cum = np.cumsum(DeltaT)
        def pfd(t):
            n_passed = int(np.sum(t >= cum))
            return n_passed % 2   # even = light (1st segment), odd = dark
    return pfd

def chlorophyll_fluo_lifetime(AllConc, kappa, k_qI, delta):
    """
    Compute fluorescence lifetime from concentration time series.
    """
    QV   = AllConc[:, 6]
    QA   = AllConc[:, 7]
    QZ   = AllConc[:, 8]
    QL   = AllConc[:, 10]
    Z    = AllConc[:, 2]
    PSIId= AllConc[:, 11]
    tau  = 1.0 / (delta
                  + kappa["QV"] * QV
                  + kappa["QA"] * QA
                  + kappa["QZ"] * QZ
                  + kappa["QL"] * QL
                  + kappa["qZ"] * Z
                  + k_qI * PSIId)
    return tau



############DERIVED##############
def P_free(Ptot, Xtot, V, A, Z):
        return Ptot - Xtot + V + A + Z




############RATES##############

def v_VA(PPFD, k_L_VA, k_D_VA, V,  E):
        """V -> A  (forward, light-dependent VDE activation)"""
        if PPFD == 0:
            return k_D_VA * E * V
        else:
            return k_L_VA * E * V  

def v_AV(k_AV, A):
    """A -> V  (reverse, ZEP)"""
    return k_AV * A

def v_AZ(k_AZ, A, E):
    """A -> Z  (light-only deepoxidation)"""
    return k_AZ * A * E

def v_ZA(k_ZA, Z):
    """Z -> A  (reverse)"""
    return k_ZA * Z

    # --- pigment–protein binding reactions ---

def v_PVf(k_PV_f, V, P):
        """V + P -> PV"""
        return k_PV_f * V * P

def v_PVb(k_PV_b, PV):
        """PV -> V + P"""
        return k_PV_b * PV

def v_PAf(A, P):
        """A + P -> PA"""
        return "k_PA_f" * A * P

def v_PAb(PA):
        """PA -> A + P"""
        return k["pab"] * PA

def v_PZf(Z, V, A):
        """Z + P -> PZ"""
        return k["pzf"] * Z * P

def v_PZb(PZ):
        """PZ -> Z + P"""
        return k["pzb"] * PZ

    # --- quenching-complex formation ---

def v_QVf(t, PV):
        li = I(t)
        return k["qvf"][li] * PV

def v_QVb(QV):
        return k["qvb"] * QV

def v_QAf(t, PA):
        li = I(t)
        return k["qaf"][li] * PA

def v_QAb(QA):
        return k["qab"] * QA

def v_QZf(t, PZ):
        li = I(t)
        return k["qzf"][li] * PZ

def v_QZb(QZ):
        return k["qzb"] * QZ

    # --- Lhcsr (PL / QL) ---

def v_QLf(t, PL):
        li = I(t)
        return k["qlf"][li] * PL

def v_QLb(QL):
        return k["qlb"] * QL

    # --- PSII damage ---
    # dPSIId/dt = k_damage(I) * tau * (1 - PSIId)
    # tau is a derived variable; we implement this as a derived-variable
    # reaction where the rate depends on the current lifetime.

def v_damage(t, V, A, Z, QV, QA, QZ, QL, PSIId, delta):
        li = I(t)
        tau = 1.0 / (delta
                     + kappa["QV"] * QV
                     + kappa["QA"] * QA
                     + kappa["QZ"] * QZ
                     + kappa["QL"] * QL
                     + kappa["qZ"] * Z
                     + k_qI * PSIId)
        return k["damage"][li] * tau * (1.0 - PSIId)

    # --- VDE enzyme activation ---

def v_VDE(t, E):
        li = I(t)
        E_eq = [gamma, 1.0][li]
        return k["vde"][li] * (E_eq - E)

############MODELS##############

def get_lam2026() -> Model:
    m = Model()
    m.add_parameters({
        "delta": 0.0,   # placeholder; set from data before simulating
        "k_L_VA": 2.47,
        "k_D_VA": 0.0014,
        "k_L_AZ": 0.5,
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
                      "E": 0,
               }
        )

    m.add_reaction("VA",    mass_action_2s,    stoichiometry={"V": -1, "A":  1}, args=["k_VA", "E", "V"])
    m.add_reaction("AV",    mass_action_1s,    stoichiometry={"A": -1, "V":  1}, args=["k_AV","A"])
    m.add_reaction("AZ",    mass_action_1s,    stoichiometry={"A": -1, "Z":  1}, args=["k_AZ", "E", "A"])
    m.add_reaction("ZA",    mass_action_1s,    stoichiometry={"Z": -1, "A":  1}, args=["k_ZA","Z"])

    m.add_reaction("PVf",   v_PVf,   stoichiometry={"V": -1, "PV":  1}, args=["V", "A", "Z"])
    m.add_reaction("PVb",   v_PVb,   stoichiometry={"PV": -1, "V":  1}, args=["PV"])
    m.add_reaction("PAf",   v_PAf,   stoichiometry={"A": -1, "PA":  1}, args=["A", "V", "Z"])
    m.add_reaction("PAb",   v_PAb,   stoichiometry={"PA": -1, "A":  1}, args=["PA"])
    m.add_reaction("PZf",   v_PZf,   stoichiometry={"Z": -1, "PZ":  1}, args=["Z", "V", "A"])
    m.add_reaction("PZb",   v_PZb,   stoichiometry={"PZ": -1, "Z":  1}, args=["PZ"])

    m.add_reaction("QVf",   v_QVf,   stoichiometry={"PV": -1, "QV":  1}, args=["t", "PV"])
    m.add_reaction("QVb",   v_QVb,   stoichiometry={"QV": -1, "PV":  1}, args=["QV"])
    m.add_reaction("QAf",   v_QAf,   stoichiometry={"PA": -1, "QA":  1}, args=["t", "PA"])
    m.add_reaction("QAb",   v_QAb,   stoichiometry={"QA": -1, "PA":  1}, args=["QA"])
    m.add_reaction("QZf",   v_QZf,   stoichiometry={"PZ": -1, "QZ":  1}, args=["t", "PZ"])
    m.add_reaction("QZb",   v_QZb,   stoichiometry={"QZ": -1, "PZ":  1}, args=["QZ"])

    m.add_reaction("QLf",   v_QLf,   stoichiometry={"PL": -1, "QL":  1}, args=["t", "PL"])
    m.add_reaction("QLb",   v_QLb,   stoichiometry={"QL": -1, "PL":  1}, args=["QL"])

    m.add_reaction("damage", v_damage, stoichiometry={"PSIId": 1},
                    args=["t", "V", "A", "Z", "QV", "QA", "QZ", "QL", "PSIId", "delta"])

    m.add_reaction("VDE",   v_VDE,   stoichiometry={"E": 1}, args=["t", "E"])

    return m