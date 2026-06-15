from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from mxlpy import Model, InitialAssignment


############INITIAL CONDITIONS##############

def P_free(Ptot, Xtot, V, A, Z):
    return Ptot - Xtot + V + A + Z

def X_tot(V, A, Z ,PV, PA, PZ, QV, QA, QZ):
    return V + A + Z + PV + PA + PZ + QV + QA + QZ

def V_free(Xtot, A, Z ,PV, PA, PZ, QV, QA, QZ):
    return Xtot - (A + Z + PV + PA + PZ + QV + QA + QZ)

def Keq(kf, kr):
    return kf/kr

def Keq_light_dark(kf_light, kr_light, kf_dark, kr_dark, ppfd):
    if ppfd == 0:
        return kf_dark/kr_dark
    else:
        return kf_light/kr_light
    
def P0(Ptot, V0, Kpv, Kpa, Ka, Kpz, Kz, Kqv, Kqa, Kqz):
    return Ptot / (1 + V0*(Kpv + Kpa*Ka + Kpz*Kz*Ka + Kqv*Kpv + Kqa*Kpa*Ka + Kqz*Kpz*Kz*Ka))

def A0(Keq, V0):
    return Keq*V0

def Z0(Ka, Kz, V0):
    return Ka*Kz*V0

def PA0(Kpa, Ka, V0, P0,):
    return Kpa*Ka*V0*P0

def PZ0(Kpz, Kz, Ka, P0, V0):
    return Kpz*Kz*Ka*V0*P0

def PV0(Kpv, V0, P0,):
    return Kpv*V0*P0

def QA0(P0, V0, Kpa, Ka, Kqa):
    return  Kpa * Ka * Kqa * P0 * V0

def QV0(P0, V0, Kpv, Kqv):
    return  Kpv * Kqv * P0 * V0

def QZ0(Kqz, Kpz, Kz, Ka, P0, V0):
    return  Kqz * Kpz * Kz * Ka * P0 * V0

def E0(kva_d, kva_l):
    return kva_d/kva_l

############DERIVED##############

def moiety(Xtot, X):
    return Xtot - X

# def chlorophyll_fluo_lifetime(kappa_QV, QV,
#                               kappa_QA, QA,
#                               kappa_QZ, QZ,
#                               kappa_QL, QL,
#                               kappa_qZ, Z,
#                               k_qI, PSIId,
#                               kappa_r_nr):
#     """
#     Compute fluorescence lifetime from concentration time series.
#     """
#     tau  = 1.0 / (kappa_r_nr + kappa_QV * QV  + kappa_QA * QA + kappa_QZ * QZ + kappa_QL * QL + kappa_qZ * Z + k_qI * PSIId )
#     return tau

def kappa_r_nr(tau_0, kappa_qZ, Z_0):
     return 1/tau_0 - kappa_qZ*Z_0

############RATES##############

def same(x):
    return x

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

def damage(ppfd, k_light, k_dark, tau, PSIId):
    tau = max(tau,0)
    if ppfd ==0:
        return max(k_dark*tau*(1-PSIId),0)
    return max(k_light*tau*(1-PSIId),0)
# --- alpha_VDE enzyme activation ---

def v_alpha_VDE(ppfd, k_L_VDE, k_D_VDE, k_L_VA, k_D_VA, alpha_VDE):
    if ppfd == 0:
        k_VDE = k_D_VDE
        alpha_VDE_eq = k_D_VA/k_L_VA
    else:
        k_VDE = k_L_VDE
        alpha_VDE_eq = 1
    return k_VDE * (alpha_VDE_eq - alpha_VDE)

def chlorophyll_fluo_lifetime(kappa_QV, QV, kappa_QA, QA,
                               kappa_QZ, QZ, kappa_QL, QL,
                               kappa_qZ, Z, k_qI, PSIId,
                               kappa_r_nr):
    denom = (kappa_r_nr
             + kappa_QV * max(QV, 0)
             + kappa_QA * max(QA, 0)
             + kappa_QZ * max(QZ, 0)
             + kappa_QL * max(QL, 0)
             + kappa_qZ * max(Z,  0)
             + k_qI     * max(PSIId, 0))
    return 1.0 / max(denom, 1e-10)
############MODELS##############

def get_lam2026() -> Model:
    m = Model()
    m.add_parameters({
        "k_L_VA": 2.47,
        "k_D_VA": 0.014,
        "k_L_AZ": 0.5,
        "k_AV": 1.12,
        "k_ZA": 0.07,
        "k_PV_f": 2.18,
        "k_PV_b": 9.43,
        "k_PA_f": 130,
        "k_PA_b": 254,
        "k_PZ_f": 295,
        "k_PZ_b": 126,
        "k_L_QV_f": 0.027,
        "k_D_QV_f": 0,
        "k_QV_b": 0.066,
        "k_L_QA_f": 0.66,
        "k_D_QA_f": 0,
        "k_QA_b": 8.57,
        "k_L_QZ_f": 0.56,
        "k_D_QZ_f": 0,
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
        "kappa_qI": 3.86,
        "kappa_qI_double_mut": 7.05,
        "ppfd":0,
        "tau_0": 1.73089079100000, # from the paper WT tau_0 for 5-10-5 dataset 
        "PSII_tot": 1,
        "X_tot": 77.023655, #need proper implementation later
    })

    m.add_derived("gamma", E0, args=["k_D_VA", "k_L_VA"])
    m.add_derived("k_D_AZ", mass_action_1s, args=["gamma", "k_L_AZ"])

    m.add_derived("Keq_pz", Keq, args=["k_PZ_f", "k_PZ_b"])
    m.add_derived("Keq_pv", Keq, args=["k_PV_f", "k_PV_b"])
    m.add_derived("Keq_pa", Keq, args=["k_PA_f", "k_PA_b"])

    m.add_derived("Keq_a", Keq_light_dark, args=["k_D_VA", "k_AV", "k_D_VA", "k_AV", "ppfd"])
    m.add_derived("Keq_z", Keq_light_dark, args=["k_D_AZ", "k_ZA", "k_D_AZ", "k_ZA", "ppfd"])
    m.add_derived("Keq_qv", Keq_light_dark, args=["k_L_QV_f", "k_QV_b", "k_D_QV_f", "k_QV_b", "ppfd"])
    m.add_derived("Keq_qa", Keq_light_dark, args=["k_L_QA_f", "k_QA_b", "k_D_QA_f", "k_QA_b", "ppfd"])
    m.add_derived("Keq_qz", Keq_light_dark, args=["k_L_QZ_f", "k_QZ_b", "k_D_QZ_f", "k_QZ_b", "ppfd"])

    m.add_derived("P0", P0, 
                  args=["P_tot", "V_tot_WT", "Keq_pv", "Keq_pa", "Keq_a", "Keq_pz", "Keq_z", "Keq_qv", "Keq_qa", "Keq_qz"]
                  )

    # Variables and initial conditions
    m.add_variables(
               {      "V": InitialAssignment(fn=same, args=["V_tot_WT"]),
                      "A": InitialAssignment(fn=A0, args=["Keq_a", "V_tot_WT"]),
                      "Z": InitialAssignment(fn=Z0, args=["Keq_a", "Keq_z", "V_tot_WT"]),
                      "PV": InitialAssignment(fn=PV0, args=["Keq_pv", "V_tot_WT", "P0"]),
                      "PA": InitialAssignment(fn=PA0, args=["Keq_pa", "Keq_a", "V_tot_WT", "P0"]),
                      "PZ": InitialAssignment(fn=PZ0, args=["Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"]),
                      "QV": InitialAssignment(fn=QV0, args=[ "P0","V_tot_WT", "Keq_pv", "Keq_qv"]),
                      "QA": InitialAssignment(fn=QA0, args=["P0","V_tot_WT", "Keq_pa", "Keq_a", "Keq_qa"]),
                      "QZ": InitialAssignment(fn=QZ0, args=["Keq_qz", "Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"]),
                      "QL": 0,
                      "PL": 165, # set from HPLC data
                      "PSIId": 0,
                      "alpha_VDE": InitialAssignment(fn=E0, args=["k_D_VA", "k_L_VA"]),
               }
        )
    
    
    m.add_derived("PSII_active", moiety, args=["PSII_tot", "PSIId"])
    # m.add_derived("X_tot", X_tot, args=["V_tot_WT", "A", "Z","PV", "PA", "PZ", "QV","QA", "QZ"])
    # m.add_derived("V", V_free, args=["X_tot", "A", "Z","PV", "PA", "PZ", "QV","QA", "QZ"])
    m.add_derived("P_free", P_free, args=["P_tot", "X_tot", "V", "A", "Z",])
    m.add_derived("Z_0", Z0, args=["Keq_a", "Keq_z", "V_tot_WT"])
    m.add_derived("kappa_r_nr", kappa_r_nr, args= ["tau_0", "kappa_qZ", "Z_0"])
    
    m.add_derived("tau_Fluo", chlorophyll_fluo_lifetime, 
                  args= ["kappa_QV", "QV",
                         "kappa_QA", "QA",
                         "kappa_QZ", "QZ",
                         "kappa_QL", "QL",
                         "kappa_qZ", "Z",
                         "kappa_qI", "PSIId",
                         "kappa_r_nr",
                         ]
                  )
    
    m.add_reaction("VA",    mass_action_light_dark_2s,    stoichiometry={"V": -1,"A":  1}, args=["ppfd","k_L_VA", "k_L_VA", "alpha_VDE", "V"])
    m.add_reaction("AV",    mass_action_1s,    stoichiometry={"A": -1, "V": 1}, args=["k_AV","A"])
    m.add_reaction("AZ",    mass_action_light_dark_2s,    stoichiometry={"A": -1, "Z":  1}, args=["ppfd", "k_L_AZ", "k_L_AZ","alpha_VDE", "A"])
    m.add_reaction("ZA",    mass_action_1s,    stoichiometry={"Z": -1, "A":  1}, args=["k_ZA","Z"])

    m.add_reaction("PVf",   mass_action_2s,   stoichiometry={ "PV":  1, "V": -1}, args=["k_PV_f", "V", "P_free"])
    m.add_reaction("PVb",   mass_action_1s,   stoichiometry={"PV": -1, "V": 1}, args=["k_PV_b", "PV"])
    m.add_reaction("PAf",   mass_action_2s,   stoichiometry={"A": -1, "PA":  1}, args=["k_PA_f", "A", "P_free"])
    m.add_reaction("PAb",   mass_action_1s,   stoichiometry={"PA": -1, "A":  1}, args=["k_PA_b", "PA"])
    m.add_reaction("PZf",   mass_action_2s,   stoichiometry={"Z": -1, "PZ":  1}, args=["k_PZ_f", "Z", "P_free"])
    m.add_reaction("PZb",   mass_action_1s,   stoichiometry={"PZ": -1, "Z":  1}, args=["k_PZ_b", "PZ"])

    m.add_reaction("QVf",   mass_action_light_dark_1s,   stoichiometry={"PV": -1, "QV":  1}, args=["ppfd", "k_L_QV_f", "k_D_QX_f", "PV"])
    m.add_reaction("QVb",   mass_action_1s,   stoichiometry={"QV": -1, "PV":  1}, args=["k_QV_b", "QV"])
    m.add_reaction("QAf",   mass_action_light_dark_1s,   stoichiometry={"PA": -1, "QA":  1}, args=["ppfd", "k_L_QA_f", "k_D_QX_f", "PA"])
    m.add_reaction("QAb",   mass_action_1s,   stoichiometry={"QA": -1, "PA":  1}, args=["k_QA_b", "QA"])
    m.add_reaction("QZf",   mass_action_light_dark_1s,   stoichiometry={"PZ": -1, "QZ":  1}, args=["ppfd", "k_L_QZ_f", "k_D_QX_f", "PZ"])
    m.add_reaction("QZb",   mass_action_1s,   stoichiometry={"QZ": -1, "PZ":  1}, args=["k_QA_b", "QZ"])

    m.add_reaction("QLf",   mass_action_light_dark_1s,   stoichiometry={"PL": -1, "QL":  1}, args=["ppfd", "k_L_QL_f", "k_D_QX_f", "PL"])
    m.add_reaction("QLb",   mass_action_1s,   stoichiometry={"QL": -1, "PL":  1}, args=["k_QL_b", "QL"])

    m.add_reaction("damage", damage, stoichiometry={"PSIId": 1},
                    args=["ppfd","k_L_damage", "k_D_damage","tau_Fluo", "PSIId"])

    m.add_reaction("v_alpha_VDE",   v_alpha_VDE,   stoichiometry={"alpha_VDE": 1}, args=["ppfd","k_L_VDE", "k_D_VDE", "k_L_VA", "k_D_VA" , "alpha_VDE"])

    return m
