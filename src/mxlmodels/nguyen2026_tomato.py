"""NPQ model for tomato.

Adapted from the original NPQ model for Arabidopsis - without the light conversion fn
Rewritten for better implentation
"""

import numpy as np
from mxlpy import Derived, InitialAssignment, Model

import mxlmodels._names as n


def same(x: float) -> float:
    return x


def divide(x: float, y: float) -> float:
    return x / y


def divide_negative(x: float, y: float) -> float:
    return -x / y


def ATPsyn_stoi(x: float, y: float, z: float) -> float:
    return -x * z / y


def inverse(x: float) -> float:
    return 1 / x


def inverse_negative(x: float) -> float:
    return -1 / x


def four_times_inverse(x: float) -> float:
    return 4 / x


def two_times_inverse(x: float) -> float:
    return 2 / x


def two_times_ratio(x: float, y: float) -> float:
    return 2 * y / x


def four_times_ratio(x: float, y: float) -> float:
    return 4 * y / x


def moiety_3(c1, c2, c3, total):
    c4 = total - c1 - c2 - c3
    return c4


def normalize_concentration(concentration, total):
    return concentration / total


def mass_action_1s(s1, kf):
    return kf * s1


def mass_actions(s1, s2, kf):
    return kf * s1 * s2


def mass_action2_rev(s1, s2, p1, p2, kf, keq):  # reverse reaction
    forward = kf * s1 * s2
    reverse = kf / keq * p1 * p2
    return forward - reverse


par = {
    # Pool sizes
    "PSIItot": 2.5,  # unchanged [mmol/molChl] total concentration of PSII
    "PQtot": 20.0,  # unchanged [mmol/molChl]
    "APtot": 50.0,  # unchanged [mmol/molChl] Bionumbers ~2.55mM (=81mmol/molChl)
    "PsbStot": 1,  # [relative] LHCs that get phosphorylated and protonated
    "Xtot": 1.0,  # unchanged [relative] xanthophylls
    "O2ex": 8.0,  # unchanged external oxygen, kept constant, corresponds to 250 microM, corr. to 20%
    "Pi": 0.01,  # unchanged
    # Rate constants and key parameters
    # Cytb6f
    n.k(
        "b6f"
    ): 0.22,  # unchanged a rough estimate of the transfer from PQ to cyt that is equal to ~ 10ms
    "pKreg": 6.4,  # pKa of pH inihibition of Cytb6f
    # ATPsynthase
    "kActATPase": 0.01,  # unchanged parameter relating the rate constant of activation of the ATPase in the light
    "kDeactATPase": 0.002,  # unchanged parameter relating the deactivation of the ATPase at night
    "kATPsynthase": 20.0,  # unchanged
    "kATPconsumption": 10.0,  # unchanged
    "HPR": 14.0 / 3.0,  # unchanged
    "pKE0": 7.211142552636095,  # fitted value for ATPsynthase pmf regulation
    "b": 3.1924977471697407,  # fitted value for ATPsynthase pmf regulation
    # PQ pool
    "kPQH2": 250.0,  # unchanged [1/(s*(mmol/molChl))]
    "kPTOX": 0.01,  # unchanged
    # PSII
    "kH_Qslope": 5e9,
    "kH0": 5e8,  # Andre assumption
    "kF": 6.25e8,  # unchanged fluorescence 16ns
    "kP": 6939318750.0,  # Fitted
    # Proton
    n.pH_stroma: 7.8,  # unchanged [1/s] leakage rate
    "kleak": 1000.0,  # unchanged
    "bH": 100,  # unchanged proton buffer: ratio total / free protons
    # Parameter associated with xanthophyll cycle
    "kDeepoxV": 0.00096,  # Fitted
    "kEpoxZ": 0.0013824,  # Fitted
    "KphSatZ": 5.8,  # Taken from Zaks model
    # [-] half-saturation pH value for activity de-epoxidase, highest activity at ~pH 5.8
    "KZsat": 0.65,  # [-], half-saturation constant (relative conc. of Z) for quenching of Z
    "nHX": 5.0,  # unchanged, the cooperativity, hill-coefficient for activity of de-epoxidase
    "nHZ": 3.0,
    # Parameter associated with PsbS protonation
    "nHL": 3,
    "kDeprot": 0.0336,  # Fitted
    "kProt": 0.07392,  # Fitted
    "KphSatLHC": 5.8,  # Taken from Zaks model
    # Fitted quencher contribution factors
    "gamma0": 0.1,  # slow quenching of Vx present despite lack of protonation
    "gamma1": 1,  # CHANGED - fast quenching present due to the protonation
    "gamma2": 8,  # slow quenching of Zx present despite lack of protonation
    "gamma3": 2,  # fastest possible quenching
    # KEA3 parameters
    "pK_KEA3": 6.75,  # Fitted
    "k_KEA3": 5,  # modified to adjust with the new stoi
    "K_lumen_conc_initial": 0.1,  # M From Meng Li model
    "K_stroma_conc_initial": 0.1,  # M From Meng Li model
    "ATP_thres_KEA3": 20.5,  # Fitted
    "c": 0.1,
    # Physical constants
    "F": 96.485,  # unchanged Faraday constant
    "R": 8.3e-3,  # unchanged universal gas constant
    "T": 298.0,  # unchanged Temperature in K - for now assumed to be constant at 25 C
    # Standard potentials and DeltaG0_ATP
    "E0QAQAm": -0.140,  # unchanged
    "E0PQPQH2": 0.354,  # unchanged
    "E0PCPCm": 0.380,  # unchanged
    "DeltaG0_ATP": 30.6,  # unchanged [kJ/mol / RT]
    # Other constant
    "e": 2.71828,  # constant
    "lumen_volume_per_area_membrane": 0.0014,
    "stroma_volume_per_area_membrane": 0.0112,
    "molChl_per_area_membrane": 350e-6,
    "thylakoid_membrane_capacitance": 0.6e-2,  # F/m^2
    "pHlumen_init": 7.2,  # Assumed initial pH condition
    # PFD
    n.light: 200.0,
}


def mmol_to_conc(n_mmol, volume_per_area_membrane, Chl_per_area_membrane):

    n_mol = n_mmol / 1000.0
    conc = ((n_mol) / volume_per_area_membrane) * Chl_per_area_membrane
    return conc


def conc_to_mmol(conc, volume_per_area_membrane, Chl_per_area_membrane):

    n_mol = (conc / Chl_per_area_membrane) * volume_per_area_membrane
    n_mmol = n_mol * 1000.0
    return n_mmol


def calculate_pHinv(pH, volume_per_area_membrane, Chl_per_area_membrane):
    """new"""
    H_conc = 10 ** (-pH)
    H_mmol_per_molChl = conc_to_mmol(
        H_conc, volume_per_area_membrane, Chl_per_area_membrane
    )
    return H_mmol_per_molChl


def calculate_pH(H_mmol_per_molChl, volume_per_area_membrane, Chl_per_area_membrane):
    "new"
    H_conc = mmol_to_conc(
        H_mmol_per_molChl, volume_per_area_membrane, Chl_per_area_membrane
    )
    return -np.log10(H_conc)


def moiety(x: float, x_tot: float) -> float:
    return x_tot - x


def propotional(x: float, y: float) -> float:
    return x * y


def _KeqQAPQ(
    F: float, E0QAQAm: float, E0PQPQH2: float, pHstroma: float, R: float, T: float
) -> float:

    DG1 = -F * E0QAQAm
    DG2 = -2 * F * E0PQPQH2 + 2 * pHstroma * np.log(10) * R * T
    DG0 = -2 * DG1 + DG2
    Keq = np.exp(-DG0 / (R * T))
    return Keq


def Keq_cytb6f(pHlumen, F, E0_PQ, E0_PC, pmf, R, T):
    """
    Equilibrium constant of cytb6f
    Adjusted from Matuszynska et al 2019 - calculated from pmf instead of deltapH
    """
    DG1 = -2 * F * E0_PQ
    DG2 = -F * E0_PC
    DG = -(DG1 + 2 * (np.log(10) * R * T) * pHlumen) + 2 * DG2 + 2 * F * pmf
    Keq = np.exp(-DG / (R * T))
    return Keq


def KeqATPsyn(
    pmf: float,
    DeltaG0_ATP: float,
    R: float,
    T: float,
    F: float,
    HPR: float,
    Pi_mol: float,
) -> float:
    """
    Equilibrium constant of ATP synthase. Adjusted for pmf description
    For more information see Matuszynska et al 2016 or Ebenhöh et al. 2011,2014
    """

    DG = DeltaG0_ATP - F * pmf * HPR
    Keq = Pi_mol * np.exp(-DG / (R * T))
    return Keq


def ATP_pmf_act(
    pmf: float,
    pK0E: float,
    b: float,
    e: float,
    F: float,  # kJ per volt–gram-equivalent
    R: float,
    T: float,  # K
) -> float:
    """pmf regulation of ATPsynthase"""

    x = np.log(10 ** (-pK0E)) + b * (pmf * F) / (R * T)
    ATP_pmf_act = e**x / (1 + e**x)
    return ATP_pmf_act


def Fluo(Q, B0, B2, kP, kF, kH_Qslope, kH0):
    kH = kH0 + kH_Qslope * Q
    return (kF * B0) / (kF + kP + kH) + (kF * B2) / (kF + kH)


def kquencher(s, q, kH_Qslope, kH0):
    return (kH0 + kH_Qslope * q) * s


def Quencher(
    PsbS: float,
    Vx: float,
    Xtot: float,
    PsbStot: float,
    Kzsat: float,
    gamma0: float,
    gamma1: float,
    gamma2: float,
    gamma3: float,
) -> float:
    """
    Quencher mechanism - Anna 2016 model

    accepts:
    Pr: fraction of non-protonated PsbS protein
    V: fraction of Violaxanthin
    """
    Z = Xtot - Vx
    P = PsbStot - PsbS
    Zs = Z / (Z + Kzsat)

    Q = (
        gamma0 * (1 - Zs) * PsbS
        + gamma1 * (1 - Zs) * P
        + gamma2 * Zs * P
        + gamma3 * Zs * PsbS
    )
    return Q


def quencher_q0(Psbs, Vx, y0):
    """co-operative 4-state quenching mechanism"""
    # gamma0: slow quenching of (Vx - protonation)
    return y0 * Vx * Psbs


def quencher_q1(Vx, Psbsp, y1):
    """co-operative 4-state quenching mechanism"""
    # gamma1: fast quenching (Vx + protonation)
    return y1 * Vx * Psbsp


def quencher_q2(Psbsp, Zx, y2):
    """co-operative 4-state quenching mechanism"""
    # gamma2: fastest possible quenching (Zx + protonation)
    return y2 * Zx * Psbsp


def quencher_q3(Psbs, Zx, y3):
    """co-operative 4-state quenching mechanism"""
    # gamma3: slow quenching of Zx present (Zx - protonation)
    return y3 * Zx * (Psbs)


def quencher_total(q1, q2, q3, q4):
    return q1 + q2 + q3 + q4


###################################################################

# ========= PMF ============#


def deltapH_to_V(delta_pH: float, R: float, T: float, F: float) -> float:
    return -np.log(10) * ((R * T) / F) * delta_pH


def delta_ph(pH_lumen: float, pH_stoma: float) -> float:
    """
    calculation of pH difference between stroma and thylakoid lumen

    Accepts:

    pH_lumen: thylakoid lumen pH
    pH_stroma: stroma pH

    """

    delta_pH = pH_lumen - pH_stoma

    return delta_pH


def voltage_turnover_molChl_per_mmol(
    capacitance_specific: float, molChl_per_area_membrane: float, F: float
) -> float:
    area_permolChl = 1 / molChl_per_area_membrane
    voltage = F / (capacitance_specific * area_permolChl)
    return voltage


def initial_delta_psi(delta_pH: float, R: float, F: float, T: float) -> float:
    """
    Estimation of delta psi in the dark - assuming delta_pH and delta_psi have equal contribution to pmf
    """
    return -np.log(10) * ((R * T) / F) * delta_pH


def proton_motive_force(
    delta_ph: float, delta_psi: float, F: float, T: float, R: float
) -> float:
    """
    proton motive force formula - taken from Lowe & Jones (1984) https://doi.org/10.1016/0968-0004(84)90038-0

    Accepts:
    delta_ph: pH different between the thylakoid lumen and the stroma
    delta_psi: thylakoid membrane potential
    F: Faraday constant
    R: gas constant
    T: temperature (K)
    """

    pmf = delta_psi - np.log(10) * ((R * T) / F) * delta_ph

    return pmf


def pHmod(pH: float, pKreg: float) -> float:
    return 1 - (1 / (10 ** (pH - pKreg) + 1))


def k_b6f(pHmod: float, k_b6f: float) -> float:
    b6f_deprot = pHmod * k_b6f
    return b6f_deprot


def reg_KEA3(reg_KEA3_ATP: float, reg_KEA3_pH: float) -> float:
    return reg_KEA3_ATP * reg_KEA3_pH


def reg_KEA3_pH(pHlumen: float, pK_KEA3: float) -> float:
    pH_act = 10 ** (pHlumen - pK_KEA3) / (10 ** (pHlumen - pK_KEA3) + 1)
    return pH_act


def reg_KEA3_ATP(ATP: float, ATP_thres: float, c: float) -> float:
    ATP_inhib = (1 - c) / (1 + np.exp((ATP - ATP_thres) / c))
    return ATP_inhib


def vKEA3_in(
    Klumen: float,
    Hlumen: float,
    Kstroma: float,
    k_KEA3: float,
    Hstroma: float,
    reg_KEA3: float,
    stroma_volume_per_area_membrane: float,
    Chl_per_area_membrane: float,
) -> float:
    return float(
        max(
            conc_to_mmol(
                (k_KEA3 * (Hlumen * Kstroma - Hstroma * Klumen) * reg_KEA3),
                stroma_volume_per_area_membrane,
                Chl_per_area_membrane,
            ),
            0,
        )
    )


def vKEA3_out(
    Klumen: float,
    Hlumen: float,
    Kstroma: float,
    k_KEA3: float,
    Hstroma: float,
    reg_KEA3: float,
    lumen_volume_per_area_membrane: float,
    Chl_per_area_membrane: float,
) -> float:

    return float(
        max(
            conc_to_mmol(
                (k_KEA3 * (Hstroma * Klumen - Hlumen * Kstroma) * reg_KEA3),
                lumen_volume_per_area_membrane,
                Chl_per_area_membrane,
            ),
            0,
        )
    )


def vps2(B1: float, kP: float) -> float:
    """Reduction of PQ due to ps2"""
    v = kP * 0.5 * B1
    return v


def vPQox(
    PQH2: float,
    pfd: float,
    kCytb6f: float,
    kPTOX: float,
    O2ex: float,
    PQtot: float,
    Keq: float,
) -> float:
    """Oxidation of the PQ pool through cytochrome and PTOX"""
    kPFD = kCytb6f * (pfd)
    kPTOX = kPTOX * O2ex
    a1 = kPFD * Keq / (Keq + 1)
    a2 = kPFD / (Keq + 1)
    v = (a1 + kPTOX) * PQH2 - a2 * (PQtot - PQH2)
    return v


def vATPactivity(
    ATPactivity: float, light: float, kActATPase: float, kDeactATPase: float
) -> float:
    """Activation of ATPsynthase by light"""
    switch = light > 0.0
    v = (
        kActATPase * switch * (1 - ATPactivity)
        - kDeactATPase * (1 - switch) * ATPactivity
    )
    return v


def vATPsynthase(
    ATP: float,
    ADP: float,
    KeqATPsyn: float,
    ATPactivity: float,
    ATP_pmf_act: float,
    kATPsynthase: float,  # mmol per mol Chl
) -> float:
    """Production of ATP by ATPsynthase - pmf regulation implemented"""

    v = ATPactivity * ATP_pmf_act * kATPsynthase * (ADP - ATP / KeqATPsyn)
    return v


def vATPcons(ATP: float, kATPconsumption: float) -> float:
    """ATP consuming reaction"""
    v = kATPconsumption * ATP
    return v


def vLeak(H_lumen_conc: float, kleak: float, H_stroma_conc: float):
    """Transmembrane proton leak"""
    v = kleak * (H_lumen_conc - H_stroma_conc)
    return v


def vXdeepox(
    Vx: float,
    H: float,
    nHX: float,
    KphSatZ: float,
    kDeepoxV: float,
    volume_per_area_membrane: float,
    Chl_per_area_membrane: float,
) -> float:
    """Deepoxidation of Vx"""
    a = H**nHX / (
        H**nHX
        + calculate_pHinv(KphSatZ, volume_per_area_membrane, Chl_per_area_membrane)
        ** nHX
    )
    v = kDeepoxV * a
    vv = v * Vx
    return vv


def vEpoxZ(
    Zx: float,
    kEpoxZ: float,
) -> float:
    """Deepoxidation of Vx"""
    return kEpoxZ * Zx


def vPsbSP(
    PsbS: float,
    H: float,
    nHL: float,
    KphSatLHC: float,
    kProt: float,
    volume_per_area_membrane: float,
    Chl_per_area_membrane: float,
) -> float:
    """Protonation of PsbS protein - Modified for Zx inhibition effect
    Zx is assmuned to inhibit the deprotonation of PsbS
    """
    a = H**nHL / (
        H**nHL
        + calculate_pHinv(KphSatLHC, volume_per_area_membrane, Chl_per_area_membrane)
        ** nHL
    )
    return kProt * a * PsbS


def deprot_act(KZsat: float, nHZ: float, Zx: float) -> float:
    """
    Inhibition effect of Zx on PsbS deprotonation.
    """
    return KZsat**nHZ / (KZsat**nHZ + Zx**nHZ)


def vPsbS(
    PsbSP: float,
    kDeprot: float,
    psbs_deprot_act: float,
) -> float:
    """Deprotonation of PsbS protein - Modified for Zx inhibition effect
    Zx is assmuned to inhibit the deprotonation of PsbS
    """
    return kDeprot * psbs_deprot_act * PsbSP


def create_model() -> Model:
    m = Model()

    m.add_variables(
        {
            n.b0(): 2.5,
            n.b1(): 0,
            n.b2(): 0,
            "PQH2": 0.0,
            "ATP": 25.0,
            n.h_lumen: InitialAssignment(
                fn=calculate_pHinv,
                args=[
                    "pHlumen_init",
                    "lumen_volume_per_area_membrane",
                    "molChl_per_area_membrane",
                ],
            ),
            n.delta_psi: InitialAssignment(
                fn=initial_delta_psi, args=[n.delta_pH, "R", "T", "F"]
            ),
            "Vx": 1.0,
            "PsbS": 1.0,
            "ATPactivity": 0.1,
            n.k_lumen: InitialAssignment(
                fn=conc_to_mmol,
                args=[
                    "K_lumen_conc_initial",
                    "lumen_volume_per_area_membrane",
                    "molChl_per_area_membrane",
                ],
            ),
            n.k_stroma: InitialAssignment(
                fn=conc_to_mmol,
                args=[
                    "K_stroma_conc_initial",
                    "stroma_volume_per_area_membrane",
                    "molChl_per_area_membrane",
                ],
            ),
        }
    )

    m.add_parameters(par)

    m.add_derived("RT", propotional, args=["R", "T"])

    m.add_derived(
        n.pH_lumen,
        calculate_pH,
        args=[n.h_lumen, "lumen_volume_per_area_membrane", "molChl_per_area_membrane"],
    )

    m.add_derived(
        "H_lumen_conc",
        mmol_to_conc,
        args=[n.h_lumen, "lumen_volume_per_area_membrane", "molChl_per_area_membrane"],
    )

    m.add_derived(
        n.h_stroma,
        calculate_pHinv,
        args=[
            n.pH_stroma,
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        "H_stroma_conc",
        mmol_to_conc,
        args=[
            n.h_stroma,
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(n.delta_pH, delta_ph, args=[n.pH_lumen, n.pH_stroma])
    m.add_derived("delta_pH_V", deltapH_to_V, args=[n.delta_pH, "R", "T", "F"])

    m.add_derived(
        n.pmf, proton_motive_force, args=[n.delta_pH, n.delta_psi, "F", "T", "R"]
    )

    m.add_derived(
        "volts_per_charge",
        voltage_turnover_molChl_per_mmol,
        args=["thylakoid_membrane_capacitance", "molChl_per_area_membrane", "F"],
    )

    m.add_derived(n.pq, moiety, args=["PQH2", "PQtot"])

    m.add_derived(n.adp, moiety, args=["ATP", "APtot"])

    m.add_derived("PsbSP", moiety, args=["PsbS", "PsbStot"])

    m.add_derived(n.zx, moiety, args=["Vx", "Xtot"])

    m.add_derived(
        "Keq_PQH2", _KeqQAPQ, args=["F", "E0QAQAm", "E0PQPQH2", n.pH_stroma, "R", "T"]
    )

    m.add_derived(
        "Keqcytb6f",
        Keq_cytb6f,
        args=[n.pH_lumen, "F", "E0PQPQH2", "E0PCPCm", n.pmf, "R", "T"],
    )

    m.add_derived(
        "KeqATPsyn", KeqATPsyn, args=[n.pmf, "DeltaG0_ATP", "R", "T", "F", "HPR", "Pi"]
    )

    m.add_derived(
        "ATP_pmf_act",
        ATP_pmf_act,
        args=[n.pmf, "pKE0", "b", "e", "F", "R", "T"],
    )

    m.add_derived(
        "pHmod",
        pHmod,
        args=[n.pH_lumen, "pKreg"],
    )

    m.add_derived(
        "k_cytb6f",
        k_b6f,
        args=["pHmod", n.k("b6f")],
    )

    m.add_derived("Q0", quencher_q0, args=[n.psbs, n.vx, "gamma0"])

    m.add_derived("Q1", quencher_q1, args=[n.vx, n.psbsp, "gamma1"])

    m.add_derived("Q2", quencher_q2, args=[n.psbsp, n.zx, "gamma2"])

    m.add_derived("Q3", quencher_q3, args=[n.psbs, n.zx, "gamma3"])

    m.add_derived(n.quencher, quencher_total, args=["Q0", "Q1", "Q2", "Q3"])

    m.add_derived(name=n.b3(), fn=moiety_3, args=[n.b0(), n.b1(), n.b2(), "PSIItot"])

    m.add_derived(name="rel_B0", fn=normalize_concentration, args=[n.b0(), "PSIItot"])
    m.add_derived(name="rel_B1", fn=normalize_concentration, args=[n.b1(), "PSIItot"])
    m.add_derived(name="rel_B2", fn=normalize_concentration, args=[n.b2(), "PSIItot"])
    m.add_derived(name="rel_B3", fn=normalize_concentration, args=[n.b3(), "PSIItot"])

    m.add_derived(
        name=n.fluo,
        fn=Fluo,
        args=[n.quencher, n.b0(), n.b2(), "kP", "kF", "kH_Qslope", "kH0"],
    )

    m.add_reaction(
        name="B01",
        fn=mass_action_1s,
        stoichiometry={n.b0(): -2, n.b1(): 2},
        args=[n.b0(), n.light],
    )

    m.add_reaction(
        name="B10Q",
        fn=kquencher,
        stoichiometry={n.b1(): -2, n.b0(): 2},
        args=[n.b1(), n.quencher, "kH_Qslope", "kH0"],
    )

    m.add_reaction(
        name="B10F",
        fn=mass_action_1s,
        stoichiometry={n.b1(): -2, n.b0(): 2},
        args=[n.b1(), "kF"],
    )

    m.add_reaction(
        name="vps2",
        fn=vps2,
        stoichiometry={
            n.b1(): -2,
            n.b2(): 2,
            n.h_lumen: Derived(fn=two_times_inverse, args=["bH"]),
            n.delta_psi: Derived(fn=two_times_ratio, args=["bH", "volts_per_charge"]),
        },
        args=[n.b1(), "kP"],
    )

    m.add_reaction(
        name="B20",
        fn=mass_action2_rev,
        stoichiometry={n.b2(): -2, n.pqh2: 1, n.b0(): 2},
        args=[n.b2(), n.pq, n.pqh2, n.b0(), "kPQH2", "Keq_PQH2"],
    )

    m.add_reaction(
        name="B23",
        fn=mass_action_1s,
        stoichiometry={n.b2(): -2},
        args=[n.b2(), n.light],
    )
    m.add_reaction(
        name="B32F", fn=mass_action_1s, stoichiometry={n.b2(): 2}, args=[n.b3(), "kF"]
    )

    m.add_reaction(
        name="B32Q",
        fn=kquencher,
        stoichiometry={n.b2(): 2},
        args=[n.b3(), n.quencher, "kH_Qslope", "kH0"],
    )

    m.add_reaction(
        "vPQox",
        vPQox,
        args=["PQH2", n.light, "k_cytb6f", "kPTOX", "O2ex", "PQtot", "Keqcytb6f"],
        stoichiometry={
            "PQH2": -1,
            n.h_lumen: Derived(fn=four_times_inverse, args=["bH"]),
            n.delta_psi: Derived(fn=four_times_ratio, args=["bH", "volts_per_charge"]),
        },
    )

    m.add_reaction(
        "vATPactivity",
        vATPactivity,
        args=["ATPactivity", n.light, "kActATPase", "kDeactATPase"],
        stoichiometry={"ATPactivity": 1},
    )

    m.add_reaction(
        name="vATPsynthase",
        fn=vATPsynthase,
        args=[
            n.atp,
            n.adp,
            "KeqATPsyn",
            n.atpact,
            "ATP_pmf_act",
            "kATPsynthase",
        ],
        stoichiometry={
            n.h_lumen: Derived(fn=divide_negative, args=["HPR", "bH"]),
            "ATP": 1,
            n.delta_psi: Derived(
                fn=ATPsyn_stoi, args=["HPR", "bH", "volts_per_charge"]
            ),
        },
    )

    m.add_reaction(
        "vATPcons", vATPcons, stoichiometry={"ATP": -1}, args=["ATP", "kATPconsumption"]
    )

    m.add_reaction(
        name="vleak",
        fn=vLeak,
        args=["H_lumen_conc", "kleak", "H_stroma_conc"],
        stoichiometry={
            n.h_lumen: Derived(fn=inverse_negative, args=["bH"]),
            n.delta_psi: Derived(fn=divide_negative, args=["volts_per_charge", "bH"]),
        },
    )

    m.add_reaction(
        "vXdeepox",
        vXdeepox,
        args=[
            "Vx",
            n.h_lumen,
            "nHX",
            "KphSatZ",
            "kDeepoxV",
            "lumen_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
        stoichiometry={"Vx": -1},
    )

    m.add_reaction(
        "vEpoxZ",
        vEpoxZ,
        args=[
            "Zx",
            "kEpoxZ",
        ],
        stoichiometry={"Vx": 1},
    )

    m.add_derived("PsbS_deprot_act", deprot_act, args=["KZsat", "nHZ", n.zx])

    m.add_reaction(
        name="vPsbSP",
        fn=vPsbSP,
        args=[
            n.psbs,
            n.h_lumen,
            "nHL",
            "KphSatLHC",
            "kProt",
            "lumen_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
        stoichiometry={
            n.psbs: -1,
        },
    )

    m.add_reaction(
        name="vPsbS",
        fn=vPsbS,
        args=[n.psbsp, "kDeprot", "PsbS_deprot_act"],
        stoichiometry={
            n.psbs: 1,
        },
    )

    m.add_derived(
        "K_stroma_conc",
        mmol_to_conc,
        args=[
            n.k_stroma,
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        "K_lumen_conc",
        mmol_to_conc,
        args=[n.k_lumen, "lumen_volume_per_area_membrane", "molChl_per_area_membrane"],
    )

    m.add_derived("reg_KEA3_ATP", reg_KEA3_ATP, args=[n.atp, "ATP_thres_KEA3", "c"])

    m.add_derived("reg_KEA3_pH", reg_KEA3_pH, args=[n.pH_lumen, "pK_KEA3"])

    m.add_derived("reg_KEA3", reg_KEA3, args=["reg_KEA3_ATP", "reg_KEA3_pH"])

    m.add_reaction(
        name="vKEA3_in",
        fn=vKEA3_in,
        args=[
            "K_lumen_conc",
            "H_lumen_conc",
            "K_stroma_conc",
            "k_KEA3",
            "H_stroma_conc",
            "reg_KEA3",
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
        stoichiometry={
            n.h_lumen: Derived(fn=inverse_negative, args=["bH"]),
            n.k_lumen: 1,
            n.k_stroma: -1,
        },
    )

    m.add_reaction(
        name="vKEA3_out",
        fn=vKEA3_out,
        args=[
            "K_lumen_conc",
            "H_lumen_conc",
            "K_stroma_conc",
            "k_KEA3",
            "H_stroma_conc",
            "reg_KEA3",
            "lumen_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
        stoichiometry={
            n.h_lumen: Derived(fn=inverse, args=["bH"]),
            n.k_lumen: -1,
            n.k_stroma: 1,
        },
    )

    return m
