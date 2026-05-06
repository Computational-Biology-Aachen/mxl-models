"""NPQ model for tomato.

Adapted from the original NPQ model for Arabidopsis - without the light conversion fn
Rewritten for better implentation
"""

import numpy as np
from mxlpy import Derived, InitialAssignment, Model

import mxlmodels._names as n


def _divide_negative(
    x: float,
    y: float,
) -> float:
    return -x / y


def _at_psyn_stoi(
    x: float,
    y: float,
    z: float,
) -> float:
    return -x * z / y


def _inverse(
    x: float,
) -> float:
    return 1 / x


def _inverse_negative(
    x: float,
) -> float:
    return -1 / x


def _four_times_inverse(
    x: float,
) -> float:
    return 4 / x


def _two_times_inverse(
    x: float,
) -> float:
    return 2 / x


def _two_times_ratio(
    x: float,
    y: float,
) -> float:
    return 2 * y / x


def _four_times_ratio(
    x: float,
    y: float,
) -> float:
    return 4 * y / x


def _moiety_3(
    c1: float,
    c2: float,
    c3: float,
    total: float,
) -> float:
    return total - c1 - c2 - c3


def _normalize_concentration(
    concentration: float,
    total: float,
) -> float:
    return concentration / total


def _mass_action_1s(
    s1: float,
    kf: float,
) -> float:
    return kf * s1


def _mass_action2_rev(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    kf: float,
    keq: float,
) -> float:  # reverse reaction
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
    n.ph_stroma: 7.8,  # unchanged [1/s] leakage rate
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


def _mmol_to_conc(
    n_mmol: float,
    volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:

    n_mol = n_mmol / 1000.0
    return ((n_mol) / volume_per_area_membrane) * chl_per_area_membrane


def _conc_to_mmol(
    conc: float,
    volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:

    n_mol = (conc / chl_per_area_membrane) * volume_per_area_membrane
    return n_mol * 1000.0


def _calculate_p_hinv(
    p_h: float,
    volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:
    """New"""
    H_conc = 10 ** (-p_h)
    return _conc_to_mmol(
        H_conc,
        volume_per_area_membrane,
        chl_per_area_membrane,
    )


def _calculate_p_h(
    h_mmol_per_mol_chl: float,
    volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:
    """New"""
    H_conc = _mmol_to_conc(
        h_mmol_per_mol_chl,
        volume_per_area_membrane,
        chl_per_area_membrane,
    )
    return -np.log10(H_conc)


def _moiety(
    x: float,
    x_tot: float,
) -> float:
    return x_tot - x


def _propotional(
    x: float,
    y: float,
) -> float:
    return x * y


def _keq_qapq(
    f: float,
    e0_qaq_am: float,
    e0_pqpqh2: float,
    p_hstroma: float,
    r: float,
    t: float,
) -> float:

    DG1 = -f * e0_qaq_am
    DG2 = -2 * f * e0_pqpqh2 + 2 * p_hstroma * np.log(10) * r * t
    DG0 = -2 * DG1 + DG2
    return np.exp(-DG0 / (r * t))


def _keq_cytb6f(
    p_hlumen: float,
    f: float,
    e0_pq: float,
    e0_pc: float,
    pmf: float,
    r: float,
    t: float,
) -> float:
    """Equilibrium constant of cytb6f.

    Adjusted from Matuszynska et al 2019 - calculated from pmf instead of deltapH
    """
    DG1 = -2 * f * e0_pq
    DG2 = -f * e0_pc
    DG = -(DG1 + 2 * (np.log(10) * r * t) * p_hlumen) + 2 * DG2 + 2 * f * pmf
    return np.exp(-DG / (r * t))


def _keq_at_psyn(
    pmf: float,
    delta_g0_atp: float,
    r: float,
    t: float,
    f: float,
    hpr: float,
    pi_mol: float,
) -> float:
    """Equilibrium constant of ATP synthase. Adjusted for pmf description.

    For more information see Matuszynska et al 2016 or Ebenhöh et al. 2011,2014
    """
    DG = delta_g0_atp - f * pmf * hpr
    return pi_mol * np.exp(-DG / (r * t))


def _atp_pmf_act(
    pmf: float,
    p_k0_e: float,
    b: float,
    e: float,
    f: float,  # kJ per volt-gram-equivalent
    r: float,
    t: float,  # K
) -> float:
    """Pmf regulation of ATPsynthase"""
    x = np.log(10 ** (-p_k0_e)) + b * (pmf * f) / (r * t)
    return e**x / (1 + e**x)


def _fluo(
    q: float,
    b0: float,
    b2: float,
    k_p: float,
    k_f: float,
    k_h_qslope: float,
    k_h0: float,
) -> float:
    kH = k_h0 + k_h_qslope * q
    return (k_f * b0) / (k_f + k_p + kH) + (k_f * b2) / (k_f + kH)


def _kquencher(
    s: float,
    q: float,
    k_h_qslope: float,
    k_h0: float,
) -> float:
    return (k_h0 + k_h_qslope * q) * s


def _quencher(
    psb_s: float,
    vx: float,
    xtot: float,
    psb_stot: float,
    kzsat: float,
    gamma0: float,
    gamma1: float,
    gamma2: float,
    gamma3: float,
) -> float:
    """Quencher mechanism - Anna 2016 model

    accepts:
    Pr: fraction of non-protonated PsbS protein
    V: fraction of Violaxanthin
    """
    Z = xtot - vx
    P = psb_stot - psb_s
    Zs = Z / (Z + kzsat)

    return (
        gamma0 * (1 - Zs) * psb_s
        + gamma1 * (1 - Zs) * P
        + gamma2 * Zs * P
        + gamma3 * Zs * psb_s
    )


def _quencher_q0(
    psbs: float,
    vx: float,
    y0: float,
) -> float:
    """co-operative 4-state quenching mechanism"""
    # gamma0: slow quenching of (Vx - protonation)
    return y0 * vx * psbs


def _quencher_q1(
    vx: float,
    psbsp: float,
    y1: float,
) -> float:
    """co-operative 4-state quenching mechanism"""
    # gamma1: fast quenching (Vx + protonation)
    return y1 * vx * psbsp


def _quencher_q2(
    psbsp: float,
    zx: float,
    y2: float,
) -> float:
    """co-operative 4-state quenching mechanism"""
    # gamma2: fastest possible quenching (Zx + protonation)
    return y2 * zx * psbsp


def _quencher_q3(
    psbs: float,
    zx: float,
    y3: float,
) -> float:
    """co-operative 4-state quenching mechanism"""
    # gamma3: slow quenching of Zx present (Zx - protonation)
    return y3 * zx * (psbs)


def _quencher_total(
    q1: float,
    q2: float,
    q3: float,
    q4: float,
) -> float:
    return q1 + q2 + q3 + q4


def _deltap_h_to_v(
    delta_p_h: float,
    r: float,
    t: float,
    f: float,
) -> float:
    return -np.log(10) * ((r * t) / f) * delta_p_h


def _delta_ph(
    p_h_lumen: float,
    p_h_stoma: float,
) -> float:
    """
    Calculation of pH difference between stroma and thylakoid lumen

    Accepts:

    pH_lumen: thylakoid lumen pH
    pH_stroma: stroma pH

    """
    return p_h_lumen - p_h_stoma


def _voltage_turnover_mol_chl_per_mmol(
    capacitance_specific: float,
    mol_chl_per_area_membrane: float,
    f: float,
) -> float:
    area_permolChl = 1 / mol_chl_per_area_membrane
    return f / (capacitance_specific * area_permolChl)


def _initial_delta_psi(
    delta_p_h: float,
    r: float,
    f: float,
    t: float,
) -> float:
    """Estimation of delta psi in the dark - assuming delta_pH and delta_psi have equal contribution to pmf"""
    return -np.log(10) * ((r * t) / f) * delta_p_h


def _proton_motive_force(
    _delta_ph: float,
    delta_psi: float,
    f: float,
    t: float,
    r: float,
) -> float:
    """Proton motive force formula - taken from Lowe & Jones (1984).

    https://doi.org/10.1016/0968-0004(84)90038-0

    Accepts:
    delta_ph: pH different between the thylakoid lumen and the stroma
    delta_psi: thylakoid membrane potential
    F: Faraday constant
    R: gas constant
    T: temperature (K)
    """
    return delta_psi - np.log(10) * ((r * t) / f) * _delta_ph


def _p_hmod(
    p_h: float,
    p_kreg: float,
) -> float:
    return 1 - (1 / (10 ** (p_h - p_kreg) + 1))


def _k_b6f(
    _p_hmod: float,
    _k_b6f: float,
) -> float:
    return _p_hmod * _k_b6f


def _reg_kea3(
    _reg_kea3_atp: float,
    _reg_kea3_p_h: float,
) -> float:
    return _reg_kea3_atp * _reg_kea3_p_h


def _reg_kea3_p_h(
    p_hlumen: float,
    p_k_kea3: float,
) -> float:
    return 10 ** (p_hlumen - p_k_kea3) / (10 ** (p_hlumen - p_k_kea3) + 1)


def _reg_kea3_atp(
    atp: float,
    atp_thres: float,
    c: float,
) -> float:
    return (1 - c) / (1 + np.exp((atp - atp_thres) / c))


def _v_kea3_in(
    klumen: float,
    hlumen: float,
    kstroma: float,
    k_kea3: float,
    hstroma: float,
    _reg_kea3: float,
    stroma_volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:
    return float(
        max(
            _conc_to_mmol(
                (k_kea3 * (hlumen * kstroma - hstroma * klumen) * _reg_kea3),
                stroma_volume_per_area_membrane,
                chl_per_area_membrane,
            ),
            0,
        )
    )


def _v_kea3_out(
    klumen: float,
    hlumen: float,
    kstroma: float,
    k_kea3: float,
    hstroma: float,
    _reg_kea3: float,
    lumen_volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:

    return float(
        max(
            _conc_to_mmol(
                (k_kea3 * (hstroma * klumen - hlumen * kstroma) * _reg_kea3),
                lumen_volume_per_area_membrane,
                chl_per_area_membrane,
            ),
            0,
        )
    )


def _vps2(
    b1: float,
    k_p: float,
) -> float:
    """Reduction of PQ due to ps2"""
    return k_p * 0.5 * b1


def _v_p_qox(
    pqh2: float,
    pfd: float,
    k_cytb6f: float,
    k_ptox: float,
    o2ex: float,
    p_qtot: float,
    keq: float,
) -> float:
    """Oxidation of the PQ pool through cytochrome and PTOX"""
    kPFD = k_cytb6f * (pfd)
    k_ptox = k_ptox * o2ex
    a1 = kPFD * keq / (keq + 1)
    a2 = kPFD / (keq + 1)
    return (a1 + k_ptox) * pqh2 - a2 * (p_qtot - pqh2)


def _v_at_pactivity(
    at_pactivity: float,
    light: float,
    k_act_at_pase: float,
    k_deact_at_pase: float,
) -> float:
    """Activation of ATPsynthase by light"""
    switch = light > 0.0
    return (
        k_act_at_pase * switch * (1 - at_pactivity)
        - k_deact_at_pase * (1 - switch) * at_pactivity
    )


def _v_at_psynthase(
    atp: float,
    adp: float,
    _keq_at_psyn: float,
    at_pactivity: float,
    _atp_pmf_act: float,
    k_at_psynthase: float,  # mmol per mol Chl
) -> float:
    """Production of ATP by ATPsynthase - pmf regulation implemented"""
    return at_pactivity * _atp_pmf_act * k_at_psynthase * (adp - atp / _keq_at_psyn)


def _v_at_pcons(
    atp: float,
    k_at_pconsumption: float,
) -> float:
    """ATP consuming reaction"""
    return k_at_pconsumption * atp


def _v_leak(
    h_lumen_conc: float,
    kleak: float,
    h_stroma_conc: float,
) -> float:
    """Transmembrane proton leak"""
    return kleak * (h_lumen_conc - h_stroma_conc)


def _v_xdeepox(
    vx: float,
    h: float,
    n_hx: float,
    kph_sat_z: float,
    k_deepox_v: float,
    volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:
    """Deepoxidation of Vx"""
    a = h**n_hx / (
        h**n_hx
        + _calculate_p_hinv(
            kph_sat_z,
            volume_per_area_membrane,
            chl_per_area_membrane,
        )
        ** n_hx
    )
    v = k_deepox_v * a
    return v * vx


def _v_epox_z(
    zx: float,
    k_epox_z: float,
) -> float:
    """Deepoxidation of Vx"""
    return k_epox_z * zx


def _v_psb_sp(
    psb_s: float,
    h: float,
    n_hl: float,
    kph_sat_lhc: float,
    k_prot: float,
    volume_per_area_membrane: float,
    chl_per_area_membrane: float,
) -> float:
    """Protonation of PsbS protein - Modified for Zx inhibition effect.

    Zx is assmuned to inhibit the deprotonation of PsbS
    """
    a = h**n_hl / (
        h**n_hl
        + _calculate_p_hinv(
            kph_sat_lhc,
            volume_per_area_membrane,
            chl_per_area_membrane,
        )
        ** n_hl
    )
    return k_prot * a * psb_s


def _deprot_act(
    k_zsat: float,
    n_hz: float,
    zx: float,
) -> float:
    """Inhibition effect of Zx on PsbS deprotonation."""
    return k_zsat**n_hz / (k_zsat**n_hz + zx**n_hz)


def _v_psb_s(
    psb_sp: float,
    k_deprot: float,
    psbs_deprot_act: float,
) -> float:
    """Deprotonation of PsbS protein - Modified for Zx inhibition effect.

    Zx is assmuned to inhibit the deprotonation of PsbS
    """
    return k_deprot * psbs_deprot_act * psb_sp


def get_nguyen2026_tomato() -> Model:
    """NPQ model for tomato.

    Adapted from the original NPQ model for Arabidopsis - without the light conversion fn
    Rewritten for better implentation
    """
    m = Model()

    m.add_variables(
        {
            n.b0(): 2.5,
            n.b1(): 0,
            n.b2(): 0,
            "PQH2": 0.0,
            "ATP": 25.0,
            n.h_lumen: InitialAssignment(
                fn=_calculate_p_hinv,
                args=[
                    "pHlumen_init",
                    "lumen_volume_per_area_membrane",
                    "molChl_per_area_membrane",
                ],
            ),
            n.delta_psi: InitialAssignment(
                fn=_initial_delta_psi, args=[n.delta_ph, "R", "T", "F"]
            ),
            "Vx": 1.0,
            "PsbS": 1.0,
            "ATPactivity": 0.1,
            n.k_lumen: InitialAssignment(
                fn=_conc_to_mmol,
                args=[
                    "K_lumen_conc_initial",
                    "lumen_volume_per_area_membrane",
                    "molChl_per_area_membrane",
                ],
            ),
            n.k_stroma: InitialAssignment(
                fn=_conc_to_mmol,
                args=[
                    "K_stroma_conc_initial",
                    "stroma_volume_per_area_membrane",
                    "molChl_per_area_membrane",
                ],
            ),
        }
    )

    m.add_parameters(par)

    m.add_derived(
        "RT",
        _propotional,
        args=["R", "T"],
    )

    m.add_derived(
        n.ph_lumen,
        _calculate_p_h,
        args=[
            n.h_lumen,
            "lumen_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        "H_lumen_conc",
        _mmol_to_conc,
        args=[
            n.h_lumen,
            "lumen_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        n.h_stroma,
        _calculate_p_hinv,
        args=[
            n.ph_stroma,
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        "H_stroma_conc",
        _mmol_to_conc,
        args=[
            n.h_stroma,
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        n.delta_ph,
        _delta_ph,
        args=[n.ph_lumen, n.ph_stroma],
    )
    m.add_derived(
        "delta_pH_V",
        _deltap_h_to_v,
        args=[n.delta_ph, "R", "T", "F"],
    )

    m.add_derived(
        n.pmf,
        _proton_motive_force,
        args=[n.delta_ph, n.delta_psi, "F", "T", "R"],
    )

    m.add_derived(
        "volts_per_charge",
        _voltage_turnover_mol_chl_per_mmol,
        args=[
            "thylakoid_membrane_capacitance",
            "molChl_per_area_membrane",
            "F",
        ],
    )

    m.add_derived(
        n.pq,
        _moiety,
        args=["PQH2", "PQtot"],
    )

    m.add_derived(
        n.adp,
        _moiety,
        args=["ATP", "APtot"],
    )

    m.add_derived(
        "PsbSP",
        _moiety,
        args=["PsbS", "PsbStot"],
    )

    m.add_derived(
        n.zx,
        _moiety,
        args=["Vx", "Xtot"],
    )

    m.add_derived(
        "Keq_PQH2",
        _keq_qapq,
        args=["F", "E0QAQAm", "E0PQPQH2", n.ph_stroma, "R", "T"],
    )

    m.add_derived(
        "Keqcytb6f",
        _keq_cytb6f,
        args=[n.ph_lumen, "F", "E0PQPQH2", "E0PCPCm", n.pmf, "R", "T"],
    )

    m.add_derived(
        "KeqATPsyn",
        _keq_at_psyn,
        args=[n.pmf, "DeltaG0_ATP", "R", "T", "F", "HPR", "Pi"],
    )

    m.add_derived(
        "ATP_pmf_act",
        _atp_pmf_act,
        args=[n.pmf, "pKE0", "b", "e", "F", "R", "T"],
    )

    m.add_derived(
        "pHmod",
        _p_hmod,
        args=[n.ph_lumen, "pKreg"],
    )

    m.add_derived(
        "k_cytb6f",
        _k_b6f,
        args=["pHmod", n.k("b6f")],
    )

    m.add_derived(
        "Q0",
        _quencher_q0,
        args=[n.psbs, n.vx, "gamma0"],
    )

    m.add_derived(
        "Q1",
        _quencher_q1,
        args=[n.vx, n.psbsp, "gamma1"],
    )

    m.add_derived(
        "Q2",
        _quencher_q2,
        args=[n.psbsp, n.zx, "gamma2"],
    )

    m.add_derived(
        "Q3",
        _quencher_q3,
        args=[n.psbs, n.zx, "gamma3"],
    )

    m.add_derived(
        n.quencher,
        _quencher_total,
        args=["Q0", "Q1", "Q2", "Q3"],
    )

    m.add_derived(
        name=n.b3(),
        fn=_moiety_3,
        args=[n.b0(), n.b1(), n.b2(), "PSIItot"],
    )

    m.add_derived(
        name="rel_B0",
        fn=_normalize_concentration,
        args=[n.b0(), "PSIItot"],
    )
    m.add_derived(
        name="rel_B1",
        fn=_normalize_concentration,
        args=[n.b1(), "PSIItot"],
    )
    m.add_derived(
        name="rel_B2",
        fn=_normalize_concentration,
        args=[n.b2(), "PSIItot"],
    )
    m.add_derived(
        name="rel_B3",
        fn=_normalize_concentration,
        args=[n.b3(), "PSIItot"],
    )

    m.add_derived(
        name=n.fluo,
        fn=_fluo,
        args=[n.quencher, n.b0(), n.b2(), "kP", "kF", "kH_Qslope", "kH0"],
    )

    m.add_reaction(
        name="B01",
        fn=_mass_action_1s,
        stoichiometry={n.b0(): -2, n.b1(): 2},
        args=[n.b0(), n.light],
    )

    m.add_reaction(
        name="B10Q",
        fn=_kquencher,
        stoichiometry={n.b1(): -2, n.b0(): 2},
        args=[n.b1(), n.quencher, "kH_Qslope", "kH0"],
    )

    m.add_reaction(
        name="B10F",
        fn=_mass_action_1s,
        stoichiometry={n.b1(): -2, n.b0(): 2},
        args=[n.b1(), "kF"],
    )

    m.add_reaction(
        name="vps2",
        fn=_vps2,
        stoichiometry={
            n.b1(): -2,
            n.b2(): 2,
            n.h_lumen: Derived(fn=_two_times_inverse, args=["bH"]),
            n.delta_psi: Derived(fn=_two_times_ratio, args=["bH", "volts_per_charge"]),
        },
        args=[n.b1(), "kP"],
    )

    m.add_reaction(
        name="B20",
        fn=_mass_action2_rev,
        stoichiometry={n.b2(): -2, n.pqh2: 1, n.b0(): 2},
        args=[n.b2(), n.pq, n.pqh2, n.b0(), "kPQH2", "Keq_PQH2"],
    )

    m.add_reaction(
        name="B23",
        fn=_mass_action_1s,
        stoichiometry={n.b2(): -2},
        args=[n.b2(), n.light],
    )
    m.add_reaction(
        name="B32F",
        fn=_mass_action_1s,
        stoichiometry={n.b2(): 2},
        args=[n.b3(), "kF"],
    )

    m.add_reaction(
        name="B32Q",
        fn=_kquencher,
        stoichiometry={n.b2(): 2},
        args=[n.b3(), n.quencher, "kH_Qslope", "kH0"],
    )

    m.add_reaction(
        "vPQox",
        _v_p_qox,
        args=["PQH2", n.light, "k_cytb6f", "kPTOX", "O2ex", "PQtot", "Keqcytb6f"],
        stoichiometry={
            "PQH2": -1,
            n.h_lumen: Derived(fn=_four_times_inverse, args=["bH"]),
            n.delta_psi: Derived(fn=_four_times_ratio, args=["bH", "volts_per_charge"]),
        },
    )

    m.add_reaction(
        "vATPactivity",
        _v_at_pactivity,
        args=["ATPactivity", n.light, "kActATPase", "kDeactATPase"],
        stoichiometry={"ATPactivity": 1},
    )

    m.add_reaction(
        name="vATPsynthase",
        fn=_v_at_psynthase,
        args=[
            n.atp,
            n.adp,
            "KeqATPsyn",
            n.atpact,
            "ATP_pmf_act",
            "kATPsynthase",
        ],
        stoichiometry={
            n.h_lumen: Derived(fn=_divide_negative, args=["HPR", "bH"]),
            "ATP": 1,
            n.delta_psi: Derived(
                fn=_at_psyn_stoi, args=["HPR", "bH", "volts_per_charge"]
            ),
        },
    )

    m.add_reaction(
        "vATPcons",
        _v_at_pcons,
        stoichiometry={"ATP": -1},
        args=["ATP", "kATPconsumption"],
    )

    m.add_reaction(
        name="vleak",
        fn=_v_leak,
        args=["H_lumen_conc", "kleak", "H_stroma_conc"],
        stoichiometry={
            n.h_lumen: Derived(fn=_inverse_negative, args=["bH"]),
            n.delta_psi: Derived(fn=_divide_negative, args=["volts_per_charge", "bH"]),
        },
    )

    m.add_reaction(
        "vXdeepox",
        _v_xdeepox,
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
        _v_epox_z,
        args=[
            "Zx",
            "kEpoxZ",
        ],
        stoichiometry={"Vx": 1},
    )

    m.add_derived(
        "PsbS_deprot_act",
        _deprot_act,
        args=["KZsat", "nHZ", n.zx],
    )

    m.add_reaction(
        name="vPsbSP",
        fn=_v_psb_sp,
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
        fn=_v_psb_s,
        args=[n.psbsp, "kDeprot", "PsbS_deprot_act"],
        stoichiometry={
            n.psbs: 1,
        },
    )

    m.add_derived(
        "K_stroma_conc",
        _mmol_to_conc,
        args=[
            n.k_stroma,
            "stroma_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        "K_lumen_conc",
        _mmol_to_conc,
        args=[
            n.k_lumen,
            "lumen_volume_per_area_membrane",
            "molChl_per_area_membrane",
        ],
    )

    m.add_derived(
        "reg_KEA3_ATP",
        _reg_kea3_atp,
        args=[n.atp, "ATP_thres_KEA3", "c"],
    )

    m.add_derived(
        "reg_KEA3_pH",
        _reg_kea3_p_h,
        args=[n.ph_lumen, "pK_KEA3"],
    )

    m.add_derived(
        "reg_KEA3",
        _reg_kea3,
        args=["reg_KEA3_ATP", "reg_KEA3_pH"],
    )

    m.add_reaction(
        name="vKEA3_in",
        fn=_v_kea3_in,
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
            n.h_lumen: Derived(fn=_inverse_negative, args=["bH"]),
            n.k_lumen: 1,
            n.k_stroma: -1,
        },
    )

    m.add_reaction(
        name="vKEA3_out",
        fn=_v_kea3_out,
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
            n.h_lumen: Derived(fn=_inverse, args=["bH"]),
            n.k_lumen: -1,
            n.k_stroma: 1,
        },
    )

    return m
