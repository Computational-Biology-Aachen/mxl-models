import math

import numpy as np
from mxlpy import Derived, Model
from mxlpy.surrogates import qss


def _mass_action_1s(
    s1: float,
    k_fwd: float,
) -> float:
    return k_fwd * s1


def _dg_ph(
    r: float,
    t: float,
) -> float:
    return np.log(10) * r * t


def _ph_lumen(
    protons: float,
) -> float:
    return -np.log10(protons * 0.00025)


def _moiety_1(
    concentration: float,
    total: float,
) -> float:
    return total - concentration


def _quencher(
    Psbs: float,
    Vx: float,
    Psbsp: float,
    Zx: float,
    y0: float,
    y1: float,
    y2: float,
    y3: float,
    kZSat: float,
) -> float:
    """co-operative 4-state quenching mechanism
    gamma0: slow quenching of (Vx - protonation)
    gamma1: fast quenching (Vx + protonation)
    gamma2: fastest possible quenching (Zx + protonation)
    gamma3: slow quenching of Zx present (Zx - protonation)
    """
    ZAnt = Zx / (Zx + kZSat)
    return y0 * Vx * Psbs + y1 * Vx * Psbsp + y2 * ZAnt * Psbsp + y3 * ZAnt * Psbs


def _keq_pq_red(
    E0_QA: float,
    F: float,
    E0_PQ: float,
    pHstroma: float,
    dG_pH: float,
    RT: float,
) -> float:
    dg1 = -E0_QA * F
    dg2 = -2 * E0_PQ * F
    dg = -2 * dg1 + dg2 + 2 * pHstroma * dG_pH
    return np.exp(-dg / RT)


def _ps2_crosssection(
    lhc: float,
    static_ant_ii: float,
    static_ant_i: float,
) -> float:
    return static_ant_ii + (1 - static_ant_ii - static_ant_i) * lhc


def _pi_cbb(
    phosphate_total: float,
    pga: float,
    bpga: float,
    gap: float,
    dhap: float,
    fbp: float,
    f6p: float,
    g6p: float,
    g1p: float,
    sbp: float,
    s7p: float,
    e4p: float,
    x5p: float,
    r5p: float,
    rubp: float,
    ru5p: float,
    atp: float,
) -> float:
    return phosphate_total - (
        pga
        + 2 * bpga
        + gap
        + dhap
        + 2 * fbp
        + f6p
        + g6p
        + g1p
        + 2 * sbp
        + s7p
        + e4p
        + x5p
        + r5p
        + 2 * rubp
        + ru5p
        + atp
    )


def _keq_atp(
    pH: float,
    DeltaG0_ATP: float,
    dG_pH: float,
    HPR: float,
    pHstroma: float,
    Pi_mol: float,
    RT: float,
) -> float:
    delta_g = DeltaG0_ATP - dG_pH * HPR * (pHstroma - pH)
    return Pi_mol * math.exp(-delta_g / RT)


def _keq_cytb6f(
    pH: float,
    F: float,
    E0_PQ: float,
    E0_PC: float,
    pHstroma: float,
    RT: float,
    dG_pH: float,
) -> float:
    DG1 = -2 * F * E0_PQ
    DG2 = -F * E0_PC
    DG = -(DG1 + 2 * dG_pH * pH) + 2 * DG2 + 2 * dG_pH * (pHstroma - pH)
    return math.exp(-DG / RT)


def _keq_fnr(
    E0_Fd: float,
    F: float,
    E0_NADP: float,
    pHstroma: float,
    dG_pH: float,
    RT: float,
) -> float:
    dg1 = -E0_Fd * F
    dg2 = -2 * E0_NADP * F
    dg = -2 * dg1 + dg2 + dG_pH * pHstroma
    return math.exp(-dg / RT)


def _keq_pcp700(
    e0_pc: float,
    f: float,
    eo_p700: float,
    rt: float,
) -> float:
    dg1 = -e0_pc * f
    dg2 = -eo_p700 * f
    dg = -dg1 + dg2
    return math.exp(-dg / rt)


def _keq_faf_d(
    e0_fa: float,
    f: float,
    e0_fd: float,
    rt: float,
) -> float:
    dg1 = -e0_fa * f
    dg2 = -e0_fd * f
    dg = -dg1 + dg2
    return math.exp(-dg / rt)


def _ps1states_2019(
    pc_px: float,
    pc_red: float,
    fd_ox: float,
    fd_red: float,
    ps2cs: float,
    psi_tot: float,
    k_fd_red: float,
    keq_fafd: float,
    keq_pcp700: float,
    k_pc_ox: float,
    pfd: float,
) -> float:
    """QSSA calculates open state of PSI
    depends on reduction states of plastocyanin and ferredoxin
    C = [PC], F = [Fd] (ox. forms)
    """
    L = (1 - ps2cs) * pfd
    return psi_tot / (
        1
        + L / (k_fd_red * fd_ox)
        + (1 + fd_red / (keq_fafd * fd_ox))
        * (pc_px / (keq_pcp700 * pc_red) + L / (k_pc_ox * pc_red))
    )


def _rate_translocator(
    pi: float,
    pga: float,
    gap: float,
    dhap: float,
    k_pxt: float,
    p_ext: float,
    k_pi: float,
    k_pga: float,
    k_gap: float,
    k_dhap: float,
) -> float:
    return 1 + (1 + k_pxt / p_ext) * (
        pi / k_pi + pga / k_pga + gap / k_gap + dhap / k_dhap
    )


def _rate_atp_synthase_2019(
    ATP: float,
    ADP: float,
    Keq_ATPsynthase: float,
    kATPsynth: float,
    convf: float,
) -> float:
    return kATPsynth * (ADP / convf - ATP / convf / Keq_ATPsynthase)


def _neg_div(
    x: float,
    y: float,
) -> float:
    return -x / y


def _value(
    x: float,
) -> float:
    return x


def _b6f(
    PC_ox: float,
    PQ_ox: float,
    PQ_red: float,
    PC_red: float,
    Keq_B6f: float,
    kCytb6f: float,
) -> float:
    return max(
        kCytb6f * (PQ_red * PC_ox**2 - PQ_ox * PC_red**2 / Keq_B6f),
        -kCytb6f,
    )


def _four_div_by(
    x: float,
) -> float:
    return 4.0 / x


def _protonation_hill(
    vx: float,
    h: float,
    nh: float,
    k_fwd: float,
    k_ph_sat: float,
) -> float:
    return k_fwd * (h**nh / (h**nh + _protons_stroma(k_ph_sat) ** nh)) * vx  # type: ignore


def _rate_cyclic_electron_flow(
    Pox: float,
    Fdred: float,
    kcyc: float,
) -> float:
    return kcyc * Fdred**2 * Pox


def _rate_protonation_hill(
    Vx: float,
    H: float,
    k_fwd: float,
    nH: float,
    kphSat: float,
) -> float:
    return k_fwd * (H**nH / (H**nH + _protons_stroma(kphSat) ** nH)) * Vx  # type: ignore


def _rate_fnr_2019(
    Fd_ox: float,
    Fd_red: float,
    NADPH: float,
    NADP: float,
    KM_FNR_F: float,
    KM_FNR_N: float,
    vmax: float,
    Keq_FNR: float,
    convf: float,
) -> float:
    fdred = Fd_red / KM_FNR_F
    fdox = Fd_ox / KM_FNR_F
    nadph = NADPH / convf / KM_FNR_N
    nadp = NADP / convf / KM_FNR_N
    return (
        vmax
        * (fdred**2 * nadp - fdox**2 * nadph / Keq_FNR)
        / ((1 + fdred + fdred**2) * (1 + nadp) + (1 + fdox + fdox**2) * (1 + nadph) - 1)
    )


def _rate_ps2(
    b1: float,
    k2: float,
) -> float:
    return 0.5 * k2 * b1


def _two_div_by(
    x: float,
) -> float:
    return 2.0 / x


def _rate_ps1(
    a: float,
    ps2cs: float,
    pfd: float,
) -> float:
    return (1 - ps2cs) * pfd * a


def _protons_stroma(
    ph: float,
) -> float:
    return 4000.0 * 10 ** (-ph)


def _rate_leak(
    protons_lumen: float,
    ph_stroma: float,
    k_leak: float,
) -> float:
    return k_leak * (protons_lumen - _protons_stroma(ph_stroma))


def _neg_one_div_by(
    x: float,
) -> float:
    return -1.0 / x


def _mass_action_2s(
    s1: float,
    s2: float,
    k_fwd: float,
) -> float:
    return k_fwd * s1 * s2


def _rate_state_transition_ps1_ps2(
    ant: float,
    pox: float,
    p_tot: float,
    k_stt7: float,
    km_st: float,
    n_st: float,
) -> float:
    return k_stt7 * (1 / (1 + (pox / p_tot / km_st) ** n_st)) * ant


def _rate_poolman_5i(
    rubp: float,
    pga: float,
    co2: float,
    vmax: float,
    kms_rubp: float,
    kms_co2: float,
    # inhibitors
    ki_pga: float,
    fbp: float,
    ki_fbp: float,
    sbp: float,
    ki_sbp: float,
    pi: float,
    ki_p: float,
    nadph: float,
    ki_nadph: float,
) -> float:
    top = vmax * rubp * co2
    btm = (
        rubp
        + kms_rubp
        * (
            1
            + pga / ki_pga
            + fbp / ki_fbp
            + sbp / ki_sbp
            + pi / ki_p
            + nadph / ki_nadph
        )
    ) * (co2 + kms_co2)
    return top / btm


def _rapid_equilibrium_2s_2p(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 * s2 - p1 * p2 / q)


def _rapid_equilibrium_3s_3p(
    s1: float,
    s2: float,
    s3: float,
    p1: float,
    p2: float,
    p3: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 * s2 * s3 - p1 * p2 * p3 / q)


def _rapid_equilibrium_1s_1p(
    s1: float,
    p1: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 - p1 / q)


def _rapid_equilibrium_2s_1p(
    s1: float,
    s2: float,
    p1: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 * s2 - p1 / q)


def _michaelis_menten_1s_2i(
    s: float,
    i1: float,
    i2: float,
    vmax: float,
    km: float,
    ki1: float,
    ki2: float,
) -> float:
    return vmax * s / (s + km * (1 + i1 / ki1 + i2 / ki2))


def _michaelis_menten_1s_1i(
    s: float,
    i: float,
    vmax: float,
    km: float,
    ki: float,
) -> float:
    return vmax * s / (s + km * (1 + i / ki))


def _rate_prk(
    ru5p: float,
    atp: float,
    pi: float,
    pga: float,
    rubp: float,
    adp: float,
    v13: float,
    km131: float,
    km132: float,
    ki131: float,
    ki132: float,
    ki133: float,
    ki134: float,
    ki135: float,
) -> float:
    return (
        v13
        * ru5p
        * atp
        / (
            (ru5p + km131 * (1 + pga / ki131 + rubp / ki132 + pi / ki133))
            * (atp * (1 + adp / ki134) + km132 * (1 + adp / ki135))
        )
    )


def _rate_out(
    s1: float,
    n_total: float,
    vmax_efflux: float,
    k_efflux: float,
) -> float:
    return vmax_efflux * s1 / (n_total * k_efflux)


def _rate_starch(
    g1p: float,
    atp: float,
    adp: float,
    pi: float,
    pga: float,
    f6p: float,
    fbp: float,
    v_st: float,
    kmst1: float,
    kmst2: float,
    ki_st: float,
    kast1: float,
    kast2: float,
    kast3: float,
) -> float:
    return (
        v_st
        * g1p
        * atp
        / (
            (g1p + kmst1)
            * (
                (1 + adp / ki_st) * (atp + kmst2)
                + kmst2 * pi / (kast1 * pga + kast2 * f6p + kast3 * fbp)
            )
        )
    )


def _ps2states_2016_phd_surrogate(
    pq_ox: float,
    pq_red: float,
    ps2cs: float,
    quencher: float,
    psii_tot: float,
    k2: float,
    k_f: float,
    _kh: float,
    keq_pq_red: float,
    k_pq_red: float,
    pfd: float,
    k_h0: float,
) -> tuple[float, float, float, float]:
    x0 = k_f**2
    x1 = k_h0**2
    x2 = k2 * k_f
    x3 = k2 * k_h0
    x4 = 2 * k_f
    x5 = k_h0 * x4
    x6 = _kh * quencher
    x7 = k2 * x6
    x8 = x4 * x6
    x9 = 2 * x6
    x10 = k_h0 * x9
    x11 = _kh**2 * quencher**2
    x12 = k2 * keq_pq_red
    x13 = k_pq_red * keq_pq_red * pq_ox
    x14 = k_pq_red * pq_red
    x15 = k2 * x14
    x16 = pfd * ps2cs
    x17 = k_f * x14
    x18 = k_h0 * x14
    x19 = x14 * x6
    x20 = x13 * x16
    x21 = keq_pq_red * x16
    x22 = (
        x0 * x14
        + x1 * x14
        + x11 * x14
        + x14 * x2
        + x14 * x3
        + x14 * x5
        + x14 * x7
        + x14 * x8
        + x18 * x9
        + x2 * x21
        + x21 * x3
        + x21 * x7
    )
    x23 = psii_tot / (
        k_f * x20
        + k_h0 * x20
        + pfd**2 * ps2cs**2 * x12
        + x0 * x13
        + x1 * x13
        + x10 * x13
        + x11 * x13
        + x13 * x2
        + x13 * x3
        + x13 * x5
        + x13 * x7
        + x13 * x8
        + x15 * x16
        + x16 * x17
        + x16 * x18
        + x16 * x19
        + x20 * x6
        + x22
    )
    x24 = x16 * x23
    _B0 = x13 * x23 * (x0 + x1 + x10 + x11 + x2 + x3 + x5 + x7 + x8)
    _B1 = x13 * x24 * (k_f + k_h0 + x6)
    _B2 = x22 * x23
    _B3 = x24 * (x12 * x16 + x15 + x17 + x18 + x19)
    return _B0, _B1, _B2, _B3


def create_model() -> Model:
    m = Model()

    m = m.add_variable(
        "3PGA",
        initial_value=0.9928653922138561,
    )
    m = m.add_variable(
        "BPGA",
        initial_value=0.0005297732935310749,
    )
    m = m.add_variable(
        "GAP",
        initial_value=0.0062663539939955834,
    )
    m = m.add_variable(
        "DHAP",
        initial_value=0.13785977143668732,
    )
    m = m.add_variable(
        "FBP",
        initial_value=0.006133532145409954,
    )
    m = m.add_variable(
        "F6P",
        initial_value=0.31271973359685457,
    )
    m = m.add_variable(
        "G6P",
        initial_value=0.719255387166192,
    )
    m = m.add_variable(
        "G1P",
        initial_value=0.041716812452951633,
    )
    m = m.add_variable(
        "SBP",
        initial_value=0.013123745088361893,
    )
    m = m.add_variable(
        "S7P",
        initial_value=0.15890073845176905,
    )
    m = m.add_variable(
        "E4P",
        initial_value=0.007322797350442026,
    )
    m = m.add_variable(
        "X5P",
        initial_value=0.022478763225333428,
    )
    m = m.add_variable(
        "R5P",
        initial_value=0.037651927659696716,
    )
    m = m.add_variable(
        "RUBP",
        initial_value=0.13184790283048484,
    )
    m = m.add_variable(
        "RU5P",
        initial_value=0.015060770937455408,
    )
    m = m.add_variable(
        "ATP",
        initial_value=1.612922506604933,
    )
    m = m.add_variable(
        "Ferredoxine (oxidised)",
        initial_value=3.8624032084329674,
    )
    m = m.add_variable(
        "protons_lumen",
        initial_value=0.002208423037307405,
    )
    m = m.add_variable(
        "Light-harvesting complex",
        initial_value=0.80137477470646,
    )
    m = m.add_variable(
        "NADPH",
        initial_value=0.491395685599137,
    )
    m = m.add_variable(
        "Plastocyanine (oxidised)",
        initial_value=1.885391998090184,
    )
    m = m.add_variable(
        "Plastoquinone (oxidised)",
        initial_value=10.991562708096392,
    )
    m = m.add_variable(
        "PsbS (de-protonated)",
        initial_value=0.9610220887579118,
    )
    m = m.add_variable(
        "Violaxanthin",
        initial_value=0.9514408605906095,
    )
    m = m.add_parameter(
        "protons",
        _value=1.2589254117941661e-05,
    )
    m = m.add_parameter(
        "pH",
        _value=7.9,
    )
    m = m.add_parameter(
        "CO2 (dissolved)",
        _value=0.2,
    )
    m = m.add_parameter(
        "O2 (dissolved)_lumen",
        _value=8.0,
    )
    m = m.add_parameter(
        "PPFD",
        _value=100.0,
    )
    m = m.add_parameter(
        "bH",
        _value=100.0,
    )
    m = m.add_parameter(
        "F",
        _value=96.485,
    )
    m = m.add_parameter(
        "E^0_PC",
        _value=0.38,
    )
    m = m.add_parameter(
        "E^0_P700",
        _value=0.48,
    )
    m = m.add_parameter(
        "E^0_FA",
        _value=-0.55,
    )
    m = m.add_parameter(
        "E^0_Fd",
        _value=-0.43,
    )
    m = m.add_parameter(
        "E^0_NADP",
        _value=-0.113,
    )
    m = m.add_parameter(
        "convf",
        _value=0.032,
    )
    m = m.add_parameter(
        "R",
        _value=0.0083,
    )
    m = m.add_parameter(
        "T",
        _value=298.0,
    )
    m = m.add_parameter(
        "Carotenoids_tot",
        _value=1.0,
    )
    m = m.add_parameter(
        "Fd*",
        _value=5.0,
    )
    m = m.add_parameter(
        "PC_tot",
        _value=4.0,
    )
    m = m.add_parameter(
        "PSBS_tot",
        _value=1.0,
    )
    m = m.add_parameter(
        "LHC_tot",
        _value=1.0,
    )
    m = m.add_parameter(
        "gamma0",
        _value=0.1,
    )
    m = m.add_parameter(
        "gamma1",
        _value=0.25,
    )
    m = m.add_parameter(
        "gamma2",
        _value=0.6,
    )
    m = m.add_parameter(
        "gamma3",
        _value=0.15,
    )
    m = m.add_parameter(
        "kZSat",
        _value=0.12,
    )
    m = m.add_parameter(
        "E^0_QA",
        _value=-0.14,
    )
    m = m.add_parameter(
        "E^0_PQ",
        _value=0.354,
    )
    m = m.add_parameter(
        "PQ_tot",
        _value=17.5,
    )
    m = m.add_parameter(
        "staticAntII",
        _value=0.1,
    )
    m = m.add_parameter(
        "staticAntI",
        _value=0.37,
    )
    m = m.add_parameter(
        "NADP*",
        _value=0.8,
    )
    m = m.add_parameter(
        "A*P",
        _value=2.55,
    )
    m = m.add_parameter(
        "Pi_tot",
        _value=17.05,
    )
    m = m.add_parameter(
        "kf_atp_synthase",
        _value=20.0,
    )
    m = m.add_parameter(
        "HPR",
        _value=4.666666666666667,
    )
    m = m.add_parameter(
        "Pi_mol",
        _value=0.01,
    )
    m = m.add_parameter(
        "DeltaG0_ATP",
        _value=30.6,
    )
    m = m.add_parameter(
        "kcat_b6f",
        _value=2.5,
    )
    m = m.add_parameter(
        "kh_lhc_protonation",
        _value=3.0,
    )
    m = m.add_parameter(
        "kf_lhc_protonation",
        _value=0.0096,
    )
    m = m.add_parameter(
        "ksat_lhc_protonation",
        _value=5.8,
    )
    m = m.add_parameter(
        "kf_lhc_deprotonation",
        _value=0.0096,
    )
    m = m.add_parameter(
        "kf_cyclic_electron_flow",
        _value=1.0,
    )
    m = m.add_parameter(
        "kf_violaxanthin_deepoxidase",
        _value=0.0024,
    )
    m = m.add_parameter(
        "kh_violaxanthin_deepoxidase",
        _value=5.0,
    )
    m = m.add_parameter(
        "ksat_violaxanthin_deepoxidase",
        _value=5.8,
    )
    m = m.add_parameter(
        "kf_zeaxanthin_epoxidase",
        _value=0.00024,
    )
    m = m.add_parameter(
        "km_fnr_Ferredoxine (reduced)",
        _value=1.56,
    )
    m = m.add_parameter(
        "km_fnr_NADP",
        _value=0.22,
    )
    m = m.add_parameter(
        "E0_fnr",
        _value=3.0,
    )
    m = m.add_parameter(
        "kcat_fnr",
        _value=500.0,
    )
    m = m.add_parameter(
        "kf_ndh",
        _value=0.002,
    )
    m = m.add_parameter(
        "PSII_total",
        _value=2.5,
    )
    m = m.add_parameter(
        "PSI_total",
        _value=2.5,
    )
    m = m.add_parameter(
        "kH0",
        _value=500000000.0,
    )
    m = m.add_parameter(
        "kPQred",
        _value=250.0,
    )
    m = m.add_parameter(
        "kPCox",
        _value=2500.0,
    )
    m = m.add_parameter(
        "kFdred",
        _value=250000.0,
    )
    m = m.add_parameter(
        "k2",
        _value=5000000000.0,
    )
    m = m.add_parameter(
        "kH",
        _value=5000000000.0,
    )
    m = m.add_parameter(
        "kF",
        _value=625000000.0,
    )
    m = m.add_parameter(
        "kf_proton_leak",
        _value=10.0,
    )
    m = m.add_parameter(
        "kPTOX",
        _value=0.01,
    )
    m = m.add_parameter(
        "kStt7",
        _value=0.0035,
    )
    m = m.add_parameter(
        "km_lhc_state_transition_12",
        _value=0.2,
    )
    m = m.add_parameter(
        "n_ST",
        _value=2.0,
    )
    m = m.add_parameter(
        "kPph1",
        _value=0.0013,
    )
    m = m.add_parameter(
        "E0_rubisco",
        _value=1.0,
    )
    m = m.add_parameter(
        "kcat_rubisco_carboxylase",
        _value=2.72,
    )
    m = m.add_parameter(
        "km_rubisco_carboxylase_RUBP",
        _value=0.02,
    )
    m = m.add_parameter(
        "km_rubisco_carboxylase_CO2 (dissolved)",
        _value=0.0107,
    )
    m = m.add_parameter(
        "ki_rubisco_carboxylase_3PGA",
        _value=0.04,
    )
    m = m.add_parameter(
        "ki_rubisco_carboxylase_FBP",
        _value=0.04,
    )
    m = m.add_parameter(
        "ki_rubisco_carboxylase_SBP",
        _value=0.075,
    )
    m = m.add_parameter(
        "ki_rubisco_carboxylase_Orthophosphate",
        _value=0.9,
    )
    m = m.add_parameter(
        "ki_rubisco_carboxylase_NADPH",
        _value=0.07,
    )
    m = m.add_parameter(
        "kre_phosphoglycerate_kinase",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_phosphoglycerate_kinase",
        _value=0.00031,
    )
    m = m.add_parameter(
        "kre_gadph",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_gadph",
        _value=16000000.0,
    )
    m = m.add_parameter(
        "kre_triose_phosphate_isomerase",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_triose_phosphate_isomerase",
        _value=22.0,
    )
    m = m.add_parameter(
        "kre_aldolase_dhap_gap",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_aldolase_dhap_gap",
        _value=7.1,
    )
    m = m.add_parameter(
        "kre_aldolase_dhap_e4p",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_aldolase_dhap_e4p",
        _value=13.0,
    )
    m = m.add_parameter(
        "E0_fbpase",
        _value=1.0,
    )
    m = m.add_parameter(
        "kcat_fbpase",
        _value=1.6,
    )
    m = m.add_parameter(
        "km_fbpase_s",
        _value=0.03,
    )
    m = m.add_parameter(
        "ki_fbpase_F6P",
        _value=0.7,
    )
    m = m.add_parameter(
        "ki_fbpase_Orthophosphate",
        _value=12.0,
    )
    m = m.add_parameter(
        "kre_transketolase_gap_f6p",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_transketolase_gap_f6p",
        _value=0.084,
    )
    m = m.add_parameter(
        "kre_transketolase_gap_s7p",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_transketolase_gap_s7p",
        _value=0.85,
    )
    m = m.add_parameter(
        "E0_SBPase",
        _value=1.0,
    )
    m = m.add_parameter(
        "kcat_SBPase",
        _value=0.32,
    )
    m = m.add_parameter(
        "km_SBPase_s",
        _value=0.013,
    )
    m = m.add_parameter(
        "ki_SBPase_Orthophosphate",
        _value=12.0,
    )
    m = m.add_parameter(
        "kre_ribose_phosphate_isomerase",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_ribose_phosphate_isomerase",
        _value=0.4,
    )
    m = m.add_parameter(
        "kre_ribulose_phosphate_epimerase",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_ribulose_phosphate_epimerase",
        _value=0.67,
    )
    m = m.add_parameter(
        "E0_phosphoribulokinase",
        _value=1.0,
    )
    m = m.add_parameter(
        "kcat_phosphoribulokinase",
        _value=7.9992,
    )
    m = m.add_parameter(
        "km_phosphoribulokinase_RU5P",
        _value=0.05,
    )
    m = m.add_parameter(
        "km_phosphoribulokinase_ATP",
        _value=0.05,
    )
    m = m.add_parameter(
        "ki_phosphoribulokinase_3PGA",
        _value=2.0,
    )
    m = m.add_parameter(
        "ki_phosphoribulokinase_RUBP",
        _value=0.7,
    )
    m = m.add_parameter(
        "ki_phosphoribulokinase_Orthophosphate",
        _value=4.0,
    )
    m = m.add_parameter(
        "ki_phosphoribulokinase_4",
        _value=2.5,
    )
    m = m.add_parameter(
        "ki_phosphoribulokinase_5",
        _value=0.4,
    )
    m = m.add_parameter(
        "kre_g6pi",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_g6pi",
        _value=2.3,
    )
    m = m.add_parameter(
        "kre_phosphoglucomutase",
        _value=800000000.0,
    )
    m = m.add_parameter(
        "keq_phosphoglucomutase",
        _value=0.058,
    )
    m = m.add_parameter(
        "Orthophosphate (external)",
        _value=0.5,
    )
    m = m.add_parameter(
        "km_ex_pga",
        _value=0.25,
    )
    m = m.add_parameter(
        "km_ex_gap",
        _value=0.075,
    )
    m = m.add_parameter(
        "km_ex_dhap",
        _value=0.077,
    )
    m = m.add_parameter(
        "km_N_translocator_Orthophosphate (external)",
        _value=0.74,
    )
    m = m.add_parameter(
        "km_N_translocator_Orthophosphate",
        _value=0.63,
    )
    m = m.add_parameter(
        "kcat_N_translocator",
        _value=2.0,
    )
    m = m.add_parameter(
        "E0_N_translocator",
        _value=1.0,
    )
    m = m.add_parameter(
        "E0_ex_g1p",
        _value=1.0,
    )
    m = m.add_parameter(
        "km_ex_g1p_G1P",
        _value=0.08,
    )
    m = m.add_parameter(
        "km_ex_g1p_ATP",
        _value=0.08,
    )
    m = m.add_parameter(
        "ki_ex_g1p",
        _value=10.0,
    )
    m = m.add_parameter(
        "ki_ex_g1p_3PGA",
        _value=0.1,
    )
    m = m.add_parameter(
        "ki_ex_g1p_F6P",
        _value=0.02,
    )
    m = m.add_parameter(
        "ki_ex_g1p_FBP",
        _value=0.02,
    )
    m = m.add_parameter(
        "kcat_ex_g1p",
        _value=0.32,
    )
    m = m.add_derived(
        "RT",
        fn=_mass_action_1s,
        args=["R", "T"],
    )
    m = m.add_derived(
        "dG_pH",
        fn=_dg_ph,
        args=["R", "T"],
    )
    m = m.add_derived(
        "pH_lumen",
        fn=_ph_lumen,
        args=["protons_lumen"],
    )
    m = m.add_derived(
        "Zeaxanthin",
        fn=_moiety_1,
        args=["Violaxanthin", "Carotenoids_tot"],
    )
    m = m.add_derived(
        "Ferredoxine (reduced)",
        fn=_moiety_1,
        args=["Ferredoxine (oxidised)", "Fd*"],
    )
    m = m.add_derived(
        "Plastocyanine (reduced)",
        fn=_moiety_1,
        args=["Plastocyanine (oxidised)", "PC_tot"],
    )
    m = m.add_derived(
        "PsbS (protonated)",
        fn=_moiety_1,
        args=["PsbS (de-protonated)", "PSBS_tot"],
    )
    m = m.add_derived(
        "Light-harvesting complex (protonated)",
        fn=_moiety_1,
        args=["Light-harvesting complex", "LHC_tot"],
    )
    m = m.add_derived(
        "Q",
        fn=_quencher,
        args=[
            "PsbS (de-protonated)",
            "Violaxanthin",
            "PsbS (protonated)",
            "Zeaxanthin",
            "gamma0",
            "gamma1",
            "gamma2",
            "gamma3",
            "kZSat",
        ],
    )
    m = m.add_derived(
        "keq_Plastoquinone (reduced)",
        fn=_keq_pq_red,
        args=["E^0_QA", "F", "E^0_PQ", "pH", "dG_pH", "RT"],
    )
    m = m.add_derived(
        "Plastoquinone (reduced)",
        fn=_moiety_1,
        args=["Plastoquinone (oxidised)", "PQ_tot"],
    )
    m = m.add_derived(
        "PSII_cross_section",
        fn=_ps2_crosssection,
        args=["Light-harvesting complex", "staticAntII", "staticAntI"],
    )
    m = m.add_derived(
        "NADP",
        fn=_moiety_1,
        args=["NADPH", "NADP*"],
    )
    m = m.add_derived(
        "ADP",
        fn=_moiety_1,
        args=["ATP", "A*P"],
    )
    m = m.add_derived(
        "Orthophosphate",
        fn=_pi_cbb,
        args=[
            "Pi_tot",
            "3PGA",
            "BPGA",
            "GAP",
            "DHAP",
            "FBP",
            "F6P",
            "G6P",
            "G1P",
            "SBP",
            "S7P",
            "E4P",
            "X5P",
            "R5P",
            "RUBP",
            "RU5P",
            "ATP",
        ],
    )
    m = m.add_derived(
        "keq_atp_synthase",
        fn=_keq_atp,
        args=["pH_lumen", "DeltaG0_ATP", "dG_pH", "HPR", "pH", "Pi_mol", "RT"],
    )
    m = m.add_derived(
        "keq_b6f",
        fn=_keq_cytb6f,
        args=["pH_lumen", "F", "E^0_PQ", "E^0_PC", "pH", "RT", "dG_pH"],
    )
    m = m.add_derived(
        "keq_fnr",
        fn=_keq_fnr,
        args=["E^0_Fd", "F", "E^0_NADP", "pH", "dG_pH", "RT"],
    )
    m = m.add_derived(
        "vmax_fnr",
        fn=_mass_action_1s,
        args=["kcat_fnr", "E0_fnr"],
    )
    m = m.add_derived(
        "keq_PCP700",
        fn=_keq_pcp700,
        args=["E^0_PC", "F", "E^0_P700", "RT"],
    )
    m = m.add_derived(
        "keq_ferredoxin_reductase",
        fn=_keq_faf_d,
        args=["E^0_FA", "F", "E^0_Fd", "RT"],
    )
    m = m.add_derived(
        "A1",
        fn=_ps1states_2019,
        args=[
            "Plastocyanine (oxidised)",
            "Plastocyanine (reduced)",
            "Ferredoxine (oxidised)",
            "Ferredoxine (reduced)",
            "PSII_cross_section",
            "PSI_total",
            "kFdred",
            "keq_ferredoxin_reductase",
            "keq_PCP700",
            "kPCox",
            "PPFD",
        ],
    )
    m = m.add_derived(
        "vmax_rubisco_carboxylase",
        fn=_mass_action_1s,
        args=["kcat_rubisco_carboxylase", "E0_rubisco"],
    )
    m = m.add_derived(
        "vmax_fbpase",
        fn=_mass_action_1s,
        args=["kcat_fbpase", "E0_fbpase"],
    )
    m = m.add_derived(
        "vmax_SBPase",
        fn=_mass_action_1s,
        args=["kcat_SBPase", "E0_SBPase"],
    )
    m = m.add_derived(
        "vmax_phosphoribulokinase",
        fn=_mass_action_1s,
        args=["kcat_phosphoribulokinase", "E0_phosphoribulokinase"],
    )
    m = m.add_derived(
        "vmax_ex_pga",
        fn=_mass_action_1s,
        args=["kcat_N_translocator", "E0_N_translocator"],
    )
    m = m.add_derived(
        "N_translocator",
        fn=_rate_translocator,
        args=[
            "Orthophosphate",
            "3PGA",
            "GAP",
            "DHAP",
            "km_N_translocator_Orthophosphate (external)",
            "Orthophosphate (external)",
            "km_N_translocator_Orthophosphate",
            "km_ex_pga",
            "km_ex_gap",
            "km_ex_dhap",
        ],
    )
    m = m.add_derived(
        "vmax_ex_g1p",
        fn=_mass_action_1s,
        args=["kcat_ex_g1p", "E0_ex_g1p"],
    )
    m = m.add_reaction(
        "atp_synthase",
        fn=_rate_atp_synthase_2019,
        args=["ATP", "ADP", "keq_atp_synthase", "kf_atp_synthase", "convf"],
        stoichiometry={
            "protons_lumen": Derived(
                fn=_neg_div,
                args=["HPR", "bH"],
            ),
            "ATP": Derived(
                fn=_value,
                args=["convf"],
            ),
        },
    )
    m = m.add_reaction(
        "b6f",
        fn=_b6f,
        args=[
            "Plastocyanine (oxidised)",
            "Plastoquinone (oxidised)",
            "Plastoquinone (reduced)",
            "Plastocyanine (reduced)",
            "keq_b6f",
            "kcat_b6f",
        ],
        stoichiometry={
            "Plastocyanine (oxidised)": -2,
            "Plastoquinone (oxidised)": 1,
            "protons_lumen": Derived(
                fn=_four_div_by,
                args=["bH"],
            ),
        },
    )
    m = m.add_reaction(
        "lhc_protonation",
        fn=_protonation_hill,
        args=[
            "PsbS (de-protonated)",
            "protons_lumen",
            "kh_lhc_protonation",
            "kf_lhc_protonation",
            "ksat_lhc_protonation",
        ],
        stoichiometry={"PsbS (de-protonated)": -1},
    )
    m = m.add_reaction(
        "lhc_deprotonation",
        fn=_mass_action_1s,
        args=["PsbS (protonated)", "kf_lhc_deprotonation"],
        stoichiometry={"PsbS (de-protonated)": 1},
    )
    m = m.add_reaction(
        "cyclic_electron_flow",
        fn=_rate_cyclic_electron_flow,
        args=[
            "Plastoquinone (oxidised)",
            "Ferredoxine (reduced)",
            "kf_cyclic_electron_flow",
        ],
        stoichiometry={"Plastoquinone (oxidised)": -1, "Ferredoxine (oxidised)": 2},
    )
    m = m.add_reaction(
        "violaxanthin_deepoxidase",
        fn=_rate_protonation_hill,
        args=[
            "Violaxanthin",
            "protons_lumen",
            "kf_violaxanthin_deepoxidase",
            "kh_violaxanthin_deepoxidase",
            "ksat_violaxanthin_deepoxidase",
        ],
        stoichiometry={"Violaxanthin": -1},
    )
    m = m.add_reaction(
        "zeaxanthin_epoxidase",
        fn=_mass_action_1s,
        args=["Zeaxanthin", "kf_zeaxanthin_epoxidase"],
        stoichiometry={"Violaxanthin": 1},
    )
    m = m.add_reaction(
        "fnr",
        fn=_rate_fnr_2019,
        args=[
            "Ferredoxine (oxidised)",
            "Ferredoxine (reduced)",
            "NADPH",
            "NADP",
            "km_fnr_Ferredoxine (reduced)",
            "km_fnr_NADP",
            "vmax_fnr",
            "keq_fnr",
            "convf",
        ],
        stoichiometry={
            "Ferredoxine (oxidised)": 2,
            "NADPH": Derived(
                fn=_value,
                args=["convf"],
            ),
        },
    )
    m = m.add_reaction(
        "ndh",
        fn=_mass_action_1s,
        args=["Plastoquinone (oxidised)", "kf_ndh"],
        stoichiometry={"Plastoquinone (oxidised)": -1},
    )
    m = m.add_reaction(
        "PSII",
        fn=_rate_ps2,
        args=["B1", "k2"],
        stoichiometry={
            "Plastoquinone (oxidised)": -1,
            "protons_lumen": Derived(
                fn=_two_div_by,
                args=["bH"],
            ),
        },
    )
    m = m.add_reaction(
        "PSI",
        fn=_rate_ps1,
        args=["A1", "PSII_cross_section", "PPFD"],
        stoichiometry={"Ferredoxine (oxidised)": -1, "Plastocyanine (oxidised)": 1},
    )
    m = m.add_reaction(
        "proton_leak",
        fn=_rate_leak,
        args=["protons_lumen", "pH", "kf_proton_leak"],
        stoichiometry={
            "protons_lumen": Derived(
                fn=_neg_one_div_by,
                args=["bH"],
            )
        },
    )
    m = m.add_reaction(
        "PTOX",
        fn=_mass_action_2s,
        args=["Plastoquinone (reduced)", "O2 (dissolved)_lumen", "kPTOX"],
        stoichiometry={"Plastoquinone (oxidised)": 1},
    )
    m = m.add_reaction(
        "lhc_state_transition_12",
        fn=_rate_state_transition_ps1_ps2,
        args=[
            "Light-harvesting complex",
            "Plastoquinone (oxidised)",
            "PQ_tot",
            "kStt7",
            "km_lhc_state_transition_12",
            "n_ST",
        ],
        stoichiometry={"Light-harvesting complex": -1},
    )
    m = m.add_reaction(
        "lhc_state_transition_21",
        fn=_mass_action_1s,
        args=["Light-harvesting complex (protonated)", "kPph1"],
        stoichiometry={"Light-harvesting complex": 1},
    )
    m = m.add_reaction(
        "rubisco_carboxylase",
        fn=_rate_poolman_5i,
        args=[
            "RUBP",
            "3PGA",
            "CO2 (dissolved)",
            "vmax_rubisco_carboxylase",
            "km_rubisco_carboxylase_RUBP",
            "km_rubisco_carboxylase_CO2 (dissolved)",
            "ki_rubisco_carboxylase_3PGA",
            "FBP",
            "ki_rubisco_carboxylase_FBP",
            "SBP",
            "ki_rubisco_carboxylase_SBP",
            "Orthophosphate",
            "ki_rubisco_carboxylase_Orthophosphate",
            "NADPH",
            "ki_rubisco_carboxylase_NADPH",
        ],
        stoichiometry={"RUBP": -1.0, "3PGA": 2.0},
    )
    m = m.add_reaction(
        "phosphoglycerate_kinase",
        fn=_rapid_equilibrium_2s_2p,
        args=[
            "3PGA",
            "ATP",
            "BPGA",
            "ADP",
            "kre_phosphoglycerate_kinase",
            "keq_phosphoglycerate_kinase",
        ],
        stoichiometry={"3PGA": -1.0, "ATP": -1.0, "BPGA": 1.0},
    )
    m = m.add_reaction(
        "gadph",
        fn=_rapid_equilibrium_3s_3p,
        args=[
            "BPGA",
            "NADPH",
            "protons",
            "GAP",
            "NADP",
            "Orthophosphate",
            "kre_gadph",
            "keq_gadph",
        ],
        stoichiometry={"NADPH": -1.0, "BPGA": -1.0, "GAP": 1.0},
    )
    m = m.add_reaction(
        "triose_phosphate_isomerase",
        fn=_rapid_equilibrium_1s_1p,
        args=[
            "GAP",
            "DHAP",
            "kre_triose_phosphate_isomerase",
            "keq_triose_phosphate_isomerase",
        ],
        stoichiometry={"GAP": -1, "DHAP": 1},
    )
    m = m.add_reaction(
        "aldolase_dhap_gap",
        fn=_rapid_equilibrium_2s_1p,
        args=[
            "GAP",
            "DHAP",
            "FBP",
            "kre_aldolase_dhap_gap",
            "keq_aldolase_dhap_gap",
        ],
        stoichiometry={"GAP": -1, "DHAP": -1, "FBP": 1},
    )
    m = m.add_reaction(
        "aldolase_dhap_e4p",
        fn=_rapid_equilibrium_2s_1p,
        args=[
            "DHAP",
            "E4P",
            "SBP",
            "kre_aldolase_dhap_e4p",
            "keq_aldolase_dhap_e4p",
        ],
        stoichiometry={"DHAP": -1, "E4P": -1, "SBP": 1},
    )
    m = m.add_reaction(
        "fbpase",
        fn=_michaelis_menten_1s_2i,
        args=[
            "FBP",
            "F6P",
            "Orthophosphate",
            "vmax_fbpase",
            "km_fbpase_s",
            "ki_fbpase_F6P",
            "ki_fbpase_Orthophosphate",
        ],
        stoichiometry={"FBP": -1, "F6P": 1},
    )
    m = m.add_reaction(
        "transketolase_gap_f6p",
        fn=_rapid_equilibrium_2s_2p,
        args=[
            "GAP",
            "F6P",
            "E4P",
            "X5P",
            "kre_transketolase_gap_f6p",
            "keq_transketolase_gap_f6p",
        ],
        stoichiometry={"GAP": -1, "F6P": -1, "E4P": 1, "X5P": 1},
    )
    m = m.add_reaction(
        "transketolase_gap_s7p",
        fn=_rapid_equilibrium_2s_2p,
        args=[
            "GAP",
            "S7P",
            "R5P",
            "X5P",
            "kre_transketolase_gap_s7p",
            "keq_transketolase_gap_s7p",
        ],
        stoichiometry={"GAP": -1, "S7P": -1, "R5P": 1, "X5P": 1},
    )
    m = m.add_reaction(
        "SBPase",
        fn=_michaelis_menten_1s_1i,
        args=[
            "SBP",
            "Orthophosphate",
            "vmax_SBPase",
            "km_SBPase_s",
            "ki_SBPase_Orthophosphate",
        ],
        stoichiometry={"SBP": -1, "S7P": 1},
    )
    m = m.add_reaction(
        "ribose_phosphate_isomerase",
        fn=_rapid_equilibrium_1s_1p,
        args=[
            "R5P",
            "RU5P",
            "kre_ribose_phosphate_isomerase",
            "keq_ribose_phosphate_isomerase",
        ],
        stoichiometry={"R5P": -1, "RU5P": 1},
    )
    m = m.add_reaction(
        "ribulose_phosphate_epimerase",
        fn=_rapid_equilibrium_1s_1p,
        args=[
            "X5P",
            "RU5P",
            "kre_ribulose_phosphate_epimerase",
            "keq_ribulose_phosphate_epimerase",
        ],
        stoichiometry={"X5P": -1, "RU5P": 1},
    )
    m = m.add_reaction(
        "phosphoribulokinase",
        fn=_rate_prk,
        args=[
            "RU5P",
            "ATP",
            "Orthophosphate",
            "3PGA",
            "RUBP",
            "ADP",
            "vmax_phosphoribulokinase",
            "km_phosphoribulokinase_RU5P",
            "km_phosphoribulokinase_ATP",
            "ki_phosphoribulokinase_3PGA",
            "ki_phosphoribulokinase_RUBP",
            "ki_phosphoribulokinase_Orthophosphate",
            "ki_phosphoribulokinase_4",
            "ki_phosphoribulokinase_5",
        ],
        stoichiometry={"RU5P": -1.0, "ATP": -1.0, "RUBP": 1.0},
    )
    m = m.add_reaction(
        "g6pi",
        fn=_rapid_equilibrium_1s_1p,
        args=["F6P", "G6P", "kre_g6pi", "keq_g6pi"],
        stoichiometry={"F6P": -1, "G6P": 1},
    )
    m = m.add_reaction(
        "phosphoglucomutase",
        fn=_rapid_equilibrium_1s_1p,
        args=["G6P", "G1P", "kre_phosphoglucomutase", "keq_phosphoglucomutase"],
        stoichiometry={"G6P": -1, "G1P": 1},
    )
    m = m.add_reaction(
        "ex_pga",
        fn=_rate_out,
        args=["3PGA", "N_translocator", "vmax_ex_pga", "km_ex_pga"],
        stoichiometry={"3PGA": -1},
    )
    m = m.add_reaction(
        "ex_gap",
        fn=_rate_out,
        args=["GAP", "N_translocator", "vmax_ex_pga", "km_ex_gap"],
        stoichiometry={"GAP": -1},
    )
    m = m.add_reaction(
        "ex_dhap",
        fn=_rate_out,
        args=["DHAP", "N_translocator", "vmax_ex_pga", "km_ex_dhap"],
        stoichiometry={"DHAP": -1},
    )
    m = m.add_reaction(
        "ex_g1p",
        fn=_rate_starch,
        args=[
            "G1P",
            "ATP",
            "ADP",
            "Orthophosphate",
            "3PGA",
            "F6P",
            "FBP",
            "vmax_ex_g1p",
            "km_ex_g1p_G1P",
            "km_ex_g1p_ATP",
            "ki_ex_g1p",
            "ki_ex_g1p_3PGA",
            "ki_ex_g1p_F6P",
            "ki_ex_g1p_FBP",
        ],
        stoichiometry={"G1P": -1.0, "ATP": -1.0},
    )
    m = m.add_surrogate(
        "ps2states",
        qss.Surrogate(
            model=_ps2states_2016_phd_surrogate,
            args=[
                "Plastoquinone (oxidised)",
                "Plastoquinone (reduced)",
                "PSII_cross_section",
                "Q",
                "PSII_total",
                "k2",
                "kF",
                "kH",
                "keq_Plastoquinone (reduced)",
                "kPQred",
                "PPFD",
                "kH0",
            ],
            outputs=["B0", "B1", "B2", "B3"],
        ),
    )
    return m
