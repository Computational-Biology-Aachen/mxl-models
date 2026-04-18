import math

import numpy as np
from mxlpy.surrogates import qss

from mxlpy import Derived, Model


def _protons_stroma(ph: float) -> float:
    return 4000.0 * 10 ** (-ph)


def _moiety_1(concentration: float, total: float) -> float:
    return total - concentration


def _mass_action_1s(s1: float, k_fwd: float) -> float:
    return k_fwd * s1


def _dg_ph(r: float, t: float) -> float:
    return np.log(10) * r * t


def _ph_lumen(protons: float) -> float:
    return -np.log10(protons * 0.00025)


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


def _rate_atp_synthase_2016(
    ATP: float,
    ADP: float,
    Keq_ATPsynthase: float,
    kATPsynth: float,
) -> float:
    return kATPsynth * (ADP - ATP / Keq_ATPsynthase)


def neg_div(x: float, y: float) -> float:
    return -x / y


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


def _four_div_by(x: float) -> float:
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


def _rate_fnr2016(
    fd_ox: float,
    fd_red: float,
    nadph: float,
    nadp: float,
    vmax: float,
    km_fd_red: float,
    km_nadph: float,
    keq: float,
) -> float:
    fdred = fd_red / km_fd_red
    fdox = fd_ox / km_fd_red
    nadph = nadph / km_nadph
    nadp = nadp / km_nadph
    return (
        vmax
        * (fdred**2 * nadp - fdox**2 * nadph / keq)
        / ((1 + fdred + fdred**2) * (1 + nadp) + (1 + fdox + fdox**2) * (1 + nadph) - 1)
    )


def _rate_ps2(
    b1: float,
    k2: float,
) -> float:
    return 0.5 * k2 * b1


def _two_div_by(x: float) -> float:
    return 2.0 / x


def _rate_ps1(
    a: float,
    ps2cs: float,
    pfd: float,
) -> float:
    return (1 - ps2cs) * pfd * a


def _rate_leak(
    protons_lumen: float,
    ph_stroma: float,
    k_leak: float,
) -> float:
    return k_leak * (protons_lumen - _protons_stroma(ph_stroma))


def _neg_one_div_by(x: float) -> float:
    return -1.0 / x


def mass_action_2s(s1: float, s2: float, k_fwd: float) -> float:
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


def _ps2states_2016b_analytic(
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
    return (
        Model()
        .add_variable("atp", initial_value=1.6999999999999997)
        .add_variable("pq_ox", initial_value=4.706348349506148)
        .add_variable("pc_ox", initial_value=3.9414515288091567)
        .add_variable("fd_ox", initial_value=3.7761613271207324)
        .add_variable("protons_lumen", initial_value=7.737821100836988)
        .add_variable("lhc", initial_value=0.5105293511676007)
        .add_variable("psbs_de", initial_value=0.5000000001374878)
        .add_variable("vx", initial_value=0.09090909090907397)
        .add_parameter("pH", value=7.9)
        .add_parameter("PPFD", value=100.0)
        .add_parameter("nadph", value=0.6)
        .add_parameter("O2_lumen", value=8.0)
        .add_parameter("bH", value=100.0)
        .add_parameter("F", value=96.485)
        .add_parameter("E^0_PC", value=0.38)
        .add_parameter("E^0_P700", value=0.48)
        .add_parameter("E^0_FA", value=-0.55)
        .add_parameter("E^0_Fd", value=-0.43)
        .add_parameter("E^0_NADP", value=-0.113)
        .add_parameter("NADP*", value=0.8)
        .add_parameter("R", value=0.0083)
        .add_parameter("T", value=298.0)
        .add_parameter("A*P", value=2.55)
        .add_parameter("Carotenoids_tot", value=1.0)
        .add_parameter("Fd*", value=5.0)
        .add_parameter("PC_tot", value=4.0)
        .add_parameter("PSBS_tot", value=1.0)
        .add_parameter("LHC_tot", value=1.0)
        .add_parameter("gamma0", value=0.1)
        .add_parameter("gamma1", value=0.25)
        .add_parameter("gamma2", value=0.6)
        .add_parameter("gamma3", value=0.15)
        .add_parameter("kZSat", value=0.12)
        .add_parameter("E^0_QA", value=-0.14)
        .add_parameter("E^0_PQ", value=0.354)
        .add_parameter("PQ_tot", value=17.5)
        .add_parameter("staticAntII", value=0.1)
        .add_parameter("staticAntI", value=0.37)
        .add_parameter("kf_atp_synthase", value=20.0)
        .add_parameter("HPR", value=4.666666666666667)
        .add_parameter("Pi_mol", value=0.01)
        .add_parameter("DeltaG0_ATP", value=30.6)
        .add_parameter("kcat_b6f", value=2.5)
        .add_parameter("kh_lhc_protonation", value=3.0)
        .add_parameter("kf_lhc_protonation", value=0.0096)
        .add_parameter("ksat_lhc_protonation", value=5.8)
        .add_parameter("kf_lhc_deprotonation", value=0.0096)
        .add_parameter("kf_cyclic_electron_flow", value=1.0)
        .add_parameter("kf_violaxanthin_deepoxidase", value=0.0024)
        .add_parameter("kh_violaxanthin_deepoxidase", value=5.0)
        .add_parameter("ksat_violaxanthin_deepoxidase", value=5.8)
        .add_parameter("kf_zeaxanthin_epoxidase", value=0.00024)
        .add_parameter("E0_fnr", value=3.0)
        .add_parameter("kcat_fnr", value=500.0)
        .add_parameter("km_fnr_fd_red", value=1.56)
        .add_parameter("km_fnr_nadp", value=0.22)
        .add_parameter("kf_ndh", value=0.002)
        .add_parameter("PSII_total", value=2.5)
        .add_parameter("PSI_total", value=2.5)
        .add_parameter("kH0", value=500000000.0)
        .add_parameter("kPQred", value=250.0)
        .add_parameter("kPCox", value=2500.0)
        .add_parameter("kFdred", value=250000.0)
        .add_parameter("k2", value=5000000000.0)
        .add_parameter("kH", value=5000000000.0)
        .add_parameter("kF", value=625000000.0)
        .add_parameter("convf", value=0.032)
        .add_parameter("kf_proton_leak", value=10.0)
        .add_parameter("kPTOX", value=0.01)
        .add_parameter("kStt7", value=0.0035)
        .add_parameter("km_lhc_state_transition_12", value=0.2)
        .add_parameter("n_ST", value=2.0)
        .add_parameter("kPph1", value=0.0013)
        .add_parameter("kf_ex_atp", value=10.0)
        .add_derived(
            "nadp",
            fn=_moiety_1,
            args=["nadph", "NADP*"],
        )
        .add_derived(
            "RT",
            fn=_mass_action_1s,
            args=["R", "T"],
        )
        .add_derived(
            "adp",
            fn=_moiety_1,
            args=["atp", "A*P"],
        )
        .add_derived(
            "dG_pH",
            fn=_dg_ph,
            args=["R", "T"],
        )
        .add_derived(
            "pH_lumen",
            fn=_ph_lumen,
            args=["protons_lumen"],
        )
        .add_derived(
            "zx",
            fn=_moiety_1,
            args=["vx", "Carotenoids_tot"],
        )
        .add_derived(
            "fd_red",
            fn=_moiety_1,
            args=["fd_ox", "Fd*"],
        )
        .add_derived(
            "pc_red",
            fn=_moiety_1,
            args=["pc_ox", "PC_tot"],
        )
        .add_derived(
            "psbs_pr",
            fn=_moiety_1,
            args=["psbs_de", "PSBS_tot"],
        )
        .add_derived(
            "lhc_prot",
            fn=_moiety_1,
            args=["lhc", "LHC_tot"],
        )
        .add_derived(
            "Q",
            fn=_quencher,
            args=[
                "psbs_de",
                "vx",
                "psbs_pr",
                "zx",
                "gamma0",
                "gamma1",
                "gamma2",
                "gamma3",
                "kZSat",
            ],
        )
        .add_derived(
            "keq_pq_red",
            fn=_keq_pq_red,
            args=["E^0_QA", "F", "E^0_PQ", "pH", "dG_pH", "RT"],
        )
        .add_derived(
            "pq_red",
            fn=_moiety_1,
            args=["pq_ox", "PQ_tot"],
        )
        .add_derived(
            "PSII_cross_section",
            fn=_ps2_crosssection,
            args=["lhc", "staticAntII", "staticAntI"],
        )
        .add_derived(
            "keq_atp_synthase",
            fn=_keq_atp,
            args=["pH_lumen", "DeltaG0_ATP", "dG_pH", "HPR", "pH", "Pi_mol", "RT"],
        )
        .add_derived(
            "keq_b6f",
            fn=_keq_cytb6f,
            args=["pH_lumen", "F", "E^0_PQ", "E^0_PC", "pH", "RT", "dG_pH"],
        )
        .add_derived(
            "keq_fnr",
            fn=_keq_fnr,
            args=["E^0_Fd", "F", "E^0_NADP", "pH", "dG_pH", "RT"],
        )
        .add_derived(
            "vmax_fnr",
            fn=_mass_action_1s,
            args=["kcat_fnr", "E0_fnr"],
        )
        .add_derived(
            "keq_PCP700",
            fn=_keq_pcp700,
            args=["E^0_PC", "F", "E^0_P700", "RT"],
        )
        .add_derived(
            "keq_ferredoxin_reductase",
            fn=_keq_faf_d,
            args=["E^0_FA", "F", "E^0_Fd", "RT"],
        )
        .add_derived(
            "A1",
            fn=_ps1states_2019,
            args=[
                "pc_ox",
                "pc_red",
                "fd_ox",
                "fd_red",
                "PSII_cross_section",
                "PSI_total",
                "kFdred",
                "keq_ferredoxin_reductase",
                "keq_PCP700",
                "kPCox",
                "PPFD",
            ],
        )
        .add_reaction(
            "atp_synthase",
            fn=_rate_atp_synthase_2016,
            args=["atp", "adp", "keq_atp_synthase", "kf_atp_synthase"],
            stoichiometry={
                "atp": 1.0,
                "protons_lumen": Derived(fn=neg_div, args=["HPR", "bH"]),
            },
        )
        .add_reaction(
            "b6f",
            fn=_b6f,
            args=["pc_ox", "pq_ox", "pq_red", "pc_red", "keq_b6f", "kcat_b6f"],
            stoichiometry={
                "pc_ox": -2,
                "pq_ox": 1,
                "protons_lumen": Derived(fn=_four_div_by, args=["bH"]),
            },
        )
        .add_reaction(
            "lhc_protonation",
            fn=_protonation_hill,
            args=[
                "psbs_de",
                "protons_lumen",
                "kh_lhc_protonation",
                "kf_lhc_protonation",
                "ksat_lhc_protonation",
            ],
            stoichiometry={"psbs_de": -1},
        )
        .add_reaction(
            "lhc_deprotonation",
            fn=_mass_action_1s,
            args=["psbs_pr", "kf_lhc_deprotonation"],
            stoichiometry={"psbs_de": 1},
        )
        .add_reaction(
            "cyclic_electron_flow",
            fn=_rate_cyclic_electron_flow,
            args=["pq_ox", "fd_red", "kf_cyclic_electron_flow"],
            stoichiometry={"pq_ox": -1, "fd_ox": 2},
        )
        .add_reaction(
            "violaxanthin_deepoxidase",
            fn=_rate_protonation_hill,
            args=[
                "vx",
                "protons_lumen",
                "kf_violaxanthin_deepoxidase",
                "kh_violaxanthin_deepoxidase",
                "ksat_violaxanthin_deepoxidase",
            ],
            stoichiometry={"vx": -1},
        )
        .add_reaction(
            "zeaxanthin_epoxidase",
            fn=_mass_action_1s,
            args=["zx", "kf_zeaxanthin_epoxidase"],
            stoichiometry={"vx": 1},
        )
        .add_reaction(
            "fnr",
            fn=_rate_fnr2016,
            args=[
                "fd_ox",
                "fd_red",
                "nadph",
                "nadp",
                "vmax_fnr",
                "km_fnr_fd_red",
                "km_fnr_nadp",
                "keq_fnr",
            ],
            stoichiometry={"fd_ox": 2},
        )
        .add_reaction(
            "ndh",
            fn=_mass_action_1s,
            args=["pq_ox", "kf_ndh"],
            stoichiometry={"pq_ox": -1},
        )
        .add_reaction(
            "PSII",
            fn=_rate_ps2,
            args=["B1", "k2"],
            stoichiometry={
                "pq_ox": -1,
                "protons_lumen": Derived(fn=_two_div_by, args=["bH"]),
            },
        )
        .add_reaction(
            "PSI",
            fn=_rate_ps1,
            args=["A1", "PSII_cross_section", "PPFD"],
            stoichiometry={"fd_ox": -1, "pc_ox": 1},
        )
        .add_reaction(
            "proton_leak",
            fn=_rate_leak,
            args=["protons_lumen", "pH", "kf_proton_leak"],
            stoichiometry={"protons_lumen": Derived(fn=_neg_one_div_by, args=["bH"])},
        )
        .add_reaction(
            "PTOX",
            fn=mass_action_2s,
            args=["pq_red", "O2_lumen", "kPTOX"],
            stoichiometry={"pq_ox": 1},
        )
        .add_reaction(
            "lhc_state_transition_12",
            fn=_rate_state_transition_ps1_ps2,
            args=[
                "lhc",
                "pq_ox",
                "PQ_tot",
                "kStt7",
                "km_lhc_state_transition_12",
                "n_ST",
            ],
            stoichiometry={"lhc": -1},
        )
        .add_reaction(
            "lhc_state_transition_21",
            fn=_mass_action_1s,
            args=["lhc_prot", "kPph1"],
            stoichiometry={"lhc": 1},
        )
        .add_reaction(
            "ex_atp",
            fn=_mass_action_1s,
            args=["atp", "kf_ex_atp"],
            stoichiometry={"atp": -1},
        )
        .add_surrogate(
            "ps2states",
            qss.Surrogate(
                model=_ps2states_2016b_analytic,
                args=[
                    "pq_ox",
                    "pq_red",
                    "PSII_cross_section",
                    "Q",
                    "PSII_total",
                    "k2",
                    "kF",
                    "kH",
                    "keq_pq_red",
                    "kPQred",
                    "PPFD",
                    "kH0",
                ],
                outputs=["B0", "B1", "B2", "B3"],
            ),
        )
    )
