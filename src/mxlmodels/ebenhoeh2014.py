"""Ebenhoeh 2014 photosynthetic electron transport chain (PETC) model.

Core dynamic model of the thylakoid electron transport chain used to study
short-term light acclimation (state transitions) in *Chlamydomonas*. Covers
PSII/PSI, cytochrome b6f, FNR, ATP synthase, cyclic electron flow, PTOX/NDH and
LHCII state transitions. Non-photochemical quenching is a constant base term
(kH0); the xanthophyll/PsbS machinery of the later Matuszynska models is absent.
ATP and NADPH are consumed by lumped linear sinks instead of a Calvin cycle.

This is the aerobic (constant oxygen supply) configuration; the anoxia switch of
the original code is not reproduced.

Reference: Ebenhoeh, Oliver, Geoffrey Fucile, Giovanni Finazzi,
Jean-David Rochaix, and Michel Goldschmidt-Clermont.
"Short-term acclimation of the photosynthetic electron transfer chain to
changing light: a mathematical model."
Philosophical Transactions of the Royal Society B 369.1640 (2014): 20130223.
doi:10.1098/rstb.2013.0223
"""

import math

import numpy as np
from mxlpy import Derived, Model
from mxlpy.surrogates import qss


def _moiety_1(
    concentration: float,
    total: float,
) -> float:
    """Conservation moiety: total - concentration."""
    return total - concentration


def _mass_action_1s(
    s1: float,
    k_fwd: float,
) -> float:
    """Mass-action rate for one substrate."""
    return k_fwd * s1


def _mass_action_2s(
    s1: float,
    s2: float,
    k_fwd: float,
) -> float:
    """Mass-action rate for two substrates."""
    return k_fwd * s1 * s2


def _dg_ph(
    r: float,
    t: float,
) -> float:
    """Thermodynamic coefficient dG/dpH = RT*ln(10) in kJ/mol."""
    return np.log(10) * r * t


def _ph_lumen(
    protons: float,
) -> float:
    """Lumenal pH from proton concentration in mmol/mmol_Chl (conversion factor 0.00025)."""
    return -np.log10(protons * 0.00025)


def _protons_stroma(
    ph: float,
) -> float:
    """Convert a pH value to proton concentration (4000 * 10^-pH)."""
    return 4000.0 * 10 ** (-ph)


def _keq_pq_red(
    e0_qa: float,
    f: float,
    e0_pq: float,
    p_hstroma: float,
    d_g_p_h: float,
    rt: float,
) -> float:
    """Equilibrium constant for PQ reduction by QA, pH-corrected via stroma proton contribution."""
    dg1 = -e0_qa * f
    dg2 = -2 * e0_pq * f
    dg = -2 * dg1 + dg2 + 2 * p_hstroma * d_g_p_h
    return np.exp(-dg / rt)


def _ps2_crosssection(
    lhc: float,
    static_ant_ii: float,
    static_ant_i: float,
) -> float:
    """Relative absorption cross-section of PSII as a function of mobile LHC association."""
    return static_ant_ii + (1 - static_ant_ii - static_ant_i) * lhc


def _keq_atp(
    p_h: float,
    delta_g0_atp: float,
    d_g_p_h: float,
    hpr: float,
    p_hstroma: float,
    pi_mol: float,
    rt: float,
) -> float:
    """Equilibrium constant for ATP synthase, driven by the transmembrane proton gradient."""
    delta_g = delta_g0_atp - d_g_p_h * hpr * (p_hstroma - p_h)
    return pi_mol * math.exp(-delta_g / rt)


def _keq_cytb6f(
    p_h: float,
    f: float,
    e0_pq: float,
    e0_pc: float,
    p_hstroma: float,
    rt: float,
    d_g_p_h: float,
) -> float:
    """Equilibrium constant of cytochrome b6f from redox potentials and transmembrane pH gradient."""
    DG1 = -2 * f * e0_pq
    DG2 = -f * e0_pc
    DG = -(DG1 + 2 * d_g_p_h * p_h) + 2 * DG2 + 2 * d_g_p_h * (p_hstroma - p_h)
    return math.exp(-DG / rt)


def _keq_fnr(
    e0_fd: float,
    f: float,
    e0_nadp: float,
    p_hstroma: float,
    d_g_p_h: float,
    rt: float,
) -> float:
    """Equilibrium constant for FNR: Fd-mediated NADP+ reduction, pH-corrected."""
    dg1 = -e0_fd * f
    dg2 = -2 * e0_nadp * f
    dg = -2 * dg1 + dg2 + d_g_p_h * p_hstroma
    return math.exp(-dg / rt)


def _keq_pcp700(
    e0_pc: float,
    f: float,
    eo_p700: float,
    rt: float,
) -> float:
    """Equilibrium constant for PC -> P700 electron transfer from standard redox potentials."""
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
    """Equilibrium constant for FA -> Fd electron transfer from standard redox potentials."""
    dg1 = -e0_fa * f
    dg2 = -e0_fd * f
    dg = -dg1 + dg2
    return math.exp(-dg / rt)


def _ps1states(
    pc_ox: float,
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
    """QSSA open state of PSI as a function of PC and Fd redox states."""
    L = (1 - ps2cs) * pfd
    return psi_tot / (
        1
        + L / (k_fd_red * fd_ox)
        + (1 + fd_red / (keq_fafd * fd_ox))
        * (pc_ox / (keq_pcp700 * pc_red) + L / (k_pc_ox * pc_red))
    )


def _rate_atp_synthase(
    atp: float,
    adp: float,
    keq_atp_synthase: float,
    k_atp_synth: float,
) -> float:
    """ATP synthase rate: linear reversible kinetics driven by the pH-dependent Keq."""
    return k_atp_synth * (adp - atp / keq_atp_synthase)


def _neg_div(
    x: float,
    y: float,
) -> float:
    """Return -x / y."""
    return -x / y


def _neg_one_div_by(
    x: float,
) -> float:
    """Return -1/x; negated unit stoichiometry scaled by buffering capacity."""
    return -1.0 / x


def _two_div_by(
    x: float,
) -> float:
    """Return 2/x; 2-proton stoichiometry of PSII scaled by buffering capacity."""
    return 2.0 / x


def _four_div_by(
    x: float,
) -> float:
    """Return 4/x; 4-proton stoichiometry of b6f scaled by buffering capacity."""
    return 4.0 / x


def _b6f(
    pc_ox: float,
    pq_ox: float,
    pq_red: float,
    pc_red: float,
    keq_b6f: float,
    k_cytb6f: float,
) -> float:
    """Cytochrome b6f rate: reversible mass action clamped to -kCytb6f to avoid runaway reverse flux."""
    return max(
        k_cytb6f * (pq_red * pc_ox**2 - pq_ox * pc_red**2 / keq_b6f),
        -k_cytb6f,
    )


def _rate_cyclic_electron_flow(
    pox: float,
    fdred: float,
    kcyc: float,
) -> float:
    """Cyclic electron flow rate: mass action on Fd_red^2 and PQ_ox."""
    return kcyc * fdred**2 * pox


def _rate_fnr(
    fd_ox: float,
    fd_red: float,
    nadph: float,
    nadp: float,
    vmax: float,
    km_fd_red: float,
    km_nadph: float,
    keq: float,
) -> float:
    """FNR rate: reversible convenience kinetics with Fd^2 stoichiometry (mmol/mmol_Chl units)."""
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
    """PSII electron transfer rate from the open-excited state B1 and photochemistry rate constant k2."""
    return 0.5 * k2 * b1


def _rate_ps1(
    a: float,
    ps2cs: float,
    pfd: float,
) -> float:
    """PSI electron transfer rate: open PSI centres (a) * light absorbed by PSI antenna."""
    return (1 - ps2cs) * pfd * a


def _rate_leak(
    protons_lumen: float,
    ph_stroma: float,
    k_leak: float,
) -> float:
    """Passive proton leak across the thylakoid membrane, proportional to the proton gradient."""
    return k_leak * (protons_lumen - _protons_stroma(ph_stroma))


def _rate_state_transition_ps1_ps2(
    ant: float,
    pox: float,
    p_tot: float,
    k_stt7: float,
    km_st: float,
    n_st: float,
) -> float:
    """STT7-kinase phosphorylation of LHC; inhibited by oxidised PQ (state 1 -> 2 transition)."""
    return k_stt7 * (1 / (1 + (pox / p_tot / km_st) ** n_st)) * ant


def _ps2states_surrogate(
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
    """PSII state populations via the analytical closed-form of the 4-state QSSA.

    Total non-radiative de-excitation rate is k_h0 + _kh*quencher; for the
    Ebenhoeh 2014 model the variable quencher is zero, leaving the constant base
    quenching k_h0.
    """
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


def _div(
    x: float,
    y: float,
) -> float:
    """Return x / y."""
    return x / y


def _rate_fluorescence(
    b0: float,
    b2: float,
    ps2cs: float,
    k2: float,
    k_f: float,
    k_h0: float,
    k_h: float,
) -> float:
    """Chlorophyll fluorescence yield from open (B0) and closed (B2) PSII centres."""
    return ps2cs * k_f * b0 / (k_f + k_h0 + k2 + k_h) + ps2cs * k_f * b2 / (
        k_f + k_h0 + k_h
    )


def get_ebenhoeh2014() -> Model:
    """Ebenhoeh 2014 photosynthetic electron transport chain (PETC) model.

    Core dynamic model of the thylakoid electron transport chain used to study
    short-term light acclimation (state transitions) in *Chlamydomonas*. Covers
    PSII/PSI, cytochrome b6f, FNR, ATP synthase, cyclic electron flow, PTOX/NDH
    and LHCII state transitions, with constant base non-photochemical quenching
    and lumped linear ATP/NADPH consumption (aerobic configuration).

    Reference: Ebenhoeh, Oliver, Geoffrey Fucile, Giovanni Finazzi,
    Jean-David Rochaix, and Michel Goldschmidt-Clermont.
    "Short-term acclimation of the photosynthetic electron transfer chain to
    changing light: a mathematical model."
    Philosophical Transactions of the Royal Society B 369.1640 (2014): 20130223.
    """
    m: Model = Model()
    # Dynamic variables (initial values from the original main.ipynb)
    m = m.add_variable("Plastoquinone (oxidised)", initial_value=17.5)
    m = m.add_variable("Plastocyanine (oxidised)", initial_value=0.0202)
    m = m.add_variable("Ferredoxine (oxidised)", initial_value=5.0)
    m = m.add_variable("ATP", initial_value=0.0)
    m = m.add_variable("NADPH", initial_value=0.0)
    m = m.add_variable("protons_lumen", initial_value=4000.0 * 10 ** (-7.2))
    m = m.add_variable("Light-harvesting complex", initial_value=0.9)
    # Stroma pH (constant) and light
    m = m.add_parameter("pH", value=7.8)
    m = m.add_parameter("PPFD", value=100.0)
    # Total pool sizes
    m = m.add_parameter("PSII_total", value=2.5)
    m = m.add_parameter("PSI_total", value=2.5)
    m = m.add_parameter("PQ_tot", value=17.5)
    m = m.add_parameter("PC_tot", value=4.0)
    m = m.add_parameter("Fd*", value=5.0)
    m = m.add_parameter("NADP*", value=25.0)
    m = m.add_parameter("A*P", value=60.0)
    m = m.add_parameter("LHC_tot", value=1.0)
    # Physical constants
    m = m.add_parameter("F", value=96.485)
    m = m.add_parameter("R", value=0.0083)
    m = m.add_parameter("T", value=298.0)
    m = m.add_parameter("bH", value=100.0)
    # Standard redox potentials (V, at pH 0)
    m = m.add_parameter("E^0_QA", value=-0.14)
    m = m.add_parameter("E^0_PQ", value=0.354)
    m = m.add_parameter("E^0_PC", value=0.38)
    m = m.add_parameter("E^0_P700", value=0.48)
    m = m.add_parameter("E^0_FA", value=-0.55)
    m = m.add_parameter("E^0_Fd", value=-0.43)
    m = m.add_parameter("E^0_NADP", value=-0.113)
    # PSII rate constants
    m = m.add_parameter("kH0", value=500000000.0)
    m = m.add_parameter("kH", value=0.0)
    m = m.add_parameter("kF", value=62500000.0)
    m = m.add_parameter("k2", value=5000000000.0)
    m = m.add_parameter("Q", value=0.0)
    # Cross-section partitioning
    m = m.add_parameter("staticAntI", value=0.2)
    m = m.add_parameter("staticAntII", value=0.0)
    # ATP synthase / consumption
    m = m.add_parameter("kf_atp_synthase", value=20.0)
    m = m.add_parameter("kf_atp_consumption", value=10.0)
    m = m.add_parameter("Pi_mol", value=0.01)
    m = m.add_parameter("DeltaG0_ATP", value=30.6)
    m = m.add_parameter("HPR", value=14.0 / 3.0)
    m = m.add_parameter("kf_nadph_consumption", value=15.0)
    # Proton leak / membrane
    m = m.add_parameter("kf_proton_leak", value=0.01)
    # Electron transfer rate constants
    m = m.add_parameter("kPQred", value=250.0)
    m = m.add_parameter("kcat_b6f", value=2.5)
    m = m.add_parameter("kPTOX", value=0.01)
    m = m.add_parameter("kPCox", value=2500.0)
    m = m.add_parameter("kFdred", value=250000.0)
    m = m.add_parameter("kcat_fnr", value=500.0)
    m = m.add_parameter("E0_fnr", value=3.0)
    m = m.add_parameter("km_fnr_Ferredoxine (reduced)", value=1.56)
    m = m.add_parameter("km_fnr_NADP", value=0.22)
    m = m.add_parameter("kf_cyclic_electron_flow", value=1.0)
    m = m.add_parameter("O2 (dissolved)_lumen", value=8.0)
    m = m.add_parameter("kf_ndh", value=0.002)
    # State transitions
    m = m.add_parameter("kStt7", value=0.0035)
    m = m.add_parameter("kPph1", value=0.0013)
    m = m.add_parameter("km_lhc_state_transition_12", value=0.2)
    m = m.add_parameter("n_ST", value=2.0)
    # Derived quantities
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
        "Plastoquinone (reduced)",
        fn=_moiety_1,
        args=["Plastoquinone (oxidised)", "PQ_tot"],
    )
    m = m.add_derived(
        "Plastocyanine (reduced)",
        fn=_moiety_1,
        args=["Plastocyanine (oxidised)", "PC_tot"],
    )
    m = m.add_derived(
        "Ferredoxine (reduced)",
        fn=_moiety_1,
        args=["Ferredoxine (oxidised)", "Fd*"],
    )
    m = m.add_derived(
        "ADP",
        fn=_moiety_1,
        args=["ATP", "A*P"],
    )
    m = m.add_derived(
        "NADP",
        fn=_moiety_1,
        args=["NADPH", "NADP*"],
    )
    m = m.add_derived(
        "Light-harvesting complex (protonated)",
        fn=_moiety_1,
        args=["Light-harvesting complex", "LHC_tot"],
    )
    m = m.add_derived(
        "PSII_cross_section",
        fn=_ps2_crosssection,
        args=["Light-harvesting complex", "staticAntII", "staticAntI"],
    )
    m = m.add_derived(
        "keq_Plastoquinone (reduced)",
        fn=_keq_pq_red,
        args=["E^0_QA", "F", "E^0_PQ", "pH", "dG_pH", "RT"],
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
        fn=_ps1states,
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
    # Reactions
    m = m.add_reaction(
        "PSII",
        fn=_rate_ps2,
        args=["B1", "k2"],
        stoichiometry={
            "Plastoquinone (oxidised)": -1,
            "protons_lumen": Derived(fn=_two_div_by, args=["bH"]),
        },
    )
    m = m.add_reaction(
        "PSI",
        fn=_rate_ps1,
        args=["A1", "PSII_cross_section", "PPFD"],
        stoichiometry={
            "Ferredoxine (oxidised)": -1,
            "Plastocyanine (oxidised)": 1,
        },
    )
    m = m.add_reaction(
        "PTOX",
        fn=_mass_action_2s,
        args=["Plastoquinone (reduced)", "O2 (dissolved)_lumen", "kPTOX"],
        stoichiometry={"Plastoquinone (oxidised)": 1},
    )
    m = m.add_reaction(
        "ndh",
        fn=_mass_action_1s,
        args=["Plastoquinone (oxidised)", "kf_ndh"],
        stoichiometry={"Plastoquinone (oxidised)": -1},
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
            "protons_lumen": Derived(fn=_four_div_by, args=["bH"]),
        },
    )
    m = m.add_reaction(
        "cyclic_electron_flow",
        fn=_rate_cyclic_electron_flow,
        args=[
            "Plastoquinone (oxidised)",
            "Ferredoxine (reduced)",
            "kf_cyclic_electron_flow",
        ],
        stoichiometry={
            "Plastoquinone (oxidised)": -1,
            "Ferredoxine (oxidised)": 2,
        },
    )
    m = m.add_reaction(
        "fnr",
        fn=_rate_fnr,
        args=[
            "Ferredoxine (oxidised)",
            "Ferredoxine (reduced)",
            "NADPH",
            "NADP",
            "vmax_fnr",
            "km_fnr_Ferredoxine (reduced)",
            "km_fnr_NADP",
            "keq_fnr",
        ],
        stoichiometry={
            "Ferredoxine (oxidised)": 2,
            "NADPH": 1,
        },
    )
    m = m.add_reaction(
        "proton_leak",
        fn=_rate_leak,
        args=["protons_lumen", "pH", "kf_proton_leak"],
        stoichiometry={"protons_lumen": Derived(fn=_neg_one_div_by, args=["bH"])},
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
        "atp_synthase",
        fn=_rate_atp_synthase,
        args=["ATP", "ADP", "keq_atp_synthase", "kf_atp_synthase"],
        stoichiometry={
            "ATP": 1.0,
            "protons_lumen": Derived(fn=_neg_div, args=["HPR", "bH"]),
        },
    )
    m = m.add_reaction(
        "atp_consumption",
        fn=_mass_action_1s,
        args=["ATP", "kf_atp_consumption"],
        stoichiometry={"ATP": -1},
    )
    m = m.add_reaction(
        "nadph_consumption",
        fn=_mass_action_1s,
        args=["NADPH", "kf_nadph_consumption"],
        stoichiometry={"NADPH": -1},
    )
    m = m.add_surrogate(
        "ps2states",
        qss.Surrogate(
            model=_ps2states_surrogate,
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
    m = m.add_readout(
        "PQ_red/tot",
        fn=_div,
        args=["Plastoquinone (reduced)", "PQ_tot"],
    )
    m = m.add_readout(
        "Fd_red/tot",
        fn=_div,
        args=["Ferredoxine (reduced)", "Fd*"],
    )
    m = m.add_readout(
        "PC_red/tot",
        fn=_div,
        args=["Plastocyanine (reduced)", "PC_tot"],
    )
    m = m.add_readout(
        "NADPH/tot",
        fn=_div,
        args=["NADPH", "NADP*"],
    )
    m = m.add_readout(
        "ATP/tot",
        fn=_div,
        args=["ATP", "A*P"],
    )
    m = m.add_readout(
        "Fluo",
        fn=_rate_fluorescence,
        args=["B0", "B2", "PSII_cross_section", "k2", "kF", "kH0", "kH"],
    )
    return m  # noqa: RET504
