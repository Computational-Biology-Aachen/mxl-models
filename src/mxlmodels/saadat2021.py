"""Saadat 2021 chloroplast model with ROS scavenging, Calvin cycle, and thioredoxin regulation.

Reference: Saadat, Nima P., et al.
"Computational analysis of alternative photosynthetic electron flows linked with oxidative stress."
Frontiers in plant science 12 (2021): 750580
"""

import math

import numpy as np
from mxlpy import Derived, Model
from mxlpy.surrogates import qss


def _mass_action_1s(
    s1: float,
    k_fwd: float,
) -> float:
    """Mass-action rate for one substrate."""
    return k_fwd * s1


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


def _moiety_1(
    concentration: float,
    total: float,
) -> float:
    """Conservation moiety: total - concentration."""
    return total - concentration


def _quencher(
    psbs: float,
    vx: float,
    psbsp: float,
    zx: float,
    y0: float,
    y1: float,
    y2: float,
    y3: float,
    k_z_sat: float,
) -> float:
    """co-operative 4-state quenching mechanism.

    gamma0: slow quenching of (Vx - protonation)
    gamma1: fast quenching (Vx + protonation)
    gamma2: fastest possible quenching (Zx + protonation)
    gamma3: slow quenching of Zx present (Zx - protonation).
    """
    ZAnt = zx / (zx + k_z_sat)
    return y0 * vx * psbs + y1 * vx * psbsp + y2 * ZAnt * psbsp + y3 * ZAnt * psbs


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
    """Equilibrium constant for PQ reduction by QA, pH-corrected via stroma proton contribution."""
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
    """Free orthophosphate from total minus all phosphorylated CBB intermediates (bisphosphates count twice)."""
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


def _moiety_2(
    x1: float,
    x2: float,
    total: float,
) -> float:
    """Conservation moiety: total - x1 - x2."""
    return total - x1 - x2


def _glutathion_moiety(
    gssg: float,
    gs_total: float,
) -> float:
    """Conservation moiety: GSH = total - 2*GSSG (one GSSG releases two GSH)."""
    return gs_total - 2 * gssg


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


def _mul(
    x: float,
    y: float,
) -> float:
    """Calculate the product of two values.

    Parameters
    ----------
    x
        First factor
    y
        Second factor

    Returns
    -------
    Float
        Product of x and y (x * y)

    Examples
    --------
    >>> mul(2.0, 3.0)
    6.0
    >>> mul(0.5, 4.0)
    2.0

    """
    return x * y


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
    """Denominator term N for the phosphate translocator shared by all triose-P export reactions."""
    return 1 + (1 + k_pxt / p_ext) * (
        pi / k_pi + pga / k_pga + gap / k_gap + dhap / k_dhap
    )


def _mass_action_2s(
    s1: float,
    s2: float,
    k_fwd: float,
) -> float:
    """Mass-action rate for two substrates."""
    return k_fwd * s1 * s2


def _rate_atp_synthase_2019(
    atp: float,
    adp: float,
    keq_at_psynthase: float,
    k_at_psynth: float,
    convf: float,
) -> float:
    """ATP synthase rate (2019 formulation): same as 2016 but ADP/ATP scaled by convf."""
    return k_at_psynth * (adp / convf - atp / convf / keq_at_psynthase)


def _neg_div(
    x: float,
    y: float,
) -> float:
    """Return -x / y."""
    return -x / y


def _value(
    x: float,
) -> float:
    """Return x unchanged."""
    return x


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


def _four_div_by(
    x: float,
) -> float:
    """Return 4/x; used for the 4-proton stoichiometry of b6f scaled by buffering capacity."""
    return 4.0 / x


def _protons_stroma_2016(
    ph: float,
) -> float:
    """Convert stromal pH to proton concentration (µmol/L).

    Introduced by the Matuszynska 2016 PhD model.
    """
    return 4000.0 * 10 ** (-ph)


def _protonation_hill(
    vx: float,
    h: float,
    nh: float,
    k_fwd: float,
    k_ph_sat: float,
) -> float:
    """Hill-type protonation rate scaled by lumenal proton concentration."""
    return k_fwd * (h**nh / (h**nh + _protons_stroma_2016(k_ph_sat) ** nh)) * vx  # type: ignore


def _rate_cyclic_electron_flow(
    pox: float,
    fdred: float,
    kcyc: float,
) -> float:
    """Cyclic electron flow rate: mass action on Fd_red^2 and PQ_ox."""
    return kcyc * fdred**2 * pox


def _rate_protonation_hill(
    vx: float,
    h: float,
    k_fwd: float,
    n_h: float,
    kph_sat: float,
) -> float:
    """Hill-type deepoxidase rate activated by lumenal proton concentration."""
    return k_fwd * (h**n_h / (h**n_h + _protons_stroma_2016(kph_sat) ** n_h)) * vx  # type: ignore


def _rate_fnr_2019(
    fd_ox: float,
    fd_red: float,
    nadph: float,
    nadp: float,
    km_fnr_f: float,
    km_fnr_n: float,
    vmax: float,
    keq_fnr: float,
    convf: float,
) -> float:
    """FNR rate (2019 formulation): same as 2016 but NADP/H concentrations scaled by convf."""
    fdred = fd_red / km_fnr_f
    fdox = fd_ox / km_fnr_f
    nadph = nadph / convf / km_fnr_n
    nadp = nadp / convf / km_fnr_n
    return (
        vmax
        * (fdred**2 * nadp - fdox**2 * nadph / keq_fnr)
        / ((1 + fdred + fdred**2) * (1 + nadp) + (1 + fdox + fdox**2) * (1 + nadph) - 1)
    )


def _rate_ps2(
    b1: float,
    k2: float,
) -> float:
    """PSII electron transfer rate from the open-excited state B1 and photochemistry rate constant k2."""
    return 0.5 * k2 * b1


def _two_div_by(
    x: float,
) -> float:
    """Return 2/x; used for the 2-proton stoichiometry of PSII scaled by buffering capacity."""
    return 2.0 / x


def _rate_ps1(
    a: float,
    ps2cs: float,
    pfd: float,
) -> float:
    """PSI electron transfer rate: open PSI centers (a) * light absorbed by PSI antenna."""
    return (1 - ps2cs) * pfd * a


def _rate_ferredoxin_reductase(
    fd: float,
    fdred: float,
    a1: float,
    a2: float,
    k_fdred: float,
    keq_fa_fd: float,
) -> float:
    """Rate of the redcution of Fd by the activity of PSI.

    used to be equall to the rate of PSI but now
    alternative electron pathway from Fd allows for the production of ROS
    hence this rate has to be separate.
    """
    return k_fdred * fd * a1 - k_fdred / keq_fa_fd * fdred * a2


def _rate_leak(
    protons_lumen: float,
    ph_stroma: float,
    k_leak: float,
) -> float:
    """Passive proton leak across the thylakoid membrane, proportional to the proton gradient."""
    return k_leak * (protons_lumen - _protons_stroma_2016(ph_stroma))


def _neg_one_div_by(
    x: float,
) -> float:
    """Return -1/x; used for negated unit stoichiometry scaled by buffering capacity."""
    return -1.0 / x


def _rate_state_transition_ps1_ps2(
    ant: float,
    pox: float,
    p_tot: float,
    k_stt7: float,
    km_st: float,
    n_st: float,
) -> float:
    """STT7-kinase phosphorylation of LHC; inhibited by oxidised PQ (state 1 → 2 transition)."""
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
    """Rubisco carboxylation rate (Poolman 2000): bi-substrate with 5 competitive inhibitors."""
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
    """Rapid-equilibrium rate for two substrates, two products."""
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
    """Rapid-equilibrium rate for three substrates, three products."""
    return k_re * (s1 * s2 * s3 - p1 * p2 * p3 / q)


def _rapid_equilibrium_1s_1p(
    s1: float,
    p1: float,
    k_re: float,
    q: float,
) -> float:
    """Rapid-equilibrium rate for one substrate, one product."""
    return k_re * (s1 - p1 / q)


def _rapid_equilibrium_2s_1p(
    s1: float,
    s2: float,
    p1: float,
    k_re: float,
    q: float,
) -> float:
    """Rapid-equilibrium rate for two substrates, one product."""
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
    """Irreversible Michaelis-Menten rate for one substrate with two inhibitors."""
    return vmax * s / (s + km * (1 + i1 / ki1 + i2 / ki2))


def _michaelis_menten_1s_1i(
    s: float,
    i: float,
    vmax: float,
    km: float,
    ki: float,
) -> float:
    """Irreversible Michaelis-Menten rate for one substrate with one inhibitor."""
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
    """Phosphoribulokinase rate: ordered bi-substrate kinetics with PGA, RuBP, Pi and ADP inhibition."""
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
    """Individual substrate export rate normalised by the translocator occupancy N."""
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
    """Starch synthesis rate via G1P+ATP with ADP inhibition and allosteric activation by PGA/F6P/FBP."""
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


def _rate_mda_reductase1(
    mda: float,
    k3: float,
) -> float:
    """MDA reductase rate: second-order disproportionation of monodehydroascorbate."""
    return k3 * mda**2


def _rate_mda_reductase2(
    nadph: float,
    mda: float,
    vmax: float,
    km_nadph: float,
    km_mda: float,
) -> float:
    """Compare Valero et al. 2016."""
    nom = vmax * nadph * mda
    denom = km_nadph * mda + km_mda * nadph + nadph * mda + km_nadph * km_mda
    return nom / denom


def _rate_ascorbate_peroxidase(
    a: float,
    h: float,
    kf1: float,
    kr1: float,
    kf2: float,
    kr2: float,
    kf3: float,
    kf4: float,
    kr4: float,
    kf5: float,
    xt: float,
) -> float:
    """Lumped reaction of ascorbate peroxidase.

    the cycle stretched to a linear chain with
    two steps producing the MDA
    two steps releasing ASC
    and one step producing hydrogen peroxide.
    """
    nom = a * h * xt
    denom = (
        a * h * (1 / kf3 + 1 / kf5)
        + a / kf1
        + h / kf4
        + h * kr4 / (kf4 * kf5)
        + h / kf2
        + h * kr2 / (kf2 * kf3)
        + kr1 / (kf1 * kf2)
        + kr1 * kr2 / (kf1 * kf2 * kf3)
    )
    return nom / denom


def _rate_glutathion_reductase(
    nadph: float,
    gssg: float,
    vmax: float,
    km_nadph: float,
    km_gssg: float,
) -> float:
    """Glutathione reductase rate: bi-substrate Michaelis-Menten for NADPH + GSSG."""
    nom = vmax * nadph * gssg
    denom = km_nadph * gssg + km_gssg * nadph + nadph * gssg + km_nadph * km_gssg
    return nom / denom


def _rate_dhar(
    dha: float,
    gsh: float,
    vmax: float,
    km_dha: float,
    km_gsh: float,
    k: float,
) -> float:
    """Dehydroascorbate reductase rate: bi-substrate random-order Michaelis-Menten."""
    nom = vmax * dha * gsh
    denom = k + km_dha * gsh + km_gsh * dha + dha * gsh
    return nom / denom


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
    """PSII state populations (PHD quenching model, 2016) via analytical closed-form surrogate."""
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


def _ps1states_2021_surrogate(
    pc_ox: float,
    pc_red: float,
    fd_ox: float,
    fd_red: float,
    ps2cs: float,
    ps1_tot: float,
    k_fd_red: float,
    keq_f: float,
    keq_c: float,
    k_pc_ox: float,
    pfd: float,
    k0: float,
    o2: float,
) -> tuple[float, float, float]:
    """QSSA calculates open state of PSI.

    depends on reduction states of plastocyanin and ferredoxin
    C = [PC], F = [Fd] (ox. forms).
    """
    kLI = (1 - ps2cs) * pfd

    y0 = (
        keq_c
        * keq_f
        * pc_red
        * ps1_tot
        * k_pc_ox
        * (fd_ox * k_fd_red + o2 * k0)
        / (
            fd_ox * keq_c * keq_f * pc_red * k_fd_red * k_pc_ox
            + fd_ox * keq_f * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + fd_red * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + keq_c * keq_f * o2 * pc_red * k0 * k_pc_ox
            + keq_c * keq_f * pc_red * kLI * k_pc_ox
            + keq_f * o2 * k0 * (keq_c * kLI + pc_ox * k_pc_ox)
        )
    )

    y1 = (
        ps1_tot
        * (
            fd_red * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + keq_c * keq_f * pc_red * kLI * k_pc_ox
        )
        / (
            fd_ox * keq_c * keq_f * pc_red * k_fd_red * k_pc_ox
            + fd_ox * keq_f * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + fd_red * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + keq_c * keq_f * o2 * pc_red * k0 * k_pc_ox
            + keq_c * keq_f * pc_red * kLI * k_pc_ox
            + keq_f * o2 * k0 * (keq_c * kLI + pc_ox * k_pc_ox)
        )
    )
    y2 = ps1_tot - y0 - y1

    return y0, y1, y2


def _div(
    x: float,
    y: float,
) -> float:
    """Return x / y."""
    return x / y


def _rate_fluorescence(
    q: float,
    b0: float,
    b2: float,
    ps2cs: float,
    k2: float,
    k_f: float,
    k_h: float,
) -> float:
    """Chlorophyll fluorescence yield from open (B0) and closed (B2) PSII centres."""
    return ps2cs * k_f * b0 / (k_f + k2 + k_h * q) + ps2cs * k_f * b2 / (k_f + k_h * q)


def create_model() -> Model:
    """Saadat 2021 chloroplast model with ROS scavenging, Calvin cycle, and thioredoxin regulation.

    Reference: Saadat, Nima P., et al.
    "Computational analysis of alternative photosynthetic electron flows linked with oxidative stress."
    Frontiers in plant science 12 (2021): 750580
    """
    m: Model = Model()
    m = m.add_variable("3PGA", initial_value=0.9167729479368978)
    m = m.add_variable("BPGA", initial_value=0.0003814495319659031)
    m = m.add_variable("GAP", initial_value=0.00580821050261484)
    m = m.add_variable("DHAP", initial_value=0.1277806166216142)
    m = m.add_variable("FBP", initial_value=0.005269452472931973)
    m = m.add_variable("F6P", initial_value=0.2874944558066638)
    m = m.add_variable("G6P", initial_value=0.6612372482712676)
    m = m.add_variable("G1P", initial_value=0.03835176039761378)
    m = m.add_variable("SBP", initial_value=0.011101373736607443)
    m = m.add_variable("S7P", initial_value=0.1494578301900007)
    m = m.add_variable("E4P", initial_value=0.00668295494870102)
    m = m.add_variable("X5P", initial_value=0.020988553174809618)
    m = m.add_variable("R5P", initial_value=0.035155825913785584)
    m = m.add_variable("RUBP", initial_value=0.11293260727162346)
    m = m.add_variable("RU5P", initial_value=0.014062330254191594)
    m = m.add_variable("ATP", initial_value=1.4612747767895344)
    m = m.add_variable("Ferredoxine (oxidised)", initial_value=3.715702384326767)
    m = m.add_variable("protons_lumen", initial_value=0.002086128887296243)
    m = m.add_variable("Light-harvesting complex", initial_value=0.7805901436176024)
    m = m.add_variable("NADPH", initial_value=0.5578718406315588)
    m = m.add_variable("Plastocyanine (oxidised)", initial_value=1.8083642974980014)
    m = m.add_variable("Plastoquinone (oxidised)", initial_value=10.251099271612473)
    m = m.add_variable("PsbS (de-protonated)", initial_value=0.9667381262477079)
    m = m.add_variable("Violaxanthin", initial_value=0.9629870646993118)
    m = m.add_variable("MDA", initial_value=2.0353396709300447e-07)
    m = m.add_variable("H2O2", initial_value=1.2034405327140102e-07)
    m = m.add_variable("DHA", initial_value=1.0296456279861962e-11)
    m = m.add_variable("GSSG", initial_value=4.99986167652437e-12)
    m = m.add_variable("Thioredoxin (oxidised)", initial_value=0.9334426859846461)
    m = m.add_variable("E_inactive", initial_value=3.6023635680406634)
    m = m.add_parameter("PPFD", value=100.0)
    m = m.add_parameter("CO2 (dissolved)", value=0.2)
    m = m.add_parameter("O2 (dissolved)_lumen", value=8.0)
    m = m.add_parameter("pH", value=7.9)
    m = m.add_parameter("protons", value=1.2589254117941661e-05)
    m = m.add_parameter("bH", value=100.0)
    m = m.add_parameter("F", value=96.485)
    m = m.add_parameter("E^0_PC", value=0.38)
    m = m.add_parameter("E^0_P700", value=0.48)
    m = m.add_parameter("E^0_FA", value=-0.55)
    m = m.add_parameter("E^0_Fd", value=-0.43)
    m = m.add_parameter("E^0_NADP", value=-0.113)
    m = m.add_parameter("convf", value=0.032)
    m = m.add_parameter("R", value=0.0083)
    m = m.add_parameter("T", value=298.0)
    m = m.add_parameter("Carotenoids_tot", value=1.0)
    m = m.add_parameter("Fd*", value=5.0)
    m = m.add_parameter("PC_tot", value=4.0)
    m = m.add_parameter("PSBS_tot", value=1.0)
    m = m.add_parameter("LHC_tot", value=1.0)
    m = m.add_parameter("gamma0", value=0.1)
    m = m.add_parameter("gamma1", value=0.25)
    m = m.add_parameter("gamma2", value=0.6)
    m = m.add_parameter("gamma3", value=0.15)
    m = m.add_parameter("kZSat", value=0.12)
    m = m.add_parameter("E^0_QA", value=-0.14)
    m = m.add_parameter("E^0_PQ", value=0.354)
    m = m.add_parameter("PQ_tot", value=17.5)
    m = m.add_parameter("staticAntII", value=0.1)
    m = m.add_parameter("staticAntI", value=0.37)
    m = m.add_parameter("Thioredoxin_tot", value=1.0)
    m = m.add_parameter("E_total", value=6.0)
    m = m.add_parameter("NADP*", value=0.8)
    m = m.add_parameter("A*P", value=2.55)
    m = m.add_parameter("Pi_tot", value=17.05)
    m = m.add_parameter("kf_ferredoxin_thioredoxin_reductase", value=1.0)
    m = m.add_parameter("kf_tr_activation", value=1.0)
    m = m.add_parameter("kf_tr_inactivation", value=0.1)
    m = m.add_parameter("ASC_tot*", value=10)
    m = m.add_parameter("Glutathion_tot", value=10.0)
    m = m.add_parameter("kf_atp_synthase", value=20.0)
    m = m.add_parameter("HPR", value=4.666666666666667)
    m = m.add_parameter("Pi_mol", value=0.01)
    m = m.add_parameter("DeltaG0_ATP", value=30.6)
    m = m.add_parameter("kcat_b6f", value=2.5)
    m = m.add_parameter("kh_lhc_protonation", value=3.0)
    m = m.add_parameter("kf_lhc_protonation", value=0.0096)
    m = m.add_parameter("ksat_lhc_protonation", value=5.8)
    m = m.add_parameter("kf_lhc_deprotonation", value=0.0096)
    m = m.add_parameter("kf_cyclic_electron_flow", value=1.0)
    m = m.add_parameter("kf_violaxanthin_deepoxidase", value=0.0024)
    m = m.add_parameter("kh_violaxanthin_deepoxidase", value=5.0)
    m = m.add_parameter("ksat_violaxanthin_deepoxidase", value=5.8)
    m = m.add_parameter("kf_zeaxanthin_epoxidase", value=0.00024)
    m = m.add_parameter("km_fnr_Ferredoxine (reduced)", value=1.56)
    m = m.add_parameter("km_fnr_NADP", value=0.22)
    m = m.add_parameter("E0_fnr", value=3.0)
    m = m.add_parameter("kcat_fnr", value=500.0)
    m = m.add_parameter("kf_ndh", value=0.002)
    m = m.add_parameter("PSII_total", value=2.5)
    m = m.add_parameter("PSI_total", value=2.5)
    m = m.add_parameter("kH0", value=500000000.0)
    m = m.add_parameter("kPQred", value=250.0)
    m = m.add_parameter("kPCox", value=2500.0)
    m = m.add_parameter("kFdred", value=250000.0)
    m = m.add_parameter("k2", value=5000000000.0)
    m = m.add_parameter("kH", value=5000000000.0)
    m = m.add_parameter("kF", value=625000000.0)
    m = m.add_parameter("kMehler", value=1.0)
    m = m.add_parameter("E0_ferredoxin_reductase", value=1.0)
    m = m.add_parameter("kcat_ferredoxin_reductase", value=250000.0)
    m = m.add_parameter("kf_proton_leak", value=10.0)
    m = m.add_parameter("kPTOX", value=0.01)
    m = m.add_parameter("kStt7", value=0.0035)
    m = m.add_parameter("km_lhc_state_transition_12", value=0.2)
    m = m.add_parameter("n_ST", value=2.0)
    m = m.add_parameter("kPph1", value=0.0013)
    m = m.add_parameter("E0_rubisco", value=1.0)
    m = m.add_parameter("kcat_rubisco_carboxylase", value=2.72)
    m = m.add_parameter("km_rubisco_carboxylase_RUBP", value=0.02)
    m = m.add_parameter("km_rubisco_carboxylase_CO2 (dissolved)", value=0.0107)
    m = m.add_parameter("ki_rubisco_carboxylase_3PGA", value=0.04)
    m = m.add_parameter("ki_rubisco_carboxylase_FBP", value=0.04)
    m = m.add_parameter("ki_rubisco_carboxylase_SBP", value=0.075)
    m = m.add_parameter("ki_rubisco_carboxylase_Orthophosphate", value=0.9)
    m = m.add_parameter("ki_rubisco_carboxylase_NADPH", value=0.07)
    m = m.add_parameter("kre_phosphoglycerate_kinase", value=800000000.0)
    m = m.add_parameter("keq_phosphoglycerate_kinase", value=0.00031)
    m = m.add_parameter("kre_gadph", value=800000000.0)
    m = m.add_parameter("keq_gadph", value=16000000.0)
    m = m.add_parameter("kre_triose_phosphate_isomerase", value=800000000.0)
    m = m.add_parameter("keq_triose_phosphate_isomerase", value=22.0)
    m = m.add_parameter("kre_aldolase_dhap_gap", value=800000000.0)
    m = m.add_parameter("keq_aldolase_dhap_gap", value=7.1)
    m = m.add_parameter("kre_aldolase_dhap_e4p", value=800000000.0)
    m = m.add_parameter("keq_aldolase_dhap_e4p", value=13.0)
    m = m.add_parameter("E0_fbpase", value=1.0)
    m = m.add_parameter("kcat_fbpase", value=1.6)
    m = m.add_parameter("km_fbpase_s", value=0.03)
    m = m.add_parameter("ki_fbpase_F6P", value=0.7)
    m = m.add_parameter("ki_fbpase_Orthophosphate", value=12.0)
    m = m.add_parameter("kre_transketolase_gap_f6p", value=800000000.0)
    m = m.add_parameter("keq_transketolase_gap_f6p", value=0.084)
    m = m.add_parameter("kre_transketolase_gap_s7p", value=800000000.0)
    m = m.add_parameter("keq_transketolase_gap_s7p", value=0.85)
    m = m.add_parameter("E0_SBPase", value=1.0)
    m = m.add_parameter("kcat_SBPase", value=0.32)
    m = m.add_parameter("km_SBPase_s", value=0.013)
    m = m.add_parameter("ki_SBPase_Orthophosphate", value=12.0)
    m = m.add_parameter("kre_ribose_phosphate_isomerase", value=800000000.0)
    m = m.add_parameter("keq_ribose_phosphate_isomerase", value=0.4)
    m = m.add_parameter("kre_ribulose_phosphate_epimerase", value=800000000.0)
    m = m.add_parameter("keq_ribulose_phosphate_epimerase", value=0.67)
    m = m.add_parameter("E0_phosphoribulokinase", value=1.0)
    m = m.add_parameter("kcat_phosphoribulokinase", value=7.9992)
    m = m.add_parameter("km_phosphoribulokinase_RU5P", value=0.05)
    m = m.add_parameter("km_phosphoribulokinase_ATP", value=0.05)
    m = m.add_parameter("ki_phosphoribulokinase_3PGA", value=2.0)
    m = m.add_parameter("ki_phosphoribulokinase_RUBP", value=0.7)
    m = m.add_parameter("ki_phosphoribulokinase_Orthophosphate", value=4.0)
    m = m.add_parameter("ki_phosphoribulokinase_4", value=2.5)
    m = m.add_parameter("ki_phosphoribulokinase_5", value=0.4)
    m = m.add_parameter("kre_g6pi", value=800000000.0)
    m = m.add_parameter("keq_g6pi", value=2.3)
    m = m.add_parameter("kre_phosphoglucomutase", value=800000000.0)
    m = m.add_parameter("keq_phosphoglucomutase", value=0.058)
    m = m.add_parameter("Orthophosphate (external)", value=0.5)
    m = m.add_parameter("km_ex_pga", value=0.25)
    m = m.add_parameter("km_ex_gap", value=0.075)
    m = m.add_parameter("km_ex_dhap", value=0.077)
    m = m.add_parameter("km_N_translocator_Orthophosphate (external)", value=0.74)
    m = m.add_parameter("km_N_translocator_Orthophosphate", value=0.63)
    m = m.add_parameter("kcat_N_translocator", value=2.0)
    m = m.add_parameter("E0_N_translocator", value=1.0)
    m = m.add_parameter("E0_ex_g1p", value=1.0)
    m = m.add_parameter("km_ex_g1p_G1P", value=0.08)
    m = m.add_parameter("km_ex_g1p_ATP", value=0.08)
    m = m.add_parameter("ki_ex_g1p", value=10.0)
    m = m.add_parameter("ki_ex_g1p_3PGA", value=0.1)
    m = m.add_parameter("ki_ex_g1p_F6P", value=0.02)
    m = m.add_parameter("ki_ex_g1p_FBP", value=0.02)
    m = m.add_parameter("kcat_ex_g1p", value=0.32)
    m = m.add_parameter("kf_mda_reductase_1", value=500.0)
    m = m.add_parameter("E0_mda_reductase_2", value=0.002)
    m = m.add_parameter("kcat_mda_reductase_2", value=300.0)
    m = m.add_parameter("km_mda_reductase_2_NADPH", value=0.023)
    m = m.add_parameter("km_mda_reductase_2_MDA", value=0.0014)
    m = m.add_parameter("kf1", value=10000.0)
    m = m.add_parameter("kr1", value=220.0)
    m = m.add_parameter("kf2", value=10000.0)
    m = m.add_parameter("kr2", value=4000.0)
    m = m.add_parameter("kf3", value=2510.0)
    m = m.add_parameter("kf4", value=10000.0)
    m = m.add_parameter("kr4", value=4000.0)
    m = m.add_parameter("kf5", value=2510.0)
    m = m.add_parameter("XT", value=0.07)
    m = m.add_parameter("E0_glutathion_reductase", value=0.0014)
    m = m.add_parameter("kcat_glutathion_reductase", value=595)
    m = m.add_parameter("km_glutathion_reductase_NADPH", value=0.003)
    m = m.add_parameter("km_glutathion_reductase_GSSG", value=0.2)
    m = m.add_parameter("km_dehydroascorbate_reductase_DHA", value=0.07)
    m = m.add_parameter("km_dehydroascorbate_reductase_GSH", value=2.5)
    m = m.add_parameter("K", value=0.5)
    m = m.add_parameter("E0_dehydroascorbate_reductase", value=0.0017)
    m = m.add_parameter("kcat_dehydroascorbate_reductase", value=142)
    m = m.add_parameter("kf_ex_atp", value=0.2)
    m = m.add_parameter("kf_ex_nadph", value=0.2)
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
        "Thioredoxin (reduced)",
        fn=_moiety_1,
        args=["Thioredoxin (oxidised)", "Thioredoxin_tot"],
    )
    m = m.add_derived(
        "E_active",
        fn=_moiety_1,
        args=["E_inactive", "E_total"],
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
        "ascorbate",
        fn=_moiety_2,
        args=["MDA", "DHA", "ASC_tot*"],
    )
    m = m.add_derived(
        "GSH",
        fn=_glutathion_moiety,
        args=["GSSG", "Glutathion_tot"],
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
        "vmax_ferredoxin_reductase",
        fn=_mass_action_1s,
        args=["kcat_ferredoxin_reductase", "E0_ferredoxin_reductase"],
    )
    m = m.add_derived(
        "E0_rubisco_active",
        fn=_mul,
        args=["E0_rubisco", "E_active"],
    )
    m = m.add_derived(
        "vmax_rubisco_carboxylase",
        fn=_mass_action_1s,
        args=["kcat_rubisco_carboxylase", "E0_rubisco_active"],
    )
    m = m.add_derived(
        "E0_fbpase_active",
        fn=_mul,
        args=["E0_fbpase", "E_active"],
    )
    m = m.add_derived(
        "vmax_fbpase",
        fn=_mass_action_1s,
        args=["kcat_fbpase", "E0_fbpase_active"],
    )
    m = m.add_derived(
        "E0_SBPase_active",
        fn=_mul,
        args=["E0_SBPase", "E_active"],
    )
    m = m.add_derived(
        "vmax_SBPase",
        fn=_mass_action_1s,
        args=["kcat_SBPase", "E0_SBPase_active"],
    )
    m = m.add_derived(
        "E0_phosphoribulokinase_active",
        fn=_mul,
        args=["E0_phosphoribulokinase", "E_active"],
    )
    m = m.add_derived(
        "vmax_phosphoribulokinase",
        fn=_mass_action_1s,
        args=["kcat_phosphoribulokinase", "E0_phosphoribulokinase_active"],
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
        "E0_ex_g1p_active",
        fn=_mul,
        args=["E0_ex_g1p", "E_active"],
    )
    m = m.add_derived(
        "vmax_ex_g1p",
        fn=_mass_action_1s,
        args=["kcat_ex_g1p", "E0_ex_g1p_active"],
    )
    m = m.add_derived(
        "vmax_mda_reductase_2",
        fn=_mass_action_1s,
        args=["kcat_mda_reductase_2", "E0_mda_reductase_2"],
    )
    m = m.add_derived(
        "vmax_glutathion_reductase",
        fn=_mass_action_1s,
        args=["kcat_glutathion_reductase", "E0_glutathion_reductase"],
    )
    m = m.add_derived(
        "vmax_dehydroascorbate_reductase",
        fn=_mass_action_1s,
        args=["kcat_dehydroascorbate_reductase", "E0_dehydroascorbate_reductase"],
    )
    m = m.add_reaction(
        "ferredoxin_thioredoxin_reductase",
        fn=_mass_action_2s,
        args=[
            "Thioredoxin (oxidised)",
            "Ferredoxine (reduced)",
            "kf_ferredoxin_thioredoxin_reductase",
        ],
        stoichiometry={"Thioredoxin (oxidised)": -1, "Ferredoxine (oxidised)": 1},
    )
    m = m.add_reaction(
        "tr_activation",
        fn=_mass_action_2s,
        args=["E_inactive", "Thioredoxin (reduced)", "kf_tr_activation"],
        stoichiometry={"E_inactive": -5, "Thioredoxin (oxidised)": 5},
    )
    m = m.add_reaction(
        "tr_inactivation",
        fn=_mass_action_1s,
        args=["E_active", "kf_tr_inactivation"],
        stoichiometry={"E_inactive": 5},
    )
    m = m.add_reaction(
        "atp_synthase",
        fn=_rate_atp_synthase_2019,
        args=["ATP", "ADP", "keq_atp_synthase", "kf_atp_synthase", "convf"],
        stoichiometry={
            "protons_lumen": Derived(fn=_neg_div, args=["HPR", "bH"]),
            "ATP": Derived(fn=_value, args=["convf"]),
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
            "protons_lumen": Derived(fn=_four_div_by, args=["bH"]),
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
            "NADPH": Derived(fn=_value, args=["convf"]),
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
            "protons_lumen": Derived(fn=_two_div_by, args=["bH"]),
        },
    )
    m = m.add_reaction(
        "PSI",
        fn=_rate_ps1,
        args=["A0", "PSII_cross_section", "PPFD"],
        stoichiometry={"Plastocyanine (oxidised)": 1},
    )
    m = m.add_reaction(
        "mehler",
        fn=_mass_action_2s,
        args=["A1", "O2 (dissolved)_lumen", "kMehler"],
        stoichiometry={"H2O2": Derived(fn=_value, args=["convf"])},
    )
    m = m.add_reaction(
        "ferredoxin_reductase",
        fn=_rate_ferredoxin_reductase,
        args=[
            "Ferredoxine (oxidised)",
            "Ferredoxine (reduced)",
            "A1",
            "A2",
            "vmax_ferredoxin_reductase",
            "keq_ferredoxin_reductase",
        ],
        stoichiometry={"Ferredoxine (oxidised)": -1},
    )
    m = m.add_reaction(
        "proton_leak",
        fn=_rate_leak,
        args=["protons_lumen", "pH", "kf_proton_leak"],
        stoichiometry={"protons_lumen": Derived(fn=_neg_one_div_by, args=["bH"])},
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
        args=["GAP", "DHAP", "FBP", "kre_aldolase_dhap_gap", "keq_aldolase_dhap_gap"],
        stoichiometry={"GAP": -1, "DHAP": -1, "FBP": 1},
    )
    m = m.add_reaction(
        "aldolase_dhap_e4p",
        fn=_rapid_equilibrium_2s_1p,
        args=["DHAP", "E4P", "SBP", "kre_aldolase_dhap_e4p", "keq_aldolase_dhap_e4p"],
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
    m = m.add_reaction(
        "mda_reductase_1",
        fn=_rate_mda_reductase1,
        args=["MDA", "kf_mda_reductase_1"],
        stoichiometry={"MDA": -2, "DHA": 1},
    )
    m = m.add_reaction(
        "mda_reductase_2",
        fn=_rate_mda_reductase2,
        args=[
            "NADPH",
            "MDA",
            "vmax_mda_reductase_2",
            "km_mda_reductase_2_NADPH",
            "km_mda_reductase_2_MDA",
        ],
        stoichiometry={"NADPH": -1, "MDA": -2},
    )
    m = m.add_reaction(
        "ascorbate_peroxidase",
        fn=_rate_ascorbate_peroxidase,
        args=[
            "ascorbate",
            "H2O2",
            "kf1",
            "kr1",
            "kf2",
            "kr2",
            "kf3",
            "kf4",
            "kr4",
            "kf5",
            "XT",
        ],
        stoichiometry={"H2O2": -1, "MDA": 2},
    )
    m = m.add_reaction(
        "glutathion_reductase",
        fn=_rate_glutathion_reductase,
        args=[
            "NADPH",
            "GSSG",
            "vmax_glutathion_reductase",
            "km_glutathion_reductase_NADPH",
            "km_glutathion_reductase_GSSG",
        ],
        stoichiometry={"NADPH": -1, "GSSG": -1},
    )
    m = m.add_reaction(
        "dehydroascorbate_reductase",
        fn=_rate_dhar,
        args=[
            "DHA",
            "GSH",
            "vmax_dehydroascorbate_reductase",
            "km_dehydroascorbate_reductase_DHA",
            "km_dehydroascorbate_reductase_GSH",
            "K",
        ],
        stoichiometry={"DHA": -1, "GSSG": 1},
    )
    m = m.add_reaction(
        "ex_atp",
        fn=_mass_action_1s,
        args=["ATP", "kf_ex_atp"],
        stoichiometry={"ATP": -1},
    )
    m = m.add_reaction(
        "ex_nadph",
        fn=_mass_action_1s,
        args=["NADPH", "kf_ex_nadph"],
        stoichiometry={"NADPH": -1},
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
    m = m.add_surrogate(
        "ps1states",
        qss.Surrogate(
            model=_ps1states_2021_surrogate,
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
                "kMehler",
                "O2 (dissolved)_lumen",
            ],
            outputs=["A0", "A1", "A2"],
        ),
    )
    m = m.add_readout(
        "PQ_ox/tot",
        fn=_div,
        args=["Plastoquinone (reduced)", "PQ_tot"],
    )
    m = m.add_readout(
        "Fd_ox/tot",
        fn=_div,
        args=["Ferredoxine (reduced)", "Fd*"],
    )
    m = m.add_readout(
        "PC_ox/tot",
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
        args=["Q", "B0", "B2", "PSII_cross_section", "k2", "kF", "kH"],
    )
    return m  # noqa: RET504
