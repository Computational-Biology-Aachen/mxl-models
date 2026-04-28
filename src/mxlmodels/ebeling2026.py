"""Ebeling 2026 extended chloroplast model with ion transport, ROS, and Calvin cycle.

Reference: tbd
"""

import math

import numpy as np
from mxlpy import Derived, InitialAssignment, Model


def _initial_delta_psi(
    p_h: float,
    p_h_lumen: float,
    r: float,
    f: float,
    t: float,
) -> float:
    """Estimate delta psi in the dark, assuming delta_pH and delta_psi contribute equally to pmf."""
    return np.log(10) * ((r * t) / f) * (p_h - p_h_lumen)


def _half(
    x: float,
) -> float:
    """Return x/2; used for halving a stoichiometric coefficient."""
    return x / 2


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


def _keq_pcp700(
    e0_pc: float,
    f: float,
    e0_p700: float,
    rt: float,
) -> float:
    """Equilibrium constant for PC -> P700 electron transfer from standard redox potentials."""
    DG = -(-e0_pc * f) + (-e0_p700 * f)
    return np.exp(-DG / rt)


def _keq_faf_d(
    e0_fa: float,
    f: float,
    e0_fd: float,
    rt: float,
) -> float:
    """Equilibrium constant for FA -> Fd electron transfer from standard redox potentials."""
    DG = -(-e0_fa * f) + (-e0_fd * f)
    return np.exp(-DG / rt)


def _moiety_3(
    c1: float,
    c2: float,
    c3: float,
    total: float,
) -> float:
    """Conservation moiety: fourth species = total - c1 - c2 - c3."""
    return total - c1 - c2 - c3


def _normalize_concentration(
    concentration: float,
    total: float,
) -> float:
    """Return concentration/total as a dimensionless fraction."""
    return concentration / total


def _normalize_2_concentrations(
    c1: float,
    c2: float,
    total: float,
) -> float:
    """Return (c1+c2)/total as a combined dimensionless fraction."""
    return (c1 + c2) / total


def _fluo(
    q: float,
    b0: float,
    b2: float,
    ps2cs: float,
    k2: float,
    k_f: float,
    k_h_qslope: float,
    k_h0: float,
) -> float:
    """Chlorophyll fluorescence yield from open (B0) and closed (B2) PSII centres with quencher-dependent kH."""
    kH = k_h0 + k_h_qslope * q
    return (ps2cs * k_f * b0) / (k_f + k2 + kH) + (ps2cs * k_f * b2) / (k_f + kH)


def _k_b6f(
    p_h: float,
    p_kreg: float,
    b6f_content: float,
    max_b6f: float,
) -> float:
    """Effective b6f rate constant modulated by lumenal pH via a sigmoid around pKreg."""
    pHmod = 1 - (1 / (10 ** (p_h - p_kreg) + 1))
    return pHmod * b6f_content * max_b6f


def _protons_lumen(
    p_h_lumen: float,
) -> float:
    """Convert lumenal pH to proton concentration in mmol/mmol_Chl (conversion factor 2.5e-4)."""
    return (10 ** (-p_h_lumen)) / 2.5e-4


def _protons_stroma_ebeling(
    p_h_stroma: float,
) -> float:
    """Convert stromal pH to proton concentration in mmol/mmol_Chl (Ebeling model, factor 3.2e-5)."""
    return (10 ** (-p_h_stroma)) / 3.2e-5


def _pmf(
    _deltap_h: float,
    delta_psi: float,
    f: float,
) -> float:
    """Total proton motive force: electrical (F*delta_psi) + chemical (delta_pH) contributions."""
    return f * delta_psi + _deltap_h


def _deltap_h(
    p_h: float,
    p_h_lumen: float,
    d_g: float,
) -> float:
    """Proton motive force component from transmembrane pH difference (in energy units)."""
    return d_g * (p_h - p_h_lumen)


def _atp_pmf_activity2(
    p_k0_e: float,
    b: float,
    p_h_lumen: float,
    p_h: float,
    f: float,
    rt: float,
    delta_psi: float,
) -> float:
    """Variant of _atp_pmf_activity used in a second ATP synthase module instance."""
    _pmf = delta_psi - np.log(10) * ((rt) / f) * (p_h_lumen - p_h)
    x = np.log(10 ** (-p_k0_e)) + b * (_pmf * f) / (rt)
    return (np.e**x) / (1 + np.e**x)


def _pmf_in_v(
    delta_psi: float,
    p_h_lumen: float,
    p_h: float,
    rt: float,
    f: float,
) -> float:
    """Total PMF expressed in volts: delta_psi - (RT/F)*ln(10)*delta_pH."""
    return delta_psi - np.log(10) * ((rt) / f) * (p_h_lumen - p_h)


def _keq_cytb6f(
    p_h: float,
    _pmf: float,
    f: float,
    e0_pq: float,
    e0_pc: float,
    rt: float,
    d_g_p_h: float,
) -> float:
    """Equilibrium constant of cytochrome b6f including PMF contribution to free energy."""
    DG1 = -2 * f * e0_pq
    DG2 = -f * e0_pc
    DG = -(DG1 + 2 * d_g_p_h * p_h) + 2 * DG2 + 2 * _pmf
    return np.exp(-DG / rt)


def _squared(
    x: float,
) -> float:
    """Return x^2; used for second-order stoichiometry terms."""
    return x**2


def _reg_kea(
    p_h: float,
    atp: float,
    kea3_p_h_reg: float,
    kea3_atp_treshold: float,
) -> float:
    """KEA3 K+/H+ antiporter regulation: product of pH and ATP inhibition sigmoids."""
    pH_inhib = (1 - 0.1) / (1 + np.exp((p_h - kea3_p_h_reg) / 0.001))
    ATP_inhib = (1 - 0.1) / (1 + np.exp((kea3_atp_treshold - atp) / 0.01))
    return pH_inhib * ATP_inhib


def _dg_k(
    klumen: float,
    kstroma: float,
    delta_psi: float,
    rt: float,
    f: float,
) -> float:
    """Electrochemical driving force for K+ transport across the thylakoid membrane."""
    return (-(rt / f) * np.log10(kstroma / klumen) + delta_psi) * f


def _cl_driving_force(
    delta_psi: float,
    cl_lumen: float,
    cl_stroma: float,
    rt: float,
    f: float,
) -> float:
    """Electrochemical driving force for Cl- transport across the thylakoid membrane."""
    return ((rt / f) * np.log10(cl_stroma / cl_lumen) + delta_psi) * f


def _keq_ndh1(
    _pmf: float,
    e0_fd: float,
    f: float,
    e0_pq: float,
    p_hstroma: float,
    d_g_p_h: float,
    rt: float,
) -> float:
    """Equilibrium constant for NDH-1 (Fd-dependent PQ reduction) including PMF and stromal pH."""
    DG1 = -e0_fd * f
    DG2 = -2 * e0_pq * f
    DG = -2 * DG1 + DG2 + 2 * d_g_p_h * p_hstroma + 4 * _pmf
    return np.exp(-DG / rt)


def _cl_ce_activation(
    atp: float,
    cl_ce_atp_threshold: float,
) -> float:
    """CLCe activation factor: sigmoid inhibited by high ATP (low-energy stress signal)."""
    return (1 - 0.1) / (1 + np.exp((atp - cl_ce_atp_threshold) / 0.01))


def _mass_action_2s(
    s1: float,
    s2: float,
    k_fwd: float,
) -> float:
    """Mass-action rate for two substrates."""
    return k_fwd * s1 * s2


def _atp_pmf_activity(
    p_k0_e: float,
    b: float,
    p_h_lumen: float,
    p_h: float,
    f: float,
    rt: float,
    delta_psi: float,
) -> float:
    """Sigmoidal ATP synthase activity as function of PMF (delta_psi and delta_pH combined)."""
    _pmf = delta_psi - np.log(10) * ((rt) / f) * (p_h_lumen - p_h)
    x = np.log(10 ** (-p_k0_e)) + b * (_pmf * f) / (rt)
    return (np.e**x) / (1 + np.e**x)


def _v_at_psynthase_mod(
    atp: float,
    atp_activity: float,
    _atp_pmf_activity: float,
    k_at_psynthase: float,
    adp: float,
    _keq_atp: float,
    convf: float,
) -> float:
    """ATP synthase flux modulated by light-activation state and PMF-dependent activity sigmoid."""
    return (
        atp_activity
        * _atp_pmf_activity
        * k_at_psynthase
        * (adp / convf - atp / convf / _keq_atp)
    )


def _value(
    x: float,
) -> float:
    """Return x unchanged."""
    return x


def _atp_div(
    hpr: float,
    x: float,
) -> float:
    """Return -hpr*x for the HPR-scaled proton stoichiometry of ATP synthase (negative = lumen consumption)."""
    return -hpr * x


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


def _rate_leak(
    protons_lumen: float,
    ph_stroma: float,
    k_leak: float,
) -> float:
    """Passive proton leak across the thylakoid membrane, proportional to the proton gradient."""
    return k_leak * (protons_lumen - _protons_stroma_2016(ph_stroma))


def _neg_one_div(
    x: float,
) -> float:
    """Negate x; used as a sign-flipping stoichiometry wrapper."""
    return -1 * x


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


def _mass_action_22_rev(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    kf: float,
    keq: float,
) -> float:
    """Reversible bimolecular mass-action rate: kf*(s1*s2) - (kf/keq)*(p1*p2)."""
    return kf * s1 * s2 - (kf / keq) * p1 * p2


def _v_ps1(
    p700_fa: float,
    ps2cs: float,
    pfd: float,
) -> float:
    """PSI electron transfer rate: open PSI fraction * light absorbed by PSI antenna."""
    return (1 - ps2cs) * pfd * p700_fa


def _v_mehler(
    psi_red_acceptor: float,
    o2ext: float,
    k_mehler: float,
) -> float:
    """Mehler reaction rate: O2 reduction by PSI-reduced acceptor (pseudo-Mehler/flavodiiron)."""
    return k_mehler * o2ext * psi_red_acceptor


def _kquencher(
    s: float,
    q: float,
    k_h_qslope: float,
    k_h0: float,
) -> float:
    """Effective quenching rate on state s: linear quencher dependence with baseline k_h0."""
    return (k_h0 + k_h_qslope * q) * s


def _one_div(
    x: float,
) -> float:
    """Identity passthrough; used as a no-op stoichiometry wrapper."""
    return x


def _vb6f_2024(
    pc: float,
    p_cred: float,
    pq: float,
    p_qred: float,
    _k_b6f: float,
    keq: float,
) -> float:
    """2024 b6f rate using PQH2/PQ fractions instead of absolute concentrations."""
    f = p_qred / (p_qred + pq)
    return f * pc * _k_b6f - (1 - f) * p_cred * (_k_b6f / keq)


def _four_div(
    x: float,
) -> float:
    """Return 4*x; used for the 4-proton lumenal stoichiometry of the b6f complex."""
    return 4 * x


def _v_at_pactivity(
    at_pactivity: float,
    light: float,
    k_act_at_pase: float,
    k_deact_at_pase: float,
) -> float:
    """Activation of ATPsynthase by light."""
    if light > 0.0:
        return k_act_at_pase * (1 - at_pactivity)
    return -k_deact_at_pase * at_pactivity


def _reversible_mass_action_1s_1p(
    s: float,
    p: float,
    kf: float,
    kb: float,
) -> float:
    """Reversible unimolecular mass-action rate: kf*s - kb*p."""
    return kf * s - kb * p


def _v_kea(
    klumen: float,
    h: float,
    kstroma: float,
    k_kea: float,
    hstroma: float,
    _reg_kea: float,
) -> float:
    """KEA3 antiporter rate: regulated K+/H+ exchange clamped to zero for back-flux."""
    v_KEA = k_kea * (h * kstroma - hstroma * klumen) * _reg_kea
    return max(
        v_KEA,
        0,
    )


def _v_voltage_k_channel(
    delta_psi_ions: float,
    klumen: float,
    kstroma: float,
    _dg_k: float,
    perm_k: float,
    k_delta_psi_treshold: float,
) -> float:
    """Voltage-gated K+ channel flux: sigmoid voltage activation times electrochemical driving force."""
    voltage_dependence = (1 - 0.1) / (
        1 + np.exp(-(delta_psi_ions - k_delta_psi_treshold) / 0.001)
    )
    return (
        perm_k * _dg_k * voltage_dependence * (klumen / kstroma)
    )  # why divided K_total/2


def _v_vccn1(
    cl_stroma: float,
    cl_lumen: float,
    _cl_driving_force: float,
    delta_psi_ions: float,
    k_vccn1: float,
    vccn_delta_psi_treshold: float,
) -> float:
    """VCCN1 anion channel flux: voltage-gated Cl- transport driven by electrochemical force."""
    voltage_gate = (1 - 0.1) / (
        1 + np.exp(-(delta_psi_ions - vccn_delta_psi_treshold) / 0.001)
    )
    return voltage_gate * k_vccn1 * _cl_driving_force * (cl_stroma / cl_lumen)


def _v_cl_leak(
    k_cl_leak: float,
    cl_lumen: float,
    cl_stroma: float,
    pq: float,
    cl_leak_pq: float,
    total_div: float,
) -> float:
    """Passive Cl- leak flux: PQ-activated quadratic concentration-difference driving force."""
    activation = (1 - 0.1) / (1 + np.exp(-(pq - cl_leak_pq) / 0.1))
    return k_cl_leak * ((cl_lumen - cl_stroma) ** 2) / (total_div) * activation


def _v_ndh1(
    a1: float,
    fdred: float,
    pq: float,
    p_hlumen: float,
    k_ndh1: float,
) -> float:
    """NDH-1 flux: Fd-dependent PQ reduction activated by open PSI fraction and lumenal pH."""
    return (
        k_ndh1
        * ((fdred**2) * pq)
        * ((1 - 0.1) / (1 + np.exp(-((a1 - 0.02) / 0.01))))
        * (10 ** (p_hlumen - 6.5) / (10 ** (p_hlumen - 6.5) + 0.5))
    )


def _cl_ce_bi(
    cl_lumen: float,
    cl_stroma: float,
    k_cl_ce: float,
    activation: float,
) -> float:  # correct
    """CLCe bidirectional flux: simple concentration-gradient driven Cl- transport."""
    return k_cl_ce * (cl_stroma - cl_lumen) * activation


def _div(
    x: float,
    y: float,
) -> float:
    """Return x / y."""
    return x / y


def create_model() -> Model:
    """Ebeling 2026 extended chloroplast model with ion transport, ROS, and Calvin cycle.

    Reference: tbd
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
    m = m.add_variable("P700FA", initial_value=1.506615384275408)
    m = m.add_variable("P700+FA-", initial_value=0.019197449388051676)
    m = m.add_variable("P700FA-", initial_value=0.028144516332212766)
    m = m.add_variable("B0", initial_value=1.9379789566530539)
    m = m.add_variable("B1", initial_value=9.786232812526368e-08)
    m = m.add_variable("B2", initial_value=0.5620208537555176)
    m = m.add_variable("pH_lumen", initial_value=6.8)
    m = m.add_variable("pH", initial_value=7.5)
    m = m.add_variable("ATPactivity", initial_value=0)
    m = m.add_variable(
        "delta_psi",
        initial_value=InitialAssignment(
            fn=_initial_delta_psi, args=["pH", "pH_lumen", "R", "F", "T"]
        ),
    )
    m = m.add_variable(
        "K_stroma",
        initial_value=InitialAssignment(fn=_half, args=["K_total"]),
    )
    m = m.add_variable(
        "Cl_stroma",
        initial_value=InitialAssignment(fn=_half, args=["Cl_total"]),
    )
    m = m.add_parameter("PPFD", value=100.0)
    m = m.add_parameter("CO2 (dissolved)", value=0.013226)
    m = m.add_parameter("O2 (dissolved)_lumen", value=8.0)
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
    m = m.add_parameter("gamma0", value=0.06260060801266355)
    m = m.add_parameter("gamma1", value=0.4053583123566203)
    m = m.add_parameter("gamma2", value=0.7040758738825375)
    m = m.add_parameter("gamma3", value=0.07834807781016208)
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
    m = m.add_parameter("kf_ferredoxin_thioredoxin_reductase", value=0.8)
    m = m.add_parameter("kf_tr_activation", value=1.0)
    m = m.add_parameter("kf_tr_inactivation", value=0.1)
    m = m.add_parameter("ASC_tot*", value=10)
    m = m.add_parameter("Glutathion_tot", value=10.0)
    m = m.add_parameter("kf_atp_synthase", value=20.0)
    m = m.add_parameter("HPR", value=4.666666666666667)
    m = m.add_parameter("Pi_mol", value=0.01)
    m = m.add_parameter("DeltaG0_ATP", value=30.6)
    m = m.add_parameter("kh_lhc_protonation", value=10)
    m = m.add_parameter("kf_lhc_protonation", value=0.15837051384170664)
    m = m.add_parameter("ksat_lhc_protonation", value=6.2539066418842255)
    m = m.add_parameter("kf_lhc_deprotonation", value=0.015892570403695704)
    m = m.add_parameter("kf_cyclic_electron_flow", value=1.0)
    m = m.add_parameter("kf_violaxanthin_deepoxidase", value=0.0006091912188339879)
    m = m.add_parameter("kh_violaxanthin_deepoxidase", value=4)
    m = m.add_parameter("ksat_violaxanthin_deepoxidase", value=6.193595407850397)
    m = m.add_parameter("kf_zeaxanthin_epoxidase", value=0.000106261953934132)
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
    m = m.add_parameter("kf_ex_atp", value=0.5)
    m = m.add_parameter("kf_ex_nadph", value=0.5)
    m = m.add_parameter("kH_Qslope", value=30000000000.0)
    m = m.add_parameter("b6f_content", value=1)
    m = m.add_parameter("max_b6f", value=500)
    m = m.add_parameter("pKreg", value=7)
    m = m.add_parameter("stroma_buffering", value=400)
    m = m.add_parameter("kActATPase", value=0.001)
    m = m.add_parameter("kDeactATPase", value=0.002)
    m = m.add_parameter("k_ATPsynthase", value=20)
    m = m.add_parameter("b", value=1.8688304401249531)
    m = m.add_parameter("pK0E", value=5.960025833706074)
    m = m.add_parameter("k_import_ATP", value=0.5)
    m = m.add_parameter("k_import_NADPH", value=0.5)
    m = m.add_parameter("volts_per_charge", value=0.000769481926574965)
    m = m.add_parameter("ClCe_PQ", value=15.87880046767565)
    m = m.add_parameter("Cl_leak_PQ", value=14.92901445507139)
    m = m.add_parameter("KEA3_ATP_treshold", value=0.26274793681796166)
    m = m.add_parameter("KEA3_pH_reg", value=7.69)
    m = m.add_parameter("K_delta_psi_treshold", value=0.08146807307624158)
    m = m.add_parameter("VCCN_delta_psi_treshold", value=0.08000900979332677)
    m = m.add_parameter("k_Cl_leak", value=25)
    m = m.add_parameter("k_NDH1", value=7.447430768265866)
    m = m.add_parameter("k_KEA", value=90)
    m = m.add_parameter("perm_K", value=1.6113135416150155)
    m = m.add_parameter("k_VCCN1", value=0.5)
    m = m.add_parameter("k_ClCe", value=0.5)
    m = m.add_parameter("K_total", value=60)
    m = m.add_parameter("Cl_total", value=50)
    m = m.add_parameter("ClCe_ATP_threshold", value=0.2)
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
    m = m.add_derived(
        "keq_PCP700",
        fn=_keq_pcp700,
        args=["E^0_PC", "F", "E^0_P700", "RT"],
    )
    m = m.add_derived(
        "keq_FAFd",
        fn=_keq_faf_d,
        args=["E^0_FA", "F", "E^0_Fd", "RT"],
    )
    m = m.add_derived(
        "B3",
        fn=_moiety_3,
        args=["B0", "B1", "B2", "PSII_total"],
    )
    m = m.add_derived(
        "P700+FA",
        fn=_moiety_3,
        args=["P700FA-", "P700FA", "P700+FA-", "PSI_total"],
    )
    m = m.add_derived(
        "rel_P700+FA",
        fn=_normalize_concentration,
        args=["P700+FA", "PSI_total"],
    )
    m = m.add_derived(
        "rel_P700FA",
        fn=_normalize_concentration,
        args=["P700FA", "PSI_total"],
    )
    m = m.add_derived(
        "rel_P700FA-",
        fn=_normalize_concentration,
        args=["P700FA-", "PSI_total"],
    )
    m = m.add_derived(
        "rel_P700+FA-",
        fn=_normalize_concentration,
        args=["P700+FA-", "PSI_total"],
    )
    m = m.add_derived(
        "rel_P700",
        fn=_normalize_2_concentrations,
        args=["P700+FA-", "P700+FA", "PSI_total"],
    )
    m = m.add_derived(
        "rel_P700+",
        fn=_normalize_2_concentrations,
        args=["P700+FA-", "P700+FA", "PSI_total"],
    )
    m = m.add_derived(
        "rel_B0",
        fn=_normalize_concentration,
        args=["B0", "PSII_total"],
    )
    m = m.add_derived(
        "rel_B1",
        fn=_normalize_concentration,
        args=["B1", "PSII_total"],
    )
    m = m.add_derived(
        "rel_B2",
        fn=_normalize_concentration,
        args=["B2", "PSII_total"],
    )
    m = m.add_derived(
        "rel_B3",
        fn=_normalize_concentration,
        args=["B3", "PSII_total"],
    )
    m = m.add_derived(
        "Fluo",
        fn=_fluo,
        args=["Q", "B0", "B2", "PSII_cross_section", "k2", "kF", "kH_Qslope", "kH0"],
    )
    m = m.add_derived(
        "keq_b6f_dyn",
        fn=_k_b6f,
        args=["pH_lumen", "pKreg", "b6f_content", "max_b6f"],
    )
    m = m.add_derived(
        "protons_lumen",
        fn=_protons_lumen,
        args=["pH_lumen"],
    )
    m = m.add_derived(
        "protons",
        fn=_protons_stroma_ebeling,
        args=["pH"],
    )
    m = m.add_derived(
        "ATP_pmf_activity",
        fn=_atp_pmf_activity2,
        args=["pK0E", "b", "pH_lumen", "pH", "F", "RT", "delta_psi"],
    )
    m = m.add_derived(
        "deltapH",
        fn=_deltap_h,
        args=["pH", "pH_lumen", "dG_pH"],
    )
    m = m.add_derived(
        "deltapH_in_volts",
        fn=_initial_delta_psi,
        args=["pH", "pH_lumen", "R", "F", "T"],
    )
    m = m.add_derived(
        "pmf",
        fn=_pmf,
        args=["deltapH", "delta_psi", "F"],
    )
    m = m.add_derived(
        "pmf_in_V",
        fn=_pmf_in_v,
        args=["delta_psi", "pH_lumen", "pH", "RT", "F"],
    )
    m = m.add_derived(
        "keq_b6f",
        fn=_keq_cytb6f,
        args=["pH_lumen", "pmf_in_V", "F", "E^0_PQ", "E^0_PC", "RT", "dG_pH"],
    )
    m = m.add_derived(
        "K_lumen",
        fn=_moiety_1,
        args=["K_stroma", "K_total"],
    )
    m = m.add_derived(
        "Cl_lumen",
        fn=_moiety_1,
        args=["Cl_stroma", "Cl_total"],
    )
    m = m.add_derived(
        "total_Cl_2",
        fn=_squared,
        args=["Cl_total"],
    )
    m = m.add_derived(
        "total_K_2",
        fn=_squared,
        args=["K_total"],
    )
    m = m.add_derived(
        "KEA3_reg",
        fn=_reg_kea,
        args=["pH", "ATP", "KEA3_pH_reg", "KEA3_ATP_treshold"],
    )
    m = m.add_derived(
        "dG_K_ions",
        fn=_dg_k,
        args=["K_lumen", "K_stroma", "delta_psi", "RT", "F"],
    )
    m = m.add_derived(
        "Cl_driving_force",
        fn=_cl_driving_force,
        args=["delta_psi", "Cl_lumen", "Cl_stroma", "RT", "F"],
    )
    m = m.add_derived(
        "Keq_NDH1",
        fn=_keq_ndh1,
        args=["pmf", "E^0_Fd", "F", "E^0_PQ", "pH", "dG_pH", "RT"],
    )
    m = m.add_derived(
        "ClCe_activation",
        fn=_cl_ce_activation,
        args=["ATP", "ClCe_ATP_threshold"],
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
        fn=_v_at_psynthase_mod,
        args=[
            "ATP",
            "ATPactivity",
            "ATP_pmf_activity",
            "k_ATPsynthase",
            "ADP",
            "kf_atp_synthase",
            "convf",
        ],
        stoichiometry={
            "ATP": Derived(fn=_value, args=["convf"]),
            "pH_lumen": 0.04666666666666667,
            "pH": -0.011666666666666667,
            "delta_psi": Derived(fn=_atp_div, args=["HPR", "volts_per_charge"]),
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
        "proton_leak",
        fn=_rate_leak,
        args=["protons_lumen", "pH", "kf_proton_leak"],
        stoichiometry={
            "pH_lumen": 0.01,
            "pH": -0.0025,
            "delta_psi": Derived(fn=_neg_one_div, args=["volts_per_charge"]),
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
        "toP700FA-",
        fn=_mass_action_22_rev,
        args=[
            "P700+FA-",
            "Plastocyanine (reduced)",
            "Plastocyanine (oxidised)",
            "P700FA-",
            "kPCox",
            "keq_PCP700",
        ],
        stoichiometry={"P700+FA-": -1, "P700FA-": 1, "Plastocyanine (oxidised)": 1},
    )
    m = m.add_reaction(
        "toP700FA_v3",
        fn=_mass_action_22_rev,
        args=[
            "P700FA-",
            "Ferredoxine (oxidised)",
            "P700FA",
            "Ferredoxine (reduced)",
            "kFdred",
            "keq_FAFd",
        ],
        stoichiometry={"P700FA-": -1, "Ferredoxine (oxidised)": -1, "P700FA": 1},
    )
    m = m.add_reaction(
        "toP700+FA",
        fn=_mass_action_22_rev,
        args=[
            "P700+FA-",
            "Ferredoxine (oxidised)",
            "P700+FA",
            "Ferredoxine (reduced)",
            "kFdred",
            "keq_FAFd",
        ],
        stoichiometry={"P700+FA-": -1, "Ferredoxine (oxidised)": -1},
    )
    m = m.add_reaction(
        "toP700FA_v5",
        fn=_mass_action_22_rev,
        args=[
            "P700+FA",
            "Plastocyanine (reduced)",
            "P700FA",
            "Plastocyanine (oxidised)",
            "kPCox",
            "keq_PCP700",
        ],
        stoichiometry={"P700FA": 1, "Plastocyanine (oxidised)": 1},
    )
    m = m.add_reaction(
        "PSI",
        fn=_v_ps1,
        args=["P700FA", "PSII_cross_section", "PPFD"],
        stoichiometry={"P700FA": -1, "P700+FA-": 1},
    )
    m = m.add_reaction(
        "mehler1",
        fn=_v_mehler,
        args=["P700FA-", "O2 (dissolved)_lumen", "kMehler"],
        stoichiometry={
            "H2O2": Derived(fn=_value, args=["convf"]),
            "P700FA": 2,
            "P700FA-": -2,
        },
    )
    m = m.add_reaction(
        "mehler2",
        fn=_v_mehler,
        args=["P700+FA-", "O2 (dissolved)_lumen", "kMehler"],
        stoichiometry={"H2O2": Derived(fn=_value, args=["convf"]), "P700+FA-": -2},
    )
    m = m.add_reaction(
        "B01",
        fn=_mass_action_2s,
        args=["B0", "PSII_cross_section", "PPFD"],
        stoichiometry={"B0": -1, "B1": 1},
    )
    m = m.add_reaction(
        "B10Q",
        fn=_kquencher,
        args=["B1", "Q", "kH_Qslope", "kH0"],
        stoichiometry={"B1": -1, "B0": 1},
    )
    m = m.add_reaction(
        "B10F",
        fn=_mass_action_1s,
        args=["B1", "kF"],
        stoichiometry={"B1": -1, "B0": 1},
    )
    m = m.add_reaction(
        "B12",
        fn=_mass_action_1s,
        args=["B1", "k2"],
        stoichiometry={
            "B1": -1,
            "B2": 1,
            "pH_lumen": -0.01,
            "delta_psi": Derived(fn=_one_div, args=["volts_per_charge"]),
        },
    )
    m = m.add_reaction(
        "B20",
        fn=_mass_action_22_rev,
        args=[
            "B2",
            "Plastoquinone (oxidised)",
            "Plastoquinone (reduced)",
            "B0",
            "kPQred",
            "keq_Plastoquinone (reduced)",
        ],
        stoichiometry={
            "B2": -1,
            "Plastoquinone (oxidised)": -0.5,
            "B0": 1,
            "pH": 0.0025,
        },
    )
    m = m.add_reaction(
        "B23",
        fn=_mass_action_2s,
        args=["B2", "PSII_cross_section", "PPFD"],
        stoichiometry={"B2": -1},
    )
    m = m.add_reaction(
        "B32F",
        fn=_mass_action_1s,
        args=["B3", "kF"],
        stoichiometry={"B2": 1},
    )
    m = m.add_reaction(
        "B32Q",
        fn=_kquencher,
        args=["B3", "Q", "kH_Qslope", "kH0"],
        stoichiometry={"B2": 1},
    )
    m = m.add_reaction(
        "b6f",
        fn=_vb6f_2024,
        args=[
            "Plastocyanine (oxidised)",
            "Plastocyanine (reduced)",
            "Plastoquinone (oxidised)",
            "Plastoquinone (reduced)",
            "keq_b6f_dyn",
            "keq_b6f",
        ],
        stoichiometry={
            "Plastocyanine (oxidised)": -2,
            "Plastoquinone (oxidised)": 1,
            "pH_lumen": -0.04,
            "pH": 0.01,
            "delta_psi": Derived(fn=_four_div, args=["volts_per_charge"]),
        },
    )
    m = m.add_reaction(
        "vATPactivity",
        fn=_v_at_pactivity,
        args=["ATPactivity", "PPFD", "kActATPase", "kDeactATPase"],
        stoichiometry={"ATPactivity": 1},
    )
    m = m.add_reaction(
        "vATP_shuttle",
        fn=_reversible_mass_action_1s_1p,
        args=["ADP", "ATP", "k_import_ATP", "kf_ex_atp"],
        stoichiometry={"ATP": 1},
    )
    m = m.add_reaction(
        "vNADPH_shuttle",
        fn=_reversible_mass_action_1s_1p,
        args=["NADP", "NADPH", "k_import_NADPH", "kf_ex_nadph"],
        stoichiometry={"NADPH": 1},
    )
    m = m.add_reaction(
        "KEA3",
        fn=_v_kea,
        args=["K_lumen", "protons_lumen", "K_stroma", "k_KEA", "protons", "KEA3_reg"],
        stoichiometry={"K_stroma": -1, "pH_lumen": 0.01, "pH": -0.0025},
    )
    m = m.add_reaction(
        "voltage_K_channel",
        fn=_v_voltage_k_channel,
        args=[
            "delta_psi",
            "K_lumen",
            "K_stroma",
            "dG_K_ions",
            "perm_K",
            "K_delta_psi_treshold",
        ],
        stoichiometry={
            "K_stroma": 1,
            "delta_psi": Derived(fn=_neg_one_div, args=["volts_per_charge"]),
        },
    )
    m = m.add_reaction(
        "VCCN1",
        fn=_v_vccn1,
        args=[
            "Cl_stroma",
            "Cl_lumen",
            "Cl_driving_force",
            "delta_psi",
            "k_VCCN1",
            "VCCN_delta_psi_treshold",
        ],
        stoichiometry={
            "Cl_stroma": -1,
            "delta_psi": Derived(fn=_neg_one_div, args=["volts_per_charge"]),
        },
    )
    m = m.add_reaction(
        "Cl_leak",
        fn=_v_cl_leak,
        args=[
            "k_Cl_leak",
            "Cl_lumen",
            "Cl_stroma",
            "Plastoquinone (oxidised)",
            "Cl_leak_PQ",
            "total_Cl_2",
        ],
        stoichiometry={
            "Cl_stroma": 1,
            "delta_psi": Derived(fn=_one_div, args=["volts_per_charge"]),
        },
    )
    m = m.add_reaction(
        "NDH1",
        fn=_v_ndh1,
        args=[
            "P700+FA-",
            "Ferredoxine (reduced)",
            "Plastoquinone (oxidised)",
            "pH_lumen",
            "k_NDH1",
        ],
        stoichiometry={
            "Ferredoxine (oxidised)": 2,
            "Plastoquinone (oxidised)": -1,
            "pH_lumen": -0.04,
            "pH": 0.01,
            "delta_psi": Derived(fn=_four_div, args=["volts_per_charge"]),
        },
    )
    m = m.add_reaction(
        "ClCe_bi",
        fn=_cl_ce_bi,
        args=["Cl_lumen", "Cl_stroma", "k_ClCe", "ClCe_activation"],
        stoichiometry={"Cl_stroma": -1},
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
    return m  # noqa: RET504
