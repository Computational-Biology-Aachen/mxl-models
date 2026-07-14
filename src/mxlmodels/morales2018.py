r"""Morales et al. 2018 dynamic leaf CO2 assimilation model.

mxlpy implementation skeleton:

- 16 state variables
- 7 environmental forcings as temporary parameters
- 103 model parameters
- light, temperature, fluorescence, NPQ, and first dynamic reaction equations
"""

import numpy as np
from mxlpy import Model

R_GAS = 8314.0
AIR_PRESSURE = 1.01e8
FLUX0 = 1e-22
TREF = 298.15
ZERO_PRESSURE = 100000.0
TZERO = 273.15
ES0 = 610780.0
ES_K = 17.269
ES_TREF = 237.3


def _safe_sqrt(
    x: float,
) -> float:
    return np.sqrt(max(x, 0.0))


def _safe_div(
    num: float,
    den: float,
    eps: float = 1e-30,
) -> float:
    return num / den if abs(den) > eps else num / eps


def _par(
    ib: float,
    ig: float,
    ir: float,
) -> float:
    return ib + ig + ir


def _para(
    ib: float,
    ig: float,
    ir: float,
    alphab: float,
    alphag: float,
    alphared: float,
) -> float:
    return ib * alphab + ig * alphag + ir * alphared


def _parap(
    ib: float,
    ig: float,
    ir: float,
    alphabp: float,
    alphagp: float,
    alpharp: float,
) -> float:
    return ib * alphabp + ig * alphagp + ir * alpharp


def _arrhenius(
    t: float,
    value25: float,
    ha: float,
) -> float:
    return value25 * np.exp((ha * (t - TREF)) / (TREF * R_GAS * t))


def _arrhenius_inverse(
    t: float,
    value25: float,
    ha: float,
) -> float:
    return value25 * np.exp((-(t - TREF) * ha) / (TREF * R_GAS * t))


def _peaked_arrhenius(
    t: float,
    value25: float,
    ha: float,
    hd: float,
    s: float,
) -> float:
    activation = np.exp((ha * (t - TREF)) / (TREF * R_GAS * t))
    deactivation_ref = 1.0 + np.exp((TREF * s - hd) / (TREF * R_GAS))
    deactivation_t = 1.0 + np.exp((t * s - hd) / (t * R_GAS))
    return value25 * activation * deactivation_ref / deactivation_t


def _phiqe(
    f_p: float,
    f_z: float,
    gamma1: float,
    gamma2: float,
    gamma3: float,
    phiq_emax: float,
) -> float:
    return (f_p * gamma1 + f_p * f_z * gamma2 + f_z * gamma3) * phiq_emax


def _rp(
    pr: float,
    k_pr: float,
) -> float:
    return 0.5 * pr * k_pr


def _sc(
    scm: float,
    falpha_sc: float,
    alphar: float,
) -> float:
    return scm * falpha_sc * alphar


def _gc(
    sc: float,
    sm: float,
    gcm: float,
) -> float:
    return (sc / sm) * gcm


def _kmapp_rubp(
    km_ru_bp: float,
    pga: float,
    vch: float,
    ki_pga: float,
) -> float:
    return km_ru_bp * (1.0 + pga / (vch * ki_pga))


def _frca(
    tl: float,
    d_hd_rca: float,
    to_rca: float,
    d_ha_rca: float,
) -> float:
    return (d_hd_rca * np.exp(((tl - to_rca) * d_ha_rca) / (to_rca * R_GAS * tl))) / (
        d_hd_rca
        - d_ha_rca * (1.0 - np.exp((d_hd_rca * (tl - to_rca)) / (to_rca * R_GAS * tl)))
    )


def _fmax(
    rca: float,
    f_rca: float,
    ka_rca: float,
) -> float:
    return (rca * f_rca) / (rca * f_rca + ka_rca)


def _para_p2(
    sigma2: float,
    alphar: float,
    pa_ra_p: float,
) -> float:
    return sigma2 * alphar * pa_ra_p


def _frubp(
    rb: float,
    vch: float,
    kmapp_ru_bp: float,
    ru_bp: float,
) -> float:
    term = rb / vch + kmapp_ru_bp + ru_bp / vch
    inner = term**2 - (4.0 * rb * ru_bp) / (vch**2)
    return (1.0 / ((2.0 * rb) / vch)) * (term - _safe_sqrt(inner))


def _phi(
    kmc: float,
    ko: float,
    o2: float,
    kmo: float,
    kc: float,
    cc: float,
) -> float:
    return _safe_div(kmc * ko * o2, kmo * kc * cc)


def _vc(
    f_rb: float,
    f_ru_bp: float,
    kc: float,
    rb: float,
    cc: float,
    kmc: float,
    o2: float,
    kmo: float,
) -> float:
    return _safe_div(f_rb * f_ru_bp * kc * rb * cc, cc + kmc * (1.0 + o2 / kmo))


def _vr_tpu(
    tpu: float,
    phi: float,
) -> float:
    return _safe_div(3.0 * tpu * (2.0 + 1.5 * phi), 1.0 - 0.5 * phi)


def _vre(
    f_r: float,
    vrmax: float,
    pga: float,
    km_pga: float,
) -> float:
    return _safe_div(f_r * vrmax * pga, pga + km_pga)


def _a(
    ci: float,
    ccyt: float,
    gw: float,
) -> float:
    return (ci - ccyt) * gw


def _gm(
    a: float,
    ci: float,
    cc: float,
) -> float:
    return _safe_div(a, ci - cc)


def _fm_d(
    kf: float,
    k_dinh: float,
) -> float:
    return kf / (kf + k_dinh)


def _fm_a(
    kf: float,
    k_d0: float,
) -> float:
    return kf / (kf + k_d0)


def _fmp_d(
    alphar: float,
    kf: float,
    k_dinh: float,
) -> float:
    return (alphar * kf) / (kf + k_dinh)


def _fo_d(
    kf: float,
    k_dinh: float,
) -> float:
    return kf / (kf + k_dinh)


def _fo_a(
    kf: float,
    k_d0: float,
    kp: float,
) -> float:
    return kf / (kf + k_d0 + kp)


def _fop_d(
    alphar: float,
    kf: float,
    k_dinh: float,
) -> float:
    return (alphar * kf) / (kf + k_dinh)


def _frbss_nr(
    ac: float,
    bc: float,
    cc: float,
) -> float:
    return min(1.0, ac + bc * cc)


def _qi(
    fm_a: float,
    psi_id: float,
    fmp_d: float,
) -> float:
    return _safe_div(fm_a, (1.0 - psi_id) * fm_a + psi_id * fmp_d) - 1.0


def _alpharss(
    ib: float,
    iac: float,
    alpharac: float,
    alphar_alpha: float,
    alpharav: float,
    thetaalphar: float,
) -> float:
    term = alphar_alpha * (ib - iac) + alpharav
    acclim = (1.0 + alpharac) - (
        term
        - _safe_sqrt(term**2 - 4.0 * alphar_alpha * thetaalphar * alpharav * (ib - iac))
    ) / (2.0 * thetaalphar)
    return min(1.0 + (ib / iac) * alpharac, acclim)


def _kinh(
    kinh0: float,
    fprot: float,
    phiq_e: float,
) -> float:
    return max(kinh0 - fprot * phiq_e, 0.0)


def _phi_iid(
    fm_a: float,
    fo_a: float,
) -> float:
    return (fm_a - fo_a) / fm_a


def _fo(
    psi_id: float,
    fo_a: float,
    fo_d: float,
) -> float:
    return (1.0 - psi_id) * fo_a + psi_id * fo_d


def _frss(
    f_r0: float,
    alphaf_r: float,
    pa_ra: float,
    thetaf_r: float,
) -> float:
    term = alphaf_r * pa_ra + (1.0 - f_r0)
    return f_r0 + (
        term - _safe_sqrt(term**2 - 4.0 * alphaf_r * thetaf_r * pa_ra * (1.0 - f_r0))
    ) / (2.0 * thetaf_r)


def _fm(
    psi_id: float,
    fm_a: float,
    fm_d: float,
) -> float:
    return (1.0 - psi_id) * fm_a + psi_id * fm_d


def _kd(
    kp: float,
    phi_i_id: float,
    phiq_e: float,
    kf: float,
) -> float:
    return (kp / (phi_i_id - phiq_e) - kf) - kp


def _fmp_a(
    alphar: float,
    kf: float,
    k_d: float,
) -> float:
    return (alphar * kf) / (kf + k_d)


def _phi_iiop(
    fm: float,
    fo: float,
) -> float:
    return (fm - fo) / fm


def _fmp(
    psi_id: float,
    fmp_a: float,
    fmp_d: float,
) -> float:
    return (1.0 - psi_id) * fmp_a + psi_id * fmp_d


def _phi_iio(
    phi_i_iop: float,
    phiq_e: float,
) -> float:
    return phi_i_iop - phiq_e


def _j2pp(
    phi_i_iop: float,
    pa_ra_p2: float,
    jmax: float,
    theta: float,
) -> float:
    term = phi_i_iop * pa_ra_p2 + jmax
    return (term - _safe_sqrt(term**2 - 4.0 * phi_i_iop * theta * jmax * pa_ra_p2)) / (
        2.0 * theta
    )


def _j2pm(
    vr_tpu: float,
    vr_e: float,
    fpseudo: float,
    fcyc: float,
    phi: float,
) -> float:
    limitation = min(vr_tpu, vr_e)
    return (
        ((limitation / (1.0 - fpseudo / (1.0 - fcyc))) * 2.0) / (2.0 + 1.5 * phi)
    ) * (2.0 + 2.0 * phi)


def _fop_a(
    alphar: float,
    kf: float,
    k_d: float,
    kp: float,
) -> float:
    return (alphar * kf) / (kf + k_d + kp)


def _qpp(
    pa_ra_p2: float,
    j2pp: float,
    phi_i_iop: float,
) -> float:
    return (j2pp / pa_ra_p2) / phi_i_iop if pa_ra_p2 > FLUX0 else 1.0


def _npq(
    fm_a: float,
    fmp: float,
) -> float:
    return fm_a / fmp - 1.0


def _qpm(
    j2pm: float,
    j2pp: float,
    q_pp: float,
) -> float:
    return _safe_div(j2pm, j2pp) * q_pp


def _qm(
    npq: float,
    fm_a: float,
    fmp: float,
    alphar: float,
) -> float:
    return npq - ((fm_a / fmp) * alphar - 1.0)


def _q_p_no_qd(
    q_pm: float,
    q_pp: float,
) -> float:
    return min(q_pm, q_pp)


def _qe(
    npq: float,
    q_m: float,
    q_i: float,
) -> float:
    return (npq - q_m) - q_i


def _fqess(
    q_pno_q_d: float,
) -> float:
    return 1.0 - q_pno_q_d


def _phiqess(
    fq_ess: float,
    gamma1: float,
    gamma2: float,
    gamma3: float,
    phiq_emax: float,
) -> float:
    return (fq_ess * gamma1 + fq_ess * fq_ess * gamma2 + fq_ess * gamma3) * phiq_emax


def _phi_iioss(
    phi_i_iop: float,
    phiq_ess: float,
) -> float:
    return phi_i_iop - phiq_ess


def _j2qe(
    phi_i_io: float,
    phi_i_ioss: float,
    j2pp: float,
) -> float:
    return (
        (1.0 - (phi_i_ioss - phi_i_io) / phi_i_ioss) * j2pp
        if phi_i_io < phi_i_ioss
        else j2pp
    )


def _vrj(
    j2q_e: float,
    fpseudo: float,
    fcyc: float,
    phi: float,
) -> float:
    return (((j2q_e * (1.0 - fpseudo / (1.0 - fcyc))) / 2.0) * (2.0 + 1.5 * phi)) / (
        2.0 + 2.0 * phi
    )


def _j2(
    j2q_e: float,
    j2pm: float,
) -> float:
    return min(j2q_e, j2pm)


def _frbss_r(
    par: float,
    vr_tpu: float,
    vr_j: float,
    phi: float,
    vc: float,
    f_rb: float,
    f_ru_bp: float,
    f_r_bmin: float,
    f_r_bmax: float,
) -> float:
    if par <= FLUX0:
        val = f_r_bmin
    else:
        val = _safe_div(min(vr_tpu, vr_j) / (2.0 + 1.5 * phi), vc / (f_rb * f_ru_bp))
    return min(val, f_r_bmax)


def _vr(
    vr_j: float,
    vr_tpu: float,
    vr_e: float,
) -> float:
    return min(vr_j, vr_tpu, vr_e)


def _qp(
    j2: float,
    pa_ra_p2: float,
    phi_i_io: float,
) -> float:
    return _safe_div(j2 / pa_ra_p2, phi_i_io)


def _phi_ii(
    pa_ra_p2: float,
    j2: float,
    phi_i_io: float,
) -> float:
    return j2 / pa_ra_p2 if pa_ra_p2 > FLUX0 else phi_i_io


def _reglimit(
    vr_j: float,
    vr_tpu: float,
    vr: float,
) -> float:
    if abs(vr_j - vr) < FLUX0:
        return 1.0
    if abs(vr_tpu - vr) < FLUX0:
        return 2.0
    return 3.0


def _frbss(
    f_r_bss_nr: float,
    f_r_bss_r: float,
) -> float:
    return min(f_r_bss_nr, f_r_bss_r)


def _d_pga_dt(
    vc: float,
    phi: float,
    vr: float,
) -> float:
    return (2.0 * vc + 1.5 * vc * phi) - vr


def _d_rubp_dt(
    phi: float,
    vr: float,
    vc: float,
) -> float:
    return ((1.0 + phi) / (2.0 + 1.5 * phi)) * vr - vc * (1.0 + phi)


def _d_pr_dt(
    vc: float,
    phi: float,
    pr: float,
    k_pr: float,
) -> float:
    return vc * phi - pr * k_pr


def _d_f_p_dt(
    fq_ess: float,
    f_p: float,
    kiq_ep: float,
    kdq_ep: float,
) -> float:
    return (fq_ess - f_p) * kiq_ep if fq_ess > f_p else (fq_ess - f_p) * kdq_ep


def _d_f_z_dt(
    fq_ess: float,
    f_z: float,
    kiq_ez: float,
    kdq_ez: float,
) -> float:
    return (fq_ess - f_z) * kiq_ez if fq_ess > f_z else (fq_ess - f_z) * kdq_ez


def _d_alphar_dt(
    alpharss: float,
    alphar: float,
    kialpha: float,
    kdalpha: float,
) -> float:
    return (
        (alpharss - alphar) * kialpha
        if alpharss > alphar
        else (alpharss - alphar) * kdalpha
    )


def _d_psiid_dt(
    psi_id: float,
    pa_ra: float,
    alphar: float,
    kinh: float,
    krep: float,
) -> float:
    return (1.0 - psi_id) * pa_ra * alphar * kinh - psi_id * krep


def _d_f_r_dt(
    f_rss: float,
    f_r: float,
    ki_r: float,
    kd_r: float,
) -> float:
    return (f_rss - f_r) * ki_r if f_rss > f_r else (f_rss - f_r) * kd_r


def _d_f_rb_dt(
    f_r_bss: float,
    f_rb: float,
    krca: float,
    rca: float,
    f_rca: float,
    kd_rb: float,
) -> float:
    return (
        (f_r_bss - f_rb) * krca * rca * f_rca
        if f_r_bss > f_rb
        else (f_r_bss - f_rb) * kd_rb
    )


def _gbc(
    gbw: float,
) -> float:
    return gbw / 1.37


def _gsc(
    gsw: float,
) -> float:
    return gsw / 1.56


def _mv(
    ta: float,
) -> float:
    return (R_GAS * ta) / AIR_PRESSURE


def _identity(
    x: float,
) -> float:
    return x


def _ea(
    h2_or: float,
) -> float:
    return h2_or * AIR_PRESSURE


def _photo(
    flow: float,
    co2_r: float,
    ca: float,
    leaf_surface: float,
) -> float:
    return (flow * co2_r - flow * ca) / leaf_surface


def _trmmol(
    flow: float,
    h2_os: float,
    h2_or: float,
    leaf_surface: float,
) -> float:
    return (flow * (h2_os - h2_or)) / (leaf_surface * (1.0 - h2_os))


def _es_leaf(
    tl: float,
) -> float:
    return ES0 * np.exp((ES_K * (tl - TZERO)) / (ES_TREF + (tl - TZERO)))


def _vpdleaf(
    es_leaf: float,
    ea: float,
) -> float:
    return max(es_leaf - ea, ZERO_PRESSURE)


def _fvpd(
    vp_dleaf: float,
    d0: float,
) -> float:
    return 1.0 / (1.0 + vp_dleaf / d0)


def _gtw(
    trmmol: float,
    es_leaf: float,
    ea: float,
    vp_dleaf: float,
) -> float:
    return (trmmol * (AIR_PRESSURE - (es_leaf + ea) / 2.0)) / vp_dleaf


def _transpiration(
    vp_dleaf: float,
    es_leaf: float,
    ea: float,
    gsw: float,
    gbw: float,
) -> float:
    return ((vp_dleaf / (AIR_PRESSURE - (es_leaf + ea) / 2.0)) * 1.0) / (
        1.0 / gsw + 1.0 / gbw
    )


def _fi(
    f_i0: float,
    alphaf_i: float,
    par: float,
    thetaf_i: float,
) -> float:
    fI_a = thetaf_i
    fI_b = -(1.0 + f_i0 + alphaf_i * par)
    fI_c = f_i0 + alphaf_i * par
    return (-fI_b - _safe_sqrt(fI_b**2 - 4.0 * fI_a * fI_c)) / (2.0 * fI_a)


def _gss(
    f_i: float,
    fvpd: float,
    gswm: float,
) -> float:
    return f_i * fvpd * gswm


def _cond(
    gtw: float,
    gbw: float,
) -> float:
    return 1.0 / (1.0 / gtw - 1.0 / gbw)


def _d_cc_dt(
    ccyt: float,
    cc: float,
    gc: float,
    vc: float,
    mv: float,
    vref: float,
) -> float:
    return (((ccyt - cc) * gc - vc) * mv) / vref


def _d_ccyt_dt(
    ci: float,
    ccyt: float,
    gw: float,
    rp: float,
    rm: float,
    cc: float,
    gc: float,
    mv: float,
    vref: float,
) -> float:
    return ((((ci - ccyt) * gw + rp + rm) - (ccyt - cc) * gc) * mv) / vref


def _d_ci_dt(
    ca: float,
    ci: float,
    gsc: float,
    gbc: float,
    ccyt: float,
    gw: float,
    mv: float,
    vref: float,
) -> float:
    return (((ca - ci) / (1.0 / gsc + 1.0 / gbc) - (ci - ccyt) * gw) * mv) / vref


def _d_ca_dt(
    flow: float,
    ca: float,
    co2_r: float,
    leaf_surface: float,
    a: float,
    ta: float,
    volume_chamber: float,
) -> float:
    return (
        (((-flow * ca + flow * co2_r) - leaf_surface * a) * R_GAS * ta) / volume_chamber
    ) / AIR_PRESSURE


def _d_h2os_dt(
    flow: float,
    leaf_surface: float,
    transpiration: float,
    h2_os: float,
    h2_or: float,
    ta: float,
    volume_chamber: float,
) -> float:
    numerator = (
        -(flow + leaf_surface * transpiration) * h2_os
        + flow * h2_or
        + leaf_surface * transpiration
    )
    return ((numerator * R_GAS * ta) / volume_chamber) / AIR_PRESSURE


def _d_gsw_dt(
    gss: float,
    gsw: float,
    kgsi: float,
    kgsd: float,
) -> float:
    return (gss - gsw) * kgsi if gss > gsw else (gss - gsw) * kgsd


def get_morales2018() -> Model:
    r"""Return Morales et al. 2018 dynamic photosynthesis model."""
    m: Model = Model()
    m.add_variables(
        {
            "PGA": 0.00005,
            "RuBP": 0.00005,
            "fRB": 0.25,
            "fP": 0.0,
            "fZ": 0.0,
            "alphar": 1.0,
            "PSIId": 0.0,
            "fR": 0.0,
            "PR": 0.0,
            "Cc": 0.00038,
            "Ccyt": 0.00038,
            "Ci": 0.00038,
            "Ca": 0.00038,
            "H2OS": 0.02000,
            "gsw": 0.09000,
            "sumA": 0.0,
        }
    )

    # Forcings
    m.add_parameters(
        {
            "Ib": 100.0,
            "Ig": 0.0,
            "Ir": 900.0,
            "Ta": 298.15,
            "Tl": 298.15,
            "H2OR": 0.015,
            "CO2R": 0.00038,
        }
    )

    m.add_parameters(
        {
            "sigma2": 5.0000e-01,
            "Jmax25": 1.3928e-04,
            "DHaJmax": 3.6210e07,
            "DHdJmax": 2.1590e08,
            "DsJmax": 6.9000e05,
            "kD0": 4.5500e08,
            "kDinh": 5.0000e09,
            "kf": 5.6000e07,
            "kp": 2.6540e09,
            "fcyc": 1.0000e-01,
            "fpseudo": 1.0000e-01,
            "theta": 7.0000e-01,
            "gamma1": 2.0000e-01,
            "gamma2": 6.0000e-01,
            "gamma3": 2.0000e-01,
            "PhiqEmax": 2.0000e-01,
            "KiqEp": 1.8700e-02,
            "KdqEp": 2.3900e-02,
            "KiqEz": 1.8700e-03,
            "KdqEz": 2.3900e-03,
            "Kinh0": 1.0000e-01,
            "fprot": 1.0000e-01,
            "Krep25": 1.9200e-04,
            "DHaKrep": 1.6080e08,
            "DHdKrep": 2.3323e08,
            "DsKrep": 7.8000e05,
            "alphar_alpha25": 6.5500e03,
            "DHaAlphar": 6.7320e07,
            "DHaKalpha": 9.0500e07,
            "DsKalpha": 1.0800e06,
            "DHdKalpha": 3.2800e08,
            "Iac": 1.6000e-06,
            "alpharac": 5.0000e-02,
            "alpharav": 2.5000e-01,
            "thetaalphar": 3.6000e-01,
            "Kialpha25": 1.4900e-03,
            "Kdalpha25": 1.8600e-03,
            "fR0": 4.0000e-02,
            "alphafR": 2.5000e03,
            "thetafR": 9.6000e-01,
            "KiR": 6.2800e-03,
            "KdR": 7.5000e-03,
            "Vrmax": 1.1865e-04,
            "KmPGA": 5.0000e-06,
            "RB": 1.5900e-05,
            "Kc25": 4.1600e00,
            "DHaKc": 4.1820e07,
            "Ko25": 1.2600e00,
            "DHaKo": 5.5150e07,
            "Kmc25": 2.6170e-04,
            "DHaKmc": 4.9430e07,
            "Kmo25": 1.9850e-01,
            "DHaKmo": 2.9080e07,
            "KaRCA": 1.0200e-02,
            "ac": 2.7000e-01,
            "bc": 1.4000e04,
            "KdRB": 6.8000e-04,
            "Krca": 8.6300e-02,
            "fRBmin": 4.8000e-01,
            "O2": 2.1000e-01,
            "RCA": 1.1737e-01,
            "DHdRCA": 2.9020e08,
            "ToRCA": 3.0040e02,
            "DHaRCA": 3.0000e07,
            "KmRuBP": 2.0000e-02,
            "Vch": 1.0000e-05,
            "KiPGA": 8.4000e-01,
            "TPU25": 7.4700e-06,
            "DHaTPU": 5.7500e07,
            "DHdTPU": 2.4670e08,
            "DsTPU": 7.9000e05,
            "DHaGc": 7.0200e07,
            "DHdGc": 9.4000e07,
            "DsGc": 3.2000e05,
            "DHaGw": 7.0200e07,
            "DHdGw": 9.4000e07,
            "DsGw": 3.2000e05,
            "Rm25": 9.9000e-07,
            "DHaRm": 5.6200e07,
            "kPR": 2.4000e-02,
            "Vref": 1.5500e-04,
            "Scm": 7.1000e00,
            "Sm": 9.8000e00,
            "falphaSc": 9.3000e-01,
            "gcm25": 3.9000e-01,
            "gw25": 7.5000e-01,
            "D0": 7.4000e05,
            "fI0": 3.9000e-01,
            "gswm": 4.8000e-01,
            "alphafI": 7.6700e02,
            "thetafI": 8.8000e-01,
            "Kgsi": 1.1400e-03,
            "Kgsd": 1.1400e-03,
            "gbw": 9.2000e00,
            "volume_chamber": 8.0000e-05,
            "leaf_surface": 2.0000e-04,
            "Flow": 5.0000e-04,
            "alphab": 9.2000e-01,
            "alphag": 7.2000e-01,
            "alphared": 8.3000e-01,
            "alphabp": 6.6000e-01,
            "alphagp": 6.0000e-01,
            "alpharp": 8.0000e-01,
        }
    )

    m.add_derived(
        "PAR",
        fn=_par,
        args=["Ib", "Ig", "Ir"],
    )
    m.add_derived(
        "PARa",
        fn=_para,
        args=["Ib", "Ig", "Ir", "alphab", "alphag", "alphared"],
    )
    m.add_derived(
        "PARaP",
        fn=_parap,
        args=["Ib", "Ig", "Ir", "alphabp", "alphagp", "alpharp"],
    )
    m.add_derived(
        "Jmax",
        fn=_peaked_arrhenius,
        args=["Tl", "Jmax25", "DHaJmax", "DHdJmax", "DsJmax"],
    )
    m.add_derived(
        "Krep",
        fn=_peaked_arrhenius,
        args=["Tl", "Krep25", "DHaKrep", "DHdKrep", "DsKrep"],
    )
    m.add_derived(
        "TPU",
        fn=_peaked_arrhenius,
        args=["Tl", "TPU25", "DHaTPU", "DHdTPU", "DsTPU"],
    )
    m.add_derived(
        "Kc",
        fn=_arrhenius,
        args=["Tl", "Kc25", "DHaKc"],
    )
    m.add_derived(
        "Ko",
        fn=_arrhenius,
        args=["Tl", "Ko25", "DHaKo"],
    )
    m.add_derived(
        "Kmc",
        fn=_arrhenius,
        args=["Tl", "Kmc25", "DHaKmc"],
    )
    m.add_derived(
        "Kmo",
        fn=_arrhenius,
        args=["Tl", "Kmo25", "DHaKmo"],
    )
    m.add_derived(
        "Rm",
        fn=_arrhenius,
        args=["Tl", "Rm25", "DHaRm"],
    )
    m.add_derived(
        "gcm",
        fn=_peaked_arrhenius,
        args=["Tl", "gcm25", "DHaGc", "DHdGc", "DsGc"],
    )
    m.add_derived(
        "gw",
        fn=_peaked_arrhenius,
        args=["Tl", "gw25", "DHaGw", "DHdGw", "DsGw"],
    )
    m.add_derived(
        "alphar_alpha",
        fn=_arrhenius_inverse,
        args=["Tl", "alphar_alpha25", "DHaAlphar"],
    )
    m.add_derived(
        "Kialpha",
        fn=_peaked_arrhenius,
        args=["Tl", "Kialpha25", "DHaKalpha", "DHdKalpha", "DsKalpha"],
    )
    m.add_derived(
        "Kdalpha",
        fn=_peaked_arrhenius,
        args=["Tl", "Kdalpha25", "DHaKalpha", "DHdKalpha", "DsKalpha"],
    )
    m.add_derived(
        "PhiqE",
        fn=_phiqe,
        args=["fP", "fZ", "gamma1", "gamma2", "gamma3", "PhiqEmax"],
    )
    m.add_derived(
        "Rp",
        fn=_rp,
        args=["PR", "kPR"],
    )
    m.add_derived(
        "Sc",
        fn=_sc,
        args=["Scm", "falphaSc", "alphar"],
    )
    m.add_derived(
        "gc",
        fn=_gc,
        args=["Sc", "Sm", "gcm"],
    )
    m.add_derived(
        "Kmapp_RuBP",
        fn=_kmapp_rubp,
        args=["KmRuBP", "PGA", "Vch", "KiPGA"],
    )
    m.add_derived(
        "fRCA",
        fn=_frca,
        args=["Tl", "DHdRCA", "ToRCA", "DHaRCA"],
    )
    m.add_derived(
        "fRBmax",
        fn=_fmax,
        args=["RCA", "fRCA", "KaRCA"],
    )
    m.add_derived(
        "PARaP2",
        fn=_para_p2,
        args=["sigma2", "alphar", "PARaP"],
    )
    m.add_derived(
        "fRuBP",
        fn=_frubp,
        args=["RB", "Vch", "Kmapp_RuBP", "RuBP"],
    )
    m.add_derived(
        "phi",
        fn=_phi,
        args=["Kmc", "Ko", "O2", "Kmo", "Kc", "Cc"],
    )
    m.add_derived(
        "Vc",
        fn=_vc,
        args=["fRB", "fRuBP", "Kc", "RB", "Cc", "Kmc", "O2", "Kmo"],
    )
    m.add_derived(
        "VrTPU",
        fn=_vr_tpu,
        args=["TPU", "phi"],
    )
    m.add_derived(
        "VrE",
        fn=_vre,
        args=["fR", "Vrmax", "PGA", "KmPGA"],
    )
    m.add_derived(
        "A",
        fn=_a,
        args=["Ci", "Ccyt", "gw"],
    )
    m.add_derived(
        "gm",
        fn=_gm,
        args=["A", "Ci", "Cc"],
    )
    m.add_derived(
        "Fm_d",
        fn=_fm_d,
        args=["kf", "kDinh"],
    )
    m.add_derived(
        "Fm_a",
        fn=_fm_a,
        args=["kf", "kD0"],
    )
    m.add_derived(
        "Fmp_d",
        fn=_fmp_d,
        args=["alphar", "kf", "kDinh"],
    )
    m.add_derived(
        "Fo_d",
        fn=_fo_d,
        args=["kf", "kDinh"],
    )
    m.add_derived(
        "Fo_a",
        fn=_fo_a,
        args=["kf", "kD0", "kp"],
    )
    m.add_derived(
        "Fop_d",
        fn=_fop_d,
        args=["alphar", "kf", "kDinh"],
    )
    m.add_derived(
        "fRBss_nr",
        fn=_frbss_nr,
        args=["ac", "bc", "Cc"],
    )
    m.add_derived(
        "qI",
        fn=_qi,
        args=["Fm_a", "PSIId", "Fmp_d"],
    )
    m.add_derived(
        "alpharss",
        fn=_alpharss,
        args=["Ib", "Iac", "alpharac", "alphar_alpha", "alpharav", "thetaalphar"],
    )
    m.add_derived(
        "Kinh",
        fn=_kinh,
        args=["Kinh0", "fprot", "PhiqE"],
    )
    m.add_derived(
        "PhiIId",
        fn=_phi_iid,
        args=["Fm_a", "Fo_a"],
    )
    m.add_derived(
        "Fo",
        fn=_fo,
        args=["PSIId", "Fo_a", "Fo_d"],
    )
    m.add_derived(
        "fRss",
        fn=_frss,
        args=["fR0", "alphafR", "PARa", "thetafR"],
    )
    m.add_derived(
        "Fm",
        fn=_fm,
        args=["PSIId", "Fm_a", "Fm_d"],
    )
    m.add_derived(
        "kD",
        fn=_kd,
        args=["kp", "PhiIId", "PhiqE", "kf"],
    )
    m.add_derived(
        "Fmp_a",
        fn=_fmp_a,
        args=["alphar", "kf", "kD"],
    )
    m.add_derived(
        "PhiIIop",
        fn=_phi_iiop,
        args=["Fm", "Fo"],
    )
    m.add_derived(
        "Fmp",
        fn=_fmp,
        args=["PSIId", "Fmp_a", "Fmp_d"],
    )
    m.add_derived(
        "PhiIIo",
        fn=_phi_iio,
        args=["PhiIIop", "PhiqE"],
    )
    m.add_derived(
        "J2pp",
        fn=_j2pp,
        args=["PhiIIop", "PARaP2", "Jmax", "theta"],
    )
    m.add_derived(
        "J2pm",
        fn=_j2pm,
        args=["VrTPU", "VrE", "fpseudo", "fcyc", "phi"],
    )
    m.add_derived(
        "Fop_a",
        fn=_fop_a,
        args=["alphar", "kf", "kD", "kp"],
    )
    m.add_derived(
        "qPp",
        fn=_qpp,
        args=["PARaP2", "J2pp", "PhiIIop"],
    )
    m.add_derived(
        "NPQ",
        fn=_npq,
        args=["Fm_a", "Fmp"],
    )
    m.add_derived(
        "qPm",
        fn=_qpm,
        args=["J2pm", "J2pp", "qPp"],
    )
    m.add_derived(
        "qM",
        fn=_qm,
        args=["NPQ", "Fm_a", "Fmp", "alphar"],
    )
    m.add_derived(
        "qPno_qD",
        fn=_q_p_no_qd,
        args=["qPm", "qPp"],
    )
    m.add_derived(
        "qE",
        fn=_qe,
        args=["NPQ", "qM", "qI"],
    )
    m.add_derived(
        "fqEss",
        fn=_fqess,
        args=["qPno_qD"],
    )
    m.add_derived(
        "PhiqEss",
        fn=_phiqess,
        args=["fqEss", "gamma1", "gamma2", "gamma3", "PhiqEmax"],
    )
    m.add_derived(
        "PhiIIoss",
        fn=_phi_iioss,
        args=["PhiIIop", "PhiqEss"],
    )
    m.add_derived(
        "J2qE",
        fn=_j2qe,
        args=["PhiIIo", "PhiIIoss", "J2pp"],
    )
    m.add_derived(
        "VrJ",
        fn=_vrj,
        args=["J2qE", "fpseudo", "fcyc", "phi"],
    )
    m.add_derived(
        "J2",
        fn=_j2,
        args=["J2qE", "J2pm"],
    )
    m.add_derived(
        "fRBss_r",
        fn=_frbss_r,
        args=["PAR", "VrTPU", "VrJ", "phi", "Vc", "fRB", "fRuBP", "fRBmin", "fRBmax"],
    )
    m.add_derived(
        "Vr",
        fn=_vr,
        args=["VrJ", "VrTPU", "VrE"],
    )
    m.add_derived(
        "qP",
        fn=_qp,
        args=["J2", "PARaP2", "PhiIIo"],
    )
    m.add_derived(
        "PhiII",
        fn=_phi_ii,
        args=["PARaP2", "J2", "PhiIIo"],
    )
    m.add_derived(
        "reg_limit",
        fn=_reglimit,
        args=["VrJ", "VrTPU", "Vr"],
    )
    m.add_derived(
        "fRBss",
        fn=_frbss,
        args=["fRBss_nr", "fRBss_r"],
    )

    # Gas-exchange and chamber-derived quantities
    m.add_derived(
        "gbc",
        fn=_gbc,
        args=["gbw"],
    )
    m.add_derived(
        "gsc",
        fn=_gsc,
        args=["gsw"],
    )
    m.add_derived(
        "Mv",
        fn=_mv,
        args=["Ta"],
    )
    m.add_derived(
        "ea",
        fn=_ea,
        args=["H2OR"],
    )
    m.add_derived(
        "Photo",
        fn=_photo,
        args=["Flow", "CO2R", "Ca", "leaf_surface"],
    )
    m.add_derived(
        "Trmmol",
        fn=_trmmol,
        args=["Flow", "H2OS", "H2OR", "leaf_surface"],
    )
    m.add_derived(
        "es_leaf",
        fn=_es_leaf,
        args=["Tl"],
    )
    m.add_derived(
        "VPDleaf",
        fn=_vpdleaf,
        args=["es_leaf", "ea"],
    )
    m.add_derived(
        "fvpd",
        fn=_fvpd,
        args=["VPDleaf", "D0"],
    )
    m.add_derived(
        "gtw",
        fn=_gtw,
        args=["Trmmol", "es_leaf", "ea", "VPDleaf"],
    )
    m.add_derived(
        "transpiration",
        fn=_transpiration,
        args=["VPDleaf", "es_leaf", "ea", "gsw", "gbw"],
    )
    m.add_derived(
        "fI",
        fn=_fi,
        args=["fI0", "alphafI", "PAR", "thetafI"],
    )
    m.add_derived(
        "gss",
        fn=_gss,
        args=["fI", "fvpd", "gswm"],
    )
    m.add_derived(
        "Cond",
        fn=_cond,
        args=["gtw", "gbw"],
    )

    m.add_reaction(
        "dPGA_dt",
        fn=_d_pga_dt,
        args=["Vc", "phi", "Vr"],
        stoichiometry={"PGA": 1.0},
    )
    m.add_reaction(
        "dRuBP_dt",
        fn=_d_rubp_dt,
        args=["phi", "Vr", "Vc"],
        stoichiometry={"RuBP": 1.0},
    )
    m.add_reaction(
        "dPR_dt",
        fn=_d_pr_dt,
        args=["Vc", "phi", "PR", "kPR"],
        stoichiometry={"PR": 1.0},
    )
    m.add_reaction(
        "dfP_dt",
        fn=_d_f_p_dt,
        args=["fqEss", "fP", "KiqEp", "KdqEp"],
        stoichiometry={"fP": 1.0},
    )
    m.add_reaction(
        "dfZ_dt",
        fn=_d_f_z_dt,
        args=["fqEss", "fZ", "KiqEz", "KdqEz"],
        stoichiometry={"fZ": 1.0},
    )
    m.add_reaction(
        "dalphar_dt",
        fn=_d_alphar_dt,
        args=["alpharss", "alphar", "Kialpha", "Kdalpha"],
        stoichiometry={"alphar": 1.0},
    )
    m.add_reaction(
        "dPSIId_dt",
        fn=_d_psiid_dt,
        args=["PSIId", "PARa", "alphar", "Kinh", "Krep"],
        stoichiometry={"PSIId": 1.0},
    )
    m.add_reaction(
        "dfR_dt",
        fn=_d_f_r_dt,
        args=["fRss", "fR", "KiR", "KdR"],
        stoichiometry={"fR": 1.0},
    )
    m.add_reaction(
        "dfRB_dt",
        fn=_d_f_rb_dt,
        args=["fRBss", "fRB", "Krca", "RCA", "fRCA", "KdRB"],
        stoichiometry={"fRB": 1.0},
    )

    m.add_reaction(
        "dCc_dt",
        fn=_d_cc_dt,
        args=["Ccyt", "Cc", "gc", "Vc", "Mv", "Vref"],
        stoichiometry={"Cc": 1.0},
    )
    m.add_reaction(
        "dCcyt_dt",
        fn=_d_ccyt_dt,
        args=["Ci", "Ccyt", "gw", "Rp", "Rm", "Cc", "gc", "Mv", "Vref"],
        stoichiometry={"Ccyt": 1.0},
    )
    m.add_reaction(
        "dCi_dt",
        fn=_d_ci_dt,
        args=["Ca", "Ci", "gsc", "gbc", "Ccyt", "gw", "Mv", "Vref"],
        stoichiometry={"Ci": 1.0},
    )
    m.add_reaction(
        "dCa_dt",
        fn=_d_ca_dt,
        args=["Flow", "Ca", "CO2R", "leaf_surface", "A", "Ta", "volume_chamber"],
        stoichiometry={"Ca": 1.0},
    )
    m.add_reaction(
        "dH2OS_dt",
        fn=_d_h2os_dt,
        args=[
            "Flow",
            "leaf_surface",
            "transpiration",
            "H2OS",
            "H2OR",
            "Ta",
            "volume_chamber",
        ],
        stoichiometry={"H2OS": 1.0},
    )
    m.add_reaction(
        "dgsw_dt",
        fn=_d_gsw_dt,
        args=["gss", "gsw", "Kgsi", "Kgsd"],
        stoichiometry={"gsw": 1.0},
    )

    return m
