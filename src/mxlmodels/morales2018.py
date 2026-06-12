"""
Morales et al. 2018 dynamic leaf CO2 assimilation model.

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


def _safe_sqrt(x):
    return np.sqrt(max(x, 0.0))


def _safe_div(num, den, eps=1e-30):
    return num / den if abs(den) > eps else num / eps


def _par(Ib, Ig, Ir):
    return Ib + Ig + Ir


def _para(Ib, Ig, Ir, alphab, alphag, alphared):
    return Ib * alphab + Ig * alphag + Ir * alphared


def _parap(Ib, Ig, Ir, alphabp, alphagp, alpharp):
    return Ib * alphabp + Ig * alphagp + Ir * alpharp


def _arrhenius(T, value25, Ha):
    return value25 * np.exp((Ha * (T - TREF)) / (TREF * R_GAS * T))


def _arrhenius_inverse(T, value25, Ha):
    return value25 * np.exp((-(T - TREF) * Ha) / (TREF * R_GAS * T))


def _peaked_arrhenius(T, value25, Ha, Hd, S):
    activation = np.exp((Ha * (T - TREF)) / (TREF * R_GAS * T))
    deactivation_ref = 1.0 + np.exp((TREF * S - Hd) / (TREF * R_GAS))
    deactivation_t = 1.0 + np.exp((T * S - Hd) / (T * R_GAS))
    return value25 * activation * deactivation_ref / deactivation_t


def _phiqe(fP, fZ, gamma1, gamma2, gamma3, PhiqEmax):
    return (fP * gamma1 + fP * fZ * gamma2 + fZ * gamma3) * PhiqEmax


def _rp(PR, kPR):
    return 0.5 * PR * kPR


def _sc(Scm, falphaSc, alphar):
    return Scm * falphaSc * alphar


def _gc(Sc, Sm, gcm):
    return (Sc / Sm) * gcm


def _kmapp_rubp(KmRuBP, PGA, Vch, KiPGA):
    return KmRuBP * (1.0 + PGA / (Vch * KiPGA))


def _frca(Tl, DHdRCA, ToRCA, DHaRCA):
    return (DHdRCA * np.exp(((Tl - ToRCA) * DHaRCA) / (ToRCA * R_GAS * Tl))) / (
        DHdRCA - DHaRCA * (1.0 - np.exp((DHdRCA * (Tl - ToRCA)) / (ToRCA * R_GAS * Tl)))
    )


def _fmax(RCA, fRCA, KaRCA):
    return (RCA * fRCA) / (RCA * fRCA + KaRCA)


def _para_p2(sigma2, alphar, PARaP):
    return sigma2 * alphar * PARaP


def _frubp(RB, Vch, Kmapp_RuBP, RuBP):
    term = RB / Vch + Kmapp_RuBP + RuBP / Vch
    inner = term**2 - (4.0 * RB * RuBP) / (Vch**2)
    return (1.0 / ((2.0 * RB) / Vch)) * (term - _safe_sqrt(inner))


def _phi(Kmc, Ko, O2, Kmo, Kc, Cc):
    return _safe_div(Kmc * Ko * O2, Kmo * Kc * Cc)


def _vc(fRB, fRuBP, Kc, RB, Cc, Kmc, O2, Kmo):
    return _safe_div(fRB * fRuBP * Kc * RB * Cc, Cc + Kmc * (1.0 + O2 / Kmo))


def _vr_tpu(TPU, phi):
    return _safe_div(3.0 * TPU * (2.0 + 1.5 * phi), 1.0 - 0.5 * phi)


def _vre(fR, Vrmax, PGA, KmPGA):
    return _safe_div(fR * Vrmax * PGA, PGA + KmPGA)


def _a(Ci, Ccyt, gw):
    return (Ci - Ccyt) * gw


def _gm(A, Ci, Cc):
    return _safe_div(A, Ci - Cc)


def _fm_d(kf, kDinh):
    return kf / (kf + kDinh)


def _fm_a(kf, kD0):
    return kf / (kf + kD0)


def _fmp_d(alphar, kf, kDinh):
    return (alphar * kf) / (kf + kDinh)


def _fo_d(kf, kDinh):
    return kf / (kf + kDinh)


def _fo_a(kf, kD0, kp):
    return kf / (kf + kD0 + kp)


def _fop_d(alphar, kf, kDinh):
    return (alphar * kf) / (kf + kDinh)


def _frbss_nr(ac, bc, Cc):
    return min(1.0, ac + bc * Cc)


def _qi(Fm_a, PSIId, Fmp_d):
    return _safe_div(Fm_a, (1.0 - PSIId) * Fm_a + PSIId * Fmp_d) - 1.0


def _alpharss(Ib, Iac, alpharac, alphar_alpha, alpharav, thetaalphar):
    term = alphar_alpha * (Ib - Iac) + alpharav
    acclim = (1.0 + alpharac) - (term - _safe_sqrt(term**2 - 4.0 * alphar_alpha * thetaalphar * alpharav * (Ib - Iac))) / (2.0 * thetaalphar)
    return min(1.0 + (Ib / Iac) * alpharac, acclim)


def _kinh(Kinh0, fprot, PhiqE):
    return max(Kinh0 - fprot * PhiqE, 0.0)


def _phi_iid(Fm_a, Fo_a):
    return (Fm_a - Fo_a) / Fm_a


def _fo(PSIId, Fo_a, Fo_d):
    return (1.0 - PSIId) * Fo_a + PSIId * Fo_d


def _frss(fR0, alphafR, PARa, thetafR):
    term = alphafR * PARa + (1.0 - fR0)
    return fR0 + (term - _safe_sqrt(term**2 - 4.0 * alphafR * thetafR * PARa * (1.0 - fR0))) / (2.0 * thetafR)


def _fm(PSIId, Fm_a, Fm_d):
    return (1.0 - PSIId) * Fm_a + PSIId * Fm_d


def _kd(kp, PhiIId, PhiqE, kf):
    return (kp / (PhiIId - PhiqE) - kf) - kp


def _fmp_a(alphar, kf, kD):
    return (alphar * kf) / (kf + kD)


def _phi_iiop(Fm, Fo):
    return (Fm - Fo) / Fm


def _fmp(PSIId, Fmp_a, Fmp_d):
    return (1.0 - PSIId) * Fmp_a + PSIId * Fmp_d


def _phi_iio(PhiIIop, PhiqE):
    return PhiIIop - PhiqE


def _j2pp(PhiIIop, PARaP2, Jmax, theta):
    term = PhiIIop * PARaP2 + Jmax
    return (term - _safe_sqrt(term**2 - 4.0 * PhiIIop * theta * Jmax * PARaP2)) / (2.0 * theta)


def _j2pm(VrTPU, VrE, fpseudo, fcyc, phi):
    limitation = min(VrTPU, VrE)
    return (((limitation / (1.0 - fpseudo / (1.0 - fcyc))) * 2.0) / (2.0 + 1.5 * phi)) * (2.0 + 2.0 * phi)


def _fop_a(alphar, kf, kD, kp):
    return (alphar * kf) / (kf + kD + kp)


def _qpp(PARaP2, J2pp, PhiIIop):
    return (J2pp / PARaP2) / PhiIIop if PARaP2 > FLUX0 else 1.0


def _npq(Fm_a, Fmp):
    return Fm_a / Fmp - 1.0


def _qpm(J2pm, J2pp, qPp):
    return _safe_div(J2pm, J2pp) * qPp


def _qm(NPQ, Fm_a, Fmp, alphar):
    return NPQ - ((Fm_a / Fmp) * alphar - 1.0)


def _q_p_no_qd(qPm, qPp):
    return min(qPm, qPp)


def _qe(NPQ, qM, qI):
    return (NPQ - qM) - qI


def _fqess(qPno_qD):
    return 1.0 - qPno_qD


def _phiqess(fqEss, gamma1, gamma2, gamma3, PhiqEmax):
    return (fqEss * gamma1 + fqEss * fqEss * gamma2 + fqEss * gamma3) * PhiqEmax


def _phi_iioss(PhiIIop, PhiqEss):
    return PhiIIop - PhiqEss


def _j2qe(PhiIIo, PhiIIoss, J2pp):
    return (1.0 - (PhiIIoss - PhiIIo) / PhiIIoss) * J2pp if PhiIIo < PhiIIoss else J2pp


def _vrj(J2qE, fpseudo, fcyc, phi):
    return (((J2qE * (1.0 - fpseudo / (1.0 - fcyc))) / 2.0) * (2.0 + 1.5 * phi)) / (2.0 + 2.0 * phi)


def _j2(J2qE, J2pm):
    return min(J2qE, J2pm)


def _frbss_r(PAR, VrTPU, VrJ, phi, Vc, fRB, fRuBP, fRBmin, fRBmax):
    if PAR <= FLUX0:
        val = fRBmin
    else:
        val = _safe_div(min(VrTPU, VrJ) / (2.0 + 1.5 * phi), Vc / (fRB * fRuBP))
    return min(val, fRBmax)


def _vr(VrJ, VrTPU, VrE):
    return min(VrJ, min(VrTPU, VrE))


def _qp(J2, PARaP2, PhiIIo):
    return _safe_div(J2 / PARaP2, PhiIIo)


def _phi_ii(PARaP2, J2, PhiIIo):
    return J2 / PARaP2 if PARaP2 > FLUX0 else PhiIIo


def _reglimit(VrJ, VrTPU, Vr):
    if abs(VrJ - Vr) < FLUX0:
        return 1.0
    if abs(VrTPU - Vr) < FLUX0:
        return 2.0
    return 3.0


def _frbss(fRBss_nr, fRBss_r):
    return min(fRBss_nr, fRBss_r)


def _d_pga_dt(Vc, phi, Vr):
    return (2.0 * Vc + 1.5 * Vc * phi) - Vr


def _d_rubp_dt(phi, Vr, Vc):
    return ((1.0 + phi) / (2.0 + 1.5 * phi)) * Vr - Vc * (1.0 + phi)


def _d_pr_dt(Vc, phi, PR, kPR):
    return Vc * phi - PR * kPR


def _d_fP_dt(fqEss, fP, KiqEp, KdqEp):
    return (fqEss - fP) * KiqEp if fqEss > fP else (fqEss - fP) * KdqEp


def _d_fZ_dt(fqEss, fZ, KiqEz, KdqEz):
    return (fqEss - fZ) * KiqEz if fqEss > fZ else (fqEss - fZ) * KdqEz


def _d_alphar_dt(alpharss, alphar, Kialpha, Kdalpha):
    return (alpharss - alphar) * Kialpha if alpharss > alphar else (alpharss - alphar) * Kdalpha


def _d_psiid_dt(PSIId, PARa, alphar, Kinh, Krep):
    return (1.0 - PSIId) * PARa * alphar * Kinh - PSIId * Krep


def _d_fR_dt(fRss, fR, KiR, KdR):
    return (fRss - fR) * KiR if fRss > fR else (fRss - fR) * KdR


def _d_fRB_dt(fRBss, fRB, Krca, RCA, fRCA, KdRB):
    return (fRBss - fRB) * Krca * RCA * fRCA if fRBss > fRB else (fRBss - fRB) * KdRB



def _gbc(gbw):
    return gbw / 1.37


def _gsc(gsw):
    return gsw / 1.56


def _mv(Ta):
    return (R_GAS * Ta) / AIR_PRESSURE

def _identity(x):
    return x


def _ea(H2OR):
    return H2OR * AIR_PRESSURE


def _photo(Flow, CO2R, Ca, leaf_surface):
    return (Flow * CO2R - Flow * Ca) / leaf_surface


def _trmmol(Flow, H2OS, H2OR, leaf_surface):
    return (Flow * (H2OS - H2OR)) / (leaf_surface * (1.0 - H2OS))


def _es_leaf(Tl):
    return ES0 * np.exp((ES_K * (Tl - TZERO)) / (ES_TREF + (Tl - TZERO)))


def _vpdleaf(es_leaf, ea):
    return max(es_leaf - ea, ZERO_PRESSURE)


def _fvpd(VPDleaf, D0):
    return 1.0 / (1.0 + VPDleaf / D0)


def _gtw(Trmmol, es_leaf, ea, VPDleaf):
    return (Trmmol * (AIR_PRESSURE - (es_leaf + ea) / 2.0)) / VPDleaf


def _transpiration(VPDleaf, es_leaf, ea, gsw, gbw):
    return ((VPDleaf / (AIR_PRESSURE - (es_leaf + ea) / 2.0)) * 1.0) / (1.0 / gsw + 1.0 / gbw)


def _fi(fI0, alphafI, PAR, thetafI):
    fI_a = thetafI
    fI_b = -((1.0 + fI0 + alphafI * PAR))
    fI_c = fI0 + alphafI * PAR
    return (-fI_b - _safe_sqrt(fI_b**2 - 4.0 * fI_a * fI_c)) / (2.0 * fI_a)


def _gss(fI, fvpd, gswm):
    return fI * fvpd * gswm


def _cond(gtw, gbw):
    return 1.0 / (1.0 / gtw - 1.0 / gbw)


def _d_cc_dt(Ccyt, Cc, gc, Vc, Mv, Vref):
    return (((Ccyt - Cc) * gc - Vc) * Mv) / Vref


def _d_ccyt_dt(Ci, Ccyt, gw, Rp, Rm, Cc, gc, Mv, Vref):
    return ((((Ci - Ccyt) * gw + Rp + Rm) - (Ccyt - Cc) * gc) * Mv) / Vref


def _d_ci_dt(Ca, Ci, gsc, gbc, Ccyt, gw, Mv, Vref):
    return (((Ca - Ci) / (1.0 / gsc + 1.0 / gbc) - (Ci - Ccyt) * gw) * Mv) / Vref


def _d_ca_dt(Flow, Ca, CO2R, leaf_surface, A, Ta, volume_chamber):
    return ((((-Flow * Ca + Flow * CO2R) - leaf_surface * A) * R_GAS * Ta) / volume_chamber) / AIR_PRESSURE


def _d_h2os_dt(Flow, leaf_surface, transpiration, H2OS, H2OR, Ta, volume_chamber):
    numerator = -((Flow + leaf_surface * transpiration)) * H2OS + Flow * H2OR + leaf_surface * transpiration
    return ((numerator * R_GAS * Ta) / volume_chamber) / AIR_PRESSURE


def _d_gsw_dt(gss, gsw, Kgsi, Kgsd):
    return (gss - gsw) * Kgsi if gss > gsw else (gss - gsw) * Kgsd


def get_morales2018() -> Model:
    """Return Morales et al. 2018 dynamic photosynthesis model."""

    m: Model = Model()

    states = {
        "PGA": 0.00005, "RuBP": 0.00005, "fRB": 0.25,
        "fP": 0.0, "fZ": 0.0, "alphar": 1.0,
        "PSIId": 0.0, "fR": 0.0, "PR": 0.0,
        "Cc": 0.00038, "Ccyt": 0.00038, "Ci": 0.00038,
        "Ca": 0.00038, "H2OS": 0.02000, "gsw": 0.09000,
        "sumA": 0.0,
    }
    for name, value in states.items():
        m = m.add_variable(name, initial_value=value)

    forcings = {
        "Ib": 100.0, "Ig": 0.0, "Ir": 900.0,
        "Ta": 298.15, "Tl": 298.15,
        "H2OR": 0.015, "CO2R": 0.00038,
    }
    for name, value in forcings.items():
        m = m.add_parameter(name, value=value)

    parameters = {
        "sigma2": 5.0000e-01, "Jmax25": 1.3928e-04, "DHaJmax": 3.6210e07,
        "DHdJmax": 2.1590e08, "DsJmax": 6.9000e05, "kD0": 4.5500e08,
        "kDinh": 5.0000e09, "kf": 5.6000e07, "kp": 2.6540e09,
        "fcyc": 1.0000e-01, "fpseudo": 1.0000e-01, "theta": 7.0000e-01,
        "gamma1": 2.0000e-01, "gamma2": 6.0000e-01, "gamma3": 2.0000e-01,
        "PhiqEmax": 2.0000e-01, "KiqEp": 1.8700e-02, "KdqEp": 2.3900e-02,
        "KiqEz": 1.8700e-03, "KdqEz": 2.3900e-03, "Kinh0": 1.0000e-01,
        "fprot": 1.0000e-01, "Krep25": 1.9200e-04, "DHaKrep": 1.6080e08,
        "DHdKrep": 2.3323e08, "DsKrep": 7.8000e05, "alphar_alpha25": 6.5500e03,
        "DHaAlphar": 6.7320e07, "DHaKalpha": 9.0500e07, "DsKalpha": 1.0800e06,
        "DHdKalpha": 3.2800e08, "Iac": 1.6000e-06, "alpharac": 5.0000e-02,
        "alpharav": 2.5000e-01, "thetaalphar": 3.6000e-01, "Kialpha25": 1.4900e-03,
        "Kdalpha25": 1.8600e-03, "fR0": 4.0000e-02, "alphafR": 2.5000e03,
        "thetafR": 9.6000e-01, "KiR": 6.2800e-03, "KdR": 7.5000e-03,
        "Vrmax": 1.1865e-04, "KmPGA": 5.0000e-06, "RB": 1.5900e-05,
        "Kc25": 4.1600e00, "DHaKc": 4.1820e07, "Ko25": 1.2600e00,
        "DHaKo": 5.5150e07, "Kmc25": 2.6170e-04, "DHaKmc": 4.9430e07,
        "Kmo25": 1.9850e-01, "DHaKmo": 2.9080e07, "KaRCA": 1.0200e-02,
        "ac": 2.7000e-01, "bc": 1.4000e04, "KdRB": 6.8000e-04,
        "Krca": 8.6300e-02, "fRBmin": 4.8000e-01, "O2": 2.1000e-01,
        "RCA": 1.1737e-01, "DHdRCA": 2.9020e08, "ToRCA": 3.0040e02,
        "DHaRCA": 3.0000e07, "KmRuBP": 2.0000e-02, "Vch": 1.0000e-05,
        "KiPGA": 8.4000e-01, "TPU25": 7.4700e-06, "DHaTPU": 5.7500e07,
        "DHdTPU": 2.4670e08, "DsTPU": 7.9000e05, "DHaGc": 7.0200e07,
        "DHdGc": 9.4000e07, "DsGc": 3.2000e05, "DHaGw": 7.0200e07,
        "DHdGw": 9.4000e07, "DsGw": 3.2000e05, "Rm25": 9.9000e-07,
        "DHaRm": 5.6200e07, "kPR": 2.4000e-02, "Vref": 1.5500e-04,
        "Scm": 7.1000e00, "Sm": 9.8000e00, "falphaSc": 9.3000e-01,
        "gcm25": 3.9000e-01, "gw25": 7.5000e-01, "D0": 7.4000e05,
        "fI0": 3.9000e-01, "gswm": 4.8000e-01, "alphafI": 7.6700e02,
        "thetafI": 8.8000e-01, "Kgsi": 1.1400e-03, "Kgsd": 1.1400e-03,
        "gbw": 9.2000e00, "volume_chamber": 8.0000e-05, "leaf_surface": 2.0000e-04,
        "Flow": 5.0000e-04, "alphab": 9.2000e-01, "alphag": 7.2000e-01,
        "alphared": 8.3000e-01, "alphabp": 6.6000e-01, "alphagp": 6.0000e-01,
        "alpharp": 8.0000e-01,
    }
    for name, value in parameters.items():
        m = m.add_parameter(name, value=value)

    m = m.add_derived("PAR", fn=_par, args=["Ib", "Ig", "Ir"])
    m = m.add_derived("PARa", fn=_para, args=["Ib", "Ig", "Ir", "alphab", "alphag", "alphared"])
    m = m.add_derived("PARaP", fn=_parap, args=["Ib", "Ig", "Ir", "alphabp", "alphagp", "alpharp"])
    m = m.add_derived("Jmax", fn=_peaked_arrhenius, args=["Tl", "Jmax25", "DHaJmax", "DHdJmax", "DsJmax"])
    m = m.add_derived("Krep", fn=_peaked_arrhenius, args=["Tl", "Krep25", "DHaKrep", "DHdKrep", "DsKrep"])
    m = m.add_derived("TPU", fn=_peaked_arrhenius, args=["Tl", "TPU25", "DHaTPU", "DHdTPU", "DsTPU"])
    m = m.add_derived("Kc", fn=_arrhenius, args=["Tl", "Kc25", "DHaKc"])
    m = m.add_derived("Ko", fn=_arrhenius, args=["Tl", "Ko25", "DHaKo"])
    m = m.add_derived("Kmc", fn=_arrhenius, args=["Tl", "Kmc25", "DHaKmc"])
    m = m.add_derived("Kmo", fn=_arrhenius, args=["Tl", "Kmo25", "DHaKmo"])
    m = m.add_derived("Rm", fn=_arrhenius, args=["Tl", "Rm25", "DHaRm"])
    m = m.add_derived("gcm", fn=_peaked_arrhenius, args=["Tl", "gcm25", "DHaGc", "DHdGc", "DsGc"])
    m = m.add_derived("gw", fn=_peaked_arrhenius, args=["Tl", "gw25", "DHaGw", "DHdGw", "DsGw"])
    m = m.add_derived("alphar_alpha", fn=_arrhenius_inverse, args=["Tl", "alphar_alpha25", "DHaAlphar"])
    m = m.add_derived("Kialpha", fn=_peaked_arrhenius, args=["Tl", "Kialpha25", "DHaKalpha", "DHdKalpha", "DsKalpha"])
    m = m.add_derived("Kdalpha", fn=_peaked_arrhenius, args=["Tl", "Kdalpha25", "DHaKalpha", "DHdKalpha", "DsKalpha"])
    m = m.add_derived("PhiqE", fn=_phiqe, args=["fP", "fZ", "gamma1", "gamma2", "gamma3", "PhiqEmax"])
    m = m.add_derived("Rp", fn=_rp, args=["PR", "kPR"])
    m = m.add_derived("Sc", fn=_sc, args=["Scm", "falphaSc", "alphar"])
    m = m.add_derived("gc", fn=_gc, args=["Sc", "Sm", "gcm"])
    m = m.add_derived("Kmapp_RuBP", fn=_kmapp_rubp, args=["KmRuBP", "PGA", "Vch", "KiPGA"])
    m = m.add_derived("fRCA", fn=_frca, args=["Tl", "DHdRCA", "ToRCA", "DHaRCA"])
    m = m.add_derived("fRBmax", fn=_fmax, args=["RCA", "fRCA", "KaRCA"])
    m = m.add_derived("PARaP2", fn=_para_p2, args=["sigma2", "alphar", "PARaP"])
    m = m.add_derived("fRuBP", fn=_frubp, args=["RB", "Vch", "Kmapp_RuBP", "RuBP"])
    m = m.add_derived("phi", fn=_phi, args=["Kmc", "Ko", "O2", "Kmo", "Kc", "Cc"])
    m = m.add_derived("Vc", fn=_vc, args=["fRB", "fRuBP", "Kc", "RB", "Cc", "Kmc", "O2", "Kmo"])
    m = m.add_derived("VrTPU", fn=_vr_tpu, args=["TPU", "phi"])
    m = m.add_derived("VrE", fn=_vre, args=["fR", "Vrmax", "PGA", "KmPGA"])
    m = m.add_derived("A", fn=_a, args=["Ci", "Ccyt", "gw"])
    m = m.add_derived("gm", fn=_gm, args=["A", "Ci", "Cc"])
    m = m.add_derived("Fm_d", fn=_fm_d, args=["kf", "kDinh"])
    m = m.add_derived("Fm_a", fn=_fm_a, args=["kf", "kD0"])
    m = m.add_derived("Fmp_d", fn=_fmp_d, args=["alphar", "kf", "kDinh"])
    m = m.add_derived("Fo_d", fn=_fo_d, args=["kf", "kDinh"])
    m = m.add_derived("Fo_a", fn=_fo_a, args=["kf", "kD0", "kp"])
    m = m.add_derived("Fop_d", fn=_fop_d, args=["alphar", "kf", "kDinh"])
    m = m.add_derived("fRBss_nr", fn=_frbss_nr, args=["ac", "bc", "Cc"])
    m = m.add_derived("qI", fn=_qi, args=["Fm_a", "PSIId", "Fmp_d"])
    m = m.add_derived("alpharss", fn=_alpharss, args=["Ib", "Iac", "alpharac", "alphar_alpha", "alpharav", "thetaalphar"])
    m = m.add_derived("Kinh", fn=_kinh, args=["Kinh0", "fprot", "PhiqE"])
    m = m.add_derived("PhiIId", fn=_phi_iid, args=["Fm_a", "Fo_a"])
    m = m.add_derived("Fo", fn=_fo, args=["PSIId", "Fo_a", "Fo_d"])
    m = m.add_derived("fRss", fn=_frss, args=["fR0", "alphafR", "PARa", "thetafR"])
    m = m.add_derived("Fm", fn=_fm, args=["PSIId", "Fm_a", "Fm_d"])
    m = m.add_derived("kD", fn=_kd, args=["kp", "PhiIId", "PhiqE", "kf"])
    m = m.add_derived("Fmp_a", fn=_fmp_a, args=["alphar", "kf", "kD"])
    m = m.add_derived("PhiIIop", fn=_phi_iiop, args=["Fm", "Fo"])
    m = m.add_derived("Fmp", fn=_fmp, args=["PSIId", "Fmp_a", "Fmp_d"])
    m = m.add_derived("PhiIIo", fn=_phi_iio, args=["PhiIIop", "PhiqE"])
    m = m.add_derived("J2pp", fn=_j2pp, args=["PhiIIop", "PARaP2", "Jmax", "theta"])
    m = m.add_derived("J2pm", fn=_j2pm, args=["VrTPU", "VrE", "fpseudo", "fcyc", "phi"])
    m = m.add_derived("Fop_a", fn=_fop_a, args=["alphar", "kf", "kD", "kp"])
    m = m.add_derived("qPp", fn=_qpp, args=["PARaP2", "J2pp", "PhiIIop"])
    m = m.add_derived("NPQ", fn=_npq, args=["Fm_a", "Fmp"])
    m = m.add_derived("qPm", fn=_qpm, args=["J2pm", "J2pp", "qPp"])
    m = m.add_derived("qM", fn=_qm, args=["NPQ", "Fm_a", "Fmp", "alphar"])
    m = m.add_derived("qPno_qD", fn=_q_p_no_qd, args=["qPm", "qPp"])
    m = m.add_derived("qE", fn=_qe, args=["NPQ", "qM", "qI"])
    m = m.add_derived("fqEss", fn=_fqess, args=["qPno_qD"])
    m = m.add_derived("PhiqEss", fn=_phiqess, args=["fqEss", "gamma1", "gamma2", "gamma3", "PhiqEmax"])
    m = m.add_derived("PhiIIoss", fn=_phi_iioss, args=["PhiIIop", "PhiqEss"])
    m = m.add_derived("J2qE", fn=_j2qe, args=["PhiIIo", "PhiIIoss", "J2pp"])
    m = m.add_derived("VrJ", fn=_vrj, args=["J2qE", "fpseudo", "fcyc", "phi"])
    m = m.add_derived("J2", fn=_j2, args=["J2qE", "J2pm"])
    m = m.add_derived("fRBss_r", fn=_frbss_r, args=["PAR", "VrTPU", "VrJ", "phi", "Vc", "fRB", "fRuBP", "fRBmin", "fRBmax"])
    m = m.add_derived("Vr", fn=_vr, args=["VrJ", "VrTPU", "VrE"])
    m = m.add_derived("qP", fn=_qp, args=["J2", "PARaP2", "PhiIIo"])
    m = m.add_derived("PhiII", fn=_phi_ii, args=["PARaP2", "J2", "PhiIIo"])
    m = m.add_derived("reg_limit", fn=_reglimit, args=["VrJ", "VrTPU", "Vr"])
    m = m.add_derived("fRBss", fn=_frbss, args=["fRBss_nr", "fRBss_r"])


    # Gas-exchange and chamber-derived quantities
    m = m.add_derived("gbc", fn=_gbc, args=["gbw"])
    m = m.add_derived("gsc", fn=_gsc, args=["gsw"])
    m = m.add_derived("Mv", fn=_mv, args=["Ta"])
    m = m.add_derived("ea", fn=_ea, args=["H2OR"])
    m = m.add_derived("Photo", fn=_photo, args=["Flow", "CO2R", "Ca", "leaf_surface"])
    m = m.add_derived("Trmmol", fn=_trmmol, args=["Flow", "H2OS", "H2OR", "leaf_surface"])
    m = m.add_derived("es_leaf", fn=_es_leaf, args=["Tl"])
    m = m.add_derived("VPDleaf", fn=_vpdleaf, args=["es_leaf", "ea"])
    m = m.add_derived("fvpd", fn=_fvpd, args=["VPDleaf", "D0"])
    m = m.add_derived("gtw", fn=_gtw, args=["Trmmol", "es_leaf", "ea", "VPDleaf"])
    m = m.add_derived("transpiration", fn=_transpiration, args=["VPDleaf", "es_leaf", "ea", "gsw", "gbw"])
    m = m.add_derived("fI", fn=_fi, args=["fI0", "alphafI", "PAR", "thetafI"])
    m = m.add_derived("gss", fn=_gss, args=["fI", "fvpd", "gswm"])
    m = m.add_derived("Cond", fn=_cond, args=["gtw", "gbw"])

    m = m.add_reaction("dPGA_dt", fn=_d_pga_dt, args=["Vc", "phi", "Vr"], stoichiometry={"PGA": 1.0})
    m = m.add_reaction("dRuBP_dt", fn=_d_rubp_dt, args=["phi", "Vr", "Vc"], stoichiometry={"RuBP": 1.0})
    m = m.add_reaction("dPR_dt", fn=_d_pr_dt, args=["Vc", "phi", "PR", "kPR"], stoichiometry={"PR": 1.0})
    m = m.add_reaction("dfP_dt", fn=_d_fP_dt, args=["fqEss", "fP", "KiqEp", "KdqEp"], stoichiometry={"fP": 1.0})
    m = m.add_reaction("dfZ_dt", fn=_d_fZ_dt, args=["fqEss", "fZ", "KiqEz", "KdqEz"], stoichiometry={"fZ": 1.0})
    m = m.add_reaction("dalphar_dt", fn=_d_alphar_dt, args=["alpharss", "alphar", "Kialpha", "Kdalpha"], stoichiometry={"alphar": 1.0})
    m = m.add_reaction("dPSIId_dt", fn=_d_psiid_dt, args=["PSIId", "PARa", "alphar", "Kinh", "Krep"], stoichiometry={"PSIId": 1.0})
    m = m.add_reaction("dfR_dt", fn=_d_fR_dt, args=["fRss", "fR", "KiR", "KdR"], stoichiometry={"fR": 1.0})
    m = m.add_reaction("dfRB_dt", fn=_d_fRB_dt, args=["fRBss", "fRB", "Krca", "RCA", "fRCA", "KdRB"], stoichiometry={"fRB": 1.0})


    m = m.add_reaction("dCc_dt", fn=_d_cc_dt, args=["Ccyt", "Cc", "gc", "Vc", "Mv", "Vref"], stoichiometry={"Cc": 1.0})
    m = m.add_reaction("dCcyt_dt", fn=_d_ccyt_dt, args=["Ci", "Ccyt", "gw", "Rp", "Rm", "Cc", "gc", "Mv", "Vref"], stoichiometry={"Ccyt": 1.0})
    m = m.add_reaction("dCi_dt", fn=_d_ci_dt, args=["Ca", "Ci", "gsc", "gbc", "Ccyt", "gw", "Mv", "Vref"], stoichiometry={"Ci": 1.0})
    m = m.add_reaction("dCa_dt", fn=_d_ca_dt, args=["Flow", "Ca", "CO2R", "leaf_surface", "A", "Ta", "volume_chamber"], stoichiometry={"Ca": 1.0})
    m = m.add_reaction("dH2OS_dt", fn=_d_h2os_dt, args=["Flow", "leaf_surface", "transpiration", "H2OS", "H2OR", "Ta", "volume_chamber"], stoichiometry={"H2OS": 1.0})
    m = m.add_reaction("dgsw_dt", fn=_d_gsw_dt, args=["gss", "gsw", "Kgsi", "Kgsd"], stoichiometry={"gsw": 1.0})
    
    # Morales 2018 observed/readout variables
    # ==========================================================
    # Morales 2018 observed outputs
    # ==========================================================
    observed = [
        "Vr", "Vc", "fRuBP", "PARaP", "PARa", "PAR",
        "fqEss", "PhiqE", "PhiIIoss", "PhiIIo",
        "qP", "PhiII", "VrJ", "NPQ", "qI", "qE",
        "qM", "Rp", "A", "gss", "VPDleaf", "Sc",
        "Photo", "transpiration", "Trmmol", "Cond",
        "gm", "reg_limit",
    ]

    for name in observed:
        m = m.add_readout(f"obs_{name}", fn=_identity, args=[name])
        
# ==========================================================
# Solver settings recovered from original Morales R wrapper
# ==========================================================

        # Original Morales R wrapper solver settings:
    # rtol = 1e-6, atol = 1e-6, method = "bdf", positive = 1
    m = m.add_parameter("solver_rtol", value=1e-6)
    m = m.add_parameter("solver_atol", value=1e-6)
    m = m.add_parameter("solver_positive", value=1.0)

    return m
