r"""Complete mxlpy port of the Zaks et al. photosynthesis model.

|             |                                                                  |
| ----------- | ---------------------------------------------------------------- |
| doi         | 10.1073/pnas.1211017109                                          |
| main author | Julia Zaks                                                       |
| paper title | A kinetic model of rapidly reversible nonphotochemical quenching |
| published   | September 25, 2012                                               |
| journal     | PNAS                                                             |
| organism    | Higher plants                                                    |
| Ported by   | Quang Huy Nguyen ( @PhotosyntheticBatman )                       |

The model provides one of the first mechanistic description of energy dependent
quenching process qE within NPQ. qE activity is simulated under 2 different
light conditions: low light and high light.

Sections: F1 - PSII (antenna + reaction centre) F2 - qE / xanthophyll cycle F3
\- PQ pool (QB site + plastoquinone) F4 - Cytochrome b6f F5 - PSI F7 - ATP
synthase F8 - Lumen ion fluxes (Mg, Cl, K)
"""

import numpy as np
from mxlpy import Derived, Model, Parameter, Slider, Variable
from mxlpy.units import (
    coulomb,
    dimensionless,
    joule,
    kelvin,
    liter,
    mol,
    ppfd,
    second,
    sqm,
    volt,
)

# States are per reaction centre (dimensionless), lumen ions and protons in M.
# Areas are cm^2, as in the original (e.g. 1 uF cm^-2 membrane capacitance)
_area = sqm / 10_000
_molar = mol / liter
_per_s = 1 / second
_conductivity = mol / (_area * volt * second)
# membrane capacitance is F cm^-2 in the original, written as C V^-1 so that it cancels
# permeabilities are cm s^-1 in the original; the flux converts with a bare 1e-3 L cm^-3,
# so they are written as L cm^-2 s^-1 for the units to close
_permeability = liter / (_area * second)

# ---------------------------------------------------------------------------
# Helper / rate functions
# ---------------------------------------------------------------------------


def _protonation_fraction(ph: float, p_ka: float, hill_n: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (hill_n * (ph - p_ka)))


def _get_p_h(p_h_start: float, b_h: float, protons: float) -> float:
    return p_h_start - protons / b_h


def _totalcharge(
    proton: float, z_cl: float, cl: float, z_k: float, k: float, z_mg: float, mg: float
) -> float:
    return proton + z_cl * cl + z_k * k + z_mg * mg


def _delta_p_h(p_h_stroma: float, p_h_lumen: float) -> float:
    return p_h_stroma - p_h_lumen


def _delta_psi(
    total_charge_lumen: float,
    total_charge_stroma: float,
    fconst: float,
    lumen_volume_per_area: float,
    membrane_capacitance: float,
) -> float:
    return (
        (total_charge_lumen - total_charge_stroma)
        * fconst
        * lumen_volume_per_area
        / membrane_capacitance
    )


def _pmf(_delta_p_h: float, _delta_psi: float, voltsperlog: float) -> float:
    return _delta_psi + np.log(10) * voltsperlog * _delta_p_h


def _delta_mu(
    z: float,
    lumen_conc: float,
    stroma_conc: float,
    _delta_psi: float,
    voltsperlog: float,
) -> float:
    diffusion_potential = voltsperlog * np.log(lumen_conc / stroma_conc)
    return z * _delta_psi + diffusion_potential


def _efield_slowdown(alpha: float, _delta_psi: float, voltsperlog: float) -> float:
    return np.exp(-alpha * _delta_psi / voltsperlog)


def _flux_to_concentration_lumen(lumen_volume_per_area: float) -> float:
    return 1.0 / lumen_volume_per_area


def _atpsyn_proton_stoi(lumen_volume_per_area: float) -> float:
    return -1.0 / lumen_volume_per_area


def _ion_flux_linear(
    lumen_conc: float,
    stroma_conc: float,
    _delta_mu: float,
    permeability: float,
    voltsperlog: float,
) -> float:
    liters_per_cc = 1e-3
    conc = lumen_conc if _delta_mu > 0 else stroma_conc
    return -conc * liters_per_cc * permeability * _delta_mu / voltsperlog


def _total_q(
    anth: float,
    zea: float,
    psbs_q: float,
    zfrac: float,
    psb_s_dose: float,
    q_trig1: float,
    q_trig2: float,
    q_trig3: float,
) -> float:
    q_x = zfrac * (zea + 0.5 * anth) * psbs_q * q_trig1 + zfrac * psbs_q * q_trig2
    q_l = (1.0 - zfrac) * psbs_q * q_trig3
    return psb_s_dose * (q_x + q_l)


def _v_from_az(xtot: float, a: float, z: float) -> float:
    return xtot - a - z


def _complement(c1: float) -> float:
    r"""1 - c1 (used for moiety complements)."""
    return 1.0 - c1


def _moeity_frac(c1: float, _frac: float) -> float:
    return (1.0 - c1) * _frac


def _mass_action_1s(kf: float, s: float) -> float:
    return kf * s


def _mass_action_1s_act(kf: float, act: float, s: float) -> float:
    return kf * act * s


# PSII-specific rates
def _v7(
    k_eetlhp680_q_aox: float,
    k_eetlhp680_q_ared: float,
    chl_ex: float,
    qa_ox: float,
    qa_red: float,
    p680_neut: float,
) -> float:
    return (
        k_eetlhp680_q_aox * chl_ex * qa_ox + k_eetlhp680_q_ared * chl_ex * qa_red
    ) * p680_neut


def _v8(
    k_eetlhp680rev_q_aox: float,
    k_eetlhp680rev_q_ared: float,
    p680_ex: float,
    qa_ox: float,
    qa_red: float,
) -> float:
    return (
        k_eetlhp680rev_q_aox * p680_ex * qa_ox
        + k_eetlhp680rev_q_ared * p680_ex * qa_red
    )


def _v12_13(
    k_etp680_phe_rc: float,
    p680_ex: float,
    phe_neut: float,
    qa: float,
    efield_slowdown_r: float,
) -> float:
    return k_etp680_phe_rc * p680_ex * phe_neut * qa * efield_slowdown_r


def _v14(
    k_et_phe_to_qa: float, phe_anion: float, qa_ox: float, efield_slowdown_r: float
) -> float:
    return k_et_phe_to_qa * phe_anion * qa_ox * efield_slowdown_r


def _v16(
    k_p680_pherecombination: float,
    p680_plus: float,
    phe_anion: float,
    efield_slowdown_r: float,
) -> float:
    return k_p680_pherecombination * p680_plus * phe_anion / efield_slowdown_r


def _v17(
    k_p680_q_arecombination: float,
    p680_plus: float,
    qa_red: float,
    phe_neut: float,
    efield_slowdown_r: float,
) -> float:
    return k_p680_q_arecombination * p680_plus * qa_red * phe_neut / efield_slowdown_r


def _v18(
    k_p680_q_arecombination_closed_rc: float,
    p680_plus: float,
    qa_red: float,
    phe_anion: float,
    efield_slowdown_r: float,
) -> float:
    return (
        k_p680_q_arecombination_closed_rc
        * p680_plus
        * qa_red
        * phe_anion
        / efield_slowdown_r
    )


# Cyt b6f
def _fraction_active_cyt(ph_lumen: float, p_ka_c: float, n_c: float) -> float:
    return 1.0 - _protonation_fraction(ph_lumen, p_ka_c, n_c)


def _fraction_pq_red(pqh2: float, pq: float) -> float:
    return pqh2 / (pq + pqh2)


def _fraction_active_pc(pcr: float, p_cper_psi: float) -> float:
    return (p_cper_psi - pcr) / p_cper_psi


def _r_cyt(
    q_reoxidation_rate: float,
    frac_active_cyt: float,
    frac_active_pc: float,
    frac_pq_red: float,
) -> float:
    return q_reoxidation_rate * frac_active_cyt * frac_active_pc * frac_pq_red


def _r_q2(k_etq_ato_qb1: float, q_ared: float, q_bneut: float, esq: float) -> float:
    return k_etq_ato_qb1 * q_ared * q_bneut * esq


def _r_q3(k_etq_ato_qb2: float, q_ared: float, q_bred1: float, esq: float) -> float:
    return k_etq_ato_qb2 * q_ared * q_bred1 * esq


def _r_q8(rate: float, pqh2frac: float) -> float:
    return rate * pqh2frac / 10.0


# Stoichiometric coefficients for lumen proton accumulation
def _psii_proton_stoi(na: float, lumen_volume: float) -> float:
    return 1.0 / (na * lumen_volume)


def _cyt_proton_stoi(na: float, lumen_volume: float) -> float:
    return 4.0 / (na * lumen_volume)


def _cyt_electron_stoi(electrons_per_pc: float) -> float:
    return 2.0 / electrons_per_pc


# PSI


def _psi_1(k_etpcp700: float, p_cr: float, p700ox: float) -> float:
    return max(k_etpcp700 * p_cr * p700ox, 0.0)


def _psi_2(
    light: float,
    ps_icross_section: float,
    k_etp700_fdx: float,
    p700r: float,
    fdxox: float,
) -> float:
    return max(light * ps_icross_section * k_etp700_fdx * p700r * fdxox, 0.0)


# ATP synthase
def _proton_flux_atp(
    atp_conductivity: float, _pmf: float, thresholdpmf: float, active_atps: float
) -> float:
    diff = _pmf - thresholdpmf
    if diff <= 0:
        return 0.0
    return atp_conductivity * diff * active_atps


def _proton_flux_leak(leak_conductivity: float, _pmf: float, leakpmf: float) -> float:
    diff = _pmf - leakpmf
    if diff <= 0:
        return 0.0
    return leak_conductivity * diff


def _atp_stoi(
    na: float, lumen_volume: float, lumen_volume_per_area: float, at_pper_proton: float
) -> float:
    return at_pper_proton * na * lumen_volume / lumen_volume_per_area


# Ion fluxes → concentration change in lumen
def _lumen_ion_flux(
    lumen_conc: float,
    stroma_conc: float,
    dmu: float,
    permeability: float,
    voltsperlog: float,
    lumen_volume_per_area: float,
) -> float:
    flux = _ion_flux_linear(lumen_conc, stroma_conc, dmu, permeability, voltsperlog)
    return flux / lumen_volume_per_area


def _same(x: float) -> float:
    return x


def _frac(x: float, xtot: float) -> float:
    return x / xtot


def _qb_moiety(qb_n: float, qb_r1: float, qb_r2: float) -> float:
    return 1.0 - qb_n - qb_r1 - qb_r2


# Output


def _k_f_rate(k_f: float, chl_ex: float) -> float:
    return k_f * chl_ex


def _kq_e_rate(k_q: float, chl_ex: float, q_total: float) -> float:
    return k_q * chl_ex * q_total


def _k_pc_rate(
    k_eetlhp680_q_aox: float,
    k_eetlhp680_q_ared: float,
    chl_ex: float,
    qa_ox: float,
    qa_red: float,
    p680_neut: float,
) -> float:
    return (
        k_eetlhp680_q_aox * chl_ex * qa_ox + k_eetlhp680_q_ared * chl_ex * qa_red
    ) * p680_neut


def _k_pcrcc_rate(
    k_eetlhp680_q_aox: float,
    k_eetlhp680_q_ared: float,
    chl_ex: float,
    p680_neut: float,
) -> float:
    # Open-RC reference: QAox=1, QAred=0
    return (
        k_eetlhp680_q_aox * chl_ex * 1.0 + k_eetlhp680_q_ared * chl_ex * 0.0
    ) * p680_neut


def _k_c_rate(
    k_n_rantenna: float,
    chl_ex: float,
    k_f_val: float,
    kquench_p680plus: float,
    p680_plus: float,
) -> float:
    return k_n_rantenna * chl_ex + k_f_val + kquench_p680plus * chl_ex * p680_plus


def _allrates_sum(k_c: float, k_pc: float, kq_e: float) -> float:
    return k_c + k_pc + kq_e


def _allrates_rcc_sum(k_c: float, k_pcrcc: float, kq_e: float) -> float:
    return k_c + k_pcrcc + kq_e


def _safe_ratio(numerator: float, denominator: float) -> float:
    # eps = np.finfo(float).eps
    # if not np.isfinite(denominator) or abs(denominator) < eps:
    #     return 1
    return numerator / denominator


# ---------------------------------------------------------------------------
# Parameters & initial conditions
# ---------------------------------------------------------------------------

PARAMS: dict[str, Parameter] = {
    # from evolvePSII: dChlEx/dt = LightIntensity * crosssection (params.txt's m^2/PSII is missing the per-umol)
    "crosssection": Parameter(0.25, unit=1 / (ppfd * second)),
    "kEETLHP680QAox": Parameter(5e9, unit=_per_s),
    "kEETLHP680QAred": Parameter(8.5e8, unit=_per_s),
    "kQ": Parameter(3e9, unit=_per_s),
    "kEETLHP680revQAox": Parameter(1e10, unit=_per_s),
    "kEETLHP680revQAred": Parameter(1e10, unit=_per_s),
    "kquenchP680plus": Parameter(5e8, unit=_per_s),
    "kNRantenna": Parameter(5e8, unit=_per_s),
    "kNRP680": Parameter(1e8, unit=_per_s),
    # confirmed: getStaticQuencherValues, TotalQ = PsbSDose * (zfrac * ... + (1 - zfrac) * ...)
    "PsbSDose": Parameter(0.6, unit=dimensionless),
    "kF": Parameter(7e7, unit=_per_s),
    "alphaRC": Parameter(0.4, unit=dimensionless),
    "alphaQ": Parameter(0.1, unit=dimensionless),
    "kETP680PheOpenRC": Parameter(3e12, unit=_per_s),
    "kETP680PheClosedRC": Parameter(1e10, unit=_per_s),
    "kETPheToQA": Parameter(3e9, unit=_per_s),
    "kETWaterOxidation": Parameter(3e7, unit=_per_s),
    "kP680Pherecombination": Parameter(5e8, unit=_per_s),
    "kP680QArecombination": Parameter(30.0, unit=_per_s),
    # confirmed from the rate equations
    "kP680QArecombinationClosedRC": Parameter(580.0, unit=_per_s),
    "kETQAtoQB1": Parameter(3500.0, unit=_per_s),
    "kETQB1toQA": Parameter(350.0, unit=_per_s),
    "kETQAtoQB2": Parameter(1600.0, unit=_per_s),
    "kETQB2toQA": Parameter(1600.0, unit=_per_s),
    "PQH2undock": Parameter(800.0, unit=_per_s),
    "QReoxidationRate": Parameter(100.0, unit=_per_s),
    "PQdockingrate": Parameter(500.0, unit=_per_s),
    "QuinonePoolSize": Parameter(10.0, unit=dimensionless),  # PQ + PQH2 per PSII
    "pKaC": Parameter(5.8, unit=dimensionless),
    "nC": Parameter(1.2, unit=dimensionless),  # Hill coefficient
    "pHStromaStart": Parameter(7.2, unit=dimensionless),
    "pHLumenStart": Parameter(7.2, unit=dimensionless),
    # confirmed: getStaticThylakoidValues adds it to ion concentrations (molar)
    "StromaProtonsStart": Parameter(1e-10, unit=_molar),
    # per pH unit; params.txt says mM/pH but comment says 0.1 = 100 mM, so value is in M
    "bufferCapacityStroma": Parameter(0.1, unit=_molar),
    # per pH unit; params.txt says mM/pH but comment says 0.03 = 30 mM, so value is in M
    "bufferCapacityLumen": Parameter(0.03, unit=_molar),
    "ATPConductivity": Parameter(6e-10, unit=_conductivity),  # mol protons
    "kATPsActivate": Parameter(0.25, unit=_per_s),  # confirmed in evolveATPsynthase
    "kATPsInactivate": Parameter(0.003, unit=_per_s),  # confirmed in evolveATPsynthase
    "thresholdpmf": Parameter(0.001, unit=volt),
    "leakpmf": Parameter(0.8, unit=volt),
    # evolveATPsynthase adds its flux to the ATPConductivity flux, so the units must match
    "leakConductivity": Parameter(1e-7, unit=_conductivity),
    "PCl": Parameter(1.8e-8, unit=_permeability),
    "PMg": Parameter(3.6e-8, unit=_permeability),
    "PK": Parameter(1.8e-8, unit=_permeability),
    "zCl": Parameter(-1.0, unit=dimensionless),  # ion valence
    "zMg": Parameter(2.0, unit=dimensionless),  # ion valence
    "zK": Parameter(1.0, unit=dimensionless),  # ion valence
    # params.txt says mM but 10e-3 = 10 mM, so value is in M
    "StromaClStart": Parameter(0.01, unit=_molar),
    # params.txt says mM but 10e-3 = 10 mM, so value is in M
    "StromaMgStart": Parameter(0.01, unit=_molar),
    # params.txt says mM but 10e-3 = 10 mM, so value is in M
    "StromaKStart": Parameter(0.01, unit=_molar),
    # confirmed: getparamsfromfilename computes voltsperlog = R*T/F
    "Rconst": Parameter(8.314, unit=joule / (mol * kelvin), display_name="R"),
    # confirmed: getStaticThylakoidValues, F * lumenVolumePerArea * charge = C/cm^2
    "Fconst": Parameter(96485.0, unit=coulomb / mol, display_name="F"),
    "Tconst": Parameter(300.0, unit=kelvin, display_name="T"),
    "LumenVolume": Parameter(6.7e-21, unit=liter),  # per PSII
    "StromaVolume": Parameter(5.36e-20, unit=liter),  # per PSII
    # confirmed: fluxToConcentration divides mol/(cm^2 s) fluxes by it to get M/s
    "lumenVolumePerArea": Parameter(8e-10, unit=liter / _area),
    # confirmed: deltapsi = (C/cm^2) / MembraneCapacitance gives volts
    "MembraneCapacitance": Parameter(1e-6, unit=coulomb / (volt * _area)),
    "Na": Parameter(6.022e23, unit=1 / mol),  # Avogadro constant
    # evolveqE: acts on fractions of the xanthophyll pool, so 1/s
    "VDErateVioToAnth": Parameter(0.04, unit=_per_s),
    # evolveqE: acts on fractions of the xanthophyll pool, so 1/s
    "VDErateAnthToZea": Parameter(0.02, unit=_per_s),
    # evolveqE: acts on fractions of the xanthophyll pool, so 1/s
    "ZErate": Parameter(0.0004, unit=_per_s),
    "TotalXanthophyll": Parameter(1.0, unit=dimensionless),  # fraction
    "VDEpKa": Parameter(6.0, unit=dimensionless),
    "nVDE": Parameter(6.0, unit=dimensionless),  # Hill coefficient
    "PsbSpKa": Parameter(6.4, unit=dimensionless),
    "nPsbS": Parameter(3.0, unit=dimensionless),  # Hill coefficient
    # confirmed: getStaticQuencherValues, TotalQ = PsbSDose * (zfrac * ... + (1 - zfrac) * ...)
    "zfrac": Parameter(0.8, unit=dimensionless),
    # evolveqE: dPsbSQ/dt = PsbSConvertRate * ActivePsbS * (1 - PsbSQ), so 1/s
    "PsbSConvertRate": Parameter(0.1, unit=_per_s),
    # same role as crosssection, in evolvePSI
    "PSIcrossSection": Parameter(0.35, unit=1 / (ppfd * second)),
    "kETPCP700": Parameter(6000.0, unit=_per_s),  # confirmed from the rate equations
    # evolvePSI: rate = LightIntensity * PSIcrossSection * kETP700Fdx * ..., so it must be a dimensionless factor
    "kETP700Fdx": Parameter(10.0, unit=dimensionless),
    "PCperPSI": Parameter(3.0, unit=dimensionless),
    "ElectronsPerPC": Parameter(1.0, unit=dimensionless),
    "fracIntactRC": Parameter(1.0, unit=dimensionless),  # fraction
    # confirmed: getparamsfromfilename overwrites it with R*T/F ('Volts per Log')
    "voltsperlog": Parameter(0.02585065036015961, unit=volt),
    # umol photons m^-2 s^-1, from the crosssection comment in evolvePSII
    "LightIntensity": Parameter(
        0.0,
        unit=ppfd,
        display_name="PPFD",
        slider=Slider(min="0", max="1000", step="100"),
    ),
    "P680neut": Parameter(1.0, unit=dimensionless),  # flag
    # quench-mode trigger flags (quenchmodel=1)
    "qtrigg1": Parameter(1.0, unit=dimensionless),  # flag
    "qtrigg2": Parameter(0.0, unit=dimensionless),  # flag
    "qtrigg3": Parameter(1.0, unit=dimensionless),  # flag
    # Additional pars
    "ATPConductivityReverse": Parameter(1e-10, unit=_conductivity),  # mol protons
    "ATPperPSI": Parameter(600.0, unit=dimensionless),
    "CytRegulateYesNO": Parameter(1.0, unit=dimensionless),  # flag
    # not used in any of the provided .m files (not in params_orig either); likely obsolete
    "F_PsbS": Parameter(0.6, unit=dimensionless),
    "NADPperPSI": Parameter(15.0, unit=dimensionless),
    "PsbSperPSII": Parameter(1.0, unit=dimensionless),
    "damageyesno": Parameter(0.0, unit=dimensionless),  # flag
    "electronsPerNADPH": Parameter(2.0, unit=dimensionless),
    "kEETP700": Parameter(14000000000.0, unit=_per_s),
    "kETFdxMV": Parameter(1000.0, unit=_per_s),  # confirmed from the rate equations
    # evolveCEF: second-order in Fdxr and PQ (per quinone per PSII)
    "kETFdxPQ": Parameter(0.005, unit=_per_s),
    "kETFdxThrdx": Parameter(1000.0, unit=_per_s),  # confirmed from the rate equations
    # evolveCEF: second-order in NADPH and PQ
    "kETNADPHPQ": Parameter(100.0, unit=_per_s),
    "kETThrdxOx": Parameter(100.0, unit=_per_s),  # confirmed from the rate equations
    "kP680PherecombinationClosedRC": Parameter(500.0, unit=_per_s),
    "kP680PherecombinationOpenRC": Parameter(200000000.0, unit=_per_s),
    "kPheQArecombination": Parameter(500.0, unit=_per_s),
    "kQuenchDamage": Parameter(0.0, unit=_per_s),
    "repairyesno": Parameter(0.0, unit=dimensionless),  # flag
    # confirmed in getTransferMatrix (rates = 1/t)
    "tF": Parameter(1.5e-09, unit=second),
    # confirmed in getTransferMatrix (rates = 1/t)
    "tHop": Parameter(1.7e-11, unit=second),
    # confirmed in getTransferMatrix (rates = 1/t); its 'in ps' comment on tCS is wrong
    "tauCS": Parameter(5.5e-12, unit=second),
    # confirmed in getTransferMatrix (rates = 1/t)
    "tauqE": Parameter(1e-11, unit=second),
    "ATPperProton": Parameter(3 / 12, unit=dimensionless),
}

VARS: dict[str, Variable] = {
    # evolveATPsynthase: dATP = flux * ATPperProton * Na * LumenVolume / lumenVolumePerArea
    "ATP": Variable(2.0, unit=dimensionless),
    "ActiveATPs": Variable(0.05, unit=dimensionless),  # fraction of ATP synthase active
    # fraction of TotalXanthophyll
    "Antheraxanthin": Variable(1e-14, unit=dimensionless),
    "Fdxox": Variable(
        1.0, unit=dimensionless, display_name="Fd_ox"
    ),  # fraction of ferredoxin
    "Fdxr": Variable(
        1e-14, unit=dimensionless, display_name="Fd_red"
    ),  # fraction of ferredoxin
    "LumenCl": Variable(0.01, unit=_molar),
    "LumenK": Variable(0.01, unit=_molar),
    "LumenMg": Variable(0.01, unit=_molar),
    # protons added to the lumen; pH via bufferCapacityLumen
    "LumenProtons": Variable(1e-14, unit=_molar),
    "P680ex": Variable(1e-14, unit=dimensionless),  # fraction of PSII reaction centres
    # fraction of PSII reaction centres
    "P680plus": Variable(1e-14, unit=dimensionless),
    "P700ox": Variable(1e-14, unit=dimensionless),  # fraction of PSI
    "P700r": Variable(1.0, unit=dimensionless),  # fraction of PSI
    "PCr": Variable(0.2, unit=dimensionless),  # PCox = PCperPSI - PCr
    "PQ": Variable(8.999, unit=dimensionless),  # PQfrac = PQ / QuinonePoolSize
    "PQH2": Variable(0.001, unit=dimensionless),
    # excited antenna chlorophylls per PSII
    "PSIIChlEx": Variable(1e-14, unit=dimensionless),
    # fraction of PSII reaction centres
    "PheAnion": Variable(1e-14, unit=dimensionless),
    # fraction of PsbS in the quenching state
    "PsbSQ": Variable(1e-14, unit=dimensionless),
    # fraction of PSII reaction centres (QAred = 1 - QAox)
    "QAox": Variable(1, unit=dimensionless, display_name="QA"),
    # fraction of QB sites (called QBox in evolvePQ)
    "QBneut": Variable(1, unit=dimensionless),
    "QBred1": Variable(1e-7, unit=dimensionless),  # fraction of QB sites
    "QBred2": Variable(1e-7, unit=dimensionless),  # fraction of QB sites
    "Thrdx": Variable(1e-14, unit=dimensionless),  # fraction of thioredoxin reduced
    # cumulative electrons delivered to Fdx
    "TotalLEF": Variable(1e-14, unit=dimensionless),
    "Zeaxanthin": Variable(1e-14, unit=dimensionless),  # fraction of TotalXanthophyll
}

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def get_zaks2012() -> Model:
    r"""Get the Zaks 2012 model."""
    m = Model()
    m.add_variables(VARS)
    m.add_parameters(PARAMS)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    # pH
    m.add_derived(
        "pH_stroma",
        _get_p_h,
        args=["pHStromaStart", "bufferCapacityStroma", "StromaProtonsStart"],
    )
    m.add_derived(
        "pH_lumen",
        _get_p_h,
        args=["pHLumenStart", "bufferCapacityLumen", "LumenProtons"],
    )

    # Electric field / pmf
    m.add_derived(
        "total_charge_lumen",
        _totalcharge,
        args=["LumenProtons", "zCl", "LumenCl", "zK", "LumenK", "zMg", "LumenMg"],
    )
    m.add_derived(
        "total_charge_stroma",
        _totalcharge,
        args=[
            "StromaProtonsStart",
            "zCl",
            "StromaClStart",
            "zK",
            "StromaKStart",
            "zMg",
            "StromaMgStart",
        ],
    )
    m.add_derived(
        "deltapsi",
        _delta_psi,
        args=[
            "total_charge_lumen",
            "total_charge_stroma",
            "Fconst",
            "lumenVolumePerArea",
            "MembraneCapacitance",
        ],
    )
    m.add_derived(
        "deltapH", _delta_p_h, args=["pH_stroma", "pH_lumen"], display_name="delta_pH"
    )
    m.add_derived("pmf", _pmf, args=["deltapH", "deltapsi", "voltsperlog"])

    # Ion electrochemical potentials
    m.add_derived(
        "deltamuCl",
        _delta_mu,
        args=["zCl", "LumenCl", "StromaClStart", "deltapsi", "voltsperlog"],
    )
    m.add_derived(
        "deltamuMg",
        _delta_mu,
        args=["zMg", "LumenMg", "StromaMgStart", "deltapsi", "voltsperlog"],
    )
    m.add_derived(
        "deltamuK",
        _delta_mu,
        args=["zK", "LumenK", "StromaKStart", "deltapsi", "voltsperlog"],
    )

    # Electric-field slowdown factors
    m.add_derived(
        "efield_slowdown_r",
        _efield_slowdown,
        args=["alphaRC", "deltapsi", "voltsperlog"],
    )
    m.add_derived(
        "efield_slowdown_q",
        _efield_slowdown,
        args=["alphaQ", "deltapsi", "voltsperlog"],
    )

    # Xanthophyll & PsbS derived states
    m.add_derived(
        "Violaxanthin",
        _v_from_az,
        args=["TotalXanthophyll", "Antheraxanthin", "Zeaxanthin"],
    )
    m.add_derived("PsbS_unprot", _complement, args=["PsbSQ"])

    # Enzyme activation states
    m.add_derived(
        "active_vde",
        _protonation_fraction,
        args=["pH_lumen", "VDEpKa", "nVDE"],
    )
    m.add_derived(
        "active_psbs",
        _protonation_fraction,
        args=["pH_lumen", "PsbSpKa", "nPsbS"],
    )
    m.add_derived("deact_psbs", _complement, args=["active_psbs"])

    # Quenching
    m.add_derived(
        "q_total",
        _total_q,
        args=[
            "Antheraxanthin",
            "Zeaxanthin",
            "PsbSQ",
            "zfrac",
            "PsbSDose",
            "qtrigg1",
            "qtrigg2",
            "qtrigg3",
        ],
    )

    # PSII moiety complements
    m.add_derived(
        "QAred", _moeity_frac, args=["QAox", "fracIntactRC"], display_name="QA_red"
    )
    m.add_derived("Pheneut", _moeity_frac, args=["PheAnion", "fracIntactRC"])

    # PQ pool fractions
    m.add_derived("PQfrac", _frac, args=["PQ", "QuinonePoolSize"])
    m.add_derived("PQH2frac", _frac, args=["PQH2", "QuinonePoolSize"])
    m.add_derived("QBempty", _qb_moiety, args=["QBneut", "QBred1", "QBred2"])

    # Cyt b6f activity factors
    m.add_derived(
        "frac_active_cyt",
        _fraction_active_cyt,
        args=["pH_lumen", "pKaC", "nC"],
    )
    m.add_derived("frac_pq_red", _fraction_pq_red, args=["PQH2", "PQ"])
    m.add_derived("frac_active_pc", _fraction_active_pc, args=["PCr", "PCperPSI"])
    m.add_derived("InactiveATPs", _complement, args=["ActiveATPs"])

    # Output
    m.add_readout("light", _same, args=["LightIntensity"])

    m.add_readout("kF_obs", _k_f_rate, args=["kF", "PSIIChlEx"])

    m.add_readout("kqE_obs", _kq_e_rate, args=["kQ", "PSIIChlEx", "q_total"])

    m.add_readout(
        "kPC_obs",
        _k_pc_rate,
        args=[
            "kEETLHP680QAox",
            "kEETLHP680QAred",
            "PSIIChlEx",
            "QAox",
            "QAred",
            "P680neut",
        ],
    )

    m.add_readout(
        "kPCRCC_obs",
        _k_pcrcc_rate,
        args=["kEETLHP680QAox", "kEETLHP680QAred", "PSIIChlEx", "P680neut"],
    )

    m.add_readout(
        "kC_obs",
        _k_c_rate,
        args=["kNRantenna", "PSIIChlEx", "kF_obs", "kquenchP680plus", "P680plus"],
    )

    m.add_readout("allrates", _allrates_sum, args=["kC_obs", "kPC_obs", "kqE_obs"])

    m.add_readout(
        "allratesRCC",
        _allrates_rcc_sum,
        args=["kC_obs", "kPCRCC_obs", "kqE_obs"],
    )

    m.add_readout("fluorescenceyield", _safe_ratio, args=["kF_obs", "allrates"])

    m.add_readout("fluorescenceyieldRCC", _safe_ratio, args=["kF_obs", "allratesRCC"])

    m.add_readout("qE_model", _safe_ratio, args=["kqE_obs", "kC_obs"])

    m.add_readout("phi_npq", _safe_ratio, args=["kqE_obs", "allrates"])

    # ------------------------------------------------------------------
    # F2 - Xanthophyll cycle & PsbS
    # ------------------------------------------------------------------

    m.add_reaction(
        "V_to_A",
        fn=_mass_action_1s_act,
        args=["VDErateVioToAnth", "active_vde", "Violaxanthin"],
        stoichiometry={"Antheraxanthin": 1},
    )
    m.add_reaction(
        "A_to_Z",
        fn=_mass_action_1s_act,
        args=["VDErateAnthToZea", "active_vde", "Antheraxanthin"],
        stoichiometry={"Antheraxanthin": -1, "Zeaxanthin": 1},
    )
    m.add_reaction(
        "Z_to_A",
        fn=_mass_action_1s,
        args=["ZErate", "Zeaxanthin"],
        stoichiometry={"Zeaxanthin": -1, "Antheraxanthin": 1},
    )
    m.add_reaction(
        "A_to_V",
        fn=_mass_action_1s,
        args=["ZErate", "Antheraxanthin"],
        stoichiometry={"Antheraxanthin": -1},
    )
    m.add_reaction(
        "PsbS_prot",
        fn=_mass_action_1s_act,
        args=["PsbSConvertRate", "active_psbs", "PsbS_unprot"],
        stoichiometry={"PsbSQ": 1},
    )
    m.add_reaction(
        "PsbS_deprot",
        fn=_mass_action_1s_act,
        args=["PsbSConvertRate", "deact_psbs", "PsbSQ"],
        stoichiometry={"PsbSQ": -1},
    )

    # ------------------------------------------------------------------
    # F1 - PSII
    # ------------------------------------------------------------------

    # v1: light absorption → antenna singlet
    m.add_reaction(
        "v1",
        fn=_mass_action_1s_act,
        args=["LightIntensity", "crosssection", "fracIntactRC"],
        stoichiometry={"PSIIChlEx": 1},
    )
    # v2: quenching by NPQ
    m.add_reaction(
        "v2",
        fn=_mass_action_1s_act,
        args=["kQ", "q_total", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v3: fluorescence
    m.add_reaction(
        "v3",
        fn=_mass_action_1s,
        args=["kF", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v4: quenching by P680+
    m.add_reaction(
        "v4",
        fn=_mass_action_1s_act,
        args=["kquenchP680plus", "P680plus", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v5: non-radiative decay
    m.add_reaction(
        "v5",
        fn=_mass_action_1s,
        args=["kNRantenna", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v7: energy transfer to open/closed RC → P680*
    m.add_reaction(
        "v7",
        fn=_v7,
        args=[
            "kEETLHP680QAox",
            "kEETLHP680QAred",
            "PSIIChlEx",
            "QAox",
            "QAred",
            "P680neut",
        ],
        stoichiometry={"PSIIChlEx": -1, "P680ex": 1},
    )
    # v8: back-transfer P680* → antenna
    m.add_reaction(
        "v8",
        fn=_v8,
        args=["kEETLHP680revQAox", "kEETLHP680revQAred", "P680ex", "QAox", "QAred"],
        stoichiometry={"PSIIChlEx": 1, "P680ex": -1},
    )
    # v9: non-radiative decay of P680*
    m.add_reaction(
        "v9",
        fn=_mass_action_1s,
        args=["kNRP680", "P680ex"],
        stoichiometry={"P680ex": -1},
    )
    # v12: primary charge separation - open RC (QA oxidised)
    m.add_reaction(
        "v12",
        fn=_v12_13,
        args=["kETP680PheOpenRC", "P680ex", "Pheneut", "QAox", "efield_slowdown_r"],
        stoichiometry={"P680ex": -1, "P680plus": 1, "PheAnion": 1},
    )
    # v13: primary charge separation - closed RC (QA reduced)
    m.add_reaction(
        "v13",
        fn=_v12_13,
        args=["kETP680PheClosedRC", "P680ex", "Pheneut", "QAred", "efield_slowdown_r"],
        stoichiometry={"P680ex": -1, "P680plus": 1, "PheAnion": 1},
    )
    # v14: Phe⁻ → QA electron transfer (QA becomes reduced, tracked as -QAox)
    m.add_reaction(
        "v14",
        fn=_v14,
        args=["kETPheToQA", "PheAnion", "QAox", "efield_slowdown_r"],
        stoichiometry={"PheAnion": -1, "QAox": -1},
    )
    # v15: water oxidation by OEC
    m.add_reaction(
        "v15",
        fn=_mass_action_1s_act,
        args=["kETWaterOxidation", "P680plus", "efield_slowdown_q"],
        stoichiometry={
            "P680plus": -1,
            "LumenProtons": Derived(fn=_psii_proton_stoi, args=["Na", "LumenVolume"]),
        },
    )
    # v16: P680+/Phe⁻ recombination
    m.add_reaction(
        "v16",
        fn=_v16,
        args=["kP680Pherecombination", "P680plus", "PheAnion", "efield_slowdown_r"],
        stoichiometry={"P680plus": -1, "PheAnion": -1},
    )
    # v17: P680+/QA⁻ recombination (Phe neutral)
    m.add_reaction(
        "v17",
        fn=_v17,
        args=[
            "kP680QArecombination",
            "P680plus",
            "QAred",
            "Pheneut",
            "efield_slowdown_r",
        ],
        stoichiometry={"P680plus": -1, "QAox": 1},
    )
    # v18: P680+/QA⁻ recombination (Phe anion)
    m.add_reaction(
        "v18",
        fn=_v18,
        args=[
            "kP680QArecombinationClosedRC",
            "P680plus",
            "QAred",
            "PheAnion",
            "efield_slowdown_r",
        ],
        stoichiometry={"P680plus": -1, "QAox": 1},
    )

    # ------------------------------------------------------------------
    # F3 - PQ pool / QB site
    # ------------------------------------------------------------------

    # r_q2: QA⁻ → QB (first reduction)
    m.add_reaction(
        "r_q2",
        fn=_r_q2,
        args=["kETQAtoQB1", "QAred", "QBneut", "efield_slowdown_q"],
        stoichiometry={"QAox": 1, "QBneut": -1, "QBred1": 1},
    )
    # r_q3: QA⁻ → QB⁻ (second reduction → PQH₂ at QB)
    m.add_reaction(
        "r_q3",
        fn=_r_q3,
        args=["kETQAtoQB2", "QAred", "QBred1", "efield_slowdown_q"],
        stoichiometry={"QAox": 1, "QBred1": -1, "QBred2": 1},
    )
    # r_q4: reverse QB⁻ → QA (first)
    m.add_reaction(
        "r_q4",
        fn=_mass_action_1s_act,
        args=["kETQB1toQA", "QAox", "QBred1"],
        stoichiometry={"QAox": -1, "QBred1": -1, "QBneut": 1},
    )
    # r_q5: reverse QB²⁻ → QA (second)
    m.add_reaction(
        "r_q5",
        fn=_mass_action_1s_act,
        args=["kETQB2toQA", "QAox", "QBred2"],
        stoichiometry={"QAox": -1, "QBred1": 1, "QBred2": -1},
    )
    # r_q6: PQ docking at QB site
    m.add_reaction(
        "r_q6",
        fn=_mass_action_1s_act,
        args=["PQdockingrate", "PQfrac", "QBempty"],
        stoichiometry={"PQ": -1, "QBneut": 1},
    )
    # r_q7: PQH₂ undocking from QB site
    m.add_reaction(
        "r_q7",
        fn=_mass_action_1s,
        args=["PQH2undock", "QBred2"],
        stoichiometry={"QBred2": -1, "PQH2": 1},
    )
    # r_q8: PQH₂ re-docking (reverse, ÷10)
    m.add_reaction(
        "r_q8",
        fn=_r_q8,
        args=["PQH2undock", "PQH2frac"],
        stoichiometry={"PQH2": -1, "QBred2": 1},
    )

    # ------------------------------------------------------------------
    # F4 - Cytochrome b6f
    # ------------------------------------------------------------------
    # r_cyt_b6f is pre-computed as a derived quantity above.
    # Stoichiometry: consumes PQH₂, produces PQ, reduces PC, pumps 4H⁺ into lumen.

    m.add_reaction(
        "r_cyt_b6f",
        fn=_r_cyt,
        args=["QReoxidationRate", "frac_active_cyt", "frac_active_pc", "frac_pq_red"],
        stoichiometry={
            "PQH2": -1,
            "PQ": 1,
            "PCr": Derived(fn=_cyt_electron_stoi, args=["ElectronsPerPC"]),
            "LumenProtons": Derived(fn=_cyt_proton_stoi, args=["Na", "LumenVolume"]),
        },
    )  # CHECK

    # ------------------------------------------------------------------
    # F5 - PSI
    # ------------------------------------------------------------------

    # r_psi_1: PC reduction of P700+
    m.add_reaction(
        "psi_1",
        fn=_psi_1,
        args=["kETPCP700", "PCr", "P700ox"],
        stoichiometry={"PCr": -1, "P700ox": -1, "P700r": 1},
    )
    # r_psi_2: P700* reduces Fdx (light-driven)
    m.add_reaction(
        "psi_2",
        fn=_psi_2,
        args=["LightIntensity", "PSIcrossSection", "kETP700Fdx", "P700r", "Fdxox"],
        stoichiometry={"P700r": -1, "P700ox": 1, "Fdxr": 1, "Fdxox": -1, "TotalLEF": 1},
    )

    # ------------------------------------------------------------------
    # F7 - ATP synthase + leak
    # ------------------------------------------------------------------

    m.add_reaction(
        "atp_synthesis",
        fn=_proton_flux_atp,
        args=["ATPConductivity", "pmf", "thresholdpmf", "ActiveATPs"],
        stoichiometry={
            "ATP": Derived(
                fn=_atp_stoi,
                args=["Na", "LumenVolume", "lumenVolumePerArea", "ATPperProton"],
            ),
            "LumenProtons": Derived(
                fn=_atpsyn_proton_stoi, args=["lumenVolumePerArea"]
            ),
        },
    )

    m.add_reaction(
        "vleak",
        fn=_proton_flux_leak,
        args=["leakConductivity", "pmf", "leakpmf"],
        stoichiometry={
            "LumenProtons": Derived(
                fn=_atpsyn_proton_stoi, args=["lumenVolumePerArea"]
            ),
        },
        display_name="proton leak",
    )

    m.add_reaction(
        "atps_activate",
        fn=_mass_action_1s_act,
        args=["kATPsActivate", "Fdxr", "InactiveATPs"],
        stoichiometry={"ActiveATPs": 1},
    )
    m.add_reaction(
        "atps_inactivate",
        fn=_mass_action_1s,
        args=["kATPsInactivate", "ActiveATPs"],
        stoichiometry={"ActiveATPs": -1},
    )

    # ------------------------------------------------------------------
    # F8 - Lumen ion fluxes
    # ------------------------------------------------------------------

    m.add_reaction(
        "flux_Mg",
        fn=_lumen_ion_flux,
        args=[
            "LumenMg",
            "StromaMgStart",
            "deltamuMg",
            "PMg",
            "voltsperlog",
            "lumenVolumePerArea",
        ],
        stoichiometry={"LumenMg": 1},
    )
    m.add_reaction(
        "flux_Cl",
        fn=_lumen_ion_flux,
        args=[
            "LumenCl",
            "StromaClStart",
            "deltamuCl",
            "PCl",
            "voltsperlog",
            "lumenVolumePerArea",
        ],
        stoichiometry={"LumenCl": 1},
    )
    # NOTE: public MATLAB uses PCl (not PK) for the K flux — preserved here.
    m.add_reaction(
        "flux_K",
        fn=_lumen_ion_flux,
        args=[
            "LumenK",
            "StromaKStart",
            "deltamuK",
            # FIXME: uses the Cl- permeability, while PK is defined but unused
            "PCl",
            "voltsperlog",
            "lumenVolumePerArea",
        ],
        stoichiometry={"LumenK": 1},
    )

    # ------------------------------------------------------------------
    # F9 - Methyl Viologen
    # ------------------------------------------------------------------

    m.add_reaction(
        "mv_1",
        fn=_mass_action_1s,
        args=["kETFdxMV", "Fdxr"],
        stoichiometry={"Fdxr": -1},
    )

    m.add_reaction(
        "mv_2",
        fn=_mass_action_1s,
        args=["kETFdxThrdx", "Fdxr"],
        stoichiometry={"Fdxox": 1, "Thrdx": 1},
    )

    m.add_reaction(
        "mv_3",
        fn=_mass_action_1s,
        args=["kETThrdxOx", "Thrdx"],
        stoichiometry={"Thrdx": -1},
    )

    return m
