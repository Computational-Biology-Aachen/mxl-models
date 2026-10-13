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
from mxlpy import Derived, Model, Parameter, Variable

import sympy
from sympy.physics import units as u
from sympy.physics.units import Dimension, Quantity
from sympy.physics.units.definitions.dimension_definitions import (
    amount_of_substance,
    length,
)
from sympy.physics.units.systems.si import SI
 
# Annotation and InitialAssignment are assumed to be defined/imported elsewhere in your project.
 
 
def _unit(
    name: str,
    abbrev: str | None = None,
    dimension: Dimension | sympy.Expr = Dimension(1),
    scale: sympy.Expr = sympy.S.One,
) -> Quantity:
    """Create a custom sympy unit and register it with the SI unit system."""
    q = Quantity(name, abbrev=abbrev or name)
    SI.set_quantity_dimension(q, dimension)
    SI.set_quantity_scale_factor(q, scale)
    return q


# --- generic helpers -------------------------------------------------------
dimensionless = sympy.Integer(1)
per_s = u.s**-1

# --- concentrations ---------------------------------------------------------
molar = _unit("molar", "M", amount_of_substance / length**3, u.mol / u.liter)
millimolar = _unit("millimolar", "mM", amount_of_substance / length**3, molar / 1000)

micromole = _unit("micromole", "umol", amount_of_substance, u.mol / 10**6)

# --- counting units (dimensionless, scale 1) --------------------------------
# These keep the "per complex" / "per molecule" bookkeeping from params.txt
# visible in the expressions, without affecting dimensional analysis.
PSII = _unit("PSII")
PSI = _unit("PSI")
quinone = _unit("quinone")
PQ = _unit("PQ")
PQH2 = _unit("PQH2")
ATP = _unit("ATP")
NADP = _unit("NADP")
NADPH = _unit("NADPH")
PC = _unit("PC")
PsbS = _unit("PsbS")
quencher = _unit("quencher", "Q")
pigment = _unit("pigment")
excitation = _unit("excitation")
electron = _unit("electron", "e")
proton = _unit("proton", "H")
pH_unit = _unit("pH_unit", "pH")

# --- placeholder for units still to be determined --------------------------
dnd = _unit("dnd", "DND")


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

PARAMS = {
    "crosssection": Parameter(value=0.25, unit=excitation * u.m**2 / (PSII * micromole)),  # from evolvePSII: dChlEx/dt = LightIntensity * crosssection (params.txt's m^2/PSII is missing the per-umol)
    "kEETLHP680QAox": Parameter(value=5e9, unit=per_s),
    "kEETLHP680QAred": Parameter(value=8.5e8, unit=per_s),
    "kQ": Parameter(value=3e9, unit=per_s),
    "kEETLHP680revQAox": Parameter(value=1e10, unit=per_s),
    "kEETLHP680revQAred": Parameter(value=1e10, unit=per_s),
    "kquenchP680plus": Parameter(value=5e8, unit=per_s),
    "kNRantenna": Parameter(value=5e8, unit=per_s),
    "kNRP680": Parameter(value=1e8, unit=per_s),
    "PsbSDose": Parameter(value=0.6, unit=dimensionless),  # confirmed: getStaticQuencherValues, TotalQ = PsbSDose * (zfrac * ... + (1 - zfrac) * ...)
    "kF": Parameter(value=7e7, unit=per_s),
    "alphaRC": Parameter(value=0.4, unit=dimensionless),
    "alphaQ": Parameter(value=0.1, unit=dimensionless),
    "kETP680PheOpenRC": Parameter(value=3e12, unit=per_s),
    "kETP680PheClosedRC": Parameter(value=1e10, unit=per_s),
    "kETPheToQA": Parameter(value=3e9, unit=per_s),
    "kETWaterOxidation": Parameter(value=3e7, unit=per_s),
    "kP680Pherecombination": Parameter(value=5e8, unit=per_s),
    "kP680QArecombination": Parameter(value=30.0, unit=per_s),
    "kP680QArecombinationClosedRC": Parameter(value=580.0, unit=per_s),  # confirmed from the rate equations
    "kETQAtoQB1": Parameter(value=3500.0, unit=per_s),
    "kETQB1toQA": Parameter(value=350.0, unit=per_s),
    "kETQAtoQB2": Parameter(value=1600.0, unit=per_s),
    "kETQB2toQA": Parameter(value=1600.0, unit=per_s),
    "PQH2undock": Parameter(value=800.0, unit=per_s),
    "QReoxidationRate": Parameter(value=100.0, unit=PQH2 / u.s),  # PQH2/s
    "PQdockingrate": Parameter(value=500.0, unit=PQ / u.s),  # PQ/s
    "QuinonePoolSize": Parameter(value=10.0, unit=quinone / PSII),  # PQ + PQH2 per PSII
    "pKaC": Parameter(value=5.8, unit=pH_unit),
    "nC": Parameter(value=1.2, unit=dimensionless),  # Hill coefficient
    "pHStromaStart": Parameter(value=7.2, unit=pH_unit),
    "pHLumenStart": Parameter(value=7.2, unit=pH_unit),
    "StromaProtonsStart": Parameter(value=1e-10, unit=molar),  # confirmed: getStaticThylakoidValues adds it to ion concentrations (molar)
    "bufferCapacityStroma": Parameter(value=0.1, unit=molar / pH_unit),  # per pH unit; params.txt says mM/pH but comment says 0.1 = 100 mM, so value is in M
    "bufferCapacityLumen": Parameter(value=0.03, unit=molar / pH_unit),  # per pH unit; params.txt says mM/pH but comment says 0.03 = 30 mM, so value is in M
    "ATPConductivity": Parameter(value=6e-10, unit=u.mol / (u.volt * u.cm**2 * u.s)),  # mol protons
    "kATPsActivate": Parameter(value=0.25, unit=per_s),  # confirmed in evolveATPsynthase
    "kATPsInactivate": Parameter(value=0.003, unit=per_s),  # confirmed in evolveATPsynthase
    "thresholdpmf": Parameter(value=0.001, unit=u.volt),
    "leakpmf": Parameter(value=0.8, unit=u.volt),
    "leakConductivity": Parameter(value=1e-7, unit=u.mol / (u.volt * u.cm**2 * u.s)),  # evolveATPsynthase adds its flux to the ATPConductivity flux, so the units must match
    "PCl": Parameter(value=1.8e-8, unit=u.cm / u.s),
    "PMg": Parameter(value=3.6e-8, unit=u.cm / u.s),
    "PK": Parameter(value=1.8e-8, unit=u.cm / u.s),
    "zCl": Parameter(value=-1.0, unit=dimensionless),  # ion valence
    "zMg": Parameter(value=2.0, unit=dimensionless),  # ion valence
    "zK": Parameter(value=1.0, unit=dimensionless),  # ion valence
    "StromaClStart": Parameter(value=0.01, unit=molar),  # params.txt says mM but 10e-3 = 10 mM, so value is in M
    "StromaMgStart": Parameter(value=0.01, unit=molar),  # params.txt says mM but 10e-3 = 10 mM, so value is in M
    "StromaKStart": Parameter(value=0.01, unit=molar),  # params.txt says mM but 10e-3 = 10 mM, so value is in M
    "Rconst": Parameter(value=8.314, unit=u.joule / (u.mol * u.kelvin)),  # confirmed: getparamsfromfilename computes voltsperlog = R*T/F
    "Fconst": Parameter(value=96485.0, unit=u.coulomb / u.mol),  # confirmed: getStaticThylakoidValues, F * lumenVolumePerArea * charge = C/cm^2
    "Tconst": Parameter(value=300.0, unit=u.kelvin),
    "LumenVolume": Parameter(value=6.7e-21, unit=u.liter),  # per PSII
    "StromaVolume": Parameter(value=5.36e-20, unit=u.liter),  # per PSII
    "lumenVolumePerArea": Parameter(value=8e-10, unit=u.liter / u.cm**2),  # confirmed: fluxToConcentration divides mol/(cm^2 s) fluxes by it to get M/s
    "MembraneCapacitance": Parameter(value=1e-6, unit=u.farad / u.cm**2),  # confirmed: deltapsi = (C/cm^2) / MembraneCapacitance gives volts
    "Na": Parameter(value=6.022e23, unit=u.mol**-1),  # Avogadro constant
    "VDErateVioToAnth": Parameter(value=0.04, unit=per_s),  # evolveqE: acts on fractions of the xanthophyll pool, so 1/s
    "VDErateAnthToZea": Parameter(value=0.02, unit=per_s),  # evolveqE: acts on fractions of the xanthophyll pool, so 1/s
    "ZErate": Parameter(value=0.0004, unit=per_s),  # evolveqE: acts on fractions of the xanthophyll pool, so 1/s
    "TotalXanthophyll": Parameter(value=1.0, unit=dimensionless),  # fraction
    "VDEpKa": Parameter(value=6.0, unit=pH_unit),
    "nVDE": Parameter(value=6.0, unit=dimensionless),  # Hill coefficient
    "PsbSpKa": Parameter(value=6.4, unit=pH_unit),
    "nPsbS": Parameter(value=3.0, unit=dimensionless),  # Hill coefficient
    "zfrac": Parameter(value=0.8, unit=dimensionless),  # confirmed: getStaticQuencherValues, TotalQ = PsbSDose * (zfrac * ... + (1 - zfrac) * ...)
    "PsbSConvertRate": Parameter(value=0.1, unit=per_s),  # evolveqE: dPsbSQ/dt = PsbSConvertRate * ActivePsbS * (1 - PsbSQ), so 1/s
    "PSIcrossSection": Parameter(value=0.35, unit=excitation * u.m**2 / (PSI * micromole)),  # same role as crosssection, in evolvePSI
    "kETPCP700": Parameter(value=6000.0, unit=per_s),  # confirmed from the rate equations
    "kETP700Fdx": Parameter(value=10.0, unit=dimensionless),  # evolvePSI: rate = LightIntensity * PSIcrossSection * kETP700Fdx * ..., so it must be a dimensionless factor
    "PCperPSI": Parameter(value=3.0, unit=PC / PSI),
    "ElectronsPerPC": Parameter(value=1.0, unit=electron / PC),
    "fracIntactRC": Parameter(value=1.0, unit=dimensionless),  # fraction
    "voltsperlog": Parameter(value=0.02585065036015961, unit=u.volt),  # confirmed: getparamsfromfilename overwrites it with R*T/F ('Volts per Log')
    "LightIntensity": Parameter(value=0.0, unit=micromole / (u.m**2 * u.s)),  # umol photons m^-2 s^-1, from the crosssection comment in evolvePSII
    "P680neut": Parameter(value=1.0, unit=dimensionless),  # flag
    # quench-mode trigger flags (quenchmodel=1)
    "qtrigg1": Parameter(value=1.0, unit=dimensionless),  # flag
    "qtrigg2": Parameter(value=0.0, unit=dimensionless),  # flag
    "qtrigg3": Parameter(value=1.0, unit=dimensionless),  # flag
    # Additional pars
    "ATPConductivityReverse": Parameter(value=1e-10, unit=u.mol / (u.volt * u.cm**2 * u.s)),  # mol protons
    "ATPperPSI": Parameter(value=600.0, unit=ATP / PSI),
    "CytRegulateYesNO": Parameter(value=1.0, unit=dimensionless),  # flag
    "F_PsbS": Parameter(value=0.6, unit=dnd),  # not used in any of the provided .m files (not in params_orig either); likely obsolete
    "NADPperPSI": Parameter(value=15.0, unit=NADP / PSI),
    "PsbSperPSII": Parameter(value=1.0, unit=PsbS / PSII),
    "damageyesno": Parameter(value=0.0, unit=dimensionless),  # flag
    "electronsPerNADPH": Parameter(value=2.0, unit=electron / NADPH),
    "kEETP700": Parameter(value=14000000000.0, unit=per_s),
    "kETFdxMV": Parameter(value=1000.0, unit=per_s),  # confirmed from the rate equations
    "kETFdxPQ": Parameter(value=0.005, unit=per_s / PQ),  # evolveCEF: second-order in Fdxr and PQ (per quinone per PSII)
    "kETFdxThrdx": Parameter(value=1000.0, unit=per_s),  # confirmed from the rate equations
    "kETNADPHPQ": Parameter(value=100.0, unit=per_s / (NADPH * PQ)),  # evolveCEF: second-order in NADPH and PQ
    "kETThrdxOx": Parameter(value=100.0, unit=per_s),  # confirmed from the rate equations
    "kP680PherecombinationClosedRC": Parameter(value=500.0, unit=per_s),
    "kP680PherecombinationOpenRC": Parameter(value=200000000.0, unit=per_s),
    "kPheQArecombination": Parameter(value=500.0, unit=per_s),
    "kQuenchDamage": Parameter(value=0.0, unit=per_s),
    "repairyesno": Parameter(value=0.0, unit=dimensionless),  # flag
    "tF": Parameter(value=1.5e-09, unit=u.s),  # confirmed in getTransferMatrix (rates = 1/t)
    "tHop": Parameter(value=1.7e-11, unit=u.s),  # confirmed in getTransferMatrix (rates = 1/t)
    "tauCS": Parameter(value=5.5e-12, unit=u.s),  # confirmed in getTransferMatrix (rates = 1/t); its 'in ps' comment on tCS is wrong
    "tauqE": Parameter(value=1e-11, unit=u.s),  # confirmed in getTransferMatrix (rates = 1/t)
    "ATPperProton": Parameter(value=3 / 12, unit=ATP / proton),
}


VARS = {
    "ATP": Variable(initial_value=2.0, unit=ATP / PSII),  # evolveATPsynthase: dATP = flux * ATPperProton * Na * LumenVolume / lumenVolumePerArea
    "ActiveATPs": Variable(initial_value=0.05, unit=dimensionless),  # fraction of ATP synthase active
    "Antheraxanthin": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of TotalXanthophyll
    "Fdxox": Variable(initial_value=1.0, unit=dimensionless),  # fraction of ferredoxin
    "Fdxr": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of ferredoxin
    "LumenCl": Variable(initial_value=0.01, unit=molar),
    "LumenK": Variable(initial_value=0.01, unit=molar),
    "LumenMg": Variable(initial_value=0.01, unit=molar),
    "LumenProtons": Variable(initial_value=1e-14, unit=molar),  # protons added to the lumen; pH via bufferCapacityLumen
    "P680ex": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of PSII reaction centres
    "P680plus": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of PSII reaction centres
    "P700ox": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of PSI
    "P700r": Variable(initial_value=1.0, unit=dimensionless),  # fraction of PSI
    "PCr": Variable(initial_value=0.2, unit=PC / PSI),  # PCox = PCperPSI - PCr
    "PQ": Variable(initial_value=8.999, unit=PQ / PSII),  # PQfrac = PQ / QuinonePoolSize
    "PQH2": Variable(initial_value=0.001, unit=PQH2 / PSII),
    "PSIIChlEx": Variable(initial_value=1e-14, unit=excitation / PSII),  # excited antenna chlorophylls per PSII
    "PheAnion": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of PSII reaction centres
    "PsbSQ": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of PsbS in the quenching state
    "QAox": Variable(initial_value=1, unit=dimensionless),  # fraction of PSII reaction centres (QAred = 1 - QAox)
    "QBneut": Variable(initial_value=1, unit=dimensionless),  # fraction of QB sites (called QBox in evolvePQ)
    "QBred1": Variable(initial_value=1e-7, unit=dimensionless),  # fraction of QB sites
    "QBred2": Variable(initial_value=1e-7, unit=dimensionless),  # fraction of QB sites
    "Thrdx": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of thioredoxin reduced
    "TotalLEF": Variable(initial_value=1e-14, unit=electron / PSI),  # cumulative electrons delivered to Fdx
    "Zeaxanthin": Variable(initial_value=1e-14, unit=dimensionless),  # fraction of TotalXanthophyll
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
    m.add_derived("deltapH", _delta_p_h, args=["pH_stroma", "pH_lumen"])
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
    m.add_derived("QAred", _moeity_frac, args=["QAox", "fracIntactRC"])
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
