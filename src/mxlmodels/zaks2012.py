"""
Complete mxlpy port of the Zaks et al. photosynthesis model.

Sections:
  F1  – PSII (antenna + reaction centre)
  F2  – qE / xanthophyll cycle
  F3  – PQ pool (QB site + plastoquinone)
  F4  – Cytochrome b6f
  F5  – PSI
  F7  – ATP synthase
  F8  – Lumen ion fluxes (Mg, Cl, K)
"""

import numpy as np
from mxlpy import Model, Derived

# ---------------------------------------------------------------------------
# Helper / rate functions
# ---------------------------------------------------------------------------

def protonation_fraction(ph: float, pKa: float, hill_n: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (hill_n * (ph - pKa)))


def get_pH(pH_start: float, bH: float, protons: float) -> float:
    return pH_start - protons / bH


def totalcharge(proton, zCl, Cl, zK, K, zMg, Mg) -> float:
    return proton + zCl * Cl + zK * K + zMg * Mg


def delta_pH(pH_stroma: float, pH_lumen: float) -> float:
    return pH_stroma - pH_lumen


def delta_psi(
    total_charge_lumen: float,
    total_charge_stroma: float,
    Fconst: float,
    lumenVolumePerArea: float,
    MembraneCapacitance: float,
) -> float:
    return (
        (total_charge_lumen - total_charge_stroma)
        * Fconst
        * lumenVolumePerArea
        / MembraneCapacitance
    )


def pmf(delta_pH: float, delta_psi: float, voltsperlog: float) -> float:
    return delta_psi + np.log(10) * voltsperlog * delta_pH


def delta_mu(
    z: float,
    lumen_conc: float,
    stroma_conc: float,
    delta_psi: float,
    voltsperlog: float,
) -> float:
    diffusion_potential = voltsperlog * np.log(lumen_conc / stroma_conc)
    return z * delta_psi + diffusion_potential


def efield_slowdown(alpha: float, delta_psi: float, voltsperlog: float) -> float:
    return np.exp(-alpha * delta_psi / voltsperlog)


def flux_to_concentration_lumen(lumenVolumePerArea: float) -> float:
    return 1.0 / lumenVolumePerArea

def atpsyn_proton_stoi(lumenVolumePerArea: float) -> float:
    return -1.0 / lumenVolumePerArea


def ion_flux_linear(
    lumen_conc: float,
    stroma_conc: float,
    delta_mu: float,
    permeability: float,
    voltsperlog: float,
) -> float:
    flowout = float(delta_mu > 0)
    liters_per_cc = 1e-3
    return (
        -(lumen_conc * flowout + stroma_conc * (1.0 - flowout))
        * liters_per_cc
        * permeability
        * delta_mu
        / voltsperlog
    )


def total_q(
    anth: float,
    zea: float,
    psbs_q: float,
    zfrac: float,
    PsbSDose: float,
    q_trig1: float,
    q_trig2: float,
    q_trig3: float,
) -> float:
    q_x = zfrac * (zea + 0.5 * anth) * psbs_q * q_trig1 + zfrac * psbs_q * q_trig2
    q_l = (1.0 - zfrac) * psbs_q * q_trig3
    return PsbSDose * (q_x + q_l)


def V_from_AZ(Xtot, A, Z):
    return Xtot - A - Z


def complement(c1):
    """1 – c1  (used for moiety complements)."""
    return 1.0 - c1


def moeity_frac(c1, frac):
    return (1.0 - c1) * frac


def mass_action_1s(kf: float, s: float) -> float:
    return kf * s


def mass_action_1s_act(kf: float, act: float, s: float) -> float:
    return kf * act * s


# PSII-specific rates
def v7(kEETLHP680QAox, kEETLHP680QAred, chl_ex, qa_ox, qa_red, p680_neut):
    return (kEETLHP680QAox * chl_ex * qa_ox + kEETLHP680QAred * chl_ex * qa_red) * p680_neut


def v8(kEETLHP680revQAox, kEETLHP680revQAred, p680_ex, qa_ox, qa_red):
    return kEETLHP680revQAox * p680_ex * qa_ox + kEETLHP680revQAred * p680_ex * qa_red


def v12_13(kETP680PheRC, p680_ex, phe_neut, qa, efield_slowdown_r):
    return kETP680PheRC * p680_ex * phe_neut * qa * efield_slowdown_r


def v14(kETPheToQA, phe_anion, qa_ox, efield_slowdown_r):
    return kETPheToQA * phe_anion * qa_ox * efield_slowdown_r


def v16(kP680Pherecombination, p680_plus, phe_anion, efield_slowdown_r):
    return kP680Pherecombination * p680_plus * phe_anion / efield_slowdown_r


def v17(kP680QArecombination, p680_plus, qa_red, phe_neut, efield_slowdown_r):
    return kP680QArecombination * p680_plus * qa_red * phe_neut / efield_slowdown_r


def v18(kP680QArecombinationClosedRC, p680_plus, qa_red, phe_anion, efield_slowdown_r):
    return kP680QArecombinationClosedRC * p680_plus * qa_red * phe_anion / efield_slowdown_r


# Cyt b6f
def fraction_active_cyt(ph_lumen, pKaC, nC):
    return 1.0 - protonation_fraction(ph_lumen, pKaC, nC)


def fraction_pq_red(pqh2, pq):
    return pqh2 / (pq + pqh2)


def fraction_active_pc(pcr, PCperPSI):
    return (PCperPSI - pcr) / PCperPSI


def r_cyt(QReoxidationRate, frac_active_cyt, frac_active_pc, frac_pq_red):
    return QReoxidationRate * frac_active_cyt * frac_active_pc * frac_pq_red

def r_q2(kETQAtoQB1, QAred, QBneut, esq) -> float:
    return kETQAtoQB1 * QAred * QBneut * esq

def r_q3(kETQAtoQB2, QAred, QBred1, esq) -> float:
    return kETQAtoQB2 * QAred * QBred1 * esq

def r_q8(rate, pqh2frac) -> float:
    return rate * pqh2frac / 10.0


# Stoichiometric coefficients for lumen proton accumulation
def psii_proton_stoi(Na, LumenVolume):
    return 1.0 / (Na * LumenVolume)


def cyt_proton_stoi(Na, LumenVolume):
    return 4.0 / (Na * LumenVolume)


def cyt_electron_stoi(ElectronsPerPC):
    return 2.0 / ElectronsPerPC

# PSI

def psi_1(kETPCP700, PCr, P700ox) -> float:
    return max(kETPCP700 * PCr * P700ox, 0.0)

def psi_2(light, PSIcrossSection, kETP700Fdx, P700r, Fdxox) -> float:
    return max(light * PSIcrossSection * kETP700Fdx * P700r * Fdxox, 0.0)


# ATP synthase
def proton_flux_atp(ATPConductivity, pmf, thresholdpmf, active_atps):
    diff = pmf - thresholdpmf
    return ATPConductivity * diff * float(diff > 0) * active_atps


def proton_flux_leak(leakConductivity, pmf, leakpmf):
    diff = pmf - leakpmf
    return leakConductivity * diff * float(diff > 0)

def atp_stoi(Na, LumenVolume, lumenVolumePerArea, ATPperProton):
    return ATPperProton * Na * LumenVolume / lumenVolumePerArea

# Ion fluxes → concentration change in lumen
def lumen_ion_flux(lumen_conc, stroma_conc, dmu, permeability, voltsperlog, lumenVolumePerArea):
    flux = ion_flux_linear(lumen_conc, stroma_conc, dmu, permeability, voltsperlog)
    return flux / lumenVolumePerArea

def same(x) -> float:
    return x

def frac(x, xtot) -> float:
    return x/xtot

def qb_moiety(qb_n, qb_r1, qb_r2) -> float:
    return 1.0 - qb_n - qb_r1 - qb_r2

# Output


def kF_rate(kF: float, chl_ex: float) -> float:
    return kF * chl_ex

def kqE_rate(kQ: float, chl_ex: float, q_total: float) -> float:
    return kQ * chl_ex * q_total

def kPC_rate(kEETLHP680QAox: float, kEETLHP680QAred: float,
             chl_ex: float, qa_ox: float, qa_red: float, p680_neut: float) -> float:
    return (kEETLHP680QAox * chl_ex * qa_ox + kEETLHP680QAred * chl_ex * qa_red) * p680_neut

def kPCRCC_rate(kEETLHP680QAox: float, kEETLHP680QAred: float,
                chl_ex: float, p680_neut: float) -> float:
    # Open-RC reference: QAox=1, QAred=0
    return (kEETLHP680QAox * chl_ex * 1.0 + kEETLHP680QAred * chl_ex * 0.0) * p680_neut

def kC_rate(kNRantenna: float, chl_ex: float, kF_val: float,
            kquenchP680plus: float, p680_plus: float) -> float:
    return kNRantenna * chl_ex + kF_val + kquenchP680plus * chl_ex * p680_plus

def allrates_sum(kC: float, kPC: float, kqE: float) -> float:
    return kC + kPC + kqE

def allratesRCC_sum(kC: float, kPCRCC: float, kqE: float) -> float:
    return kC + kPCRCC + kqE

def safe_ratio(numerator: float, denominator: float) -> float:
    eps = np.finfo(float).eps
    if not np.isfinite(denominator) or abs(denominator) < eps:
        return 1
    return numerator / denominator

# def fluorescenceyieldRCC(kF_rate: float, kC_rate: float, kPC_rate: float, kqE_rate: float) -> float:
#     return kF_rate /(kC_rate + kPC_rate + kqE_rate )

# ---------------------------------------------------------------------------
# Parameters & initial conditions
# ---------------------------------------------------------------------------

PARAMS = {
    "crosssection": 0.25,
    "kEETLHP680QAox": 5e9,
    "kEETLHP680QAred": 8.5e8,
    "kQ": 3e9,
    "kEETLHP680revQAox": 1e10,
    "kEETLHP680revQAred": 1e10,
    "kquenchP680plus": 5e8,
    "kNRantenna": 5e8,
    "kNRP680": 1e8,
    "PsbSDose": 0.6,
    "kF": 7e7,
    "alphaRC": 0.4,
    "alphaQ": 0.1,
    "kETP680PheOpenRC": 3e12,
    "kETP680PheClosedRC": 1e10,
    "kETPheToQA": 3e9,
    "kETWaterOxidation": 3e7,
    "kP680Pherecombination": 5e8,
    "kP680QArecombination": 30.0,
    "kP680QArecombinationClosedRC": 580.0,
    "kETQAtoQB1": 3500.0,
    "kETQB1toQA": 350.0,
    "kETQAtoQB2": 1600.0,
    "kETQB2toQA": 1600.0,
    "PQH2undock": 800.0,
    "QReoxidationRate": 100.0,
    "PQdockingrate": 500.0,
    "QuinonePoolSize": 10.0,
    "pKaC": 5.8,
    "nC": 1.2,
    "pHStromaStart": 7.2,
    "pHLumenStart": 7.2,
    "StromaProtonsStart": 1e-10,
    "bufferCapacityStroma": 0.1,
    "bufferCapacityLumen": 0.03,
    "ATPConductivity": 6e-10,
    "kATPsActivate": 0.25,
    "kATPsInactivate": 0.003,
    "thresholdpmf": 0.001,
    "leakpmf": 0.8,
    "leakConductivity": 1e-7,
    "PCl": 1.8e-8,
    "PMg": 3.6e-8,
    "PK": 1.8e-8,
    "zCl": -1.0,
    "zMg": 2.0,
    "zK": 1.0,
    "StromaClStart": 0.01,
    "StromaMgStart": 0.01,
    "StromaKStart": 0.01,
    "Rconst": 8.314,
    "Fconst": 96485.0,
    "Tconst": 300.0,
    "LumenVolume": 6.7e-21,
    "StromaVolume": 5.36e-20,
    "lumenVolumePerArea": 8e-10,
    "MembraneCapacitance": 1e-6,
    "Na": 6.022e23,
    "VDErateVioToAnth": 0.04,
    "VDErateAnthToZea": 0.02,
    "ZErate": 0.0004,
    "TotalXanthophyll": 1.0,
    "VDEpKa": 6.0,
    "nVDE": 6.0,
    "PsbSpKa": 6.4,
    "nPsbS": 3.0,
    "zfrac": 0.8,
    "PsbSConvertRate": 0.1,
    "PSIcrossSection": 0.35,
    "kETPCP700": 6000.0,
    "kETP700Fdx": 10.0,
    "PCperPSI": 3.0,
    "ElectronsPerPC": 1.0,
    "fracIntactRC": 1.0,
    "voltsperlog": 0.02585065036015961,
    "LightIntensity": 0.0,
    "P680neut": 1.0,
    # quench-mode trigger flags (quenchmodel=1)
    "qtrigg1": 1.0,
    "qtrigg2": 0.0,
    "qtrigg3": 1.0,

    #Additional pars
    "ATPConductivityReverse": 1e-10,
    "ATPperPSI": 600.0,
    "CytRegulateYesNO": 1.0,
    "F_PsbS": 0.6,
    "NADPperPSI": 15.0,
    "PsbSperPSII": 1.0,
    "damageyesno": 0.0,
    "electronsPerNADPH": 2.0,
    "kEETP700": 14000000000.0,
    "kETFdxMV": 1000.0,
    "kETFdxPQ": 0.005,
    "kETFdxThrdx": 1000.0,
    "kETNADPHPQ": 100.0,
    "kETThrdxOx": 100.0,
    "kP680PherecombinationClosedRC": 500.0,
    "kP680PherecombinationOpenRC": 200000000.0,
    "kPheQArecombination": 500.0,
    "kQuenchDamage": 0.0,
    "repairyesno": 0.0,
    "tF": 1.5e-09,
    "tHop": 1.7e-11,
    "tauCS": 5.5e-12,
    "tauqE": 1e-11,

    "ATPperProton":3/12,
}

VARS = {
    "ATP": 2.0,
    "ActiveATPs": 0.05,
    
    "Antheraxanthin": 1e-14,
    
    "Fdxox": 1.0,
    "Fdxr": 1e-14,
    
    "LumenCl": 0.01,
    "LumenK": 0.01,
    "LumenMg": 0.01,
    "LumenProtons": 1e-14,
    
    "P680ex": 1e-14,
    "P680plus": 1e-14,
    
    "P700ox": 1e-14,
    "P700r": 1.0,
    
    "PCr": 0.2,

    "PQ": 8.999,
    "PQH2": 0.001,
    
    "PSIIChlEx": 1e-14,
    
    "PheAnion": 1e-14,
    
    "PsbSQ": 1e-14,
    
    "QAox": 1,
    
    "QBneut": 1,
    "QBred1": 1e-7,
    "QBred2": 1e-7,

    "Thrdx": 1e-14,
    
    "TotalLEF": 1e-14,
    
    "Zeaxanthin": 1e-14,
}

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def get_zaks2012() -> Model:
    m = Model()
    m.add_variables(VARS)
    m.add_parameters(PARAMS)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    # pH
    m.add_derived("pH_stroma", get_pH,
                  args=["pHStromaStart", "bufferCapacityStroma", "StromaProtonsStart"])
    m.add_derived("pH_lumen", get_pH,
                  args=["pHLumenStart", "bufferCapacityLumen", "LumenProtons"])

    # Electric field / pmf
    m.add_derived("total_charge_lumen", totalcharge,
                  args=["LumenProtons", "zCl", "LumenCl", "zK", "LumenK", "zMg", "LumenMg"])
    m.add_derived("total_charge_stroma", totalcharge,
                  args=["StromaProtonsStart", "zCl", "StromaClStart", "zK", "StromaKStart",
                        "zMg", "StromaMgStart"])
    m.add_derived("deltapsi", delta_psi,
                  args=["total_charge_lumen", "total_charge_stroma",
                        "Fconst", "lumenVolumePerArea", "MembraneCapacitance"])
    m.add_derived("deltapH", delta_pH, args=["pH_stroma", "pH_lumen"])
    m.add_derived("pmf", pmf, args=["deltapH", "deltapsi", "voltsperlog"])

    # Ion electrochemical potentials
    m.add_derived("deltamuCl", delta_mu,
                  args=["zCl", "LumenCl", "StromaClStart", "deltapsi", "voltsperlog"])
    m.add_derived("deltamuMg", delta_mu,
                  args=["zMg", "LumenMg", "StromaMgStart", "deltapsi", "voltsperlog"])
    m.add_derived("deltamuK", delta_mu,
                  args=["zK", "LumenK", "StromaKStart", "deltapsi", "voltsperlog"])

    # Electric-field slowdown factors
    m.add_derived("efield_slowdown_r", efield_slowdown,
                  args=["alphaRC", "deltapsi", "voltsperlog"])
    m.add_derived("efield_slowdown_q", efield_slowdown,
                  args=["alphaQ", "deltapsi", "voltsperlog"])

    # Xanthophyll & PsbS derived states
    m.add_derived("Violaxanthin", V_from_AZ,
                  args=["TotalXanthophyll", "Antheraxanthin", "Zeaxanthin"])
    m.add_derived("PsbS_unprot", complement, args=["PsbSQ"])

    # Enzyme activation states
    m.add_derived("active_vde", protonation_fraction,
                  args=["pH_lumen", "VDEpKa", "nVDE"])
    m.add_derived("active_psbs", protonation_fraction,
                  args=["pH_lumen", "PsbSpKa", "nPsbS"])
    m.add_derived("deact_psbs", complement, args=["active_psbs"])

    # Quenching
    m.add_derived("q_total", total_q,
                  args=["Antheraxanthin", "Zeaxanthin", "PsbSQ",
                        "zfrac", "PsbSDose", "qtrigg1", "qtrigg2", "qtrigg3"])

    # PSII moiety complements
    m.add_derived("QAred", moeity_frac, args=["QAox", "fracIntactRC"])
    m.add_derived("Pheneut", moeity_frac, args=["PheAnion", "fracIntactRC"])

    # PQ pool fractions
    m.add_derived("PQfrac", frac, args=["PQ", "QuinonePoolSize"])
    m.add_derived("PQH2frac", frac, args=["PQH2", "QuinonePoolSize"])
    m.add_derived("QBempty", qb_moiety, args=["QBneut", "QBred1", "QBred2"])

    # Cyt b6f activity factors
    m.add_derived("frac_active_cyt", fraction_active_cyt,
                  args=["pH_lumen", "pKaC", "nC"])
    m.add_derived("frac_pq_red", fraction_pq_red, args=["PQH2", "PQ"])
    m.add_derived("frac_active_pc", fraction_active_pc, args=["PCr", "PCperPSI"])
    m.add_derived("InactiveATPs", complement, args=["ActiveATPs"])

    #Output
    m.add_readout("light", same, args=["LightIntensity"])

    m.add_readout("kF_obs", kF_rate,
                  args=["kF", "PSIIChlEx"])

    m.add_readout("kqE_obs", kqE_rate,
                  args=["kQ", "PSIIChlEx", "q_total"])

    m.add_readout("kPC_obs", kPC_rate,
                  args=["kEETLHP680QAox", "kEETLHP680QAred",
                        "PSIIChlEx", "QAox", "QAred", "P680neut"])

    m.add_readout("kPCRCC_obs", kPCRCC_rate,
                  args=["kEETLHP680QAox", "kEETLHP680QAred",
                        "PSIIChlEx", "P680neut"])

    m.add_readout("kC_obs", kC_rate,
                  args=["kNRantenna", "PSIIChlEx", "kF_obs",
                        "kquenchP680plus", "P680plus"])

    m.add_readout("allrates", allrates_sum,
                  args=["kC_obs", "kPC_obs", "kqE_obs"])

    m.add_readout("allratesRCC", allratesRCC_sum,
                  args=["kC_obs", "kPCRCC_obs", "kqE_obs"])

    m.add_readout("fluorescenceyield", safe_ratio,
                  args=["kF_obs", "allrates"])

    m.add_readout("fluorescenceyieldRCC", safe_ratio,
                  args=["kF_obs", "allratesRCC"])

    m.add_readout("qE_model", safe_ratio,
                  args=["kqE_obs", "kC_obs"])

    m.add_readout("phi_npq", safe_ratio,
                  args=["kqE_obs", "allrates"])

    # ------------------------------------------------------------------
    # F2 – Xanthophyll cycle & PsbS
    # ------------------------------------------------------------------

    m.add_reaction(
        "V_to_A",
        fn=mass_action_1s_act,
        args=["VDErateVioToAnth", "active_vde", "Violaxanthin"],
        stoichiometry={"Antheraxanthin": 1},
    )
    m.add_reaction(
        "A_to_Z",
        fn=mass_action_1s_act,
        args=["VDErateAnthToZea", "active_vde", "Antheraxanthin"],
        stoichiometry={"Antheraxanthin": -1, "Zeaxanthin": 1},
    )
    m.add_reaction(
        "Z_to_A",
        fn=mass_action_1s,
        args=["ZErate", "Zeaxanthin"],
        stoichiometry={"Zeaxanthin": -1, "Antheraxanthin": 1},
    )
    m.add_reaction(
        "A_to_V",
        fn=mass_action_1s,
        args=["ZErate", "Antheraxanthin"],
        stoichiometry={"Antheraxanthin": -1},
    )
    m.add_reaction(
        "PsbS_prot",
        fn=mass_action_1s_act,
        args=["PsbSConvertRate", "active_psbs", "PsbS_unprot"],
        stoichiometry={"PsbSQ": 1},
    )
    m.add_reaction(
        "PsbS_deprot",
        fn=mass_action_1s_act,
        args=["PsbSConvertRate", "deact_psbs", "PsbSQ"],
        stoichiometry={"PsbSQ": -1},
    )

    # ------------------------------------------------------------------
    # F1 – PSII
    # ------------------------------------------------------------------

    # v1: light absorption → antenna singlet
    m.add_reaction(
        "v1",
        fn=mass_action_1s_act,
        args=["LightIntensity", "crosssection", "fracIntactRC"],
        stoichiometry={"PSIIChlEx": 1},
    )
    # v2: quenching by NPQ
    m.add_reaction(
        "v2",
        fn=mass_action_1s_act,
        args=["kQ", "q_total", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v3: fluorescence
    m.add_reaction(
        "v3",
        fn=mass_action_1s,
        args=["kF", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v4: quenching by P680+
    m.add_reaction(
        "v4",
        fn=mass_action_1s_act,
        args=["kquenchP680plus", "P680plus", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v5: non-radiative decay
    m.add_reaction(
        "v5",
        fn=mass_action_1s,
        args=["kNRantenna", "PSIIChlEx"],
        stoichiometry={"PSIIChlEx": -1},
    )
    # v7: energy transfer to open/closed RC → P680*
    m.add_reaction(
        "v7",
        fn=v7,
        args=["kEETLHP680QAox", "kEETLHP680QAred",
              "PSIIChlEx", "QAox", "QAred", "P680neut"],
        stoichiometry={"PSIIChlEx": -1, "P680ex": 1},
    )
    # v8: back-transfer P680* → antenna
    m.add_reaction(
        "v8",
        fn=v8,
        args=["kEETLHP680revQAox", "kEETLHP680revQAred",
              "P680ex", "QAox", "QAred"],
        stoichiometry={"PSIIChlEx": 1, "P680ex": -1},
    )
    # v9: non-radiative decay of P680*
    m.add_reaction(
        "v9",
        fn=mass_action_1s,
        args=["kNRP680", "P680ex"],
        stoichiometry={"P680ex": -1},
    )
    # v12: primary charge separation – open RC (QA oxidised)
    m.add_reaction(
        "v12",
        fn=v12_13,
        args=["kETP680PheOpenRC", "P680ex", "Pheneut", "QAox", "efield_slowdown_r"],
        stoichiometry={"P680ex": -1, "P680plus": 1, "PheAnion": 1},
    )
    # v13: primary charge separation – closed RC (QA reduced)
    m.add_reaction(
        "v13",
        fn=v12_13,
        args=["kETP680PheClosedRC", "P680ex", "Pheneut", "QAred", "efield_slowdown_r"],
        stoichiometry={"P680ex": -1, "P680plus": 1, "PheAnion": 1},
    )
    # v14: Phe⁻ → QA electron transfer (QA becomes reduced, tracked as –QAox)
    m.add_reaction(
        "v14",
        fn=v14,
        args=["kETPheToQA", "PheAnion", "QAox", "efield_slowdown_r"],
        stoichiometry={"PheAnion": -1, "QAox": -1},
    )
    # v15: water oxidation by OEC
    m.add_reaction(
        "v15",
        fn=mass_action_1s_act,
        args=["kETWaterOxidation", "P680plus", "efield_slowdown_q"],
        stoichiometry={
            "P680plus": -1,
            "LumenProtons": Derived(fn=psii_proton_stoi, args=["Na", "LumenVolume"]),
        },
    )
    # v16: P680+/Phe⁻ recombination
    m.add_reaction(
        "v16",
        fn=v16,
        args=["kP680Pherecombination", "P680plus", "PheAnion", "efield_slowdown_r"],
        stoichiometry={"P680plus": -1, "PheAnion": -1},
    )
    # v17: P680+/QA⁻ recombination (Phe neutral)
    m.add_reaction(
        "v17",
        fn=v17,
        args=["kP680QArecombination", "P680plus", "QAred", "Pheneut", "efield_slowdown_r"],
        stoichiometry={"P680plus": -1, "QAox": 1},
    )
    # v18: P680+/QA⁻ recombination (Phe anion)
    m.add_reaction(
        "v18",
        fn=v18,
        args=["kP680QArecombinationClosedRC", "P680plus", "QAred", "PheAnion",
              "efield_slowdown_r"],
        stoichiometry={"P680plus": -1, "QAox": 1},
    )

    # ------------------------------------------------------------------
    # F3 – PQ pool / QB site
    # ------------------------------------------------------------------

    # r_q2: QA⁻ → QB (first reduction)
    m.add_reaction(
        "r_q2",
        fn=r_q2,
        args=["kETQAtoQB1", "QAred", "QBneut", "efield_slowdown_q"],
        stoichiometry={"QAox": 1, "QBneut": -1, "QBred1": 1},
    )
    # r_q3: QA⁻ → QB⁻ (second reduction → PQH₂ at QB)
    m.add_reaction(
        "r_q3",
        fn= r_q3,
        args=["kETQAtoQB2", "QAred", "QBred1", "efield_slowdown_q"],
        stoichiometry={"QAox": 1, "QBred1": -1, "QBred2": 1},
    )
    # r_q4: reverse QB⁻ → QA (first)
    m.add_reaction(
        "r_q4",
        fn=mass_action_1s_act,
        args=["kETQB1toQA", "QAox", "QBred1"],
        stoichiometry={"QAox": -1, "QBred1": -1, "QBneut": 1},
    )
    # r_q5: reverse QB²⁻ → QA (second)
    m.add_reaction(
        "r_q5",
        fn=mass_action_1s_act,
        args=["kETQB2toQA", "QAox", "QBred2"],
        stoichiometry={"QAox": -1, "QBred1": 1, "QBred2": -1},
    )
    # r_q6: PQ docking at QB site
    m.add_reaction(
        "r_q6",
        fn=mass_action_1s_act,
        args=["PQdockingrate", "PQfrac", "QBempty"],
        stoichiometry={"PQ": -1, "QBneut": 1},
    )
    # r_q7: PQH₂ undocking from QB site
    m.add_reaction(
        "r_q7",
        fn= mass_action_1s,
        args=["PQH2undock", "QBred2"],
        stoichiometry={"QBred2": -1, "PQH2": 1},
    )
    # r_q8: PQH₂ re-docking (reverse, ÷10)
    m.add_reaction(
        "r_q8",
        fn= r_q8,
        args=["PQH2undock", "PQH2frac"],
        stoichiometry={"PQH2": -1, "QBred2": 1},
    )

    # ------------------------------------------------------------------
    # F4 – Cytochrome b6f
    # ------------------------------------------------------------------
    # r_cyt_b6f is pre-computed as a derived quantity above.
    # Stoichiometry: consumes PQH₂, produces PQ, reduces PC, pumps 4H⁺ into lumen.

    m.add_reaction(
        "r_cyt_b6f", 
        fn = r_cyt,
        args=["QReoxidationRate", "frac_active_cyt", "frac_active_pc", "frac_pq_red"],
        stoichiometry={
            "PQH2": -1,
            "PQ": 1,
            "PCr": Derived(fn=cyt_electron_stoi, args=["ElectronsPerPC"]),
            "LumenProtons": Derived(fn=cyt_proton_stoi, args=["Na", "LumenVolume"]),
        },
    ) # CHECK

    # ------------------------------------------------------------------
    # F5 – PSI
    # ------------------------------------------------------------------

    # r_psi_1: PC reduction of P700+
    m.add_reaction(
        "psi_1",
        fn= psi_1,
        args=["kETPCP700", "PCr", "P700ox"],
        stoichiometry={"PCr": -1, "P700ox": -1, "P700r": 1},
    )
    # r_psi_2: P700* reduces Fdx (light-driven)
    m.add_reaction(
        "psi_2",
        fn= psi_2,
        args=["LightIntensity", "PSIcrossSection", "kETP700Fdx", "P700r", "Fdxox"],
        stoichiometry={"P700r": -1, "P700ox": 1, "Fdxr": 1, "Fdxox": -1, "TotalLEF": 1},
    )

    # ------------------------------------------------------------------
    # F7 – ATP synthase + leak
    # ------------------------------------------------------------------

    m.add_reaction(
        "atp_synthesis",
        fn=proton_flux_atp,
        args=["ATPConductivity", "pmf", "thresholdpmf", "ActiveATPs"],
        stoichiometry={
            "ATP": Derived( fn=atp_stoi, args=["Na", "LumenVolume", "lumenVolumePerArea", "ATPperProton"]),
            "LumenProtons": Derived(fn=atpsyn_proton_stoi, args=["lumenVolumePerArea"]),
        },
    )

    m.add_reaction(
        "vleak",
        fn=proton_flux_leak,
        args=["leakConductivity", "pmf", "leakpmf"],
        stoichiometry={
            "LumenProtons": Derived(fn=atpsyn_proton_stoi, args=["lumenVolumePerArea"]),
        },
    )

    m.add_reaction(
        "atps_activate",
        fn=mass_action_1s_act,
        args=["kATPsActivate", "Fdxr", "InactiveATPs"],
        stoichiometry={"ActiveATPs": 1},
    )
    m.add_reaction(
        "atps_inactivate",
        fn=mass_action_1s,
        args=["kATPsInactivate", "ActiveATPs"],
        stoichiometry={"ActiveATPs": -1},
    )

    # ------------------------------------------------------------------
    # F8 – Lumen ion fluxes
    # ------------------------------------------------------------------

    m.add_reaction(
        "flux_Mg",
        fn=lumen_ion_flux,
        args=["LumenMg", "StromaMgStart", "deltamuMg",
              "PMg", "voltsperlog", "lumenVolumePerArea"],
        stoichiometry={"LumenMg": 1},
    )
    m.add_reaction(
        "flux_Cl",
        fn=lumen_ion_flux,
        args=["LumenCl", "StromaClStart", "deltamuCl",
              "PCl", "voltsperlog", "lumenVolumePerArea"],
        stoichiometry={"LumenCl": 1},
    )
    # NOTE: public MATLAB uses PCl (not PK) for the K flux — preserved here.
    m.add_reaction(
        "flux_K",
        fn=lumen_ion_flux,
        args=["LumenK", "StromaKStart", "deltamuK",
              "PCl", "voltsperlog", "lumenVolumePerArea"],
        stoichiometry={"LumenK": 1},
    )

    # ------------------------------------------------------------------
    # F9 – Methyl Viologen
    # ------------------------------------------------------------------

    m.add_reaction(
        "mv_1",
        fn=mass_action_1s,
        args=["kETFdxMV", "Fdxr"],
        stoichiometry={"Fdxr": -1},
    )

    m.add_reaction(
        "mv_2",
        fn=mass_action_1s,
        args=["kETFdxThrdx", "Fdxr"],
        stoichiometry={"Fdxox": 1, "Thrdx": 1},
    )

    m.add_reaction(
        "mv_3",
        fn=mass_action_1s,
        args=["kETThrdxOx", "Thrdx"],
        stoichiometry={"Thrdx": -1},
    )

    return m

    