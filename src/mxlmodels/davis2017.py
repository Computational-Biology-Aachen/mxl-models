r"""Davis 2017 model of pmf-induced photosystem II photodamage.

|             |                                                                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| doi         | 10.1098/rstb.2016.0381                                                                                                                                           |
| main author | Geoffry A. Davis                                                                                                                                                 |
| paper title | Hacking the thylakoid proton motive force for improved photosynthesis: modulating ion flux rates that control proton motive force partitioning into Dpsi and DpH |
| published   | 31 May 2017                                                                                                                                                      |
| journal     | Philosophical Transactions of the Royal Society B                                                                                                                |
| organism    | higher plants (thylakoid)                                                                                                                                        |
| Ported by   | Quang Huy Nguyen ( @PhotosyntheticBatman )                                                                                                                       |

A mechanistic model of the photosynthetic electron transport chain and
thylakoid proton motive force (pmf) that resolves the pmf into its two
thermodynamically distinct components: the transmembrane electric field (Dpsi)
and the lumen pH gradient (delta_pH). The central result is that the Dpsi
component, rather than lumen acidification, drives elevated PSII charge
recombination, which produces singlet oxygen (singO2) and subsequent PSII
photodamage.

Electron transport runs PSII -> PQ pool -> cytochrome b6f -> plastocyanin ->
PSI -> ferredoxin -> FNR -> NADPH -> CBB. Proton accumulation in the lumen and
charge separation across the membrane build the pmf, which drives ATP synthase.
Counter-ion fluxes (KEA3 K+/H+ antiport, VKC K+ channel) partition the pmf
between Dpsi and delta_pH. NPQ is modelled via PsbS protonation and the
xanthophyll cycle (violaxanthin \<-> zeaxanthin).
"""

import numpy as np
from mxlpy import Derived, Model, Parameter, Variable
from mxlpy.units import dimensionless, liter, mol, per_second, ppfd, volt
from sympy.physics import units as u

# Pools are relative to one reaction centre (dimensionless), K+ and protons are molar
_molar = mol / liter
_kj_mol = u.kilo * u.joule / u.mol


def _neg_div(x: float, y: float) -> float:
    return -x / y


def _mul(x: float, y: float) -> float:
    return x * y


def _value(x: float) -> float:
    return x


def _moiety_1(concentration: float, total: float) -> float:
    return total - concentration


def _div(x: float, y: float) -> float:
    return x / y


def _neg(x: float) -> float:
    return -x


def _neg_proportional(x: float, y: float) -> float:
    return -x * y


def _neg_thrice(x: float) -> float:
    return x * -3


def _neg_2_div(x: float, y: float) -> float:
    return -2 * x / y


def _atp_stoi(x: float, y: float, z: float) -> float:
    return x * y / z


def _calc_psb_s_protonation(p_h_lumen: float, p_ka_psb_s: float) -> float:
    return 1 - (1 - (1 / (10 ** (p_h_lumen - p_ka_psb_s) + 1)))  # checked


def _calc_npq(z: float, psb_s_h: float, npq_max: float) -> float:
    return npq_max * psb_s_h * z  # checked


def _calc_phi2(npq: float, qa: float) -> float:
    return 1 / (1 + (1 + npq) / (4.88 * qa))  # checked


def _delta_gatp_to_volt(delta_gatp: float) -> float:
    return 0.06 * delta_gatp / 5.7


def _at_psynthase_driving_force(pmf: float, delta_gatp: float, n: float) -> float:
    return pmf - (delta_gatp / n)


def _calc_h(p_h: float) -> float:
    return 10 ** (-1 * p_h)  # checked


def _calc_pmf(dpsi: float, p_h_lumen: float, p_h_stroma: float) -> float:
    return dpsi + 0.06 * (p_h_stroma - p_h_lumen)  # checked


def _k_b6f(p_h_lumen: float, p_ka_reg: float, c_b6f: float) -> float:
    pHmod = 1 - (1 / (10 ** (p_h_lumen - p_ka_reg) + 1))
    return pHmod * c_b6f


def _delta_p_h_in_volts(delta_p_h: float) -> float:
    return 0.06 * delta_p_h


def _v_psii_recomb(
    dpsi: float, q_am: float, p_h_lumen: float, k_recomb: float
) -> float:  # checked
    delta_delta_g_recomb = dpsi + 0.06 * (7.0 - p_h_lumen)
    return k_recomb * q_am * 10 ** (delta_delta_g_recomb / 0.06)


def _v_psii_ch_sep(ppfd: float, phi_psii: float) -> float:  # checked
    return ppfd * phi_psii


def _v_psii(q_am: float, pq: float, k_qa: float) -> float:
    return q_am * pq * k_qa  # checked


def _v_pq(pqh2: float, qa: float, k_qa: float, keq_qa: float) -> float:
    return pqh2 * qa * k_qa / keq_qa  # checked


def _v_b6f(
    p_h_lumen: float,
    pqh2: float,
    pq: float,
    pc_ox: float,
    pc_red: float,
    p_ka_reg: float,
    c_b6f: float,
    em_pc_p_h7: float,
    em_pqh2_p_h7: float,
    pmf: float,
    vmax_b6f: float,
) -> float:  # checked
    pHmod = 1 - (1 / (10 ** (p_h_lumen - p_ka_reg) + 1))
    b6f_deprot = pHmod * c_b6f

    Em_PC = em_pc_p_h7
    Em_PQH2 = em_pqh2_p_h7 - 0.06 * (p_h_lumen - 7.0)

    Keq_b6f = 10 ** ((Em_PC - Em_PQH2 - pmf) / 0.06)
    _k_b6f = b6f_deprot * vmax_b6f

    k_b6f_reverse = _k_b6f / Keq_b6f
    f_PQH2 = pqh2 / (pqh2 + pq)
    f_PQ = 1 - f_PQH2
    return f_PQH2 * pc_ox * _k_b6f - f_PQ * pc_red * k_b6f_reverse


def _psi_ch_sep(
    fd_ox: float, p700_red: float, psi_antenna_size: float, ppfd: float
) -> float:  # checked
    return p700_red * ppfd * psi_antenna_size * fd_ox


def _v_psi_p_coxid(
    pc_red: float, p700_ox: float, k_p_cto_p700: float
) -> float:  # checked
    return pc_red * k_p_cto_p700 * p700_ox


def _v_fnr(fd_red: float, nadp_pool: float, k_fdto_nadp: float) -> float:  # checked
    return k_fdto_nadp * nadp_pool * fd_red


def _v_at_psynthase(
    vmax_at_psynth: float, atp_synthase_driving_force: float
) -> float:  # checked
    return vmax_at_psynth * atp_synthase_driving_force


def _v_kea3(
    k_lu: float, h_lumen: float, h_st: float, k_stroma: float, k_kea3: float
) -> float:  # checked
    return k_kea3 * (h_lumen * k_stroma - h_st * k_lu)


def _k_delta_g(k_lu: float, dpsi: float, k_stroma: float) -> float:
    return -0.06 * np.log10(k_stroma / k_lu) + dpsi


def _v_vkc(
    k_lu: float, k_delta_g: float, k_stroma: float, p_k: float
) -> float:  # checked
    return p_k * k_delta_g * (k_lu + k_stroma) / 2


def _v_epox(z: float, k_ez: float) -> float:  # checked
    return z * k_ez


def _v_vde(
    v: float, p_h_lumen: float, nh_vde: float, p_ka_vde: float, vmax_vde: float
) -> float:  # checked
    pHmod = 1 - (1 - (1 / (10 ** (nh_vde * (p_h_lumen - p_ka_vde)) + 1)))
    return v * vmax_vde * pHmod


def _v_cbb(nadph: float, k_cbb: float) -> float:  # checked
    return k_cbb * nadph


def get_davis2017() -> Model:
    r"""Build the Davis 2017 pmf / PSII photodamage model.

    Assembles the photosynthetic electron transport chain, thylakoid proton motive
    force, and photoprotection machinery into a single mxlpy Model.

    Variables (14): QA_red (reduced Q_A), PQH_2 (plastoquinol), pH_lumen, Dpsi
    (membrane potential, V), K_lu (lumen K+), PC_ox (oxidised plastocyanin), Zx
    (zeaxanthin), singO2 (cumulative singlet oxygen), P700_ox, Fd_red (reduced
    ferredoxin), NADPH_st, LEF (cumulative linear electron flow), ATP_made
    (cumulative ATP).

    Key reactions: vPSII_ChSep / vPSII_recomb (PSII charge separation and the
    Dpsi-driven recombination producing singO2), v_PSII / v_PQ (Q_A \<-> PQ pool),
    v_b6f (cytochrome b6f), PSI_ChSep / v_PSI_PCoxid (PSI), v_FNR (ferredoxin ->
    NADPH), vATPsynthase, v_CBB (Calvin-Benson-Bassham sink), v_KEA3 / v_VKC (K+/H+
    counter-ion fluxes partitioning the pmf), and v_Epox / v_Deepox (xanthophyll
    cycle).

    pmf is split into Dpsi and delta_pH via the volt_per_charge and b_H buffering
    parameters; light input is the PPFD parameter (default 0).

    Returns the fully configured mxlpy Model instance.
    """
    m = Model()

    m.add_parameters(
        {
            "PPFD": Parameter(0, unit=ppfd),  # not in SI
            "k_recomb": Parameter(0.33, unit=per_second),  # SI
            "phi_triplet": Parameter(0.45, unit=dimensionless),  # SI: fraction
            "phi_1O2": Parameter(1, unit=dimensionless),  # SI: fraction
            "sigma0_II": Parameter(
                1, unit=dimensionless
            ),  # SI: ratio (PSII_antenna_size)
            "c_b6f": Parameter(
                1, unit=dimensionless
            ),  # SI: b6f complex per standard PSII content
            "pKa_reg": Parameter(6.5, unit=dimensionless),
            "Em_PC_pH7": Parameter(0.37, unit=volt),
            "Em_PQH2_pH7": Parameter(0.11, unit=volt),
            "Vmax_b6f": Parameter(500, unit=per_second),
            "pKa_PsbS": Parameter(6.4, unit=dimensionless),
            "NPQ_max": Parameter(5, unit=dimensionless),
            "pH_stroma": Parameter(7.8, unit=dimensionless),  # SI: pH units
            "PSI_antenna_size": Parameter(1, unit=dimensionless),  # SI: ratio
            "k_QA": Parameter(
                1000, unit=per_second
            ),  # SI: s^-1 (second-order QA- x PQ, per standard PSII)
            "Keq_QA": Parameter(200, unit=dimensionless),  # not in SI
            "k_PCtoP700": Parameter(500, unit=per_second),  # SI: PC^-1 s^-1
            "k_FdtoNADP": Parameter(1000, unit=per_second),  # not in SI
            "K_st": Parameter(0.04, unit=_molar),  # SI: M
            "k_KEA3": Parameter(
                0, unit=_molar**-1 * per_second
            ),  # v_KEA is M/s, so 1/(M s)
            "P_K": Parameter(
                6000, unit=per_second / volt
            ),  # SI: ions s^-1 V^-1 PSII^-1; dK/dt is M/s, so 1/(V s)
            "lumen_protons_per_turnover": Parameter(
                1.4e-05, unit=_molar
            ),  # SI text: M change per H+ per PSII
            "n": Parameter(14 / 3, unit=dimensionless),  # SI: H+ per ATP
            "DeltaGATP": Parameter(42, unit=_kj_mol),  # SI: kJ/mol
            "Vmax_ATPsynth": Parameter(
                1000, unit=per_second / volt
            ),  # SI :ATP s^-1 complex^-1
            "b_H": Parameter(0.03, unit=_molar),  # SI: M/pH unit
            "volt_per_charge": Parameter(
                0.033, unit=volt
            ),  # SI: V charge^-1 standard PSII content^-1
            "k_EZ": Parameter(0.03, unit=per_second),  # (table value 0.01)
            "nh_VDE": Parameter(4, unit=dimensionless),
            "pKa_VDE": Parameter(5.8, unit=dimensionless),  # SI: pH units
            "Vmax_VDE": Parameter(1, unit=per_second),  # SI: s^-1 PSII^-1
            "QA_total": Parameter(
                1, unit=dimensionless
            ),  # SI: relative to standard PSII
            "PQ_tot": Parameter(6, unit=dimensionless),  # SI: relative to standard PSII
            "P700_total": Parameter(1, unit=dimensionless),  # SI: relative to PSI
            "PC_tot": Parameter(2, unit=dimensionless),  # SI: relative to standard PSII
            "Fd_tot": Parameter(1, unit=dimensionless),  # SI: relative to standard PSII
            "NADP_tot": Parameter(
                1, unit=dimensionless
            ),  # not in SI: same normalisation assumed
            "Xanthophyll_tot": Parameter(1, unit=dimensionless),  # SI: relative units
            "k_CBB": Parameter(3000, unit=per_second),  # not in SI
        }
    )

    m.add_variables(
        {
            "QA_red": Variable(0, unit=dimensionless),  # SI: relative to standard PSII
            "PQH_2": Variable(0, unit=dimensionless),  # SI: relative to standard PSII
            "pH_lumen": Variable(7, unit=dimensionless),
            "Dpsi": Variable(0.1, unit=volt),
            "K_lu": Variable(0.04, unit=_molar),
            "PC_ox": Variable(0, unit=dimensionless),  # SI: relative to standard PSII
            "Zx": Variable(0, unit=dimensionless),  # SI: relative units
            # "PsbS": Variable(0),
            "singO2": Variable(
                0, unit=dimensionless
            ),  # SI: 1O2 molecules per standard PSII
            "P700_ox": Variable(0, unit=dimensionless),  # SI: relative to PSI
            "Fd_red": Variable(0, unit=dimensionless),  # SI: relative to standard PSII
            "NADPH_st": Variable(
                0, unit=dimensionless
            ),  # not in SI; same normalisation as NADP_tot
            "LEF": Variable(0, unit=dimensionless),  # not in SI; cumulative electrons
            "ATP_made": Variable(0, unit=dimensionless),
        }
    )

    m.add_derived(
        name="QA",
        fn=_moiety_1,
        args=["QA_red", "QA_total"],
    )

    m.add_derived(
        name="P700_red",
        fn=_moiety_1,
        args=["P700_ox", "P700_total"],
    )

    m.add_derived(
        name="PQ",
        fn=_moiety_1,
        args=["PQH_2", "PQ_tot"],
    )

    m.add_derived(
        name="PC_red",
        fn=_moiety_1,
        args=["PC_ox", "PC_tot"],
    )

    m.add_derived(
        name="Fd_ox",
        fn=_moiety_1,
        args=["Fd_red", "Fd_tot"],
    )

    m.add_derived(
        name="NADP_st",
        fn=_moiety_1,
        args=["NADPH_st", "NADP_tot"],
    )

    m.add_derived(
        name="Vx",
        fn=_moiety_1,
        args=["Zx", "Xanthophyll_tot"],
    )

    # Several functions hide 0.06 V per pH unit (or similar) in a constant, so their
    # units are set explicitly
    m.add_derived(
        name="DeltaGATP_V",
        fn=_delta_gatp_to_volt,
        args=["DeltaGATP"],
        unit=volt,
    )

    m.add_derived(
        name="PsbSP",
        fn=_calc_psb_s_protonation,
        args=["pH_lumen", "pKa_PsbS"],
    )

    m.add_derived(
        name="NPQ",
        fn=_calc_npq,
        args=["Zx", "PsbSP", "NPQ_max"],
    )

    m.add_derived(
        name="PhiPSII",
        fn=_calc_phi2,
        args=["NPQ", "QA"],
    )

    m.add_derived(
        name="H_lumen",
        fn=_calc_h,
        args=["pH_lumen"],
        unit=_molar,
    )

    m.add_derived(
        name="H_stroma",
        fn=_calc_h,
        args=["pH_stroma"],
        unit=_molar,
    )

    m.add_derived(
        name="pmf",
        fn=_calc_pmf,
        args=["Dpsi", "pH_lumen", "pH_stroma"],
        unit=volt,
    )

    m.add_derived(
        name="delta_pH",
        fn=_moiety_1,
        args=["pH_lumen", "pH_stroma"],
    )

    m.add_derived(
        name="delta_pH_inVolts",
        fn=_delta_p_h_in_volts,
        args=["delta_pH"],
        unit=volt,
    )

    m.add_derived(
        name="ATP_synthase_driving_force",
        fn=_at_psynthase_driving_force,
        args=["pmf", "DeltaGATP_V", "n"],
    )

    m.add_derived(
        name="K_deltaG",
        fn=_k_delta_g,
        args=["K_lu", "Dpsi", "K_st"],
        unit=volt,
    )
    m.add_derived(
        name="k_b6f",
        fn=_k_b6f,
        args=["pH_lumen", "pKa_reg", "c_b6f"],
    )

    m.add_reaction(
        name="vPSII_recomb",
        fn=_v_psii_recomb,
        args=["Dpsi", "QA_red", "pH_lumen", "k_recomb"],
        stoichiometry={
            "singO2": Derived(
                fn=_mul,
                args=["phi_triplet", "phi_1O2"],
                unit=None,
            ),  # checked
            "QA_red": -1,  # checked
            "pH_lumen": Derived(
                fn=_div,
                args=["lumen_protons_per_turnover", "b_H"],
                unit=None,
            ),  # checked
            "Dpsi": Derived(fn=_neg, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="vPSII_ChSep",  # checked
        fn=_v_psii_ch_sep,
        args=["PPFD", "PhiPSII"],
        stoichiometry={
            "QA_red": 1,
            "pH_lumen": Derived(
                fn=_neg_div,
                args=["lumen_protons_per_turnover", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(fn=_value, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_PSII",
        fn=_v_psii,
        args=["QA_red", "PQ", "k_QA"],
        stoichiometry={
            "QA_red": -1,
            "PQH_2": 0.5,
        },
    )
    m.add_reaction(
        name="v_PQ",
        fn=_v_pq,
        args=["PQH_2", "QA", "k_QA", "Keq_QA"],
        stoichiometry={
            "QA_red": 1,
            "PQH_2": -0.5,
        },
    )
    m.add_reaction(
        name="v_b6f",
        fn=_v_b6f,
        args=[
            "pH_lumen",
            "PQH_2",
            "PQ",
            "PC_ox",
            "PC_red",
            "pKa_reg",
            "c_b6f",
            "Em_PC_pH7",
            "Em_PQH2_pH7",
            "pmf",
            "Vmax_b6f",
        ],
        stoichiometry={
            "PQH_2": -0.5,
            "PC_ox": -1,
            "pH_lumen": Derived(
                fn=_neg_2_div,
                args=["lumen_protons_per_turnover", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(fn=_value, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="PSI_ChSep",
        fn=_psi_ch_sep,
        args=["Fd_ox", "P700_red", "PSI_antenna_size", "PPFD"],
        stoichiometry={
            "P700_ox": 1,
            "Fd_red": 1,
            "Dpsi": Derived(fn=_value, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_PSI_PCoxid",
        fn=_v_psi_p_coxid,
        args=["PC_red", "P700_ox", "k_PCtoP700"],
        stoichiometry={
            "P700_ox": -1,
            "PC_ox": 1,
        },
    )

    m.add_reaction(
        name="v_FNR",
        fn=_v_fnr,
        args=["Fd_red", "NADP_st", "k_FdtoNADP"],
        stoichiometry={"Fd_red": -1, "NADPH_st": 0.5, "LEF": 1},
    )

    m.add_reaction(
        name="vATPsynthase",
        fn=_v_at_psynthase,
        args=["Vmax_ATPsynth", "ATP_synthase_driving_force"],
        stoichiometry={
            "ATP_made": 1,
            "pH_lumen": Derived(
                fn=_atp_stoi,
                args=["lumen_protons_per_turnover", "n", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg_proportional,
                args=["n", "volt_per_charge"],
                unit=None,
            ),
        },
    )

    m.add_reaction(
        name="v_CBB",
        fn=_v_cbb,
        args=["NADPH_st", "k_CBB"],
        stoichiometry={"NADPH_st": -1},
    )

    m.add_reaction(
        name="v_KEA3",
        fn=_v_kea3,
        args=["K_lu", "H_lumen", "H_stroma", "K_st", "k_KEA3"],
        stoichiometry={
            "K_lu": Derived(fn=_value, args=["lumen_protons_per_turnover"], unit=None),
            "pH_lumen": Derived(
                fn=_div,
                args=["lumen_protons_per_turnover", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(fn=_neg, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_VKC",
        fn=_v_vkc,
        args=["K_lu", "K_deltaG", "K_st", "P_K"],
        stoichiometry={
            "K_lu": Derived(fn=_neg, args=["lumen_protons_per_turnover"], unit=None),
            "Dpsi": Derived(fn=_neg, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_Epox",
        fn=_v_epox,
        args=["Zx", "k_EZ"],
        stoichiometry={
            "Zx": -1,
        },
    )
    m.add_reaction(
        name="v_Deepox",
        fn=_v_vde,
        args=["Vx", "pH_lumen", "nh_VDE", "pKa_VDE", "Vmax_VDE"],
        stoichiometry={
            "Zx": 1,
        },
    )

    return m
