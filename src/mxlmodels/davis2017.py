r"""Davis 2017 model of pmf-induced photosystem II photodamage.

|             |                                                                                         |
| ----------- | --------------------------------------------------------------------------------------- |
| doi         | 10.1098/rstb.2016.0381                                                                     |
| main author | Geoffry A. Davis                                                                        |
| paper title | Hacking the thylakoid proton motive force
for improved photosynthesis: modulating
ion flux rates that control proton motive
force partitioning into Dpsi and DpH |
| published   | 31 May 2017                                                                           |
| journal     |  Philosophical Transactions of the Royal Society B |
| organism    | higher plants (thylakoid)                                                               |
| Ported by   | Quang Huy Nguyen ( @PhotosyntheticBatman )                                              |

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
from mxlpy import Derived, Model, Variable


def neg_div(x: float, y: float) -> float:
    return -x / y


def mul(x: float, y: float) -> float:
    return x * y


def value(x: float) -> float:
    return x


def moiety_1(concentration: float, total: float) -> float:
    return total - concentration


def twice(x: float) -> float:
    return x * 2


def div(x: float, y: float) -> float:
    return x / y


def neg(x: float) -> float:
    return -x


def neg_point_one_val(x: float):
    return -0.1 * x


def neg_point_two_val(x: float):
    return -0.2 * x


def neg_proportional(x, y):
    return -x * y


def neg_thrice(x: float):
    return x * -3


def neg_2_div(x: float, y: float):
    return -2 * x / y


def ATP_stoi(x: float, y: float, z: float):
    return x * y / z


def calc_PsbS_Protonation(pH_lumen: float, pKa_PsbS: float):
    return 1 - (1 - (1 / (10 ** (pH_lumen - pKa_PsbS) + 1)))  # checked


def calc_NPQ(Z, PsbS_H, NPQ_max):
    return NPQ_max * PsbS_H * Z  # checked


def calc_phi2(NPQ, QA):
    return 1 / (1 + (1 + NPQ) / (4.88 * QA))  # checked


def DeltaGATP_to_volt(DeltaGATP):
    return 0.06 * DeltaGATP / 5.7


def ATPsynthase_driving_force(pmf, DeltaGATP, n):
    return pmf - (DeltaGATP / n)


def calc_h(pH):
    return 10 ** (-1 * pH)  # checked


def calc_pmf(Dpsi, pH_lumen, pH_stroma):
    return Dpsi + 0.06 * (pH_stroma - pH_lumen)  # checked


def k_b6f(pH_lumen, pKa_reg, c_b6f, Vmax_b6f):
    pHmod = 1 - (1 / (10 ** (pH_lumen - pKa_reg) + 1))
    b6f_deprot = pHmod * c_b6f
    return b6f_deprot


def _delta_pH_inVolts(delta_pH: float):
    return 0.06 * delta_pH


def _vPSII_recomb(Dpsi, QAm, pH_lumen, k_recomb):  # checked
    delta_delta_g_recomb = Dpsi + 0.06 * (7.0 - pH_lumen)
    return k_recomb * QAm * 10 ** (delta_delta_g_recomb / 0.06)


def _vPSII_ChSep(PPFD, PhiPSII):  # checked
    return PPFD * PhiPSII


def _v_PSII(QAm, PQ, k_QA):
    return QAm * PQ * k_QA  # checked


def _v_PQ(PQH2, QA, k_QA, Keq_QA):
    return PQH2 * QA * k_QA / Keq_QA  # checked


def _v_b6f(
    pH_lumen,
    PQH2,
    PQ,
    PC_ox,
    PC_red,
    pKa_reg,
    c_b6f,
    Em_PC_pH7,
    Em_PQH2_pH7,
    pmf,
    Vmax_b6f,
):  # checked
    pHmod = 1 - (1 / (10 ** (pH_lumen - pKa_reg) + 1))
    b6f_deprot = pHmod * c_b6f

    Em_PC = Em_PC_pH7
    Em_PQH2 = Em_PQH2_pH7 - 0.06 * (pH_lumen - 7.0)

    Keq_b6f = 10 ** ((Em_PC - Em_PQH2 - pmf) / 0.06)
    k_b6f = b6f_deprot * Vmax_b6f

    k_b6f_reverse = k_b6f / Keq_b6f
    f_PQH2 = PQH2 / (PQH2 + PQ)
    f_PQ = 1 - f_PQH2
    return f_PQH2 * PC_ox * k_b6f - f_PQ * PC_red * k_b6f_reverse


def _PSI_ChSep(Fd_ox, P700_red, PSI_antenna_size, PPFD):  # checked
    return P700_red * PPFD * PSI_antenna_size * Fd_ox


def _v_PSI_PCoxid(PC_red, P700_ox, k_PCtoP700):  # checked
    return PC_red * k_PCtoP700 * P700_ox


def _v_FNR(Fd_red, NADP_pool, k_FdtoNADP):  # checked
    return k_FdtoNADP * NADP_pool * Fd_red


def vATPsynthase(Vmax_ATPsynth, ATP_synthase_driving_force):  # checked
    return Vmax_ATPsynth * ATP_synthase_driving_force


def _v_KEA3(K_lu, H_lumen, H_st, K_stroma, k_KEA3):  # checked
    return k_KEA3 * (H_lumen * K_stroma - H_st * K_lu)


def _v_VKC(K_lu, Dpsi, K_stroma, P_K):  # checked
    K_deltaG = -0.06 * np.log10(K_stroma / K_lu) + Dpsi
    return P_K * K_deltaG * (K_lu + K_stroma) / 2


def _v_Epox(Z, k_EZ):  # checked
    return Z * k_EZ


def _v_VDE(V, pH_lumen, nh_VDE, pKa_VDE, Vmax_VDE):  # checked
    pHmod = 1 - (1 - (1 / (10 ** (nh_VDE * (pH_lumen - pKa_VDE)) + 1)))
    return V * Vmax_VDE * pHmod


def vCBB(NADPH, kCBB):  # checked
    return kCBB * NADPH


def get_davis2017() -> Model:
    """Build the Davis 2017 pmf / PSII photodamage model.

    `````
    ````
    ```
    Assembles the photosynthetic electron transport chain, thylakoid proton
    motive force, and photoprotection machinery into a single mxlpy Model.

    Variables (14): QA_red (reduced Q_A), PQH_2 (plastoquinol), pH_lumen,
    Dpsi (membrane potential, V), K_lu (lumen K+), PC_ox (oxidised
    plastocyanin), Zx (zeaxanthin), singO2 (cumulative singlet oxygen),
    P700_ox, Fd_red (reduced ferredoxin), NADPH_st, LEF (cumulative linear
    electron flow), ATP_made (cumulative ATP).

    Key reactions: vPSII_ChSep / vPSII_recomb (PSII charge separation and the
    Dpsi-driven recombination producing singO2), v_PSII / v_PQ (Q_A <-> PQ
    pool), v_b6f (cytochrome b6f), PSI_ChSep / v_PSI_PCoxid (PSI), v_FNR
    (ferredoxin -> NADPH), vATPsynthase, v_CBB (Calvin-Benson-Bassham sink),
    v_KEA3 / v_VKC (K+/H+ counter-ion fluxes partitioning the pmf), and
    v_Epox / v_Deepox (xanthophyll cycle).

    pmf is split into Dpsi and delta_pH via the volt_per_charge and b_H
    buffering parameters; light input is the PPFD parameter (default 0).

    Returns
    -------
        Model: the configured mxlpy model.
    ```
    ````
    `````
    """
    m = Model()

    m.add_parameters(
        {
            "PPFD": 0,  # checked
            "k_recomb": 0.33,  # checked
            "phi_triplet": 0.45,  # checked
            "phi_1O2": 1,  # checked
            "sigma0_II": 1,  # checked
            "c_b6f": 1,  # checked
            "pKa_reg": 6.5,  # checked
            "Em_PC_pH7": 0.37,  # checked
            "Em_PQH2_pH7": 0.11,  # checked
            "Vmax_b6f": 500,  # checked
            "pKa_PsbS": 6.4,  # checked
            "NPQ_max": 5,  # checked
            "pH_stroma": 7.8,  # checked
            "PSI_antenna_size": 1,  # checked
            "k_QA": 1000,  # checked
            "Keq_QA": 200,  # checked
            "k_PCtoP700": 500,  # checked
            "k_FdtoNADP": 1000,  # checked or 5000
            "K_st": 0.04,  # checked
            "k_KEA3": 0,  # checked
            "P_K": 6000,  # checked
            "lumen_protons_per_turnover": 1.4e-05,  # checked
            "n": 14 / 3,  # checked
            "DeltaGATP": 42,  # checked
            "Vmax_ATPsynth": 1000,  # checked
            "b_H": 0.03,  # checked
            "volt_per_charge": 0.033,  # checked
            "k_EZ": 0.03,  # checked
            "nh_VDE": 4,  # checked
            "pKa_VDE": 5.8,  # checked
            "Vmax_VDE": 1,  # checked
            "QA_total": 1,  # checked
            "PQ_tot": 6,  # checked
            "P700_total": 1,  # checked
            "PC_tot": 2,  # checked
            "Fd_tot": 1,  # checked
            "NADP_tot": 1,  # checked
            "Xanthophyll_tot": 1,  # checked
            "k_CBB": 3000,  # checked
        }
    )

    m.add_variables(
        {
            "QA_red": Variable(0),  # checked
            "PQH_2": Variable(0),  # checked
            "pH_lumen": Variable(7),  # checked
            "Dpsi": Variable(0.1),  # checked
            "K_lu": Variable(0.04),  # checked
            "PC_ox": Variable(0),  # checked
            "Zx": Variable(0),  # checked
            # "PsbS": Variable(0), # checked
            "singO2": Variable(0),  # checked
            "P700_ox": Variable(0),  # checked
            "Fd_red": Variable(0),  # checked
            "NADPH_st": Variable(0),  # checked
            "LEF": Variable(0),  # checked
            "ATP_made": Variable(0),  # checked
        }
    )

    m.add_derived(
        name="QA",
        fn=moiety_1,
        args=["QA_red", "QA_total"],
    )

    m.add_derived(
        name="P700_red",
        fn=moiety_1,
        args=["P700_ox", "P700_total"],
    )

    m.add_derived(
        name="PQ",
        fn=moiety_1,
        args=["PQH_2", "PQ_tot"],
    )

    m.add_derived(
        name="PC_red",
        fn=moiety_1,
        args=["PC_ox", "PC_tot"],
    )

    m.add_derived(
        name="Fd_ox",
        fn=moiety_1,
        args=["Fd_red", "Fd_tot"],
    )

    m.add_derived(
        name="NADP_st",
        fn=moiety_1,
        args=["NADPH_st", "NADP_tot"],
    )

    m.add_derived(
        name="Vx",
        fn=moiety_1,
        args=["Zx", "Xanthophyll_tot"],
    )

    m.add_derived(
        name="DeltaGATP_V",
        fn=DeltaGATP_to_volt,
        args=["DeltaGATP"],
    )

    m.add_derived(
        name="PsbSP",
        fn=calc_PsbS_Protonation,
        args=["pH_lumen", "pKa_PsbS"],
    )

    m.add_derived(
        name="NPQ",
        fn=calc_NPQ,
        args=["Zx", "PsbSP", "NPQ_max"],
    )

    m.add_derived(
        name="PhiPSII",
        fn=calc_phi2,
        args=["NPQ", "QA"],
    )

    m.add_derived(
        name="H_lumen",
        fn=calc_h,
        args=["pH_lumen"],
    )

    m.add_derived(
        name="H_stroma",
        fn=calc_h,
        args=["pH_stroma"],
    )

    m.add_derived(
        name="pmf",
        fn=calc_pmf,
        args=["Dpsi", "pH_lumen", "pH_stroma"],
    )

    m.add_derived(
        name="delta_pH",
        fn=moiety_1,
        args=["pH_lumen", "pH_stroma"],
    )

    m.add_derived(
        name="delta_pH_inVolts",
        fn=_delta_pH_inVolts,
        args=["delta_pH"],
    )

    m.add_derived(
        name="ATP_synthase_driving_force",
        fn=ATPsynthase_driving_force,
        args=["pmf", "DeltaGATP_V", "n"],
    )

    m.add_derived(
        name="k_b6f",
        fn=k_b6f,
        args=["pH_lumen", "pKa_reg", "c_b6f", "Vmax_b6f"],
    )

    m.add_reaction(
        name="vPSII_recomb",
        fn=_vPSII_recomb,
        args=["Dpsi", "QA_red", "pH_lumen", "k_recomb"],
        stoichiometry={
            "singO2": Derived(
                fn=mul, args=["phi_triplet", "phi_1O2"], unit=None
            ),  # checked
            "QA_red": -1,  # checked
            "pH_lumen": Derived(
                fn=div, args=["lumen_protons_per_turnover", "b_H"], unit=None
            ),  # checked
            "Dpsi": Derived(fn=neg, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="vPSII_ChSep",  # checked
        fn=_vPSII_ChSep,
        args=["PPFD", "PhiPSII"],
        stoichiometry={
            "QA_red": 1,
            "pH_lumen": Derived(
                fn=neg_div, args=["lumen_protons_per_turnover", "b_H"], unit=None
            ),
            "Dpsi": Derived(fn=value, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_PSII",
        fn=_v_PSII,
        args=["QA_red", "PQ", "k_QA"],
        stoichiometry={
            "QA_red": -1,
            "PQH_2": 0.5,
        },
    )
    m.add_reaction(
        name="v_PQ",
        fn=_v_PQ,
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
                fn=neg_2_div, args=["lumen_protons_per_turnover", "b_H"], unit=None
            ),
            "Dpsi": Derived(fn=value, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="PSI_ChSep",
        fn=_PSI_ChSep,
        args=["Fd_ox", "P700_red", "PSI_antenna_size", "PPFD"],
        stoichiometry={
            "P700_ox": 1,
            "Fd_red": 1,
            "Dpsi": Derived(fn=value, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_PSI_PCoxid",
        fn=_v_PSI_PCoxid,
        args=["PC_red", "P700_ox", "k_PCtoP700"],
        stoichiometry={
            "P700_ox": -1,
            "PC_ox": 1,
        },
    )

    m.add_reaction(
        name="v_FNR",
        fn=_v_FNR,
        args=["Fd_red", "NADP_st", "k_FdtoNADP"],
        stoichiometry={"Fd_red": -1, "NADPH_st": 0.5, "LEF": 1},
    )

    m.add_reaction(
        name="vATPsynthase",
        fn=vATPsynthase,
        args=["Vmax_ATPsynth", "ATP_synthase_driving_force"],
        stoichiometry={
            "ATP_made": 1,
            "pH_lumen": Derived(
                fn=ATP_stoi, args=["lumen_protons_per_turnover", "n", "b_H"], unit=None
            ),
            "Dpsi": Derived(
                fn=neg_proportional, args=["n", "volt_per_charge"], unit=None
            ),
        },
    )

    m.add_reaction(
        name="v_CBB",
        fn=vCBB,
        args=["NADPH_st", "k_CBB"],
        stoichiometry={"NADPH_st": -1},
    )

    m.add_reaction(
        name="v_KEA3",
        fn=_v_KEA3,
        args=["K_lu", "H_lumen", "H_stroma", "K_st", "k_KEA3"],
        stoichiometry={
            "K_lu": Derived(fn=value, args=["lumen_protons_per_turnover"], unit=None),
            "pH_lumen": Derived(
                fn=div, args=["lumen_protons_per_turnover", "b_H"], unit=None
            ),
            "Dpsi": Derived(fn=neg, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_VKC",
        fn=_v_VKC,
        args=["K_lu", "Dpsi", "K_st", "P_K"],
        stoichiometry={
            "K_lu": Derived(fn=neg, args=["lumen_protons_per_turnover"], unit=None),
            "Dpsi": Derived(fn=neg, args=["volt_per_charge"], unit=None),
        },
    )
    m.add_reaction(
        name="v_Epox",
        fn=_v_Epox,
        args=["Zx", "k_EZ"],
        stoichiometry={
            "Zx": -1,
        },
    )
    m.add_reaction(
        name="v_Deepox",
        fn=_v_VDE,
        args=["Vx", "pH_lumen", "nh_VDE", "pKa_VDE", "Vmax_VDE"],
        stoichiometry={
            "Zx": 1,
        },
    )

    return m
