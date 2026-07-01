r"""Li 2021 model.

|             |                                                                                                          |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| doi         | 10.1038/s41477-021-00947-5                                                                               |
| main author | Meng Li                                                                                                  |
| paper title | Impact of ion fluxes across thylakoid membranes on photosynthetic electron transport and photoprotection |
| published   | June 2021                                                                                                |
| journal     | Nature Plants                                                                                            |
| organism    | Arabidopsis thaliana                                                                                     |
| Ported by   | Marvin van Aalst ( @marvinvanaalst )                                                                     |

The Li2021 model is a kinetic model of photosynthesis that focuses on ion
fluxes across the thylakoid membrane and their effects on the proton motive
force (PMF). Built upon the Davis2017 model, it provides a more detailed
representation of photosynthetic reactions directly linked to PMF generation,
including water splitting at photosystem II and plastoquinone oxidation at the
cytochrome b6f complex. Other photosynthetic processes are represented with
minimal complexity. The model introduces two potassium (K+) transport channels
and two chloride (Cl-) transport channels in the thylakoid membrane to
investigate their roles in PMF regulation.

Model validation was performed by comparing simulation results with
experimental data from multiple studies. The authors demonstrated that the
model reproduces both wild-type behavior and the phenotypes of several knockout
mutants, including VCCN1, CLCE, KEA3, and combinations thereof. Following
validation, the model was used to investigate how these ion channels influence
PMF formation and photosynthetic efficiency. Simulation protocols included
light oscillation experiments, enzyme abundance scans, and other analyses that
demonstrate the model's capabilities. The primary objective of the model was to
address the long-standing question of how ion fluxes across the thylakoid
membrane contribute to photosynthetic regulation.

The model description in the publication is limited, although the authors
provide access to a public GitHub repository containing the implementation. The
code is written in Python and includes extensive comments. However, the
repository contains multiple model components and simulation protocols, making
it difficult to determine precisely which parts correspond to the published
model. The version summarized here was reduced to the components most relevant
to the publication, but this interpretation may not exactly match the authors'
intent. Discrepancies between the code, publication, and supplementary
materials further complicate reconstruction of the complete model and its
parameterization. Despite these challenges, the model illustrates the value and
versatility of photosynthesis modeling and was therefore included in
GreenSloth.
"""

import numpy as np
from mxlpy import Derived, Model, Variable


def _neg_div(
    x: float,
    y: float,
) -> float:
    return -x / y


def _mul(
    x: float,
    y: float,
) -> float:
    return x * y


def _value(
    x: float,
) -> float:
    return x


def _moiety_1(
    concentration: float,
    total: float,
) -> float:
    return total - concentration


def _twice(
    x: float,
) -> float:
    return x * 2


def _div(
    x: float,
    y: float,
) -> float:
    return x / y


def _neg(
    x: float,
) -> float:
    return -x


def _calc_k_cbb(
    par: float,
) -> float:
    return 60 * (par / (par + 250))


def _light_per_l(
    par: float,
) -> float:
    return 0.84 * par / 0.7


def _driving_force_cl(
    cl_st: float,
    cl_lu: float,
    dpsi: float,
) -> float:
    return 0.06 * np.log10(cl_st / cl_lu) + dpsi


def _calc_psb_s_protonation(
    p_h_lumen: float,
    p_ka_psb_s: float,
) -> float:
    return 1 / (10 ** (3 * (p_h_lumen - p_ka_psb_s)) + 1)


def _calc_npq(
    z: float,
    psb_s_h: float,
    npq_max: float,
) -> float:
    return 0.4 * npq_max * psb_s_h * z + 0.5 * npq_max * psb_s_h + 0.1 * npq_max * z


def _calc_phi2(
    npq: float,
    qa: float,
) -> float:
    return 1 / (1 + (1 + npq) / (4.88 * qa))


def _calc_h(
    p_h: float,
) -> float:
    return 10 ** (-1 * p_h)


def _calc_pmf(
    dpsi: float,
    p_h_lumen: float,
    p_h_st: float,
) -> float:
    return dpsi + 0.06 * (p_h_st - p_h_lumen)


def _delta_p_h_in_volts(
    delta_p_h: float,
) -> float:
    return 0.06 * delta_p_h


def _ql_act(
    qa: float,
) -> float:
    return qa**3 / (qa**3 + 0.15**3)


def _p_h_act(
    p_h_lumen: float,
) -> float:
    return 1 / (10 ** (1 * (p_h_lumen - 6.0)) + 1)


def _v_psii_recomb(
    dpsi: float,
    q_am: float,
    p_h_lumen: float,
    k_recomb: float,
) -> float:  # correct
    delta_delta_g_recomb = dpsi + 0.06 * (7.0 - p_h_lumen)
    return k_recomb * q_am * 10 ** (delta_delta_g_recomb / 0.06)


def _v_psii_ch_sep(
    antenna_size: float,
    light_per_l: float,
    phi_psii: float,
) -> float:  # correct
    return antenna_size * light_per_l * phi_psii


def _v_psii(
    q_am: float,
    pq: float,
    k_qa: float,
) -> float:
    return q_am * pq * k_qa


def _v_pq(
    pqh2: float,
    qa: float,
    k_qa: float,
    keq_qa: float,
) -> float:
    return pqh2 * qa * k_qa / keq_qa


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
) -> float:  # correct
    pHmod = 1 - (1 / (10 ** (p_h_lumen - p_ka_reg) + 1))
    b6f_deprot = pHmod * c_b6f

    Em_PC = em_pc_p_h7
    Em_PQH2 = em_pqh2_p_h7 - 0.06 * (p_h_lumen - 7.0)

    Keq_b6f = 10 ** ((Em_PC - Em_PQH2 - pmf) / 0.06)
    k_b6f = b6f_deprot * vmax_b6f

    k_b6f_reverse = k_b6f / Keq_b6f
    f_PQH2 = pqh2 / (
        pqh2 + pq
    )  # want to keep the rates in terms of fraction of PQHs, not total number
    f_PQ = 1 - f_PQH2
    return f_PQH2 * pc_ox * k_b6f - f_PQ * pc_red * k_b6f_reverse


def _neg_2_div(
    x: float,
    y: float,
) -> float:
    return -2 * x / y


def _v_ndh(
    fd_red: float,
    pq: float,
    fd_ox: float,
    pqh2: float,
    p_h_st: float,
    em_pqh2_p_h7: float,
    em_fd: float,
    k_ndh1: float,
    pmf: float,
) -> float:
    Em_PQH2 = em_pqh2_p_h7 - 0.06 * (p_h_st - 7.0)
    deltaEm = Em_PQH2 - em_fd
    Keq_NDH = 10 ** ((deltaEm - pmf * 2) / 0.06)
    k_NDH_reverse = k_ndh1 / Keq_NDH
    return k_ndh1 * fd_red * pq - k_NDH_reverse * fd_ox * pqh2


def _v_pgr(
    fd_red: float,
    pq: float,
    pqh2: float,
    vmax_pgr: float,
) -> float:
    return vmax_pgr * (fd_red**4 / (fd_red**4 + 0.1**4)) * pq / (pq + pqh2)


def _psi_ch_sep(
    fd_ox: float,
    y0: float,
    sigma0_i: float,
    light_per_l: float,
) -> float:
    return y0 * light_per_l * sigma0_i * fd_ox


def _v_psi_p_coxid(
    pc_red: float,
    y2: float,
    k_p_cto_p700: float,
) -> float:
    return pc_red * k_p_cto_p700 * y2


def _v_fnr(
    fd_red: float,
    nadp_pool: float,
    k_fdto_nadp: float,
) -> float:
    return k_fdto_nadp * nadp_pool * fd_red


def _v_mehler(
    fd_red: float,
    fd_ox: float,
) -> float:
    return 4 * 0.000265 * fd_red / (fd_red + fd_ox)


def _v_cbb(
    nadph_pool: float,
    nadp_pool: float,
    t: float,
    k_cbb: float,
) -> float:
    return (
        k_cbb
        * (1.0 - np.exp(-t / 600))
        * (np.log(nadph_pool / nadp_pool) - np.log(1.25))
        / (np.log(3.5 / 1.25))
    )


def _v_kea3(
    q_l_act: float,
    p_h_act: float,
    k_lu: float,
    h_lumen: float,
    h_st: float,
    k_stroma: float,
    k_kea3: float,
) -> float:
    f_KEA_act = q_l_act * p_h_act
    return k_kea3 * (h_lumen * k_stroma - h_st * k_lu) * f_KEA_act


def _v_vkc(
    k_lu: float,
    dpsi: float,
    k_stroma: float,
    p_k: float,
) -> float:
    K_deltaG = -0.06 * np.log10(k_stroma / k_lu) + dpsi
    return p_k * K_deltaG * (k_lu + k_stroma) / 2


def _v_vccn1(
    cl_lu: float,
    cl_st: float,
    driving_force_cl: float,
    k_vccn1: float,
) -> float:
    relative_Cl_flux = (
        332 * (driving_force_cl**3)
        + 30.8 * (driving_force_cl**2)
        + 3.6 * driving_force_cl
    )
    return k_vccn1 * relative_Cl_flux * (cl_st + cl_lu) / 2


def _neg_point_one_val(
    x: float,
) -> float:
    return -0.1 * x


def _v_cl_ce(
    cl_lu: float,
    cl_st: float,
    h_lumen: float,
    h_st: float,
    driving_force_cl: float,
    pmf: float,
    k_cl_ce: float,
) -> float:
    return (
        k_cl_ce * (driving_force_cl * 2 + pmf) * (cl_st + cl_lu) * (h_lumen + h_st) / 4
    )


def _neg_point_two_val(
    x: float,
) -> float:
    return -0.2 * x


def _neg_thrice(
    x: float,
) -> float:
    return x * -3


def _v_leak(
    h_lumen: float,
    pmf: float,
    k_leak: float,
) -> float:
    return pmf * k_leak * h_lumen


def _v_pmf_protons_activity(
    t: float,
    pmf: float,
    hpr: float,
    vmax_at_psynth: float,
    light_per_l: float,
) -> float:
    x = t / 165
    actvt = 0.2 + 0.8 * (x**4 / (x**4 + 1))
    v_proton_active = 1 - (
        1 / (10 ** ((pmf - 0.132) * 1.5 / 0.06) + 1)
    )  # reduced ATP synthase
    v_proton_inert = 1 - (
        1 / (10 ** ((pmf - 0.204) * 1.5 / 0.06) + 1)
    )  # oxidized ATP synthase

    v_active = actvt * v_proton_active * hpr * vmax_at_psynth
    v_inert = (1 - actvt) * v_proton_inert * hpr * vmax_at_psynth

    v_proton_ATP = v_active + v_inert

    if light_per_l > 0:
        return v_proton_ATP
    return 0


def _v_epox(
    z: float,
    k_ez: float,
) -> float:
    return z * k_ez


def _v_vde(
    v: float,
    p_h_lumen: float,
    nh_vde: float,
    p_ka_vde: float,
    vmax_vde: float,
) -> float:
    pHmod = 1 / (10 ** (nh_vde * (p_h_lumen - p_ka_vde)) + 1)
    return v * vmax_vde * pHmod


def get_li_2021() -> Model:
    r"""Li 2021 model.

    The Li2021 model is a kinetic model of photosynthesis that focuses on ion
    fluxes across the thylakoid membrane and their effects on the proton motive
    force (PMF). Built upon the Davis2017 model, it provides a more detailed
    representation of photosynthetic reactions directly linked to PMF generation,
    including water splitting at photosystem II and plastoquinone oxidation at the
    cytochrome b6f complex. Other photosynthetic processes are represented with
    minimal complexity. The model introduces two potassium (K+) transport channels
    and two chloride (Cl-) transport channels in the thylakoid membrane to
    investigate their roles in PMF regulation.

    Model validation was performed by comparing simulation results with
    experimental data from multiple studies. The authors demonstrated that the
    model reproduces both wild-type behavior and the phenotypes of several knockout
    mutants, including VCCN1, CLCE, KEA3, and combinations thereof. Following
    validation, the model was used to investigate how these ion channels influence
    PMF formation and photosynthetic efficiency. Simulation protocols included
    light oscillation experiments, enzyme abundance scans, and other analyses that
    demonstrate the model's capabilities. The primary objective of the model was to
    address the long-standing question of how ion fluxes across the thylakoid
    membrane contribute to photosynthetic regulation.

    The model description in the publication is limited, although the authors
    provide access to a public GitHub repository containing the implementation. The
    code is written in Python and includes extensive comments. However, the
    repository contains multiple model components and simulation protocols, making
    it difficult to determine precisely which parts correspond to the published
    model. The version summarized here was reduced to the components most relevant
    to the publication, but this interpretation may not exactly match the authors'
    intent. Discrepancies between the code, publication, and supplementary
    materials further complicate reconstruction of the complete model and its
    parameterization. Despite these challenges, the model illustrates the value and
    versatility of photosynthesis modeling and was therefore included in
    GreenSloth.
    """
    m = Model()

    m.add_parameters(
        {
            "PPFD": 50,
            "k_recomb": 0.33,
            "phi_triplet": 0.45,
            "phi_1O2": 1,
            "sigma0_II": 0.5,
            "c_b6f": 0.433,
            "pKa_reg": 6.2,
            "Em_PC_pH7": 0.37,
            "Em_PQH2_pH7": 0.11,
            "Vmax_b6f": 300,
            "pKa_PsbS": 6.2,
            "NPQ_max": 3,
            "pH_st": 7.8,
            "Em_Fd": -0.42,
            "k_NDH1": 1000,
            "Vmax_PGR": 0,
            "sigma0_I": 0.5,
            "k_QA": 1000,
            "Keq_QA": 200,
            "k_PCtoP700": 5000,
            "k_FdtoNADP": 1000,
            "K_st": 0.1,
            "k_KEA3": 2500000,
            "P_K": 150,
            "ipt_lu": 0.000587,
            "k_VCCN1": 12,
            "k_ClCe": 800000,
            "HPR": 4.666666666666667,
            "Vmax_ATPsynth": 200,
            "b_H": 0.014,
            "vpc": 0.047,
            "k_EZ": 0.004,
            "nh_VDE": 4,
            "pKa_VDE": 5.65,
            "Vmax_VDE": 0.08,
            "k_leak": 30000000.0,
            "QA_total": 1,
            "PQ_tot": 7,
            "P700_total": 0.667,
            "PC_tot": 2,
            "Fd_tot": 1,
            "NADP_tot": 5,
            "Carotenoids_tot": 1,
        }
    )

    m.add_variables(
        {
            "QA_red": Variable(0),
            "PQH_2": Variable(0),
            "pH_lumen": Variable(7.8),
            "Dpsi": Variable(0),
            "K_lu": Variable(0.1),
            "PC_ox": Variable(0),
            "Y2": Variable(0),
            "Zx": Variable(0),
            "singO2": Variable(0),
            "Fd_red": Variable(0),
            "NADPH_st": Variable(1.5),
            "Cl_lu": Variable(0.04),
            "Cl_st": Variable(0.04),
        }
    )

    m.add_derived(
        name="QA",
        fn=_moiety_1,
        args=["QA_red", "QA_total"],
    )

    m.add_derived(
        name="Y0",
        fn=_moiety_1,
        args=["Y2", "P700_total"],
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
        args=["Zx", "Carotenoids_tot"],
    )

    m.add_derived(
        name="light_per_L",
        fn=_light_per_l,
        args=["PPFD"],
    )

    m.add_derived(
        name="driving_force_Cl",
        fn=_driving_force_cl,
        args=["Cl_st", "Cl_lu", "Dpsi"],
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
    )

    m.add_derived(
        name="H_st",
        fn=_calc_h,
        args=["pH_st"],
    )

    m.add_derived(
        name="pmf",
        fn=_calc_pmf,
        args=["Dpsi", "pH_lumen", "pH_st"],
    )

    m.add_derived(
        name="kCBB",
        fn=_calc_k_cbb,
        args=["PPFD"],
    )

    m.add_derived(
        name="delta_pH",
        fn=_moiety_1,
        args=["pH_lumen", "pH_st"],
    )

    m.add_derived(
        name="delta_pH_inVolts",
        fn=_delta_p_h_in_volts,
        args=["delta_pH"],
    )

    m.add_derived(
        name="qL_act",
        fn=_ql_act,
        args=["QA"],
    )

    m.add_derived(
        name="pH_act",
        fn=_p_h_act,
        args=["pH_lumen"],
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
            ),
            "QA_red": -1,
            "pH_lumen": Derived(
                fn=_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="vPSII_ChSep",
        fn=_v_psii_ch_sep,
        args=["sigma0_II", "light_per_L", "PhiPSII"],
        stoichiometry={
            "QA_red": 1,
            "pH_lumen": Derived(
                fn=_neg_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_value,
                args=["vpc"],
                unit=None,
            ),
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
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_value,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_NDH",
        fn=_v_ndh,
        args=[
            "Fd_red",
            "PQ",
            "Fd_ox",
            "PQH_2",
            "pH_st",
            "Em_PQH2_pH7",
            "Em_Fd",
            "k_NDH1",
            "pmf",
        ],
        stoichiometry={
            "PQH_2": 0.5,
            "Fd_red": -1,
            "pH_lumen": Derived(
                fn=_neg_2_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_twice,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_PGR",
        fn=_v_pgr,
        args=["Fd_red", "PQ", "PQH_2", "Vmax_PGR"],
        stoichiometry={
            "PQH_2": 0.5,
            "Fd_red": -1,
        },
    )
    m.add_reaction(
        name="PSI_ChSep",
        fn=_psi_ch_sep,
        args=["Fd_ox", "Y0", "sigma0_I", "light_per_L"],
        stoichiometry={
            "Y2": 1,
            "Fd_red": 1,
            "Dpsi": Derived(
                fn=_value,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_PSI_PCoxid",
        fn=_v_psi_p_coxid,
        args=["PC_red", "Y2", "k_PCtoP700"],
        stoichiometry={
            "Y2": -1,
            "PC_ox": 1,
        },
    )
    m.add_reaction(
        name="v_FNR",
        fn=_v_fnr,
        args=["Fd_red", "NADP_st", "k_FdtoNADP"],
        stoichiometry={
            "Fd_red": -1,
            "NADPH_st": 0.5,
        },
    )
    m.add_reaction(
        name="v_Mehler",
        fn=_v_mehler,
        args=["Fd_red", "Fd_ox"],
        stoichiometry={
            "Fd_red": -1,
        },
    )
    m.add_reaction(
        name="v_CBB",
        fn=_v_cbb,
        args=["NADPH_st", "NADP_st", "time", "kCBB"],
        stoichiometry={
            "NADPH_st": -1,
        },
    )
    m.add_reaction(
        name="v_KEA3",
        fn=_v_kea3,
        args=["qL_act", "pH_act", "K_lu", "H_lumen", "H_st", "K_st", "k_KEA3"],
        stoichiometry={
            "K_lu": Derived(
                fn=_value,
                args=["ipt_lu"],
                unit=None,
            ),
            "pH_lumen": Derived(
                fn=_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_VKC",
        fn=_v_vkc,
        args=["K_lu", "Dpsi", "K_st", "P_K"],
        stoichiometry={
            "K_lu": Derived(
                fn=_neg,
                args=["ipt_lu"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_VCCN1",
        fn=_v_vccn1,
        args=["Cl_lu", "Cl_st", "driving_force_Cl", "k_VCCN1"],
        stoichiometry={
            "Cl_lu": Derived(
                fn=_value,
                args=["ipt_lu"],
                unit=None,
            ),
            "Cl_st": Derived(
                fn=_neg_point_one_val,
                args=["ipt_lu"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_ClCe",
        fn=_v_cl_ce,
        args=["Cl_lu", "Cl_st", "H_lumen", "H_st", "driving_force_Cl", "pmf", "k_ClCe"],
        stoichiometry={
            "Cl_lu": Derived(
                fn=_twice,
                args=["ipt_lu"],
                unit=None,
            ),
            "Cl_st": Derived(
                fn=_neg_point_two_val,
                args=["ipt_lu"],
                unit=None,
            ),
            "pH_lumen": Derived(
                fn=_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg_thrice,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_Leak",
        fn=_v_leak,
        args=["H_lumen", "pmf", "k_leak"],
        stoichiometry={
            "pH_lumen": Derived(
                fn=_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg,
                args=["vpc"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_pmf_protons_activity",
        fn=_v_pmf_protons_activity,
        args=["time", "pmf", "HPR", "Vmax_ATPsynth", "light_per_L"],
        stoichiometry={
            "pH_lumen": Derived(
                fn=_div,
                args=["ipt_lu", "b_H"],
                unit=None,
            ),
            "Dpsi": Derived(
                fn=_neg,
                args=["vpc"],
                unit=None,
            ),
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
