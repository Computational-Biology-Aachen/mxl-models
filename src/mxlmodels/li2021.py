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
from mxlpy import Derived, Model, Parameter, Variable
from mxlpy.units import dimensionless, liter, mol, ppfd, second, volt

# Pools are relative to one reaction centre (dimensionless), ions are molar
_molar = mol / liter
_per_s = 1 / second
_per_molar_s = 1 / (_molar * second)
_per_molar2_s = 1 / (_molar**2 * second)
_per_volt_molar_s = 1 / (volt * _molar * second)
_per_volt_molar2_s = 1 / (volt * _molar**2 * second)


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
    k_mehler: float,
) -> float:
    return k_mehler * fd_red / (fd_red + fd_ox)


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


def _k_delta_g(
    k_lu: float,
    dpsi: float,
    k_stroma: float,
) -> float:
    return -0.06 * np.log10(k_stroma / k_lu) + dpsi


def _v_vkc(
    k_lu: float,
    k_delta_g: float,
    k_stroma: float,
    p_k: float,
) -> float:
    return p_k * k_delta_g * (k_lu + k_stroma) / 2


def _relative_cl_flux_vccn1(
    driving_force_cl: float,
) -> float:
    return (
        332 * (driving_force_cl**3)
        + 30.8 * (driving_force_cl**2)
        + 3.6 * driving_force_cl
    )


def _v_vccn1(
    cl_lu: float,
    cl_st: float,
    relative_cl_flux: float,
    k_vccn1: float,
) -> float:
    return k_vccn1 * relative_cl_flux * (cl_st + cl_lu) / 2


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


def _atp_synthase_activity(
    t: float,
) -> float:
    x = t / 165
    return 0.2 + 0.8 * (x**4 / (x**4 + 1))


def _v_pmf_protons_activity(
    actvt: float,
    pmf: float,
    hpr: float,
    vmax_at_psynth: float,
    light_per_l: float,
) -> float:
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
            "PPFD": Parameter(50, unit=ppfd),
            "k_recomb": Parameter(0.33, unit=_per_s),
            "phi_triplet": Parameter(0.45, unit=dimensionless),
            "phi_1O2": Parameter(1, unit=dimensionless),
            "sigma0_II": Parameter(0.5, unit=dimensionless),
            "c_b6f": Parameter(0.433, unit=dimensionless),
            "pKa_reg": Parameter(6.2, unit=dimensionless),
            "Em_PC_pH7": Parameter(0.37, unit=volt),
            "Em_PQH2_pH7": Parameter(0.11, unit=volt),
            "Vmax_b6f": Parameter(300, unit=_per_s),
            "pKa_PsbS": Parameter(6.2, unit=dimensionless),
            "NPQ_max": Parameter(3, unit=dimensionless),
            "pH_st": Parameter(7.8, unit=dimensionless, display_name="pH"),
            "Em_Fd": Parameter(-0.42, unit=volt),
            "k_NDH1": Parameter(1000, unit=_per_s),
            "Vmax_PGR": Parameter(0, unit=_per_s),
            "sigma0_I": Parameter(0.5, unit=dimensionless),
            "k_QA": Parameter(1000, unit=_per_s),
            "Keq_QA": Parameter(200, unit=dimensionless),
            "k_PCtoP700": Parameter(5000, unit=_per_s),
            "k_FdtoNADP": Parameter(1000, unit=_per_s),
            "K_st": Parameter(0.1, unit=_molar),
            "k_KEA3": Parameter(2500000, unit=_per_molar2_s),
            "P_K": Parameter(150, unit=_per_volt_molar_s),
            "ipt_lu": Parameter(0.000587, unit=_molar),
            "k_VCCN1": Parameter(12, unit=_per_molar_s),
            "k_ClCe": Parameter(800000, unit=_per_volt_molar2_s),
            "HPR": Parameter(4.666666666666667, unit=dimensionless),
            "Vmax_ATPsynth": Parameter(200, unit=_per_s),
            "b_H": Parameter(0.014, unit=_molar, display_name="bH"),
            "vpc": Parameter(0.047, unit=volt),
            "k_EZ": Parameter(0.004, unit=_per_s),
            "nh_VDE": Parameter(4, unit=dimensionless),
            "pKa_VDE": Parameter(5.65, unit=dimensionless),
            "Vmax_VDE": Parameter(0.08, unit=_per_s),
            "k_leak": Parameter(30000000.0, unit=_per_volt_molar_s),
            "QA_total": Parameter(1, unit=dimensionless),
            "PQ_tot": Parameter(7, unit=dimensionless),
            "P700_total": Parameter(0.667, unit=dimensionless),
            "PC_tot": Parameter(2, unit=dimensionless),
            "Fd_tot": Parameter(1, unit=dimensionless),
            "NADP_tot": Parameter(5, unit=dimensionless),
            "Carotenoids_tot": Parameter(1, unit=dimensionless),
            # previously a constant inside the Mehler rate
            "k_Mehler": Parameter(4 * 0.000265, unit=_per_s),
        }
    )

    m.add_variables(
        {
            "QA_red": Variable(0, unit=dimensionless),
            "PQH_2": Variable(0, unit=dimensionless, display_name="PQH2"),
            "pH_lumen": Variable(7.8, unit=dimensionless),
            "Dpsi": Variable(0, unit=volt),
            "K_lu": Variable(0.1, unit=_molar),
            "PC_ox": Variable(0, unit=dimensionless),
            "Y2": Variable(0, unit=dimensionless),
            "Zx": Variable(0, unit=dimensionless, display_name="zx"),
            "singO2": Variable(0, unit=dimensionless),
            "Fd_red": Variable(0, unit=dimensionless),
            "NADPH_st": Variable(1.5, unit=dimensionless, display_name="NADPH"),
            "Cl_lu": Variable(0.04, unit=_molar),
            "Cl_st": Variable(0.04, unit=_molar),
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
        name="NADP_st", fn=_moiety_1, args=["NADPH_st", "NADP_tot"], display_name="NADP"
    )

    m.add_derived(
        name="Vx", fn=_moiety_1, args=["Zx", "Carotenoids_tot"], display_name="vx"
    )

    # These functions hide unit-carrying constants (0.06 V per pH unit, the PPFD to
    # excitation rate conversion, the maximal CBB rate), so their units are set explicitly
    m.add_derived(
        name="light_per_L",
        fn=_light_per_l,
        args=["PPFD"],
        unit=_per_s,
    )

    m.add_derived(
        name="driving_force_Cl",
        fn=_driving_force_cl,
        args=["Cl_st", "Cl_lu", "Dpsi"],
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
        display_name="protons_lumen",
    )

    m.add_derived(
        name="H_st", fn=_calc_h, args=["pH_st"], unit=_molar, display_name="H_stroma"
    )

    m.add_derived(
        name="pmf",
        fn=_calc_pmf,
        args=["Dpsi", "pH_lumen", "pH_st"],
        unit=volt,
    )

    m.add_derived(
        name="kCBB", fn=_calc_k_cbb, args=["PPFD"], unit=_per_s, display_name="k_CBB"
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
        unit=volt,
    )

    m.add_derived(
        name="K_deltaG",
        fn=_k_delta_g,
        args=["K_lu", "Dpsi", "K_st"],
        unit=volt,
    )
    m.add_derived(
        name="relative_Cl_flux_VCCN1",
        fn=_relative_cl_flux_vccn1,
        args=["driving_force_Cl"],
        unit=dimensionless,
    )
    m.add_derived(
        name="ATPsynth_activity",
        fn=_atp_synthase_activity,
        args=["time"],
        unit=dimensionless,
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
        display_name="PSII",
    )
    m.add_reaction(
        name="v_PQ",
        fn=_v_pq,
        args=["PQH_2", "QA", "k_QA", "Keq_QA"],
        stoichiometry={
            "QA_red": 1,
            "PQH_2": -0.5,
        },
        display_name="PQ reduction",
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
        display_name="cyt b6f",
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
        display_name="FNR",
    )
    m.add_reaction(
        name="v_Mehler",
        fn=_v_mehler,
        args=["Fd_red", "Fd_ox", "k_Mehler"],
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
        args=["K_lu", "K_deltaG", "K_st", "P_K"],
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
        args=["Cl_lu", "Cl_st", "relative_Cl_flux_VCCN1", "k_VCCN1"],
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
        display_name="proton leak",
    )
    m.add_reaction(
        name="v_pmf_protons_activity",
        fn=_v_pmf_protons_activity,
        args=["ATPsynth_activity", "pmf", "HPR", "Vmax_ATPsynth", "light_per_L"],
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
