r"""Bellasio 2019 model.

|             |                                                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| doi         | 10.1007/s11120-018-0601-1                                                                                              |
| main author | Chandra Bellasio                                                                                                       |
| paper title | A generalised dynamic model of leaf-level C3 photosynthesis combining light and dark reactions with stomatal behaviour |
| published   | January 2019                                                                                                           |
| journal     | Photosynthesis Research                                                                                                |
| organism    | C3 leaf                                                                                                                |
| Ported by   | Marvin van Aalst ( @marvinvanaalst )                                                                                   |

The [Bellasio2019](https://doi.org/10.1007/s11120-018-0601-1) model is a
generalized C3 leaf-photosynthesis model that includes simplified
representations of the light and dark reactions and a stomatal behavior
submodule. A lot of its implementation is based on past work by the same author
and is mainly inspired by the common Farquhar-von Caemmerer-Berry model. The
light reactions are modified from Yin et al. (2004) and include the potential
rates of ATP and NADPH production based on light intensity. This model has been
created with the simple user in mind, and the author has made an effort to show
its simplicity by giving access to a Microsoft Excel Workbook containing the
entire model. To showcase the model's capabilities, the author creates common
steady-state carbon assimilation curves against intercellular CO2 concentration
and light intensity, and compares them to experimental data from the
literature. As many models of photosynthesis rely on purely stead-state
assumptions, this model is also validated in dynamic conditions, showing for
example the response of the model to a fluctuation of ambient oxygen
concentration.

This model was created to stay as simple as possible, while still being able to
accurately represent the main features of C<sub>3</sub> photosynthesis. As
such, it can be used as a base for more complex models, or as a starting block
in larger models of plant physiology. While giving access to the entire model
in an Excel Workbook format is transparent and great, the execution of said
practice has been inefficient in this instance. The entire mathematical
description of the model is also given in the Appendix of the publication,
however there are missing or different equations between the publication and
the Excel Workbook, which can lead to confusion. On top of that, the simulation
protocols used for each figure are only given in small details, which leads to
further confusion when trying to reproduce the results and see which equations
are correct or not.
"""

import numpy as np
from mxlpy import Derived, InitialAssignment, Model, Parameter, Slider, Variable
from mxlpy.units import (
    dimensionless,
    liter,
    mmol,
    mol,
    mumol,
    pascal,
    ppfd,
    second,
    sqm,
)

# Metabolites in mM, leaf-level fluxes in mmol m^-2 s^-1 (converted to mM/s via the
# mesophyll volume V_m), partial pressures in ubar = umol/mol (Table S2)
_mm = mmol / liter
_flux = mmol / sqm / second
_mol_m2_s = mol / sqm / second
_umol_mol = mumol / mol
_mpa = pascal  # pressures of the stomatal model are given in MPa


def _value(
    x: float,
) -> float:
    return x


def _one_div(
    x: float,
) -> float:
    return 1.0 / x


def _moiety_1(
    concentration: float,
    total: float,
) -> float:
    return total - concentration


def _div(
    x: float,
    y: float,
) -> float:
    return x / y


def _co2_initial(
    ca: float,
    kh_co2: float,
) -> float:
    return 0.3 * ca / kh_co2


def _ci_initial(
    ca: float,
) -> float:
    return 0.65 * ca


def _pi_bellasio2019(
    total: float,
    pga: float,
    dhap: float,
    ru5p: float,
    rubp: float,
    atp: float,
) -> float:
    return total - pga - dhap - ru5p - 2 * rubp - atp


def _et(
    vmax_rub: float,
    kcat_rub: float,
    v_m: float,
) -> float:
    return (vmax_rub / kcat_rub) / v_m


def _km_rubp_extra(
    pga: float,
    nadp: float,
    adp: float,
    pi: float,
    km_rubp: float,
    ki_pga: float,
    ki_nadp: float,
    ki_adp: float,
    ki_pi: float,
) -> float:
    return km_rubp * (1 + pga / ki_pga + nadp / ki_nadp + adp / ki_adp + pi / ki_pi)


def _f_rubp(
    rubp: float,
    et: float,
    k_extra_rubp: float,
) -> float:
    top = (
        et
        + k_extra_rubp
        + rubp
        - np.sqrt((et + k_extra_rubp + rubp) ** 2 - 4 * rubp * et)
    )
    bottom = 2 * et
    return top / bottom


def _non_rect_hyperbole(
    x: float,
    alpha: float,
    v0: float,
    theta: float,
) -> float:
    # print(np.sqrt((alpha * x + 1 - V0)**2 - 4 * alpha * x * theta))
    # top = alpha * x + 1 - V0 - np.sqrt((alpha * x + 1 - V0)**2 - 4 * alpha * x * theta)
    # bottom = 2 * theta
    return (
        (alpha * x + 1 - v0) / (2 * theta)
        - np.sqrt((alpha * x + 1 - v0) ** 2 - 4 * alpha * x * theta * (1 - v0))
        / (2 * theta)
        + v0
    )


def _ract_eq(
    co2: float,
    ppfd: float,
    alpha_ppfd: float,
    v0_ppfd: float,
    theta_ppfd: float,
    alpha_co2: float,
    v0_co2: float,
    theta_co2: float,
) -> float:
    f_ppfd = _non_rect_hyperbole(
        ppfd,
        alpha_ppfd,
        v0_ppfd,
        theta_ppfd,
    )
    f_co2 = _non_rect_hyperbole(
        co2,
        alpha_co2,
        v0_co2,
        theta_co2,
    )
    return f_ppfd * f_co2


def _i20(
    pfd: float,
    s: float,
) -> float:
    return pfd * s


def _i10(
    i_20: float,
    y_ii_ll: float,
    y_i_ll: float,
) -> float:
    return i_20 * y_ii_ll / y_i_ll


def _chi(
    f_cyc: float,
    y_ii_ll: float,
) -> float:
    return f_cyc / (1 + f_cyc + y_ii_ll)


def _i1(
    chi: float,
    i10: float,
) -> float:
    return (1 + chi) * i10


def _f_cyc(
    j_atp: float,
    j_nadph: float,
    v_atp: float,
    v_fnr: float,
) -> float:
    return max(0, -1 + 15 ** (v_atp / j_atp - v_fnr / j_nadph))


def _i2(
    y_ii_ll: float,
    chi: float,
    i20: float,
) -> float:
    return (1 / y_ii_ll - chi) * i20 * y_ii_ll


def _y_ii(
    y_ii_ll: float,
    v_atp: float,
    j_atp: float,
    v_fnr: float,
    j_nadph: float,
    pfd: float,
    alpha: float,
    v0: float,
    theta: float,
) -> float:
    f_ppfd = _non_rect_hyperbole(
        pfd,
        alpha,
        v0,
        theta,
    )
    return y_ii_ll * (v_atp / j_atp) * (v_fnr / j_nadph) * (1 - max(0, f_ppfd))


def _j2(
    i2: float,
    y_ii: float,
) -> float:
    return i2 * y_ii


def _j1(
    j2: float,
    f_cyc: float,
) -> float:
    return j2 / 1 - f_cyc


def _f_pseudocyc(
    j_nadph: float,
    o2: float,
    v_fnr: float,
    f_pseudocyc_nr: float,
) -> float:
    return f_pseudocyc_nr + 4 * o2 * (1 - v_fnr / j_nadph)


def _j_nadph_steady(
    j1: float,
    f_cyc: float,
    f_pseudocyc: float,
) -> float:
    top = 1 - f_cyc - f_pseudocyc
    bottom = 2
    return (j1 * top / bottom) / 1000  # Added from Excel


def _j_atp_steady(
    j2: float,
    j1: float,
    f_cyc: float,
    fq: float,
    f_ndh: float,
    h: float,
) -> float:
    jcyt = (1 - fq) * j1
    jq = fq * j1
    jndh = f_cyc * f_ndh * j1

    return ((j2 + jcyt + 2 * jq + 2 * jndh) / h) / 1000  # Added from Excel


def _gs_steady(
    tau0: float,
    f_rubp: float,
    chi_beta: float,
    phi: float,
    pi_e: float,
    kh: float,
    ds: float,
    gs0: float,
) -> float:

    tau = tau0 + f_rubp
    top = chi_beta * tau * (phi + pi_e)
    bottom = 1 + chi_beta * tau * (1 / kh) * ds

    return max(gs0, top / bottom)


def _calc_ass(
    vc: float,
    vo: float,
    r_light: float,
) -> float:
    return vc - 0.5 * vo - r_light


def _ract_gs_time_dependance(
    x: float,
    x_steady: float,
    inc: float,
    dec: float,
) -> float:
    if x < x_steady:
        return (x_steady - x) / inc
    return (x_steady - x) / dec


def _atp_nadph_time_dependance(
    j_x: float,
    j_x_steady: float,
    kj_x: float,
) -> float:
    if j_x < j_x_steady:
        return (j_x_steady - j_x) / kj_x
    return (j_x_steady - j_x) / (0.1 * kj_x)


def _rubisco_carboxylation_bellasio(
    rubp: float,
    co2: float,
    ract: float,
    km_co2: float,
    o2: float,
    km_o2: float,
    vmax_rc: float,
    f_rubp: float,
    k_extra_rubp: float,
) -> float:
    k_extra_co2 = km_co2 * (1 + o2 / km_o2)

    top = vmax_rc * ract * f_rubp * rubp * co2
    bottom = (k_extra_co2 + co2) * (k_extra_rubp + rubp)

    return top / bottom


def _neg_one_div(
    x: float,
) -> float:
    return -1.0 / x


def _two_div(
    x: float,
) -> float:
    return 2.0 / x


def _rubisco_oxygenase_bellasio(
    co2: float,
    o2: float,
    s_co_gas: float,
    v_c: float,
    kh_o2: float,
    kh_co2: float,
) -> float:
    S_co_liq = s_co_gas / kh_o2 * kh_co2
    gamma_star = 1 / (2 * S_co_liq)

    return v_c * 2 * gamma_star * o2 / co2


def _neg_half_div(
    x: float,
) -> float:
    return -0.5 / x


def _half_div(
    x: float,
) -> float:
    return 0.5 / x


def _prkase(
    atp: float,
    rubp: float,
    ru5p: float,
    pga: float,
    adp: float,
    pi: float,
    vmax: float,
    k_eq: float,
    km_atp: float,
    ki_adp: float,
    km_ru5p: float,
    ki_pga: float,
    ki_rubp: float,
    ki_pi: float,
) -> float:
    top = vmax * (atp * ru5p - adp * rubp / k_eq)  # ERROR: Different from paper (typo?)
    bottom = (atp + km_atp * (1 + adp / ki_adp)) * (
        ru5p + km_ru5p * (1 + pga / ki_pga + rubp / ki_rubp + pi / ki_pi)
    )
    return top / bottom


def _neg_fivethirds_div(
    x: float,
) -> float:
    return -(5 / 3) * (1 / x)


def _v_pgareduction(
    atp: float,
    pga: float,
    nadph: float,
    adp: float,
    vmax: float,
    km_atp: float,
    km_pga: float,
    km_nadph: float,
    ki_adp: float,
) -> float:
    top = vmax * atp * pga * nadph
    bottom = (
        (pga + km_pga * (1 + adp / ki_adp))
        * (atp + km_atp * (1 + adp / ki_adp))
        * (nadph + km_nadph * (1 + adp / ki_adp))
    )
    return top / bottom


def _v_carbohydrate_synthesis(
    dhap: float,
    pi: float,
    adp: float,
    vmax: float,
    v_pgareduction: float,
    keq: float,
    km_dhap: float,
    ki_adp: float,
) -> float:
    top = vmax * (dhap - 0.4) * (1 - np.abs(v_pgareduction) * pi / keq)
    bottom = dhap + km_dhap * (1 + adp / ki_adp)
    return top / bottom


def _v_rpp(
    dhap: float,
    ru5p: float,
    vmax: float,
    k_eq: float,
    km_dhap: float,
) -> float:
    top = vmax * (dhap - ru5p / k_eq)
    bottom = dhap + km_dhap
    return top / bottom


def _v_co2_hydration(
    co2: float,
    hco3: float,
    proton: float,
    vmax: float,
    k_eq: float,
    km_co2: float,
    km_hco3: float,
) -> float:
    top = vmax * (co2 - hco3 * proton / k_eq)
    bottom = km_co2 * (1 + co2 / km_co2 + hco3 / km_hco3)
    return top / bottom


def _neg_onethirds_div(
    x: float,
) -> float:
    return -(1 / 3) * (1 / x)


def _v_fnr(
    nadph: float,
    nadp: float,
    j_nadph: float,
    k_eq: float,
    km_nadp: float,
    km_nadph: float,
) -> float:
    top = j_nadph * (nadp - nadph / k_eq)
    bottom = km_nadp * (1 + nadp / km_nadp + nadph / km_nadph)
    return top / bottom


def _v_atp(
    atp: float,
    adp: float,
    pi: float,
    j_atp: float,
    k_eq: float,
    km_adp: float,
    km_pi: float,
    km_atp: float,
) -> float:
    top = j_atp * (adp * pi - atp / k_eq)
    bottom = (
        km_adp
        * km_pi
        * (1 + adp / km_adp + atp / km_atp + pi / km_pi + adp * pi / (km_adp * km_pi))
    )
    return top / bottom


def _co2_diss(
    ci: float,
    co2: float,
    gm: float,
    kh_co2: float,
) -> float:
    return (gm * (ci - co2 * kh_co2)) / 1000


def _stom_diff(
    ci: float,
    gs: float,
    ca: float,
) -> float:
    return (gs * (ca - ci)) / 1000


def get_bellasio_2019() -> Model:
    r"""Bellassio 2019 model

    The [Bellasio2019](https://doi.org/10.1007/s11120-018-0601-1) model is a
    generalized C3 leaf-photosynthesis model that includes simplified
    representations of the light and dark reactions and a stomatal behavior
    submodule. A lot of its implementation is based on past work by the same author
    and is mainly inspired by the common Farquhar-von Caemmerer-Berry model. The
    light reactions are modified from Yin et al. (2004) and include the potential
    rates of ATP and NADPH production based on light intensity. This model has been
    created with the simple user in mind, and the author has made an effort to show
    its simplicity by giving access to a Microsoft Excel Workbook containing the
    entire model. To showcase the model's capabilities, the author creates common
    steady-state carbon assimilation curves against intercellular CO2 concentration
    and light intensity, and compares them to experimental data from the
    literature. As many models of photosynthesis rely on purely stead-state
    assumptions, this model is also validated in dynamic conditions, showing for
    example the response of the model to a fluctuation of ambient oxygen
    concentration.

    This model was created to stay as simple as possible, while still being able to
    accurately represent the main features of C<sub>3</sub> photosynthesis. As
    such, it can be used as a base for more complex models, or as a starting block
    in larger models of plant physiology. While giving access to the entire model
    in an Excel Workbook format is transparent and great, the execution of said
    practice has been inefficient in this instance. The entire mathematical
    description of the model is also given in the Appendix of the publication,
    however there are missing or different equations between the publication and
    the Excel Workbook, which can lead to confusion. On top of that, the simulation
    protocols used for each figure are only given in small details, which leads to
    further confusion when trying to reproduce the results and see which equations
    are correct or not.
    """
    m = Model()

    m.add_parameters(
        {
            "AP_tot": Parameter(1.5, unit=_mm),
            "Pi_tot": Parameter(15, unit=_mm),
            "p_o2": Parameter(210000, unit=_umol_mol),
            "Kh_o2": Parameter(833300, unit=_umol_mol / _mm),
            "V_m": Parameter(0.03, unit=liter / sqm),
            "PPFD": Parameter(
                1500, unit=ppfd, slider=Slider(min="100", max="2000", step="10")
            ),
            # Table S2 lists 0.0012 umol m^-2 s^-1, but it is a flux in mmol m^-2 s^-1 like all others
            "RLight": Parameter(0.001, unit=_flux),
            "s": Parameter(0.43, unit=dimensionless),
            "PhiPSII_LL": Parameter(0.72, unit=dimensionless),
            "PhiPSI_LL": Parameter(1, unit=dimensionless),
            "alpha_ppfd_PhiPSII": Parameter(0.00125, unit=1 / ppfd),
            "V0_ppfd_PhiPSII": Parameter(-0.8, unit=dimensionless),
            "theta_ppfd_PhiPSII": Parameter(0.7, unit=dimensionless),
            "f_pseudocycNR": Parameter(0.01, unit=dimensionless),
            "fq": Parameter(1, unit=dimensionless),
            "f_ndh": Parameter(0, unit=dimensionless),
            "h": Parameter(4, unit=dimensionless),
            "Ca": Parameter(350, unit=_umol_mol),
            "alpha_ppfd_rub": Parameter(0.0018, unit=1 / ppfd),
            "V0_ppfd_rub": Parameter(0.16, unit=dimensionless),
            "theta_ppfd_rub": Parameter(0.95, unit=dimensionless),
            "alpha_co2": Parameter(400, unit=1 / _mm),
            "V0_co2": Parameter(-0.2, unit=dimensionless),
            "theta_co2": Parameter(0.98, unit=dimensionless),
            "tau_i": Parameter(360, unit=second),
            "tau_d": Parameter(1200, unit=second),
            "km_v_RuBisCO_c_CO2": Parameter(
                0.014, unit=_mm, display_name="km_rubisco_carboxylase_CO2"
            ),
            "km_v_RuBisCO_c_RUBP": Parameter(
                0.02, unit=_mm, display_name="km_rubisco_carboxylase_RUBP"
            ),
            "km_v_RuBisCO_c_O2": Parameter(0.222, unit=_mm),
            "ki_v_RuBisCO_c_PGA": Parameter(
                2.52, unit=_mm, display_name="ki_rubisco_carboxylase_3PGA"
            ),
            "ki_v_RuBisCO_c_NADP_st": Parameter(0.21, unit=_mm),
            "ki_v_RuBisCO_c_ADP_st": Parameter(0.2, unit=_mm),
            "ki_v_RuBisCO_c_Pi_st": Parameter(
                3.6, unit=_mm, display_name="ki_rubisco_carboxylase_Pi"
            ),
            "vmax_v_RuBisCO_c": Parameter(
                0.2, unit=_flux, display_name="vmax_rubisco_carboxylase"
            ),
            "kcat_v_RuBisCO_c": Parameter(
                4.7, unit=1 / second, display_name="kcat_rubisco_carboxylase"
            ),
            "S_co_gas": Parameter(2200, unit=dimensionless),
            "vmax_v_PRKase": Parameter(
                1.17, unit=_flux, display_name="vmax_phosphoribulokinase"
            ),
            # Table S1 lists mM, but v_PRKase requires it dimensionless
            "keq_v_PRKase": Parameter(6846, unit=dimensionless),
            "km_v_PRKase_ATP_st": Parameter(
                0.625, unit=_mm, display_name="km_phosphoribulokinase_ATP"
            ),
            "ki_v_PRKase_ADP_st": Parameter(0.1, unit=_mm),
            "km_v_PRKase_RU5P": Parameter(
                0.034, unit=_mm, display_name="km_phosphoribulokinase_RU5P"
            ),
            "ki_v_PRKase_PGA": Parameter(
                2, unit=_mm, display_name="ki_phosphoribulokinase_3PGA"
            ),
            "ki_v_PRKase_RUBP": Parameter(
                0.7, unit=_mm, display_name="ki_phosphoribulokinase_RUBP"
            ),
            "ki_v_PRKase_Pi_st": Parameter(
                4, unit=_mm, display_name="ki_phosphoribulokinase_Pi"
            ),
            "vmax_v_pgareduction": Parameter(0.4, unit=_flux),
            "km_v_pgareduction_ATP_st": Parameter(0.3, unit=_mm),
            "km_v_pgareduction_PGA": Parameter(10, unit=_mm),
            "km_v_pgareduction_NADPH_st": Parameter(0.05, unit=_mm),
            "ki_v_pgareduction_ADP_st": Parameter(0.89, unit=_mm),
            "vmax_v_carbohydrate_synthesis": Parameter(0.2235, unit=_flux),
            # Table S1 lists it dimensionless, but it divides Pi_st * |v_pgareduction|
            "keq_v_carbohydrate_synthesis": Parameter(0.8, unit=_mm * _flux),
            "km_v_carbohydrate_synthesis_DHAP": Parameter(22, unit=_mm),
            "ki_v_carbohydrate_synthesis_ADP_st": Parameter(1, unit=_mm),
            "vmax_v_rpp": Parameter(0.0585, unit=_flux),
            "keq_v_rpp": Parameter(0.06, unit=dimensionless),
            "km_v_rpp_DHAP": Parameter(0.5, unit=_mm),
            "H": Parameter(5.012e-05, unit=_mm),
            "vmax_v_co2_hydration": Parameter(200, unit=_flux),
            # Table S1 lists it dimensionless, but it divides H * HCO3
            "keq_v_co2_hydration": Parameter(0.00056, unit=_mm),
            "km_v_co2_hydration_CO2": Parameter(2.8, unit=_mm),
            "km_v_co2_hydration_HCO3": Parameter(34, unit=_mm),
            # Table S1 lists mM, but v_FNR requires it dimensionless
            "keq_v_FNR": Parameter(502, unit=dimensionless),
            "km_v_FNR_NADP_st": Parameter(0.0072, unit=_mm, display_name="km_fnr_NADP"),
            "km_v_FNR_NADPH_st": Parameter(0.036, unit=_mm),
            "Kj_NADPH": Parameter(200, unit=second),
            # Table S1 lists it dimensionless, but v_ATPsynth requires mM^-1 (ADP * Pi vs ATP / keq)
            "keq_v_ATPsynth": Parameter(
                5734, unit=1 / _mm, display_name="keq_atp_synthase"
            ),
            "km_v_ATPsynth_ADP_st": Parameter(0.014, unit=_mm),
            "km_v_ATPsynth_Pi_st": Parameter(0.3, unit=_mm),
            "km_v_ATPsynth_ATP_st": Parameter(0.3, unit=_mm),
            "Kj_ATP": Parameter(200, unit=second),
            "gm": Parameter(0.5, unit=_mol_m2_s),
            "Kh_co2": Parameter(30303.0303030303, unit=_umol_mol / _mm),
            "Kd": Parameter(150, unit=second),
            "Ki": Parameter(900, unit=second),
            # Table S2 lists mM, but it is added to the dimensionless f_rubp
            "tau0": Parameter(-0.1, unit=dimensionless),
            "chi_beta": Parameter(0.5, unit=_mol_m2_s / _mpa),
            "phi": Parameter(0, unit=_mpa),
            "pi_e": Parameter(1.2, unit=_mpa),
            "Kh": Parameter(12, unit=_flux / _mpa),
            # leaf-to-air water vapour mole fraction difference
            "Ds": Parameter(10, unit=mmol / mol),
            "gs0": Parameter(0.01, unit=_mol_m2_s),
            "NADP_tot": Parameter(0.5, unit=_mm),
        }
    )

    m.add_variables(
        {
            "CO2": Variable(
                InitialAssignment(fn=_co2_initial, args=["Ca", "Kh_co2"]), unit=_mm
            ),
            "HCO3": Variable(0.1327, unit=_mm),
            "RUBP": Variable(2, unit=_mm),
            "PGA": Variable(4, unit=_mm),
            "DHAP": Variable(4, unit=_mm),
            "ATP_st": Variable(0.68, unit=_mm, display_name="ATP"),
            "NADPH_st": Variable(0.21, unit=_mm, display_name="NADPH"),
            "RU5P": Variable(0.34, unit=_mm),
            "Ract": Variable(1, unit=dimensionless),
            "J_NADPH": Variable(0.1, unit=_flux),
            "J_ATP": Variable(0.16, unit=_flux),
            "Ci": Variable(
                InitialAssignment(fn=_ci_initial, args=["Ca"]), unit=_umol_mol
            ),
            "gs": Variable(0.334934046786077, unit=_mol_m2_s),
        }
    )

    m.add_derived(
        name="ADP_st", fn=_moiety_1, args=["ATP_st", "AP_tot"], display_name="ADP"
    )

    m.add_derived(
        name="NADP_st", fn=_moiety_1, args=["NADPH_st", "NADP_tot"], display_name="NADP"
    )

    m.add_derived(
        name="Pi_st",
        fn=_pi_bellasio2019,
        args=["Pi_tot", "PGA", "DHAP", "RU5P", "RUBP", "ATP_st"],
    )

    m.add_derived(
        name="Et",
        fn=_et,
        args=["vmax_v_RuBisCO_c", "kcat_v_RuBisCO_c", "V_m"],
    )

    m.add_derived(
        name="km_v_RuBisCO_c_RUBP_extra",
        fn=_km_rubp_extra,
        args=[
            "PGA",
            "NADP_st",
            "ADP_st",
            "Pi_st",
            "km_v_RuBisCO_c_RUBP",
            "ki_v_RuBisCO_c_PGA",
            "ki_v_RuBisCO_c_NADP_st",
            "ki_v_RuBisCO_c_ADP_st",
            "ki_v_RuBisCO_c_Pi_st",
        ],
    )

    m.add_derived(
        name="f_rubp",
        fn=_f_rubp,
        args=["RUBP", "Et", "km_v_RuBisCO_c_RUBP_extra"],
    )

    m.add_derived(
        name="O2",
        fn=_div,
        args=["p_o2", "Kh_o2"],
    )

    m.add_derived(
        name="Ract_eq",
        fn=_ract_eq,
        args=[
            "CO2",
            "PPFD",
            "alpha_ppfd_rub",
            "V0_ppfd_rub",
            "theta_ppfd_rub",
            "alpha_co2",
            "V0_co2",
            "theta_co2",
        ],
    )

    m.add_derived(
        name="I2_0",
        fn=_i20,
        args=["PPFD", "s"],
    )

    m.add_derived(
        name="I1_0",
        fn=_i10,
        args=["I2_0", "PhiPSII_LL", "PhiPSI_LL"],
    )

    m.add_derived(
        name="chi",
        fn=_chi,
        args=["f_cyc", "PhiPSII_LL"],
    )

    m.add_derived(
        name="I1",
        fn=_i1,
        args=["chi", "I1_0"],
    )

    m.add_derived(
        name="f_cyc",
        fn=_f_cyc,
        args=["J_ATP", "J_NADPH", "v_ATPsynth", "v_FNR"],
    )

    m.add_derived(
        name="I2",
        fn=_i2,
        args=["PhiPSII_LL", "chi", "I2_0"],
    )

    m.add_derived(
        name="PhiPSII",
        fn=_y_ii,
        args=[
            "PhiPSII_LL",
            "v_ATPsynth",
            "J_ATP",
            "v_FNR",
            "J_NADPH",
            "PPFD",
            "alpha_ppfd_PhiPSII",
            "V0_ppfd_PhiPSII",
            "theta_ppfd_PhiPSII",
        ],
    )

    m.add_derived(
        name="J2",
        fn=_j2,
        args=["I2", "PhiPSII"],
    )

    m.add_derived(
        name="J1",
        fn=_j1,
        args=["J2", "f_cyc"],
    )

    m.add_derived(
        name="f_pseudocyc",
        fn=_f_pseudocyc,
        args=["J_NADPH", "O2", "v_FNR", "f_pseudocycNR"],
    )

    # converts electron flow in umol m^-2 s^-1 to mmol m^-2 s^-1 with a hidden factor
    m.add_derived(
        name="J_NADPH_steady",
        fn=_j_nadph_steady,
        args=["J1", "f_cyc", "f_pseudocyc"],
        unit=_flux,
    )

    m.add_derived(
        name="J_ATP_steady",
        fn=_j_atp_steady,
        args=["J2", "J1", "f_cyc", "fq", "f_ndh", "h"],
        unit=_flux,
    )

    m.add_derived(
        name="gs_steady",
        fn=_gs_steady,
        args=["tau0", "f_rubp", "chi_beta", "phi", "pi_e", "Kh", "Ds", "gs0"],
    )

    m.add_derived(
        name="A",
        fn=_calc_ass,
        args=["v_RuBisCO_c", "rubisco_oxygenase", "RLight"],
    )

    m.add_reaction(
        name="Ract_rate",
        fn=_ract_gs_time_dependance,
        args=["Ract", "Ract_eq", "tau_i", "tau_d"],
        stoichiometry={
            "Ract": 1,
        },
    )
    m.add_reaction(
        name="v_J_NADPH",
        fn=_atp_nadph_time_dependance,
        args=["J_NADPH", "J_NADPH_steady", "Kj_NADPH"],
        stoichiometry={
            "J_NADPH": 1,
        },
    )
    m.add_reaction(
        name="v_J_ATP",
        fn=_atp_nadph_time_dependance,
        args=["J_ATP", "J_ATP_steady", "Kj_ATP"],
        stoichiometry={
            "J_ATP": 1,
        },
    )
    m.add_reaction(
        name="v_gs",
        fn=_ract_gs_time_dependance,
        args=["gs", "gs_steady", "Ki", "Kd"],
        stoichiometry={
            "gs": 1,
        },
    )
    m.add_reaction(
        name="v_RuBisCO_c",
        fn=_rubisco_carboxylation_bellasio,
        args=[
            "RUBP",
            "CO2",
            "Ract",
            "km_v_RuBisCO_c_CO2",
            "O2",
            "km_v_RuBisCO_c_O2",
            "vmax_v_RuBisCO_c",
            "f_rubp",
            "km_v_RuBisCO_c_RUBP_extra",
        ],
        stoichiometry={
            "CO2": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "RUBP": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "PGA": Derived(
                fn=_two_div,
                args=["V_m"],
                unit=None,
            ),
        },
        display_name="rubisco carboxylase",
    )
    m.add_reaction(
        name="rubisco_oxygenase",
        fn=_rubisco_oxygenase_bellasio,
        args=["CO2", "O2", "S_co_gas", "v_RuBisCO_c", "Kh_o2", "Kh_co2"],
        stoichiometry={
            "RUBP": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "PGA": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
            "ATP_st": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "NADPH_st": Derived(
                fn=_neg_half_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="glycine_decarboxylase",
        fn=_value,
        args=["rubisco_oxygenase"],
        stoichiometry={
            "CO2": Derived(
                fn=_half_div,
                args=["V_m"],
                unit=None,
            ),
            "PGA": Derived(
                fn=_half_div,
                args=["V_m"],
                unit=None,
            ),
        },
        display_name="glycine decarboxylase",
    )
    m.add_reaction(
        name="v_PRKase",
        fn=_prkase,
        args=[
            "ATP_st",
            "RUBP",
            "RU5P",
            "PGA",
            "ADP_st",
            "Pi_st",
            "vmax_v_PRKase",
            "keq_v_PRKase",
            "km_v_PRKase_ATP_st",
            "ki_v_PRKase_ADP_st",
            "km_v_PRKase_RU5P",
            "ki_v_PRKase_PGA",
            "ki_v_PRKase_RUBP",
            "ki_v_PRKase_Pi_st",
        ],
        stoichiometry={
            "RUBP": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
            "DHAP": Derived(
                fn=_neg_fivethirds_div,
                args=["V_m"],
                unit=None,
            ),
            "ATP_st": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "RU5P": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
        display_name="phosphoribulokinase",
    )
    m.add_reaction(
        name="v_pgareduction",
        fn=_v_pgareduction,
        args=[
            "ATP_st",
            "PGA",
            "NADPH_st",
            "ADP_st",
            "vmax_v_pgareduction",
            "km_v_pgareduction_ATP_st",
            "km_v_pgareduction_PGA",
            "km_v_pgareduction_NADPH_st",
            "ki_v_pgareduction_ADP_st",
        ],
        stoichiometry={
            "PGA": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "DHAP": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
            "ATP_st": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "NADPH_st": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_carbohydrate_synthesis",
        fn=_v_carbohydrate_synthesis,
        args=[
            "DHAP",
            "Pi_st",
            "ADP_st",
            "vmax_v_carbohydrate_synthesis",
            "v_pgareduction",
            "keq_v_carbohydrate_synthesis",
            "km_v_carbohydrate_synthesis_DHAP",
            "ki_v_carbohydrate_synthesis_ADP_st",
        ],
        stoichiometry={
            "DHAP": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "ATP_st": Derived(
                fn=_neg_half_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_rpp",
        fn=_v_rpp,
        args=["DHAP", "RU5P", "vmax_v_rpp", "keq_v_rpp", "km_v_rpp_DHAP"],
        stoichiometry={
            "RU5P": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_co2_hydration",
        fn=_v_co2_hydration,
        args=[
            "CO2",
            "HCO3",
            "H",
            "vmax_v_co2_hydration",
            "keq_v_co2_hydration",
            "km_v_co2_hydration_CO2",
            "km_v_co2_hydration_HCO3",
        ],
        stoichiometry={
            "CO2": Derived(
                fn=_neg_one_div,
                args=["V_m"],
                unit=None,
            ),
            "HCO3": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_RLight",
        fn=_value,
        args=["RLight"],
        stoichiometry={
            "CO2": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
            "PGA": Derived(
                fn=_neg_onethirds_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_FNR",
        fn=_v_fnr,
        args=[
            "NADPH_st",
            "NADP_st",
            "J_NADPH",
            "keq_v_FNR",
            "km_v_FNR_NADP_st",
            "km_v_FNR_NADPH_st",
        ],
        stoichiometry={
            "NADPH_st": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
        display_name="FNR",
    )
    m.add_reaction(
        name="v_ATPsynth",
        fn=_v_atp,
        args=[
            "ATP_st",
            "ADP_st",
            "Pi_st",
            "J_ATP",
            "keq_v_ATPsynth",
            "km_v_ATPsynth_ADP_st",
            "km_v_ATPsynth_Pi_st",
            "km_v_ATPsynth_ATP_st",
        ],
        stoichiometry={
            "ATP_st": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
        display_name="ATP synthase",
    )
    m.add_reaction(
        name="CO2_dissolution",
        fn=_co2_diss,
        args=["Ci", "CO2", "gm", "Kh_co2"],
        stoichiometry={
            "Ci": -1,
            "CO2": Derived(
                fn=_one_div,
                args=["V_m"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="CO2_stomatal_diffusion",
        fn=_stom_diff,
        args=["Ci", "gs", "Ca"],
        stoichiometry={
            "Ci": 1,
        },
    )
    return m
