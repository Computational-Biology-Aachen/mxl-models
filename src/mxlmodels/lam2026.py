r"""Lam 2026 model

|             |                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------- |
| doi         | 10.1038/s41467-026-70414-2                                                                         |
| main author | Lam Lam, Rebecca Lee                                                                               |
| paper title | Dissecting the contributions to non-photochemical quenching in a land plant under ﬂuctuating light |
| published   | 09 March 2026                                                                                      |
| journal     | Nature Communications                                                                              |
| organism    | Higher plants (Nicotiana benthamiana)                                                              |
| Ported by   | Quang Huy Nguyen ( @PhotosyntheticBatman )                                                         |

The model provide a detailed, up-to-date descriptions regarding NPQ compoents
within higher plants, along with detailed dataset of fluorescence lifetime and
pigment qunatification from mutants that lacks one or more compoents of NPQ.
Combined with a rigorous fitting and optimization schedule for parameters, the
model brings up quantified contributions of different NPQ components and how it
changes overtime.

Compoents: energy-depedent quenching (qE), zeaxanthin-depdent quenching (qZ)
and photoinhibition (qI)
"""

from __future__ import annotations

from mxlpy import InitialAssignment, Model, Parameter, Variable
from mxlpy.units import dimensionless, mmol_mol_chl, ppfd, second

# Pigments and binding sites in mmol (mol Chl)^-1, the photoinhibited fraction PSIId and
# the VDE activity are dimensionless
_mmol_mol_chl = mmol_mol_chl
_per_s = 1 / second
_per_mmol_mol_chl_s = 1 / (mmol_mol_chl * second)

############INITIAL CONDITIONS##############


def _p_free(ptot: float, xtot: float, v: float, a: float, z: float) -> float:
    return ptot - xtot + v + a + z


def _x_tot(
    v: float,
    a: float,
    z: float,
    pv: float,
    pa: float,
    pz: float,
    qv: float,
    qa: float,
    qz: float,
) -> float:
    return v + a + z + pv + pa + pz + qv + qa + qz


def _v_free(
    xtot: float,
    a: float,
    z: float,
    pv: float,
    pa: float,
    pz: float,
    qv: float,
    qa: float,
    qz: float,
) -> float:
    return xtot - (a + z + pv + pa + pz + qv + qa + qz)


def _keq(kf: float, kr: float) -> float:
    return kf / kr


# FIXME @ Huy: what is up with kf_light & kf_dark here?
def _keq_light_dark(
    kf_light: float,  # noqa: ARG001
    kr_light: float,  # noqa: ARG001
    kf_dark: float,
    kr_dark: float,
    ppfd: float,
) -> float:
    if ppfd == 0:
        return kf_dark / kr_dark
    return kf_dark / kr_dark  # kf_light/kr_light


def _p0(
    ptot: float,
    v0: float,
    kpv: float,
    kpa: float,
    ka: float,
    kpz: float,
    kz: float,
    kqv: float,
    kqa: float,
    kqz: float,
) -> float:
    return ptot / (
        1
        + v0
        * (
            kpv
            + kpa * ka
            + kpz * kz * ka
            + kqv * kpv
            + kqa * kpa * ka
            + kqz * kpz * kz * ka
        )
    )


def _a0(_keq: float, v0: float) -> float:
    return _keq * v0


def _z0(ka: float, kz: float, v0: float) -> float:
    return ka * kz * v0


def _pa0(
    kpa: float,
    ka: float,
    v0: float,
    _p0: float,
) -> float:
    return kpa * ka * v0 * _p0


def _pz0(kpz: float, kz: float, ka: float, _p0: float, v0: float) -> float:
    return kpz * kz * ka * v0 * _p0


def _pv0(
    kpv: float,
    v0: float,
    _p0: float,
) -> float:
    return kpv * v0 * _p0


def _qa0(_p0: float, v0: float, kpa: float, ka: float, kqa: float) -> float:
    return kpa * ka * kqa * _p0 * v0


def _qv0(_p0: float, v0: float, kpv: float, kqv: float) -> float:
    return kpv * kqv * _p0 * v0


def _qz0(kqz: float, kpz: float, kz: float, ka: float, _p0: float, v0: float) -> float:
    return kqz * kpz * kz * ka * _p0 * v0


def _e0(kva_d: float, kva_l: float) -> float:
    return kva_d / kva_l


############DERIVED##############


def _moiety(xtot: float, x: float) -> float:
    return xtot - x


def _kappa_r_nr(tau_0: float, kappa_q_z: float, z_0: float) -> float:
    return 1 / tau_0 - kappa_q_z * z_0


############RATES##############


def _same(x: float) -> float:
    return x


def _mul(x: float, y: float) -> float:
    return x * y


def _mass_action_2s(k: float, s1: float, s2: float) -> float:
    return k * s1 * s2


def _mass_action_1s(k: float, s: float) -> float:
    return k * s


def _mass_action_light_dark_1s(
    ppfd: float, k_light: float, k_dark: float, s: float
) -> float:
    if ppfd == 0:
        return k_dark * s
    return k_light * s


def _mass_action_light_dark_2s(
    ppfd: float, k_light: float, k_dark: float, s1: float, s2: float
) -> float:
    if ppfd == 0:
        return k_dark * s1 * s2
    return k_light * s1 * s2


def _damage(
    ppfd: float, k_light: float, k_dark: float, tau: float, psi_id: float
) -> float:
    tau = max(tau, 0)
    if ppfd == 0:
        return max(k_dark * tau * (1 - psi_id), 0)
    return max(k_light * tau * (1 - psi_id), 0)


# --- alpha_VDE enzyme activation ---


def _v_alpha_vde(
    ppfd: float,
    k_l_vde: float,
    k_d_vde: float,
    k_l_va: float,
    k_d_va: float,
    alpha_vde: float,
) -> float:
    if ppfd == 0:
        k_VDE = k_d_vde
        alpha_VDE_eq = k_d_va / k_l_va
    else:
        k_VDE = k_l_vde
        alpha_VDE_eq = 1
    return k_VDE * (alpha_VDE_eq - alpha_vde)


def _chlorophyll_fluo_lifetime(
    kappa_qv: float,
    qv: float,
    kappa_qa: float,
    qa: float,
    kappa_qz: float,
    qz: float,
    kappa_ql: float,
    ql: float,
    kappa_q_z: float,
    z: float,
    k_q_i: float,
    psi_id: float,
    _kappa_r_nr: float,
) -> float:
    denom = (
        _kappa_r_nr
        + kappa_qv * max(qv, 0)
        + kappa_qa * max(qa, 0)
        + kappa_qz * max(qz, 0)
        + kappa_ql * max(ql, 0)
        + kappa_q_z * max(z, 0)
        + k_q_i * max(psi_id, 0)
    )
    return 1.0 / max(denom, 1e-10)


############MODELS##############


def get_lam2026() -> Model:
    r"""Get Lam 2026 model."""
    m = Model()
    m.add_parameters(
        {
            "k_L_VA": Parameter(2.47, unit=_per_s),
            "k_D_VA": Parameter(0.014, unit=_per_s),
            "k_L_AZ": Parameter(0.5, unit=_per_s),
            "k_AV": Parameter(1.12, unit=_per_s),
            "k_ZA": Parameter(0.07, unit=_per_s),
            "k_PV_f": Parameter(2.18, unit=_per_mmol_mol_chl_s),
            "k_PV_b": Parameter(9.43, unit=_per_s),
            "k_PA_f": Parameter(130, unit=_per_mmol_mol_chl_s),
            "k_PA_b": Parameter(254, unit=_per_s),
            "k_PZ_f": Parameter(295, unit=_per_mmol_mol_chl_s),
            "k_PZ_b": Parameter(126, unit=_per_s),
            "k_L_QV_f": Parameter(0.027, unit=_per_s),
            "k_D_QV_f": Parameter(0, unit=_per_s),
            "k_QV_b": Parameter(0.066, unit=_per_s),
            "k_L_QA_f": Parameter(0.66, unit=_per_s),
            "k_D_QA_f": Parameter(0, unit=_per_s),
            "k_QA_b": Parameter(8.57, unit=_per_s),
            "k_L_QZ_f": Parameter(0.56, unit=_per_s),
            "k_D_QZ_f": Parameter(0, unit=_per_s),
            "k_QZ_b": Parameter(1.22, unit=_per_s),
            "k_L_QL_f": Parameter(0.056, unit=_per_s),
            "k_QL_b": Parameter(3.68, unit=_per_s),
            "k_D_QX_f": Parameter(
                0, unit=_per_s
            ),  # for all QX complex the rate in the dark is set to 0
            "k_L_damage": Parameter(0.0222, unit=1 / second**2),
            "k_D_damage": Parameter(0.0161, unit=1 / second**2),
            "k_D_VDE": Parameter(0.24, unit=_per_s),
            "k_L_VDE": Parameter(0.28, unit=_per_s),
            "k_AV_aba1": Parameter(0.006, unit=_per_s),
            "k_ZA_aba1": Parameter(0.038, unit=_per_s),
            "k_PV_f_lut2": Parameter(1.43, unit=_per_mmol_mol_chl_s),
            "k_PV_b_lut2": Parameter(13.1, unit=_per_s),
            "k_PA_f_lut2": Parameter(34.4, unit=_per_mmol_mol_chl_s),
            "k_PA_b_lut2": Parameter(294, unit=_per_s),
            "k_PZ_f_lut2": Parameter(74.1, unit=_per_mmol_mol_chl_s),
            "k_PZ_b_lut2": Parameter(168, unit=_per_s),
            "V_tot_npq1": Parameter(49.8, unit=_mmol_mol_chl),
            "V_tot_lut2": Parameter(71.2, unit=_mmol_mol_chl),
            "V_tot_npq4": Parameter(40.6, unit=_mmol_mol_chl),
            "V_tot_aba1": Parameter(10.7, unit=_mmol_mol_chl),
            "V_tot_WT": Parameter(35.9, unit=_mmol_mol_chl),
            "P_tot": Parameter(45.4, unit=_mmol_mol_chl),
            "P_tot_lut2": Parameter(49.9, unit=_mmol_mol_chl),
            "kappa_QV": Parameter(0.040, unit=_per_mmol_mol_chl_s),
            "kappa_QA": Parameter(0.174, unit=_per_mmol_mol_chl_s),
            "kappa_QZ": Parameter(0.177, unit=_per_mmol_mol_chl_s),
            "kappa_QL": Parameter(0.262, unit=_per_mmol_mol_chl_s),
            "kappa_qZ": Parameter(0.030, unit=_per_mmol_mol_chl_s),
            "kappa_qI": Parameter(3.86, unit=_per_s),
            "kappa_qI_double_mut": Parameter(7.05, unit=_per_s),
            "ppfd": Parameter(0, unit=ppfd, display_name="PPFD"),
            "tau_0": Parameter(
                1.73089079100000, unit=second
            ),  # from the paper WT tau_0 for 5-10-5 dataset
            "PSII_tot": Parameter(1, unit=dimensionless, display_name="PSII_total"),
        }
    )

    m.add_derived("gamma", _e0, args=["k_D_VA", "k_L_VA"])
    m.add_derived("k_D_AZ", _mass_action_1s, args=["gamma", "k_L_AZ"])

    m.add_derived("Keq_pz", _keq, args=["k_PZ_f", "k_PZ_b"])
    m.add_derived("Keq_pv", _keq, args=["k_PV_f", "k_PV_b"])
    m.add_derived("Keq_pa", _keq, args=["k_PA_f", "k_PA_b"])

    m.add_derived(
        "Keq_a",
        _keq_light_dark,
        args=["k_D_VA", "k_AV", "k_D_VA", "k_AV", "ppfd"],
    )
    m.add_derived(
        "Keq_z",
        _keq_light_dark,
        args=["k_D_AZ", "k_ZA", "k_D_AZ", "k_ZA", "ppfd"],
    )
    m.add_derived(
        "Keq_qv",
        _keq_light_dark,
        args=["k_L_QV_f", "k_QV_b", "k_D_QV_f", "k_QV_b", "ppfd"],
    )
    m.add_derived(
        "Keq_qa",
        _keq_light_dark,
        args=["k_L_QA_f", "k_QA_b", "k_D_QA_f", "k_QA_b", "ppfd"],
    )
    m.add_derived(
        "Keq_qz",
        _keq_light_dark,
        args=["k_L_QZ_f", "k_QZ_b", "k_D_QZ_f", "k_QZ_b", "ppfd"],
    )

    m.add_derived(
        "P0",
        _p0,
        args=[
            "P_tot",
            "V_tot_WT",
            "Keq_pv",
            "Keq_pa",
            "Keq_a",
            "Keq_pz",
            "Keq_z",
            "Keq_qv",
            "Keq_qa",
            "Keq_qz",
        ],
    )

    m.add_derived("A_0", fn=_a0, args=["Keq_a", "V_tot_WT"])
    m.add_derived("Z_0", fn=_z0, args=["Keq_a", "Keq_z", "V_tot_WT"])
    m.add_derived("PV_0", fn=_pv0, args=["Keq_pv", "V_tot_WT", "P0"])
    m.add_derived("PA_0", fn=_pa0, args=["Keq_pa", "Keq_a", "V_tot_WT", "P0"])
    m.add_derived("PZ_0", fn=_pz0, args=["Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"])
    m.add_derived("QV_0", fn=_qv0, args=["P0", "V_tot_WT", "Keq_pv", "Keq_qv"])
    m.add_derived("QA_0", fn=_qa0, args=["P0", "V_tot_WT", "Keq_pa", "Keq_a", "Keq_qa"])
    m.add_derived(
        "QZ_0",
        fn=_qz0,
        args=["Keq_qz", "Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"],
    )

    m.add_derived(
        "X_tot",
        fn=_x_tot,
        args=["V_tot_WT", "A_0", "Z_0", "PV_0", "PA_0", "PZ_0", "QV_0", "QA_0", "QZ_0"],
    )

    m.add_derived("kappa_r_nr", _kappa_r_nr, args=["tau_0", "kappa_qZ", "Z_0"])

    # Variables and initial conditions
    m.add_variables(
        {
            "V": Variable(
                InitialAssignment(fn=_same, args=["V_tot_WT"]), unit=_mmol_mol_chl
            ),
            "A": Variable(
                InitialAssignment(fn=_a0, args=["Keq_a", "V_tot_WT"]),
                unit=_mmol_mol_chl,
            ),
            "Z": Variable(
                InitialAssignment(fn=_z0, args=["Keq_a", "Keq_z", "V_tot_WT"]),
                unit=_mmol_mol_chl,
            ),
            "PV": Variable(
                InitialAssignment(fn=_pv0, args=["Keq_pv", "V_tot_WT", "P0"]),
                unit=_mmol_mol_chl,
            ),
            "PA": Variable(
                InitialAssignment(fn=_pa0, args=["Keq_pa", "Keq_a", "V_tot_WT", "P0"]),
                unit=_mmol_mol_chl,
            ),
            "PZ": Variable(
                InitialAssignment(
                    fn=_pz0, args=["Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"]
                ),
                unit=_mmol_mol_chl,
            ),
            "QV": Variable(
                InitialAssignment(fn=_qv0, args=["P0", "V_tot_WT", "Keq_pv", "Keq_qv"]),
                unit=_mmol_mol_chl,
            ),
            "QA": Variable(
                InitialAssignment(
                    fn=_qa0, args=["P0", "V_tot_WT", "Keq_pa", "Keq_a", "Keq_qa"]
                ),
                unit=_mmol_mol_chl,
            ),
            "QZ": Variable(
                InitialAssignment(
                    fn=_qz0,
                    args=["Keq_qz", "Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"],
                ),
                unit=_mmol_mol_chl,
            ),
            "QL": Variable(0, unit=_mmol_mol_chl),
            "PL": Variable(165, unit=_mmol_mol_chl),  # set from HPLC data
            "PSIId": Variable(0, unit=dimensionless),
            "alpha_VDE": Variable(
                InitialAssignment(fn=_e0, args=["k_D_VA", "k_L_VA"]), unit=dimensionless
            ),
        }
    )

    m.add_derived("PSII_active", _moiety, args=["PSII_tot", "PSIId"])
    m.add_derived(
        "P_free",
        _p_free,
        args=[
            "P_tot",
            "X_tot",
            "V",
            "A",
            "Z",
        ],
    )

    # the lifetime floor max(1e-10, ...) hides a rate unit, so set it explicitly
    m.add_derived(
        "tau_Fluo",
        _chlorophyll_fluo_lifetime,
        args=[
            "kappa_QV",
            "QV",
            "kappa_QA",
            "QA",
            "kappa_QZ",
            "QZ",
            "kappa_QL",
            "QL",
            "kappa_qZ",
            "Z",
            "kappa_qI",
            "PSIId",
            "kappa_r_nr",
        ],
        unit=second,
    )

    m.add_reaction(
        "VA",
        _mass_action_light_dark_2s,
        stoichiometry={"V": -1, "A": 1},
        args=["ppfd", "k_L_VA", "k_L_VA", "alpha_VDE", "V"],
    )
    m.add_reaction(
        "AV",
        _mass_action_1s,
        stoichiometry={"A": -1, "V": 1},
        args=["k_AV", "A"],
    )
    m.add_reaction(
        "AZ",
        _mass_action_light_dark_2s,
        stoichiometry={"A": -1, "Z": 1},
        args=["ppfd", "k_L_AZ", "k_L_AZ", "alpha_VDE", "A"],
    )
    m.add_reaction(
        "ZA",
        _mass_action_1s,
        stoichiometry={"Z": -1, "A": 1},
        args=["k_ZA", "Z"],
    )

    m.add_reaction(
        "PVf",
        _mass_action_2s,
        stoichiometry={"PV": 1, "V": -1},
        args=["k_PV_f", "V", "P_free"],
    )
    m.add_reaction(
        "PVb",
        _mass_action_1s,
        stoichiometry={"PV": -1, "V": 1},
        args=["k_PV_b", "PV"],
    )
    m.add_reaction(
        "PAf",
        _mass_action_2s,
        stoichiometry={"A": -1, "PA": 1},
        args=["k_PA_f", "A", "P_free"],
    )
    m.add_reaction(
        "PAb",
        _mass_action_1s,
        stoichiometry={"PA": -1, "A": 1},
        args=["k_PA_b", "PA"],
    )
    m.add_reaction(
        "PZf",
        _mass_action_2s,
        stoichiometry={"Z": -1, "PZ": 1},
        args=["k_PZ_f", "Z", "P_free"],
    )
    m.add_reaction(
        "PZb",
        _mass_action_1s,
        stoichiometry={"PZ": -1, "Z": 1},
        args=["k_PZ_b", "PZ"],
    )

    m.add_reaction(
        "QVf",
        _mass_action_light_dark_1s,
        stoichiometry={"PV": -1, "QV": 1},
        args=["ppfd", "k_L_QV_f", "k_D_QX_f", "PV"],
    )
    m.add_reaction(
        "QVb",
        _mass_action_1s,
        stoichiometry={"QV": -1, "PV": 1},
        args=["k_QV_b", "QV"],
    )
    m.add_reaction(
        "QAf",
        _mass_action_light_dark_1s,
        stoichiometry={"PA": -1, "QA": 1},
        args=["ppfd", "k_L_QA_f", "k_D_QX_f", "PA"],
    )
    m.add_reaction(
        "QAb",
        _mass_action_1s,
        stoichiometry={"QA": -1, "PA": 1},
        args=["k_QA_b", "QA"],
    )
    m.add_reaction(
        "QZf",
        _mass_action_light_dark_1s,
        stoichiometry={"PZ": -1, "QZ": 1},
        args=["ppfd", "k_L_QZ_f", "k_D_QX_f", "PZ"],
    )
    m.add_reaction(
        "QZb",
        _mass_action_1s,
        stoichiometry={"QZ": -1, "PZ": 1},
        # FIXME: uses k_QA_b, while k_QZ_b is only used for Keq_qz
        args=["k_QA_b", "QZ"],
    )

    m.add_reaction(
        "QLf",
        _mass_action_light_dark_1s,
        stoichiometry={"PL": -1, "QL": 1},
        args=["ppfd", "k_L_QL_f", "k_D_QX_f", "PL"],
    )
    m.add_reaction(
        "QLb",
        _mass_action_1s,
        stoichiometry={"QL": -1, "PL": 1},
        args=["k_QL_b", "QL"],
    )

    m.add_reaction(
        "damage",
        _damage,
        stoichiometry={"PSIId": 1},
        args=["ppfd", "k_L_damage", "k_D_damage", "tau_Fluo", "PSIId"],
    )

    m.add_reaction(
        "v_alpha_VDE",
        _v_alpha_vde,
        stoichiometry={"alpha_VDE": 1},
        args=["ppfd", "k_L_VDE", "k_D_VDE", "k_L_VA", "k_D_VA", "alpha_VDE"],
    )

    m.add_readout("NPQ_V", _mul, args=["kappa_QV", "QV"])
    m.add_readout("NPQ_A", _mul, args=["kappa_QA", "QA"])
    m.add_readout("NPQ_Z_qE", _mul, args=["kappa_QZ", "QZ"])
    m.add_readout("NPQ_L", _mul, args=["kappa_QL", "QL"])
    m.add_readout("NPQ_Z_qZ", _mul, args=["kappa_qZ", "Z"])
    m.add_readout("NPQ_qI", _mul, args=["kappa_qI", "PSIId"])
    return m
