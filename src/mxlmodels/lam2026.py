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

The model provide a detailed, up-to-date descriptions regarding NPQ compoents within higher plants, along with detailed dataset of fluorescence lifetime and pigment qunatification from mutants that
lacks one or more compoents of NPQ. Combined with a rigorous fitting and optimization schedule for parameters, the model brings up quantified contributions of different NPQ components and how it changes overtime.

Compoents: energy-depedent quenching (qE), zeaxanthin-depdent quenching (qZ)
and photoinhibition (qI)
"""

from __future__ import annotations

from mxlpy import InitialAssignment, Model

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
    kf_light: float, kr_light: float, kf_dark: float, kr_dark: float, ppfd: float
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
    """Get Lam 2026 model."""
    m = Model()
    m.add_parameters(
        {
            "k_L_VA": 2.47,
            "k_D_VA": 0.014,
            "k_L_AZ": 0.5,
            "k_AV": 1.12,
            "k_ZA": 0.07,
            "k_PV_f": 2.18,
            "k_PV_b": 9.43,
            "k_PA_f": 130,
            "k_PA_b": 254,
            "k_PZ_f": 295,
            "k_PZ_b": 126,
            "k_L_QV_f": 0.027,
            "k_D_QV_f": 0,
            "k_QV_b": 0.066,
            "k_L_QA_f": 0.66,
            "k_D_QA_f": 0,
            "k_QA_b": 8.57,
            "k_L_QZ_f": 0.56,
            "k_D_QZ_f": 0,
            "k_QZ_b": 1.22,
            "k_L_QL_f": 0.056,
            "k_QL_b": 3.68,
            "k_D_QX_f": 0,  # for all QX complex the rate in the dark is set to 0
            "k_L_damage": 0.0222,
            "k_D_damage": 0.0161,
            "k_D_VDE": 0.24,
            "k_L_VDE": 0.28,
            "k_AV_aba1": 0.006,
            "k_ZA_aba1": 0.038,
            "k_PV_f_lut2": 1.43,
            "k_PV_b_lut2": 13.1,
            "k_PA_f_lut2": 34.4,
            "k_PA_b_lut2": 294,
            "k_PZ_f_lut2": 74.1,
            "k_PZ_b_lut2": 168,
            "V_tot_npq1": 49.8,
            "V_tot_lut2": 71.2,
            "V_tot_npq4": 40.6,
            "V_tot_aba1": 10.7,
            "V_tot_WT": 35.9,
            "P_tot": 45.4,
            "P_tot_lut2": 49.9,
            "kappa_QV": 0.040,
            "kappa_QA": 0.174,
            "kappa_QZ": 0.177,
            "kappa_QL": 0.262,
            "kappa_qZ": 0.030,
            "kappa_qI": 3.86,
            "kappa_qI_double_mut": 7.05,
            "PPFD": 0,
            "tau_0": 1.73089079100000,  # from the paper WT tau_0 for 5-10-5 dataset
            "PSII_tot": 1,
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
        args=["k_D_VA", "k_AV", "k_D_VA", "k_AV", "PPFD"],
    )
    m.add_derived(
        "Keq_z",
        _keq_light_dark,
        args=["k_D_AZ", "k_ZA", "k_D_AZ", "k_ZA", "PPFD"],
    )
    m.add_derived(
        "Keq_qv",
        _keq_light_dark,
        args=["k_L_QV_f", "k_QV_b", "k_D_QV_f", "k_QV_b", "PPFD"],
    )
    m.add_derived(
        "Keq_qa",
        _keq_light_dark,
        args=["k_L_QA_f", "k_QA_b", "k_D_QA_f", "k_QA_b", "PPFD"],
    )
    m.add_derived(
        "Keq_qz",
        _keq_light_dark,
        args=["k_L_QZ_f", "k_QZ_b", "k_D_QZ_f", "k_QZ_b", "PPFD"],
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
            "V": InitialAssignment(fn=_same, args=["V_tot_WT"]),
            "A": InitialAssignment(fn=_a0, args=["Keq_a", "V_tot_WT"]),
            "Z": InitialAssignment(fn=_z0, args=["Keq_a", "Keq_z", "V_tot_WT"]),
            "PV": InitialAssignment(fn=_pv0, args=["Keq_pv", "V_tot_WT", "P0"]),
            "PA": InitialAssignment(
                fn=_pa0, args=["Keq_pa", "Keq_a", "V_tot_WT", "P0"]
            ),
            "PZ": InitialAssignment(
                fn=_pz0, args=["Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"]
            ),
            "QV": InitialAssignment(
                fn=_qv0, args=["P0", "V_tot_WT", "Keq_pv", "Keq_qv"]
            ),
            "QA": InitialAssignment(
                fn=_qa0, args=["P0", "V_tot_WT", "Keq_pa", "Keq_a", "Keq_qa"]
            ),
            "QZ": InitialAssignment(
                fn=_qz0, args=["Keq_qz", "Keq_pz", "Keq_z", "Keq_a", "P0", "V_tot_WT"]
            ),
            "QL": 0,
            "PL": 165,  # set from HPLC data
            "PSIId": 0,
            "alpha_VDE": InitialAssignment(fn=_e0, args=["k_D_VA", "k_L_VA"]),
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
    )

    m.add_reaction(
        "VA",
        _mass_action_light_dark_2s,
        stoichiometry={"V": -1, "A": 1},
        args=["PPFD", "k_L_VA", "k_L_VA", "alpha_VDE", "V"],
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
        args=["PPFD", "k_L_AZ", "k_L_AZ", "alpha_VDE", "A"],
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
        args=["PPFD", "k_L_QV_f", "k_D_QX_f", "PV"],
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
        args=["PPFD", "k_L_QA_f", "k_D_QX_f", "PA"],
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
        args=["PPFD", "k_L_QZ_f", "k_D_QX_f", "PZ"],
    )
    m.add_reaction(
        "QZb",
        _mass_action_1s,
        stoichiometry={"QZ": -1, "PZ": 1},
        args=["k_QA_b", "QZ"],
    )

    m.add_reaction(
        "QLf",
        _mass_action_light_dark_1s,
        stoichiometry={"PL": -1, "QL": 1},
        args=["PPFD", "k_L_QL_f", "k_D_QX_f", "PL"],
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
        args=["PPFD", "k_L_damage", "k_D_damage", "tau_Fluo", "PSIId"],
    )

    m.add_reaction(
        "v_alpha_VDE",
        _v_alpha_vde,
        stoichiometry={"alpha_VDE": 1},
        args=["PPFD", "k_L_VDE", "k_D_VDE", "k_L_VA", "k_D_VA", "alpha_VDE"],
    )

    m.add_readout("NPQ_V", _mul, args=["kappa_QV", "QV"])
    m.add_readout("NPQ_A", _mul, args=["kappa_QA", "QA"])
    m.add_readout("NPQ_Z_qE", _mul, args=["kappa_QZ", "QZ"])
    m.add_readout("NPQ_L", _mul, args=["kappa_QL", "QL"])
    m.add_readout("NPQ_Z_qZ", _mul, args=["kappa_qZ", "Z"])
    m.add_readout("NPQ_qI", _mul, args=["kappa_qI", "PSIId"])
    return m
