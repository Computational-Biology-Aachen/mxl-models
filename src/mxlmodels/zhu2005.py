r"""Zhu et al. 2005 chlorophyll fluorescence induction model.

mxlpy implementation:

- 40 state variables
- 1 environmental forcing as a temporary parameter
- model and structural parameters
- antenna excitation, fluorescence, heat dissipation, P680/Pheo electron
  transfer, OEC S-state transitions, QA/QB redox reactions, and plastoquinone
  exchange and oxidation
"""

import math

from mxlpy import Model

EPSILON = 1.0e-30


def _identity(x: float) -> float:
    return x


def _sum3(a: float, b: float, c: float) -> float:
    return a + b + c


def _complement1(total: float, a: float) -> float:
    return total - a


def _complement3(total: float, a: float, b: float, c: float) -> float:
    return total - a - b - c


def _ratio(a: float, b: float) -> float:
    den = a + b
    return a / den if den > 0.0 else 0.0


def _divide(a: float, b: float) -> float:
    return a / b if abs(b) > EPSILON else 0.0


def _fraction3(a: float, b: float, c: float) -> float:
    den = a + b + c
    return a / den if den > 0.0 else 0.0


def _incident_peripheral(iin: float, n_psi_psii: float, x: float) -> float:
    return 220.0 * iin / ((290.0 + 200.0 * n_psi_psii) * (1.0 + x))


def _incident_core(iin: float, n_psi_psii: float, x: float) -> float:
    return 70.0 * iin / ((290.0 + 200.0 * n_psi_psii) * (1.0 + x))


def _incident_nonreducing(
    iin: float, n_psi_psii: float, x: float, chlorophylls: float
) -> float:
    return chlorophylls * x * iin / ((290.0 + 200.0 * n_psi_psii) * (1.0 + x))


def _equilibrium_ratio(
    h: float,
    c_light: float,
    k_boltzmann: float,
    temperature: float,
    lambda_chl: float,
    lambda_p680: float,
) -> float:
    # Wavelengths are stored in metres.
    exponent = (
        -h
        * c_light
        / (k_boltzmann * temperature)
        * (1.0 / lambda_chl - 1.0 / lambda_p680)
    )
    return math.exp(exponent)


def _p680_excited(
    u: float,
    p680_pheo: float,
    equilibrium_ratio: float,
    n_core_chl: float,
) -> float:
    return u * p680_pheo / ((1.0 + equilibrium_ratio) * n_core_chl)


def _q_open(qa_oxidised: float, qa_reduced: float) -> float:
    return _ratio(qa_oxidised, qa_reduced)


def _v_transfer(x: float, k: float) -> float:
    return x * k


def _v_u_dissipation(u: float, q: float, k_ud_closed: float, k_ud_open: float) -> float:
    return u * ((1.0 - q) * k_ud_closed + q * k_ud_open)


def _v_primary_charge_separation(
    p680_excited: float,
    q: float,
    connectivity: float,
    k1_open: float,
    k1_closed: float,
) -> float:
    return p680_excited * (
        q * k1_open
        + (1.0 - q) * (1.0 - connectivity) * k1_closed
        + (1.0 - q) * connectivity * k1_open
    )


def _v_charge_recombination(
    p680plus_pheominus: float,
    q: float,
    kminus1_open: float,
    kminus1_closed: float,
) -> float:
    return p680plus_pheominus * (q * kminus1_open + (1.0 - q) * kminus1_closed)


def _v_oec_donation(
    s_state: float,
    p680_state: float,
    k_z: float,
    p680_pheo_total: float,
) -> float:
    if abs(p680_pheo_total) <= EPSILON:
        return 0.0
    return s_state * k_z * p680_state / p680_pheo_total


def _v_qa_reduction(
    p680_state: float, k2: float, q: float, qb_fraction: float
) -> float:
    return p680_state * k2 * q * qb_fraction


def _v_qa_reverse(
    qa_state: float,
    p680_fraction: float,
    k2: float,
    equilibrium_constant: float,
) -> float:
    return qa_state * p680_fraction * k2 / equilibrium_constant


def _v_p680_quenching(
    antenna_excitation: float,
    p680plus_pheo: float,
    p680plus_pheominus: float,
    k_c: float,
) -> float:
    return antenna_excitation * (p680plus_pheo + p680plus_pheominus) * k_c


def _v_pq_quenching(antenna_excitation: float, pq: float, k_q: float) -> float:
    return antenna_excitation * pq * k_q


def _kq(k_f: float, k_h: float, pq_total: float) -> float:
    return 0.15 * (k_f + k_h) / pq_total


def _v_pq_exchange(
    qb_state: float, k_exchange: float, pq_state: float, pq_total: float
) -> float:
    return qb_state * k_exchange * pq_state / pq_total


def _fluorescence(ap: float, u: float, k_fa: float, k_fu: float) -> float:
    return k_fa * ap + k_fu * u


def _fluorescence_full(
    ap: float,
    u: float,
    aip: float,
    ui: float,
    uifc: float,
    k_fa: float,
    k_fu: float,
) -> float:
    # Appendix 2: F = kfa Ap + kfu(U + Ui) + kfa Aip + kfu Uifc.
    return k_fa * (ap + aip) + k_fu * (u + ui + uifc)


def _combined_reduction_fraction(
    reduced: float, oxidised: float, reduced_i: float, oxidised_i: float
) -> float:
    return _ratio(reduced + reduced_i, oxidised + oxidised_i)


def _add_nonreducing_center(m: Model, pool_size: float) -> Model:
    r"""Add the paper-described QB-nonreducing PSII population.

    Zhu et al. state that this population uses the same differential-equation
    system as the printed QB-reducing population. The duplicated system below
    follows that instruction and sets all electron-transfer rates beyond QA to
    zero, which is the defining kinetic difference of a QB-nonreducing centre. The
    duplicated PQ exchange reactions retain the paper's shared-PQ-pool topology and
    are inactive under those zero downstream rates.
    """
    m = m.add_parameters(
        {
            "P680Pheo_total_i": pool_size,
            "kAB1_i": 0.0,
            "kAB2_i": 0.0,
            "kBA1_i": 0.0,
            "kBA2_i": 0.0,
            "k3_i": 0.0,
            "kr3_i": 0.0,
        }
    )
    m = m.add_variables(
        {
            "P680plus_Pheominus_i": 0.0,
            "P680plus_Pheo_i": 0.0,
            "P680_Pheominus_i": 0.0,
            "S0T_i": 0.2 * pool_size,
            "S1T_i": 0.8 * pool_size,
            "S2T_i": 0.0,
            "S3T_i": 0.0,
            "S0Tp_i": 0.0,
            "S1Tp_i": 0.0,
            "S2Tp_i": 0.0,
            "S3Tp_i": 0.0,
            "QA_QB_i": pool_size,
            "QAred_QB_i": 0.0,
            "QA_QBred_i": 0.0,
            "QAred_QBred_i": 0.0,
            "QA_QB2red_i": 0.0,
            "QAred_QB2red_i": 0.0,
        }
    )
    m = m.add_derived(
        "P680_Pheo_i",
        fn=_complement3,
        args=[
            "P680Pheo_total_i",
            "P680plus_Pheominus_i",
            "P680plus_Pheo_i",
            "P680_Pheominus_i",
        ],
    )
    m = m.add_derived(
        "QA_oxidised_i",
        fn=_sum3,
        args=["QA_QB_i", "QA_QBred_i", "QA_QB2red_i"],
    )
    m = m.add_derived(
        "QA_reduced_i",
        fn=_sum3,
        args=["QAred_QB_i", "QAred_QBred_i", "QAred_QB2red_i"],
    )
    m = m.add_derived("q_i", fn=_q_open, args=["QA_oxidised_i", "QA_reduced_i"])
    m = m.add_derived(
        "a_QB_i", fn=_fraction3, args=["QA_QB_i", "QA_QBred_i", "QA_QB2red_i"]
    )
    m = m.add_derived(
        "b_QBred_i",
        fn=_fraction3,
        args=["QA_QBred_i", "QA_QB_i", "QA_QB2red_i"],
    )
    m = m.add_derived(
        "c_QB2red_i",
        fn=_fraction3,
        args=["QA_QB2red_i", "QA_QB_i", "QA_QBred_i"],
    )
    m = m.add_derived(
        "P680_excited_i",
        fn=_p680_excited,
        args=["Ui", "P680_Pheo_i", "equilibrium_ratio", "n_nonreducing_core_chl"],
    )
    m = m.add_derived(
        "P680plus_fraction_i",
        fn=_divide,
        args=["P680plus_Pheo_i", "P680Pheo_total_i"],
    )
    m = m.add_derived(
        "P680_ground_fraction_i",
        fn=_divide,
        args=["P680_Pheo_i", "P680Pheo_total_i"],
    )

    m = m.add_reaction(
        "vUid",
        fn=_v_u_dissipation,
        args=["Ui", "q_i", "kUd_closed", "kUd_open"],
        stoichiometry={"Ui": -1.0},
    )
    # Ui is attached to the inactive reaction centre. Aip and Uifc are
    # detached antenna pools and therefore only fluoresce or dissipate heat.
    m = m.add_reaction(
        "vP680q_Ui",
        fn=_v_p680_quenching,
        args=["Ui", "P680plus_Pheo_i", "P680plus_Pheominus_i", "k_c"],
        stoichiometry={"Ui": -1.0},
    )
    m = m.add_reaction(
        "vPQq_Ui",
        fn=_v_pq_quenching,
        args=["Ui", "PQ", "k_q"],
        stoichiometry={"Ui": -1.0},
    )

    m = m.add_reaction(
        "v1_i",
        fn=_v_primary_charge_separation,
        args=["P680_excited_i", "q_i", "p", "k1_open", "k1_closed"],
        stoichiometry={"Ui": -1.0, "P680plus_Pheominus_i": 1.0},
    )
    m = m.add_reaction(
        "vminus1_i",
        fn=_v_charge_recombination,
        args=["P680plus_Pheominus_i", "q_i", "kminus1_open", "kminus1_closed"],
        stoichiometry={"P680plus_Pheominus_i": -1.0, "Ui": 1.0},
    )

    for s in range(4):
        m = m.add_reaction(
            f"v{s}z_1_i",
            fn=_v_oec_donation,
            args=[f"S{s}T_i", "P680plus_Pheominus_i", "kz", "P680Pheo_total_i"],
            stoichiometry={
                f"S{s}T_i": -1.0,
                f"S{s}Tp_i": 1.0,
                "P680plus_Pheominus_i": -1.0,
                "P680_Pheominus_i": 1.0,
            },
        )
        m = m.add_reaction(
            f"v{s}z_2_i",
            fn=_v_oec_donation,
            args=[f"S{s}T_i", "P680plus_Pheo_i", "kz", "P680Pheo_total_i"],
            stoichiometry={
                f"S{s}T_i": -1.0,
                f"S{s}Tp_i": 1.0,
                "P680plus_Pheo_i": -1.0,
            },
        )
    for source, target, rate in (
        ("S0Tp_i", "S1T_i", "k01"),
        ("S1Tp_i", "S2T_i", "k12"),
        ("S2Tp_i", "S3T_i", "k23"),
        ("S3Tp_i", "S0T_i", "k30"),
    ):
        m = m.add_reaction(
            f"v{source}_{target}",
            fn=_v_transfer,
            args=[source, rate],
            stoichiometry={source: -1.0, target: 1.0},
        )

    qb_states = (
        ("QA_QB_i", "QAred_QB_i", "a_QB_i"),
        ("QA_QBred_i", "QAred_QBred_i", "b_QBred_i"),
        ("QA_QB2red_i", "QAred_QB2red_i", "c_QB2red_i"),
    )
    for idx, (qaox, qared, fraction) in enumerate(qb_states):
        m = m.add_reaction(
            f"v2_{idx}_1_i",
            fn=_v_qa_reduction,
            args=["P680plus_Pheominus_i", "k2", "q_i", fraction],
            stoichiometry={
                "P680plus_Pheominus_i": -1.0,
                "P680plus_Pheo_i": 1.0,
                qaox: -1.0,
                qared: 1.0,
            },
        )
        m = m.add_reaction(
            f"vr2_{idx}_1_i",
            fn=_v_qa_reverse,
            args=[qared, "P680plus_fraction_i", "k2", "Ke"],
            stoichiometry={
                "P680plus_Pheo_i": -1.0,
                "P680plus_Pheominus_i": 1.0,
                qared: -1.0,
                qaox: 1.0,
            },
        )
        m = m.add_reaction(
            f"v2_{idx}_2_i",
            fn=_v_qa_reduction,
            args=["P680_Pheominus_i", "k2", "q_i", fraction],
            stoichiometry={"P680_Pheominus_i": -1.0, qaox: -1.0, qared: 1.0},
        )
        m = m.add_reaction(
            f"vr2_{idx}_2_i",
            fn=_v_qa_reverse,
            args=[qared, "P680_ground_fraction_i", "k2", "Ke"],
            stoichiometry={"P680_Pheominus_i": 1.0, qared: -1.0, qaox: 1.0},
        )

    m = m.add_reaction(
        "vAB1_i",
        fn=_v_transfer,
        args=["QAred_QB_i", "kAB1_i"],
        stoichiometry={"QAred_QB_i": -1.0, "QA_QBred_i": 1.0},
    )
    m = m.add_reaction(
        "vBA1_i",
        fn=_v_transfer,
        args=["QA_QBred_i", "kBA1_i"],
        stoichiometry={"QA_QBred_i": -1.0, "QAred_QB_i": 1.0},
    )
    m = m.add_reaction(
        "vAB2_i",
        fn=_v_transfer,
        args=["QAred_QBred_i", "kAB2_i"],
        stoichiometry={"QAred_QBred_i": -1.0, "QA_QB2red_i": 1.0},
    )
    m = m.add_reaction(
        "vBA2_i",
        fn=_v_transfer,
        args=["QA_QB2red_i", "kBA2_i"],
        stoichiometry={"QA_QB2red_i": -1.0, "QAred_QBred_i": 1.0},
    )
    m = m.add_reaction(
        "v3_i",
        fn=_v_pq_exchange,
        args=["QA_QB2red_i", "k3_i", "PQ", "PQ_total"],
        stoichiometry={"QA_QB2red_i": -1.0, "QA_QB_i": 1.0, "PQH2": 1.0},
    )
    m = m.add_reaction(
        "vr3_i",
        fn=_v_pq_exchange,
        args=["QA_QB_i", "kr3_i", "PQH2", "PQ_total"],
        stoichiometry={"QA_QB_i": -1.0, "QA_QB2red_i": 1.0, "PQH2": -1.0},
    )
    m = m.add_reaction(
        "v3_n_i",
        fn=_v_pq_exchange,
        args=["QAred_QB2red_i", "k3_i", "PQ", "PQ_total"],
        stoichiometry={"QAred_QB2red_i": -1.0, "QAred_QB_i": 1.0, "PQH2": 1.0},
    )
    return m.add_reaction(
        "vr3_n_i",
        fn=_v_pq_exchange,
        args=["QAred_QB_i", "kr3_i", "PQH2", "PQ_total"],
        stoichiometry={"QAred_QB_i": -1.0, "QAred_QB2red_i": 1.0, "PQH2": -1.0},
    )


def get_zhu_2005() -> Model:
    r"""Return Zhu et al. 2005 chlorophyll fluorescence model."""
    m: Model = Model()

    m = m.add_parameters(
        {
            "Iin": 3000.0,
            "x": 0.0,
            "p": 0.5,
            "c_light": 3.0e8,
            "h": 6.62e-34,
            "k_boltzmann": 1.38e-23,
            "temperature": 298.0,
            "lambda_chl": 673.0e-9,
            "lambda_p680": 680.0e-9,
            "n_psi_psii": 1.0,
            "n_core_chl": 70.0,
            "n_nonreducing_peripheral_chl": 220.0,
            "n_nonreducing_core_chl": 35.0,
            "n_nonreducing_detached_chl": 35.0,
            "P680Pheo_total": 1.0,
            "PQ_total": 6.0,
            "k2": 2.0e9,
            "k3": 800.0,
            "kr3": 80.0,
            "kAB1": 2500.0,
            "kAB2": 3300.0,
            "kBA1": 175.0,
            "kBA2": 250.0,
            "kAd": 1.0e8,
            "kAf": 3.0e7,
            "kAU": 1.0e10,
            "kUA": 1.0e10,
            "kUd_closed": 1.0e8,
            "kUd_open": 0.0,
            "kUf": 3.0e7,
            "k_c": 1.0e9,
            "kminus1_closed": 9.0e8,
            "kminus1_open": 3.0e8,
            "k1_closed": 4.0e9,
            "k1_open": 2.5e10,
            "Ke": 1.0e6,
            "k01": 50.0,
            "k12": 3.0e4,
            "k23": 1.0e4,
            "k30": 3.0e3,
            "kox": 250.0,
            "kz": 5.0e6,
        }
    )

    pool_scale = 1.0

    m = m.add_variables(
        {
            "Ap": 0.0,
            "U": 0.0,
            "P680plus_Pheominus": 0.0,
            "P680plus_Pheo": 0.0,
            "P680_Pheominus": 0.0,
            "S0T": 0.2 * pool_scale,
            "S1T": 0.8 * pool_scale,
            "S2T": 0.0,
            "S3T": 0.0,
            "S0Tp": 0.0,
            "S1Tp": 0.0,
            "S2Tp": 0.0,
            "S3Tp": 0.0,
            "QA_QB": pool_scale,
            "QAred_QB": 0.0,
            "QA_QBred": 0.0,
            "QAred_QBred": 0.0,
            "QA_QB2red": 0.0,
            "QAred_QB2red": 0.0,
            "PQH2": 3.0 * pool_scale,
            # Antenna pools of the QB-nonreducing population and chlorophylls
            # detached from its reduced-size core antenna.
            "Aip": 0.0,
            "Ui": 0.0,
            "Uifc": 0.0,
        }
    )

    # Conserved and algebraic quantities.
    m = m.add_derived(
        "P680_Pheo",
        fn=_complement3,
        args=[
            "P680Pheo_total",
            "P680plus_Pheominus",
            "P680plus_Pheo",
            "P680_Pheominus",
        ],
    )
    m = m.add_derived("PQ", fn=_complement1, args=["PQ_total", "PQH2"])
    m = m.add_derived("QA_oxidised", fn=_sum3, args=["QA_QB", "QA_QBred", "QA_QB2red"])
    m = m.add_derived(
        "QA_reduced", fn=_sum3, args=["QAred_QB", "QAred_QBred", "QAred_QB2red"]
    )
    m = m.add_derived("q", fn=_q_open, args=["QA_oxidised", "QA_reduced"])
    m = m.add_derived("a_QB", fn=_fraction3, args=["QA_QB", "QA_QBred", "QA_QB2red"])
    m = m.add_derived("b_QBred", fn=_fraction3, args=["QA_QBred", "QA_QB", "QA_QB2red"])
    m = m.add_derived(
        "c_QB2red", fn=_fraction3, args=["QA_QB2red", "QA_QB", "QA_QBred"]
    )
    m = m.add_derived("Ia", fn=_incident_peripheral, args=["Iin", "n_psi_psii", "x"])
    m = m.add_derived("Ic", fn=_incident_core, args=["Iin", "n_psi_psii", "x"])
    m = m.add_derived(
        "Ai",
        fn=_incident_nonreducing,
        args=["Iin", "n_psi_psii", "x", "n_nonreducing_peripheral_chl"],
    )
    m = m.add_derived(
        "Iui",
        fn=_incident_nonreducing,
        args=["Iin", "n_psi_psii", "x", "n_nonreducing_core_chl"],
    )
    m = m.add_derived(
        "Iuif",
        fn=_incident_nonreducing,
        args=["Iin", "n_psi_psii", "x", "n_nonreducing_detached_chl"],
    )
    m = m.add_derived(
        "equilibrium_ratio",
        fn=_equilibrium_ratio,
        args=[
            "h",
            "c_light",
            "k_boltzmann",
            "temperature",
            "lambda_chl",
            "lambda_p680",
        ],
    )
    m = m.add_derived(
        "P680_excited",
        fn=_p680_excited,
        args=["U", "P680_Pheo", "equilibrium_ratio", "n_core_chl"],
    )
    m = m.add_derived("k_q", fn=_kq, args=["kUf", "kUd_closed", "PQ_total"])
    m = m.add_derived(
        "P680plus_fraction",
        fn=_divide,
        args=["P680plus_Pheo", "P680Pheo_total"],
    )
    m = m.add_derived(
        "P680_ground_fraction",
        fn=_divide,
        args=["P680_Pheo", "P680Pheo_total"],
    )

    # Antenna excitation and dissipation.
    m = m.add_reaction(
        "light_to_Ap", fn=_identity, args=["Ia"], stoichiometry={"Ap": 1.0}
    )
    m = m.add_reaction(
        "light_to_U", fn=_identity, args=["Ic"], stoichiometry={"U": 1.0}
    )
    m = m.add_reaction(
        "light_to_Aip", fn=_identity, args=["Ai"], stoichiometry={"Aip": 1.0}
    )
    m = m.add_reaction(
        "light_to_Ui", fn=_identity, args=["Iui"], stoichiometry={"Ui": 1.0}
    )
    m = m.add_reaction(
        "light_to_Uifc", fn=_identity, args=["Iuif"], stoichiometry={"Uifc": 1.0}
    )
    # Fluorescence losses from the inactive-centre antenna and fluorescence
    # plus heat loss from its detached chlorophyll pool.
    for pool, kf in (("Aip", "kAf"), ("Ui", "kUf"), ("Uifc", "kUf")):
        m = m.add_reaction(
            f"v{pool}f", fn=_v_transfer, args=[pool, kf], stoichiometry={pool: -1.0}
        )
    for pool, kd in (("Aip", "kAd"), ("Uifc", "kUd_closed")):
        m = m.add_reaction(
            f"v{pool}d", fn=_v_transfer, args=[pool, kd], stoichiometry={pool: -1.0}
        )
    m = m.add_reaction(
        "vAf", fn=_v_transfer, args=["Ap", "kAf"], stoichiometry={"Ap": -1.0}
    )
    m = m.add_reaction(
        "vAd", fn=_v_transfer, args=["Ap", "kAd"], stoichiometry={"Ap": -1.0}
    )
    m = m.add_reaction(
        "vAU", fn=_v_transfer, args=["Ap", "kAU"], stoichiometry={"Ap": -1.0, "U": 1.0}
    )
    m = m.add_reaction(
        "vUA", fn=_v_transfer, args=["U", "kUA"], stoichiometry={"U": -1.0, "Ap": 1.0}
    )
    m = m.add_reaction(
        "vUf", fn=_v_transfer, args=["U", "kUf"], stoichiometry={"U": -1.0}
    )
    m = m.add_reaction(
        "vUd",
        fn=_v_u_dissipation,
        args=["U", "q", "kUd_closed", "kUd_open"],
        stoichiometry={"U": -1.0},
    )
    for antenna in ("Ap", "U"):
        suffix = "A" if antenna == "Ap" else "U"
        m = m.add_reaction(
            f"vP680q{suffix}",
            fn=_v_p680_quenching,
            args=[antenna, "P680plus_Pheo", "P680plus_Pheominus", "k_c"],
            stoichiometry={antenna: -1.0},
        )
        m = m.add_reaction(
            f"vPQq{suffix}",
            fn=_v_pq_quenching,
            args=[antenna, "PQ", "k_q"],
            stoichiometry={antenna: -1.0},
        )

    # Primary charge separation and recombination.
    m = m.add_reaction(
        "v1",
        fn=_v_primary_charge_separation,
        args=["P680_excited", "q", "p", "k1_open", "k1_closed"],
        stoichiometry={"U": -1.0, "P680plus_Pheominus": 1.0},
    )
    m = m.add_reaction(
        "vminus1",
        fn=_v_charge_recombination,
        args=["P680plus_Pheominus", "q", "kminus1_open", "kminus1_closed"],
        stoichiometry={"P680plus_Pheominus": -1.0, "U": 1.0},
    )

    # OEC electron donation. The first set reduces P680+Pheo-; the second
    # reduces P680+Pheo. Each donation moves SnT to SnTp.
    for s in range(4):
        m = m.add_reaction(
            f"v{s}z_1",
            fn=_v_oec_donation,
            args=[f"S{s}T", "P680plus_Pheominus", "kz", "P680Pheo_total"],
            stoichiometry={
                f"S{s}T": -1.0,
                f"S{s}Tp": 1.0,
                "P680plus_Pheominus": -1.0,
                "P680_Pheominus": 1.0,
            },
        )
        m = m.add_reaction(
            f"v{s}z_2",
            fn=_v_oec_donation,
            args=[f"S{s}T", "P680plus_Pheo", "kz", "P680Pheo_total"],
            stoichiometry={f"S{s}T": -1.0, f"S{s}Tp": 1.0, "P680plus_Pheo": -1.0},
        )

    for source, target, rate in (
        ("S0Tp", "S1T", "k01"),
        ("S1Tp", "S2T", "k12"),
        ("S2Tp", "S3T", "k23"),
        ("S3Tp", "S0T", "k30"),
    ):
        m = m.add_reaction(
            f"v{source}_{target}",
            fn=_v_transfer,
            args=[source, rate],
            stoichiometry={source: -1.0, target: 1.0},
        )

    # QA reduction and reverse transfer, split by the redox state of QB.
    qb_states = (
        ("QA_QB", "QAred_QB", "a_QB"),
        ("QA_QBred", "QAred_QBred", "b_QBred"),
        ("QA_QB2red", "QAred_QB2red", "c_QB2red"),
    )
    for idx, (qaox, qared, fraction) in enumerate(qb_states):
        m = m.add_reaction(
            f"v2_{idx}_1",
            fn=_v_qa_reduction,
            args=["P680plus_Pheominus", "k2", "q", fraction],
            stoichiometry={
                "P680plus_Pheominus": -1.0,
                "P680plus_Pheo": 1.0,
                qaox: -1.0,
                qared: 1.0,
            },
        )
        m = m.add_reaction(
            f"vr2_{idx}_1",
            fn=_v_qa_reverse,
            args=[qared, "P680plus_fraction", "k2", "Ke"],
            stoichiometry={
                "P680plus_Pheo": -1.0,
                "P680plus_Pheominus": 1.0,
                qared: -1.0,
                qaox: 1.0,
            },
        )
        m = m.add_reaction(
            f"v2_{idx}_2",
            fn=_v_qa_reduction,
            args=["P680_Pheominus", "k2", "q", fraction],
            stoichiometry={"P680_Pheominus": -1.0, qaox: -1.0, qared: 1.0},
        )
        m = m.add_reaction(
            f"vr2_{idx}_2",
            fn=_v_qa_reverse,
            args=[qared, "P680_ground_fraction", "k2", "Ke"],
            stoichiometry={"P680_Pheominus": 1.0, qared: -1.0, qaox: 1.0},
        )

    # Sequential QB reduction and reversible electron transfer.
    m = m.add_reaction(
        "vAB1",
        fn=_v_transfer,
        args=["QAred_QB", "kAB1"],
        stoichiometry={"QAred_QB": -1.0, "QA_QBred": 1.0},
    )
    m = m.add_reaction(
        "vBA1",
        fn=_v_transfer,
        args=["QA_QBred", "kBA1"],
        stoichiometry={"QA_QBred": -1.0, "QAred_QB": 1.0},
    )
    m = m.add_reaction(
        "vAB2",
        fn=_v_transfer,
        args=["QAred_QBred", "kAB2"],
        stoichiometry={"QAred_QBred": -1.0, "QA_QB2red": 1.0},
    )
    m = m.add_reaction(
        "vBA2",
        fn=_v_transfer,
        args=["QA_QB2red", "kBA2"],
        stoichiometry={"QA_QB2red": -1.0, "QAred_QBred": 1.0},
    )

    # Exchange of QBH2 with the shared PQ pool.
    m = m.add_reaction(
        "v3",
        fn=_v_pq_exchange,
        args=["QA_QB2red", "k3", "PQ", "PQ_total"],
        stoichiometry={"QA_QB2red": -1.0, "QA_QB": 1.0, "PQH2": 1.0},
    )
    m = m.add_reaction(
        "vr3",
        fn=_v_pq_exchange,
        args=["QA_QB", "kr3", "PQH2", "PQ_total"],
        stoichiometry={"QA_QB": -1.0, "QA_QB2red": 1.0, "PQH2": -1.0},
    )
    m = m.add_reaction(
        "v3_n",
        fn=_v_pq_exchange,
        args=["QAred_QB2red", "k3", "PQ", "PQ_total"],
        stoichiometry={"QAred_QB2red": -1.0, "QAred_QB": 1.0, "PQH2": 1.0},
    )
    m = m.add_reaction(
        "vr3_n",
        fn=_v_pq_exchange,
        args=["QAred_QB", "kr3", "PQH2", "PQ_total"],
        stoichiometry={"QAred_QB": -1.0, "QAred_QB2red": 1.0, "PQH2": -1.0},
    )
    m = _add_nonreducing_center(m, 0.0)
    m = m.add_reaction(
        "v_pq_ox", fn=_v_transfer, args=["PQH2", "kox"], stoichiometry={"PQH2": -1.0}
    )

    m = m.add_readout(
        "F",
        fn=_fluorescence_full,
        args=["Ap", "U", "Aip", "Ui", "Uifc", "kAf", "kUf"],
    )
    m = m.add_readout(
        "QA_reduction_fraction", fn=_ratio, args=["QA_reduced", "QA_oxidised"]
    )
    m = m.add_readout(
        "QA_reduction_fraction_i",
        fn=_ratio,
        args=["QA_reduced_i", "QA_oxidised_i"],
    )
    m = m.add_readout(
        "QA_reduction_fraction_total",
        fn=_combined_reduction_fraction,
        args=["QA_reduced", "QA_oxidised", "QA_reduced_i", "QA_oxidised_i"],
    )
    observed = (
        "q",
        "q_i",
        "QA_oxidised",
        "QA_reduced",
        "QA_oxidised_i",
        "QA_reduced_i",
        "PQ",
        "P680_excited",
        "P680_excited_i",
        "Ia",
        "Ic",
        "Ai",
        "Iui",
        "Iuif",
    )
    for name in observed:
        m = m.add_readout(f"obs_{name}", fn=_identity, args=[name])
    return m
