"""Poolman 2000 Calvin-Benson-Bassham cycle model."""

from mxlpy import Model


def _moiety_1(concentration: float, total: float) -> float:
    return total - concentration


def _pi_cbb(
    phosphate_total: float,
    pga: float,
    bpga: float,
    gap: float,
    dhap: float,
    fbp: float,
    f6p: float,
    g6p: float,
    g1p: float,
    sbp: float,
    s7p: float,
    e4p: float,
    x5p: float,
    r5p: float,
    rubp: float,
    ru5p: float,
    atp: float,
) -> float:
    return phosphate_total - (
        pga
        + 2 * bpga
        + gap
        + dhap
        + 2 * fbp
        + f6p
        + g6p
        + g1p
        + 2 * sbp
        + s7p
        + e4p
        + x5p
        + r5p
        + 2 * rubp
        + ru5p
        + atp
    )


def _mass_action_1s(s1: float, k_fwd: float) -> float:
    return k_fwd * s1


def _rate_translocator(
    pi: float,
    pga: float,
    gap: float,
    dhap: float,
    k_pxt: float,
    p_ext: float,
    k_pi: float,
    k_pga: float,
    k_gap: float,
    k_dhap: float,
) -> float:
    return 1 + (1 + k_pxt / p_ext) * (
        pi / k_pi + pga / k_pga + gap / k_gap + dhap / k_dhap
    )


def _rate_poolman_5i(
    rubp: float,
    pga: float,
    co2: float,
    vmax: float,
    kms_rubp: float,
    kms_co2: float,
    # inhibitors
    ki_pga: float,
    fbp: float,
    ki_fbp: float,
    sbp: float,
    ki_sbp: float,
    pi: float,
    ki_p: float,
    nadph: float,
    ki_nadph: float,
) -> float:
    top = vmax * rubp * co2
    btm = (
        rubp
        + kms_rubp
        * (
            1
            + pga / ki_pga
            + fbp / ki_fbp
            + sbp / ki_sbp
            + pi / ki_p
            + nadph / ki_nadph
        )
    ) * (co2 + kms_co2)
    return top / btm


def _rapid_equilibrium_2s_2p(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 * s2 - p1 * p2 / q)


def _rapid_equilibrium_3s_3p(
    s1: float,
    s2: float,
    s3: float,
    p1: float,
    p2: float,
    p3: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 * s2 * s3 - p1 * p2 * p3 / q)


def _rapid_equilibrium_1s_1p(
    s1: float,
    p1: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 - p1 / q)


def _rapid_equilibrium_2s_1p(
    s1: float,
    s2: float,
    p1: float,
    k_re: float,
    q: float,
) -> float:
    return k_re * (s1 * s2 - p1 / q)


def _michaelis_menten_1s_2i(
    s: float,
    i1: float,
    i2: float,
    vmax: float,
    km: float,
    ki1: float,
    ki2: float,
) -> float:
    return vmax * s / (s + km * (1 + i1 / ki1 + i2 / ki2))


def _michaelis_menten_1s_1i(
    s: float,
    i: float,
    vmax: float,
    km: float,
    ki: float,
) -> float:
    return vmax * s / (s + km * (1 + i / ki))


def _rate_prk(
    ru5p: float,
    atp: float,
    pi: float,
    pga: float,
    rubp: float,
    adp: float,
    v13: float,
    km131: float,
    km132: float,
    ki131: float,
    ki132: float,
    ki133: float,
    ki134: float,
    ki135: float,
) -> float:
    return (
        v13
        * ru5p
        * atp
        / (
            (ru5p + km131 * (1 + pga / ki131 + rubp / ki132 + pi / ki133))
            * (atp * (1 + adp / ki134) + km132 * (1 + adp / ki135))
        )
    )


def _rate_out(
    s1: float,
    n_total: float,
    vmax_efflux: float,
    k_efflux: float,
) -> float:
    return vmax_efflux * s1 / (n_total * k_efflux)


def _rate_starch(
    g1p: float,
    atp: float,
    adp: float,
    pi: float,
    pga: float,
    f6p: float,
    fbp: float,
    v_st: float,
    kmst1: float,
    kmst2: float,
    ki_st: float,
    kast1: float,
    kast2: float,
    kast3: float,
) -> float:
    return (
        v_st
        * g1p
        * atp
        / (
            (g1p + kmst1)
            * (
                (1 + adp / ki_st) * (atp + kmst2)
                + kmst2 * pi / (kast1 * pga + kast2 * f6p + kast3 * fbp)
            )
        )
    )


def _rate_atp_synthase_2000(
    adp: float,
    pi: float,
    v16: float,
    km161: float,
    km162: float,
) -> float:
    return v16 * adp * pi / ((adp + km161) * (pi + km162))


def create_model() -> Model:
    """Build the Poolman 2000 Calvin-Benson-Bassham cycle model."""
    return (
        Model()
        .add_variable("3PGA", initial_value=0.6387788347932627)
        .add_variable("BPGA", initial_value=0.0013570885908749779)
        .add_variable("GAP", initial_value=0.011259431827358068)
        .add_variable("DHAP", initial_value=0.24770748227012374)
        .add_variable("FBP", initial_value=0.01980222074817044)
        .add_variable("F6P", initial_value=1.093666906864421)
        .add_variable("G6P", initial_value=2.5154338857582377)
        .add_variable("G1P", initial_value=0.14589516537322303)
        .add_variable("SBP", initial_value=0.09132688566151095)
        .add_variable("S7P", initial_value=0.23281380022778891)
        .add_variable("E4P", initial_value=0.02836065066520614)
        .add_variable("X5P", initial_value=0.03647242425941113)
        .add_variable("R5P", initial_value=0.06109130988031577)
        .add_variable("RUBP", initial_value=0.2672164362349537)
        .add_variable("RU5P", initial_value=0.0244365238237522)
        .add_variable("atp", initial_value=0.43633201706180874)
        .add_parameter("CO2 (dissolved)", value=0.2)
        .add_parameter("nadph", value=0.21)
        .add_parameter("protons", value=1.2589254117941661e-05)
        .add_parameter("A*P", value=0.5)
        .add_parameter("NADP*", value=0.5)
        .add_parameter("Pi_tot", value=15.0)
        .add_parameter("E0_rubisco_carboxylase", value=1.0)
        .add_parameter("kcat_rubisco_carboxylase", value=2.72)
        .add_parameter("km_rubisco_carboxylase_RUBP", value=0.02)
        .add_parameter("km_rubisco_carboxylase_CO2 (dissolved)", value=0.0107)
        .add_parameter("ki_rubisco_carboxylase_3PGA", value=0.04)
        .add_parameter("ki_rubisco_carboxylase_FBP", value=0.04)
        .add_parameter("ki_rubisco_carboxylase_SBP", value=0.075)
        .add_parameter("ki_rubisco_carboxylase_pi", value=0.9)
        .add_parameter("ki_rubisco_carboxylase_nadph", value=0.07)
        .add_parameter("kre_phosphoglycerate_kinase", value=800000000.0)
        .add_parameter("keq_phosphoglycerate_kinase", value=0.00031)
        .add_parameter("kre_gadph", value=800000000.0)
        .add_parameter("keq_gadph", value=16000000.0)
        .add_parameter("kre_triose_phosphate_isomerase", value=800000000.0)
        .add_parameter("keq_triose_phosphate_isomerase", value=22.0)
        .add_parameter("kre_aldolase_dhap_gap", value=800000000.0)
        .add_parameter("keq_aldolase_dhap_gap", value=7.1)
        .add_parameter("kre_aldolase_dhap_e4p", value=800000000.0)
        .add_parameter("keq_aldolase_dhap_e4p", value=13.0)
        .add_parameter("E0_fbpase", value=1.0)
        .add_parameter("kcat_fbpase", value=1.6)
        .add_parameter("km_fbpase_s", value=0.03)
        .add_parameter("ki_fbpase_F6P", value=0.7)
        .add_parameter("ki_fbpase_pi", value=12.0)
        .add_parameter("kre_transketolase_gap_f6p", value=800000000.0)
        .add_parameter("keq_transketolase_gap_f6p", value=0.084)
        .add_parameter("kre_transketolase_gap_s7p", value=800000000.0)
        .add_parameter("keq_transketolase_gap_s7p", value=0.85)
        .add_parameter("E0_SBPase", value=1.0)
        .add_parameter("kcat_SBPase", value=0.32)
        .add_parameter("km_SBPase_s", value=0.013)
        .add_parameter("ki_SBPase_pi", value=12.0)
        .add_parameter("kre_ribose_phosphate_isomerase", value=800000000.0)
        .add_parameter("keq_ribose_phosphate_isomerase", value=0.4)
        .add_parameter("kre_ribulose_phosphate_epimerase", value=800000000.0)
        .add_parameter("keq_ribulose_phosphate_epimerase", value=0.67)
        .add_parameter("E0_phosphoribulokinase", value=1.0)
        .add_parameter("kcat_phosphoribulokinase", value=7.9992)
        .add_parameter("km_phosphoribulokinase_RU5P", value=0.05)
        .add_parameter("km_phosphoribulokinase_atp", value=0.05)
        .add_parameter("ki_phosphoribulokinase_3PGA", value=2.0)
        .add_parameter("ki_phosphoribulokinase_RUBP", value=0.7)
        .add_parameter("ki_phosphoribulokinase_pi", value=4.0)
        .add_parameter("ki_phosphoribulokinase_4", value=2.5)
        .add_parameter("ki_phosphoribulokinase_5", value=0.4)
        .add_parameter("kre_g6pi", value=800000000.0)
        .add_parameter("keq_g6pi", value=2.3)
        .add_parameter("kre_phosphoglucomutase", value=800000000.0)
        .add_parameter("keq_phosphoglucomutase", value=0.058)
        .add_parameter("pi_ext", value=0.5)
        .add_parameter("km_ex_pga", value=0.25)
        .add_parameter("km_ex_gap", value=0.075)
        .add_parameter("km_ex_dhap", value=0.077)
        .add_parameter("km_N_translocator_pi_ext", value=0.74)
        .add_parameter("km_N_translocator_pi", value=0.63)
        .add_parameter("kcat_N_translocator", value=2.0)
        .add_parameter("E0_N_translocator", value=1.0)
        .add_parameter("km_ex_g1p_G1P", value=0.08)
        .add_parameter("km_ex_g1p_atp", value=0.08)
        .add_parameter("ki_ex_g1p", value=10.0)
        .add_parameter("ki_ex_g1p_3PGA", value=0.1)
        .add_parameter("ki_ex_g1p_F6P", value=0.02)
        .add_parameter("ki_ex_g1p_FBP", value=0.02)
        .add_parameter("E0_ex_g1p", value=1.0)
        .add_parameter("kcat_ex_g1p", value=0.32)
        .add_parameter("km_atp_synthase_adp", value=0.014)
        .add_parameter("km_atp_synthase_pi", value=0.3)
        .add_parameter("kcat_atp_synthase", value=2.8)
        .add_parameter("E0_atp_synthase", value=1.0)
        .add_derived(
            "adp",
            fn=_moiety_1,
            args=["atp", "A*P"],
        )
        .add_derived(
            "nadp",
            fn=_moiety_1,
            args=["nadph", "NADP*"],
        )
        .add_derived(
            "pi",
            fn=_pi_cbb,
            args=[
                "Pi_tot",
                "3PGA",
                "BPGA",
                "GAP",
                "DHAP",
                "FBP",
                "F6P",
                "G6P",
                "G1P",
                "SBP",
                "S7P",
                "E4P",
                "X5P",
                "R5P",
                "RUBP",
                "RU5P",
                "atp",
            ],
        )
        .add_derived(
            "vmax_rubisco_carboxylase",
            fn=_mass_action_1s,
            args=["kcat_rubisco_carboxylase", "E0_rubisco_carboxylase"],
        )
        .add_derived(
            "vmax_fbpase",
            fn=_mass_action_1s,
            args=["kcat_fbpase", "E0_fbpase"],
        )
        .add_derived(
            "vmax_SBPase",
            fn=_mass_action_1s,
            args=["kcat_SBPase", "E0_SBPase"],
        )
        .add_derived(
            "vmax_phosphoribulokinase",
            fn=_mass_action_1s,
            args=["kcat_phosphoribulokinase", "E0_phosphoribulokinase"],
        )
        .add_derived(
            "vmax_ex_pga",
            fn=_mass_action_1s,
            args=["kcat_N_translocator", "E0_N_translocator"],
        )
        .add_derived(
            "N_translocator",
            fn=_rate_translocator,
            args=[
                "pi",
                "3PGA",
                "GAP",
                "DHAP",
                "km_N_translocator_pi_ext",
                "pi_ext",
                "km_N_translocator_pi",
                "km_ex_pga",
                "km_ex_gap",
                "km_ex_dhap",
            ],
        )
        .add_derived(
            "vmax_ex_g1p",
            fn=_mass_action_1s,
            args=["kcat_ex_g1p", "E0_ex_g1p"],
        )
        .add_derived(
            "vmax_atp_synthase",
            fn=_mass_action_1s,
            args=["kcat_atp_synthase", "E0_atp_synthase"],
        )
        .add_reaction(
            "rubisco_carboxylase",
            fn=_rate_poolman_5i,
            args=[
                "RUBP",
                "3PGA",
                "CO2 (dissolved)",
                "vmax_rubisco_carboxylase",
                "km_rubisco_carboxylase_RUBP",
                "km_rubisco_carboxylase_CO2 (dissolved)",
                "ki_rubisco_carboxylase_3PGA",
                "FBP",
                "ki_rubisco_carboxylase_FBP",
                "SBP",
                "ki_rubisco_carboxylase_SBP",
                "pi",
                "ki_rubisco_carboxylase_pi",
                "nadph",
                "ki_rubisco_carboxylase_nadph",
            ],
            stoichiometry={"RUBP": -1.0, "3PGA": 2.0},
        )
        .add_reaction(
            "phosphoglycerate_kinase",
            fn=_rapid_equilibrium_2s_2p,
            args=[
                "3PGA",
                "atp",
                "BPGA",
                "adp",
                "kre_phosphoglycerate_kinase",
                "keq_phosphoglycerate_kinase",
            ],
            stoichiometry={"3PGA": -1.0, "atp": -1.0, "BPGA": 1.0},
        )
        .add_reaction(
            "gadph",
            fn=_rapid_equilibrium_3s_3p,
            args=[
                "BPGA",
                "nadph",
                "protons",
                "GAP",
                "nadp",
                "pi",
                "kre_gadph",
                "keq_gadph",
            ],
            stoichiometry={"BPGA": -1.0, "GAP": 1.0},
        )
        .add_reaction(
            "triose_phosphate_isomerase",
            fn=_rapid_equilibrium_1s_1p,
            args=[
                "GAP",
                "DHAP",
                "kre_triose_phosphate_isomerase",
                "keq_triose_phosphate_isomerase",
            ],
            stoichiometry={"GAP": -1, "DHAP": 1},
        )
        .add_reaction(
            "aldolase_dhap_gap",
            fn=_rapid_equilibrium_2s_1p,
            args=[
                "GAP",
                "DHAP",
                "FBP",
                "kre_aldolase_dhap_gap",
                "keq_aldolase_dhap_gap",
            ],
            stoichiometry={"GAP": -1, "DHAP": -1, "FBP": 1},
        )
        .add_reaction(
            "aldolase_dhap_e4p",
            fn=_rapid_equilibrium_2s_1p,
            args=[
                "DHAP",
                "E4P",
                "SBP",
                "kre_aldolase_dhap_e4p",
                "keq_aldolase_dhap_e4p",
            ],
            stoichiometry={"DHAP": -1, "E4P": -1, "SBP": 1},
        )
        .add_reaction(
            "fbpase",
            fn=_michaelis_menten_1s_2i,
            args=[
                "FBP",
                "F6P",
                "pi",
                "vmax_fbpase",
                "km_fbpase_s",
                "ki_fbpase_F6P",
                "ki_fbpase_pi",
            ],
            stoichiometry={"FBP": -1, "F6P": 1},
        )
        .add_reaction(
            "transketolase_gap_f6p",
            fn=_rapid_equilibrium_2s_2p,
            args=[
                "GAP",
                "F6P",
                "E4P",
                "X5P",
                "kre_transketolase_gap_f6p",
                "keq_transketolase_gap_f6p",
            ],
            stoichiometry={"GAP": -1, "F6P": -1, "E4P": 1, "X5P": 1},
        )
        .add_reaction(
            "transketolase_gap_s7p",
            fn=_rapid_equilibrium_2s_2p,
            args=[
                "GAP",
                "S7P",
                "R5P",
                "X5P",
                "kre_transketolase_gap_s7p",
                "keq_transketolase_gap_s7p",
            ],
            stoichiometry={"GAP": -1, "S7P": -1, "R5P": 1, "X5P": 1},
        )
        .add_reaction(
            "SBPase",
            fn=_michaelis_menten_1s_1i,
            args=["SBP", "pi", "vmax_SBPase", "km_SBPase_s", "ki_SBPase_pi"],
            stoichiometry={"SBP": -1, "S7P": 1},
        )
        .add_reaction(
            "ribose_phosphate_isomerase",
            fn=_rapid_equilibrium_1s_1p,
            args=[
                "R5P",
                "RU5P",
                "kre_ribose_phosphate_isomerase",
                "keq_ribose_phosphate_isomerase",
            ],
            stoichiometry={"R5P": -1, "RU5P": 1},
        )
        .add_reaction(
            "ribulose_phosphate_epimerase",
            fn=_rapid_equilibrium_1s_1p,
            args=[
                "X5P",
                "RU5P",
                "kre_ribulose_phosphate_epimerase",
                "keq_ribulose_phosphate_epimerase",
            ],
            stoichiometry={"X5P": -1, "RU5P": 1},
        )
        .add_reaction(
            "phosphoribulokinase",
            fn=_rate_prk,
            args=[
                "RU5P",
                "atp",
                "pi",
                "3PGA",
                "RUBP",
                "adp",
                "vmax_phosphoribulokinase",
                "km_phosphoribulokinase_RU5P",
                "km_phosphoribulokinase_atp",
                "ki_phosphoribulokinase_3PGA",
                "ki_phosphoribulokinase_RUBP",
                "ki_phosphoribulokinase_pi",
                "ki_phosphoribulokinase_4",
                "ki_phosphoribulokinase_5",
            ],
            stoichiometry={"RU5P": -1.0, "atp": -1.0, "RUBP": 1.0},
        )
        .add_reaction(
            "g6pi",
            fn=_rapid_equilibrium_1s_1p,
            args=["F6P", "G6P", "kre_g6pi", "keq_g6pi"],
            stoichiometry={"F6P": -1, "G6P": 1},
        )
        .add_reaction(
            "phosphoglucomutase",
            fn=_rapid_equilibrium_1s_1p,
            args=["G6P", "G1P", "kre_phosphoglucomutase", "keq_phosphoglucomutase"],
            stoichiometry={"G6P": -1, "G1P": 1},
        )
        .add_reaction(
            "ex_pga",
            fn=_rate_out,
            args=["3PGA", "N_translocator", "vmax_ex_pga", "km_ex_pga"],
            stoichiometry={"3PGA": -1},
        )
        .add_reaction(
            "ex_gap",
            fn=_rate_out,
            args=["GAP", "N_translocator", "vmax_ex_pga", "km_ex_gap"],
            stoichiometry={"GAP": -1},
        )
        .add_reaction(
            "ex_dhap",
            fn=_rate_out,
            args=["DHAP", "N_translocator", "vmax_ex_pga", "km_ex_dhap"],
            stoichiometry={"DHAP": -1},
        )
        .add_reaction(
            "ex_g1p",
            fn=_rate_starch,
            args=[
                "G1P",
                "atp",
                "adp",
                "pi",
                "3PGA",
                "F6P",
                "FBP",
                "vmax_ex_g1p",
                "km_ex_g1p_G1P",
                "km_ex_g1p_atp",
                "ki_ex_g1p",
                "ki_ex_g1p_3PGA",
                "ki_ex_g1p_F6P",
                "ki_ex_g1p_FBP",
            ],
            stoichiometry={"G1P": -1.0, "atp": -1.0},
        )
        .add_reaction(
            "atp_synthase",
            fn=_rate_atp_synthase_2000,
            args=[
                "adp",
                "pi",
                "vmax_atp_synthase",
                "km_atp_synthase_adp",
                "km_atp_synthase_pi",
            ],
            stoichiometry={"atp": 1.0},
        )
    )
