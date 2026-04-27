"""Yokota 1985 photorespiratory carbon oxidation cycle model."""

from mxlpy import Model


def _value(
    x: float,
) -> float:
    return x


def _mass_action_1s(
    s1: float,
    k_fwd: float,
) -> float:
    return k_fwd * s1


def _michaelis_menten_1s(
    s: float,
    vmax: float,
    km: float,
) -> float:
    return vmax * s / (km + s)


def _ping_pong_bi_bi(
    s1: float,
    s2: float,
    vmax: float,
    km_s1: float,
    km_s2: float,
) -> float:
    return vmax * s1 * s2 / (1 / (km_s1 * km_s2) + s1 / km_s1 + s2 / km_s2 + s1 * s2)


def create_model() -> Model:
    """Build the Yokota 1985 photorespiratory carbon oxidation cycle model."""
    return (
        Model()
        .add_variable(
            "glycolate",
            initial_value=0.09,
        )
        .add_variable(
            "glyoxylate",
            initial_value=0.7964601770483386,
        )
        .add_variable(
            "glycine",
            initial_value=8.999999999424611,
        )
        .add_variable(
            "serine",
            initial_value=2.5385608670239126,
        )
        .add_variable(
            "hydroxypyruvate",
            initial_value=0.009782608695111009,
        )
        .add_variable(
            "H2O2",
            initial_value=0.010880542843616855,
        )
        .add_parameter(
            "kf_phosphoglycolate_phosphatase",
            value=60.0,
        )
        .add_parameter(
            "E0_glycolate_oxidase",
            value=1.0,
        )
        .add_parameter(
            "kcat_glycolate_oxidase",
            value=100,
        )
        .add_parameter(
            "km_glycolate_oxidase_s",
            value=0.06,
        )
        .add_parameter(
            "E0_glycine_transaminase",
            value=1.0,
        )
        .add_parameter(
            "kcat_glycine_transaminase",
            value=143.0,
        )
        .add_parameter(
            "km_glycine_transaminase_s",
            value=3.0,
        )
        .add_parameter(
            "E0_glycine_decarboxylase",
            value=0.5,
        )
        .add_parameter(
            "kcat_glycine_decarboxylase",
            value=100.0,
        )
        .add_parameter(
            "km_glycine_decarboxylase_s",
            value=6.0,
        )
        .add_parameter(
            "E0_serine_glyoxylate_transaminase",
            value=1.0,
        )
        .add_parameter(
            "kcat_serine_glyoxylate_transaminase",
            value=159.0,
        )
        .add_parameter(
            "km_serine_glyoxylate_transaminase_glyoxylate",
            value=0.15,
        )
        .add_parameter(
            "km_serine_glyoxylate_transaminase_serine",
            value=2.72,
        )
        .add_parameter(
            "E0_glycerate_dehydrogenase",
            value=1.0,
        )
        .add_parameter(
            "kcat_glycerate_dehydrogenase",
            value=398.0,
        )
        .add_parameter(
            "km_glycerate_dehydrogenase_s",
            value=0.12,
        )
        .add_parameter(
            "E0_catalase",
            value=1.0,
        )
        .add_parameter(
            "kcat_catalase",
            value=760500.0,
        )
        .add_parameter(
            "km_catalase_s",
            value=137.9,
        )
        .add_derived(
            "vmax_glycolate_oxidase",
            fn=_mass_action_1s,
            args=["kcat_glycolate_oxidase", "E0_glycolate_oxidase"],
        )
        .add_derived(
            "vmax_glycine_transaminase",
            fn=_mass_action_1s,
            args=["kcat_glycine_transaminase", "E0_glycine_transaminase"],
        )
        .add_derived(
            "vmax_glycine_decarboxylase",
            fn=_mass_action_1s,
            args=["kcat_glycine_decarboxylase", "E0_glycine_decarboxylase"],
        )
        .add_derived(
            "vmax_serine_glyoxylate_transaminase",
            fn=_mass_action_1s,
            args=[
                "kcat_serine_glyoxylate_transaminase",
                "E0_serine_glyoxylate_transaminase",
            ],
        )
        .add_derived(
            "vmax_glycerate_dehydrogenase",
            fn=_mass_action_1s,
            args=["kcat_glycerate_dehydrogenase", "E0_glycerate_dehydrogenase"],
        )
        .add_derived(
            "vmax_catalase",
            fn=_mass_action_1s,
            args=["kcat_catalase", "E0_catalase"],
        )
        .add_reaction(
            "phosphoglycolate_phosphatase",
            fn=_value,
            args=["kf_phosphoglycolate_phosphatase"],
            stoichiometry={"glycolate": 1},
        )
        .add_reaction(
            "glycolate_oxidase",
            fn=_michaelis_menten_1s,
            args=["glycolate", "vmax_glycolate_oxidase", "km_glycolate_oxidase_s"],
            stoichiometry={"glycolate": -1, "glyoxylate": 1, "H2O2": 1},
        )
        .add_reaction(
            "glycine_transaminase",
            fn=_michaelis_menten_1s,
            args=[
                "glyoxylate",
                "vmax_glycine_transaminase",
                "km_glycine_transaminase_s",
            ],
            stoichiometry={"glyoxylate": -1.0, "glycine": 1.0},
        )
        .add_reaction(
            "glycine_decarboxylase",
            fn=_michaelis_menten_1s,
            args=[
                "glycine",
                "vmax_glycine_decarboxylase",
                "km_glycine_decarboxylase_s",
            ],
            stoichiometry={"glycine": -2.0, "serine": 1.0},
        )
        .add_reaction(
            "serine_glyoxylate_transaminase",
            fn=_ping_pong_bi_bi,
            args=[
                "glyoxylate",
                "serine",
                "vmax_serine_glyoxylate_transaminase",
                "km_serine_glyoxylate_transaminase_glyoxylate",
                "km_serine_glyoxylate_transaminase_serine",
            ],
            stoichiometry={
                "glyoxylate": -1.0,
                "serine": -1.0,
                "glycine": 1.0,
                "hydroxypyruvate": 1.0,
            },
        )
        .add_reaction(
            "glycerate_dehydrogenase",
            fn=_michaelis_menten_1s,
            args=[
                "hydroxypyruvate",
                "vmax_glycerate_dehydrogenase",
                "km_glycerate_dehydrogenase_s",
            ],
            stoichiometry={"hydroxypyruvate": -1.0},
        )
        .add_reaction(
            "catalase",
            fn=_michaelis_menten_1s,
            args=["H2O2", "vmax_catalase", "km_catalase_s"],
            stoichiometry={"H2O2": -1},
        )
    )
