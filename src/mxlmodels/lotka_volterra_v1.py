r"""Lotka-Volterra predator-prey model (v1).

Explicit prey growth and predation reactions.

|             |                                      |
| ----------- | ------------------------------------ |
| doi         | N/A                                  |
| main author | Alfred J. Lotka; Vito Volterra       |
| paper title | classic predator-prey model          |
| published   | 1925 / 1926                          |
| journal     | N/A                                  |
| organism    | N/A (abstract predator-prey)         |
| Ported by   | Marvin van Aalst ( @marvinvanaalst ) |
"""

from mxlpy import Model


def _v0(
    alpha: float,
    prey: float,
) -> float:
    r"""Prey intrinsic growth: Alpha * Prey."""
    return alpha * prey


def _v1(
    predator: float,
    beta: float,
    prey: float,
) -> float:
    r"""Predation rate (prey loss): Beta * Predator * Prey."""
    return beta * predator * prey


def _v2(
    delta: float,
    predator: float,
    prey: float,
) -> float:
    r"""Predator growth from predation: Delta * Predator * Prey."""
    return delta * predator * prey


def _v3(
    predator: float,
    gamma: float,
) -> float:
    r"""Predator natural death: Gamma * Predator."""
    return gamma * predator


def get_lotka_volterra_v1() -> Model:
    r"""Build the Lotka-Volterra predator-prey model (v1)."""
    return (
        Model()
        .add_variable(
            "Prey",
            initial_value=10.0,
        )
        .add_variable(
            "Predator",
            initial_value=10.0,
        )
        .add_parameter(
            "Alpha",
            value=0.1,
        )
        .add_parameter(
            "Beta",
            value=0.02,
        )
        .add_parameter(
            "Gamma",
            value=0.4,
        )
        .add_parameter(
            "Delta",
            value=0.02,
        )
        .add_reaction(
            "prey_growth",
            fn=_v0,
            args=["Alpha", "Prey"],
            stoichiometry={"Prey": 1.0},
        )
        .add_reaction(
            "predation",
            fn=_v1,
            args=["Predator", "Beta", "Prey"],
            stoichiometry={"Prey": -1.0},
        )
        .add_reaction(
            "predator_death",
            fn=_v3,
            args=["Predator", "Gamma"],
            stoichiometry={"Predator": -1.0},
        )
        .add_reaction(
            "predator_growth",
            fn=_v2,
            args=["Delta", "Predator", "Prey"],
            stoichiometry={"Predator": 1.0},
        )
    )
