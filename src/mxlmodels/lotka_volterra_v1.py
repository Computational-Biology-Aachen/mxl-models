"""Lotka-Volterra predator-prey model (v1): explicit prey growth and predation reactions."""

from mxlpy import Model


def v0(Alpha: float, Prey: float) -> float:
    """Prey intrinsic growth: Alpha * Prey."""
    return Alpha * Prey


def v1(Predator: float, Beta: float, Prey: float) -> float:
    """Predation rate (prey loss): Beta * Predator * Prey."""
    return Beta * Predator * Prey


def v2(Delta: float, Predator: float, Prey: float) -> float:
    """Predator growth from predation: Delta * Predator * Prey."""
    return Delta * Predator * Prey


def v3(Predator: float, Gamma: float) -> float:
    """Predator natural death: Gamma * Predator."""
    return Gamma * Predator


def create_model() -> Model:
    """Build the Lotka-Volterra predator-prey model (v1)."""
    return (
        Model()
        .add_variable("Prey", initial_value=10.0)
        .add_variable("Predator", initial_value=10.0)
        .add_parameter("Alpha", value=0.1)
        .add_parameter("Beta", value=0.02)
        .add_parameter("Gamma", value=0.4)
        .add_parameter("Delta", value=0.02)
        .add_reaction(
            "prey_growth",
            fn=v0,
            args=["Alpha", "Prey"],
            stoichiometry={"Prey": 1.0},
        )
        .add_reaction(
            "predation",
            fn=v1,
            args=["Predator", "Beta", "Prey"],
            stoichiometry={"Prey": -1.0},
        )
        .add_reaction(
            "predator_death",
            fn=v3,
            args=["Predator", "Gamma"],
            stoichiometry={"Predator": -1.0},
        )
        .add_reaction(
            "predator_growth",
            fn=v2,
            args=["Delta", "Predator", "Prey"],
            stoichiometry={"Predator": 1.0},
        )
    )
