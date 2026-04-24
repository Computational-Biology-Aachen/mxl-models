from mxlpy import Model, fns
from mxlpy.types import Derived


def prey_growth(Alpha: float, Prey: float) -> float:
    return Alpha * Prey


def predation(Predator: float, Prey: float) -> float:
    return Predator * Prey


def predator_death(Predator: float, Gamma: float) -> float:
    return Gamma * Predator


def create_model() -> Model:
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
            fn=prey_growth,
            args=["Alpha", "Prey"],
            stoichiometry={"Prey": 1.0},
        )
        .add_reaction(
            "predation",
            fn=predation,
            args=["Predator", "Prey"],
            stoichiometry={
                "Prey": Derived(fn=fns.neg, args=["Beta"]),
                "Predator": "Delta",
            },
        )
        .add_reaction(
            "predator_death",
            fn=predator_death,
            args=["Predator", "Gamma"],
            stoichiometry={"Predator": -1.0},
        )
    )
