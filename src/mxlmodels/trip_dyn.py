"""Three-strain public-goods game: cooperators, cheaters, and private-goods producers."""

from mxlpy import Model


def dPdt(
    beta: float,
    alpha: float,
    Public: float,
    eta: float,
    r_p: float,
    Cheater: float,
    Private: float,
) -> float:
    """Net growth rate of public-goods producers; lost to cheaters and private producers."""
    return (
        -Cheater * Public * alpha
        - Private * Public * beta
        + Public * r_p
        - Public**2.0 * eta
    )


def dCdt(Public: float, alpha: float, Cheater: float, nu: float) -> float:
    """Net growth rate of cheaters; exploits public-goods producers, density-limited."""
    return Cheater * Public * alpha - Cheater**2.0 * nu


def dMdt(beta: float, Public: float, gamma: float, r_m: float, Private: float) -> float:
    """Net growth rate of private-goods producers; grows on public goods, density-limited."""
    return -Private * Public * beta + Private * r_m - Private**2.0 * gamma


def create_model() -> Model:
    """Build the three-strain public-goods game model (Public / Cheater / Private)."""
    return (
        Model()
        .add_variable("Public", initial_value=1.0)
        .add_variable("Cheater", initial_value=1.0)
        .add_variable("Private", initial_value=1.0)
        .add_parameter("r_p", value=0.4)
        .add_parameter("eta", value=0.0001)
        .add_parameter("nu", value=1.0e-5)
        .add_parameter("r_m", value=0.2)
        .add_parameter("gamma", value=0.0001)
        .add_parameter("alpha", value=0.0002)
        .add_parameter("beta", value=0.0001)
        .add_reaction(
            "dPdt",
            fn=dPdt,
            args=[
                "beta",
                "alpha",
                "Public",
                "eta",
                "r_p",
                "Cheater",
                "Private",
            ],
            stoichiometry={"Public": 1.0},
        )
        .add_reaction(
            "dCdt",
            fn=dCdt,
            args=["Public", "alpha", "Cheater", "nu"],
            stoichiometry={"Cheater": 1.0},
        )
        .add_reaction(
            "dMdt",
            fn=dMdt,
            args=["beta", "Public", "gamma", "r_m", "Private"],
            stoichiometry={"Private": 1.0},
        )
    )
