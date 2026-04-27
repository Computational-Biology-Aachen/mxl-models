"""Three-strain public-goods game: cooperators, cheaters, and private-goods producers."""

from mxlpy import Model


def _d_pdt(
    beta: float,
    alpha: float,
    public: float,
    eta: float,
    r_p: float,
    cheater: float,
    private: float,
) -> float:
    """Net growth rate of public-goods producers; lost to cheaters and private producers."""
    return (
        -cheater * public * alpha
        - private * public * beta
        + public * r_p
        - public**2.0 * eta
    )


def _d_cdt(
    public: float,
    alpha: float,
    cheater: float,
    nu: float,
) -> float:
    """Net growth rate of cheaters; exploits public-goods producers, density-limited."""
    return cheater * public * alpha - cheater**2.0 * nu


def _d_mdt(
    beta: float,
    public: float,
    gamma: float,
    r_m: float,
    private: float,
) -> float:
    """Net growth rate of private-goods producers; grows on public goods, density-limited."""
    return -private * public * beta + private * r_m - private**2.0 * gamma


def create_model() -> Model:
    """Build the three-strain public-goods game model (Public / Cheater / Private)."""
    return (
        Model()
        .add_variable(
            "Public",
            initial_value=1.0,
        )
        .add_variable(
            "Cheater",
            initial_value=1.0,
        )
        .add_variable(
            "Private",
            initial_value=1.0,
        )
        .add_parameter(
            "r_p",
            value=0.4,
        )
        .add_parameter(
            "eta",
            value=0.0001,
        )
        .add_parameter(
            "nu",
            value=1.0e-5,
        )
        .add_parameter(
            "r_m",
            value=0.2,
        )
        .add_parameter(
            "gamma",
            value=0.0001,
        )
        .add_parameter(
            "alpha",
            value=0.0002,
        )
        .add_parameter(
            "beta",
            value=0.0001,
        )
        .add_reaction(
            "dPdt",
            fn=_d_pdt,
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
            fn=_d_cdt,
            args=["Public", "alpha", "Cheater", "nu"],
            stoichiometry={"Cheater": 1.0},
        )
        .add_reaction(
            "dMdt",
            fn=_d_mdt,
            args=["beta", "Public", "gamma", "r_m", "Private"],
            stoichiometry={"Private": 1.0},
        )
    )
