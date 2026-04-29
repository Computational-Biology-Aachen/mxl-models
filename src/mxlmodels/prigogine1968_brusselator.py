"""Brusselator (Prigogine 1968): autocatalytic chemical oscillator with limit cycle dynamics.

Reference: Prigogine, I. and Lefever, R.
"Symmetry Breaking Instabilities in Dissipative Systems."
The Journal of Chemical Physics 48 (1968): 1695-1700.
DOI: 10.1063/1.1668896
"""

from mxlpy import Model


def _production(
    a: float,
) -> float:
    """Constant source of X from reservoir species A."""
    return a


def _autocatalysis(
    x: float,
    y: float,
) -> float:
    """Autocatalytic step: 2X + Y → 3X, rate = X²Y."""
    return x**2 * y


def _conversion(
    b: float,
    x: float,
) -> float:
    """Conversion of X to Y, rate = B·X."""
    return b * x


def _removal(
    x: float,
) -> float:
    """First-order removal of X."""
    return x


def create_model() -> Model:
    """Build the Brusselator model: two-variable autocatalytic oscillator exhibiting limit cycle behavior."""
    return (
        Model()
        .add_variable("X", initial_value=1.5)
        .add_variable("Y", initial_value=3.0)
        .add_parameter("A", value=1.0)
        .add_parameter("B", value=3.0)
        .add_reaction(
            "production",
            fn=_production,
            args=["A"],
            stoichiometry={"X": 1.0},
        )
        .add_reaction(
            "autocatalysis",
            fn=_autocatalysis,
            args=["X", "Y"],
            stoichiometry={"X": 1.0, "Y": -1.0},
        )
        .add_reaction(
            "conversion",
            fn=_conversion,
            args=["B", "X"],
            stoichiometry={"X": -1.0, "Y": 1.0},
        )
        .add_reaction(
            "removal",
            fn=_removal,
            args=["X"],
            stoichiometry={"X": -1.0},
        )
    )
