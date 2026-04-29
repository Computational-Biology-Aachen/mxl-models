"""Sel'kov glycolysis oscillator (Sel'kov 1968): autocatalytic ADP activation of PFK drives sustained oscillations.

Reference: E E Sel'kov.
"Self-oscillations in glycolysis. 1. A simple kinetic model."
European Journal of Biochemistry 4.1 (1968): 79-86.
DOI: 10.1111/j.1432-1033.1968.tb00175.x
"""

from mxlpy import Model


def _pfk(
    a: float,
    x: float,
    y: float,
) -> float:
    """PFK rate: product-activated (ADP²) consumption of F6P."""
    return (a + x**2) * y


def _atp_consumption(
    x: float,
) -> float:
    """First-order ADP removal."""
    return x


def _f6p_influx(
    b: float,
) -> float:
    """Constant F6P influx."""
    return b


def create_model() -> Model:
    """Build the Sel'kov oscillator: two-variable dimensionless model of glycolytic oscillations via autocatalytic PFK activation."""
    return (
        Model()
        .add_variable("X", initial_value=0.5)
        .add_variable("Y", initial_value=0.5)
        .add_parameter("a", value=0.05)
        .add_parameter("b", value=0.5)
        .add_reaction(
            "pfk",
            fn=_pfk,
            args=["a", "X", "Y"],
            stoichiometry={"X": 1.0, "Y": -1.0},
        )
        .add_reaction(
            "atp_consumption",
            fn=_atp_consumption,
            args=["X"],
            stoichiometry={"X": -1.0},
        )
        .add_reaction(
            "f6p_influx",
            fn=_f6p_influx,
            args=["b"],
            stoichiometry={"Y": 1.0},
        )
    )
