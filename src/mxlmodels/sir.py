r"""Compartmental models of infectious disease spread.

|             |                                                        |
| ----------- | ------------------------------------------------------ |
| doi         | 10.1098/rspa.1927.0118                                 |
| main author | William Ogilvy Kermack                                 |
| paper title | A Contribution to the Mathematical Theory of Epidemics |
| published   | August 1927                                            |
| journal     | Proceedings of the Royal Society A                     |
| organism    | N/A (epidemiological compartments)                     |
| Ported by   | Marvin van Aalst ( @marvinvanaalst )                   |

These models are based on the SIR (Susceptible-Infectious-Recovered) framework,
which is commonly used to model the spread of infectious diseases.
"""

from __future__ import annotations

from mxlpy import Model, Parameter, Variable, fns
from mxlpy.units import dimensionless, second

# Population fractions are dimensionless
_per_s = 1 / second

__all__ = ["get_sir", "get_sird"]


def get_sir() -> Model:
    r"""Create a simple SIR model."""
    return (
        Model()
        .add_variables(
            {
                "s": Variable(0.9, unit=dimensionless),
                "i": Variable(0.1, unit=dimensionless),
                "r": Variable(0.0, unit=dimensionless),
            }
        )
        .add_parameters(
            {
                "beta": Parameter(0.2, unit=_per_s),
                "gamma": Parameter(0.1, unit=_per_s),
            }
        )
        .add_reaction(
            "infection",
            fns.mass_action_2s,
            args=["s", "i", "beta"],
            stoichiometry={"s": -1, "i": 1},
        )
        .add_reaction(
            "recovery",
            fns.mass_action_1s,
            args=["i", "gamma"],
            stoichiometry={"i": -1, "r": 1},
        )
    )


def get_sird() -> Model:
    r"""Create a simple SIR model with death."""
    return (
        get_sir()
        .add_variable("d", 0.0, unit=dimensionless)
        .add_parameter("mu", 0.01, unit=_per_s)
        .add_reaction(
            "death",
            fns.mass_action_1s,
            args=["i", "mu"],
            stoichiometry={"i": -1, "d": 1},
        )
    )
