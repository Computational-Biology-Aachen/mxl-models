import pandas as pd
from mxlpy.units import dimensionless

from mxlmodels import get_prigogine1968_brusselator


def test_rhs() -> None:
    model = get_prigogine1968_brusselator()
    expected = pd.Series({"X": 1.75, "Y": -2.25})
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    check = get_prigogine1968_brusselator().check_units(time_unit=dimensionless)
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
