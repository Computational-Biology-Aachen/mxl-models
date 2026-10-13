import pandas as pd
from mxlpy.units import second

from mxlmodels import get_sir


def test_rhs() -> None:
    model = get_sir()
    expected = pd.Series(
        {"s": -0.018000000000000002, "i": 0.008, "r": 0.010000000000000002}
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    check = get_sir().check_units(time_unit=second)
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
