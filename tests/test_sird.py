import pandas as pd
from mxlpy.units import second

from mxlmodels import get_sird


def test_rhs() -> None:
    model = get_sird()
    expected = pd.Series(
        {"s": -0.018000000000000002, "i": 0.007, "r": 0.010000000000000002, "d": 0.001}
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_sird()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
