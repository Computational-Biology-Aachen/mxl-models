import pandas as pd
from mxlpy.units import second

from mxlmodels import get_lotka_volterra_v2


def test_rhs() -> None:
    model = get_lotka_volterra_v2()
    expected = pd.Series({"Prey": -1.0, "Predator": -2.0})
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_lotka_volterra_v2()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
