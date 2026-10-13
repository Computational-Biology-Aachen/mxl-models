import pandas as pd
from mxlpy.units import hour

from mxlmodels import get_dynamic_enterobactin


def test_rhs() -> None:
    model = get_dynamic_enterobactin()
    expected = pd.Series(
        {
            "x1": 0.12093348356070453,
            "x2": 0.2355964874705504,
            "s1": -0.7399340495204443,
            "p1": 0.0002614990261410044,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_dynamic_enterobactin()
    check = model.check_units(time_unit=hour)
    assert set(check.per_variable) == set(model.get_raw_variables())
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
