import pandas as pd
from mxlpy.units import dimensionless

from mxlmodels import get_selkov1968_glycolysis_oscillator


def test_rhs() -> None:
    model = get_selkov1968_glycolysis_oscillator()
    expected = pd.Series({"X": -0.35, "Y": 0.35})
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    check = get_selkov1968_glycolysis_oscillator().check_units(time_unit=dimensionless)
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
