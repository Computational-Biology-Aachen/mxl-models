import pandas as pd
from mxlpy.units import dimensionless

from mxlmodels import get_elowitz2000_repressilator


def test_rhs() -> None:
    model = get_elowitz2000_repressilator()
    expected = pd.Series(
        {
            "MlacI": 21.816000000000003,
            "MtetR": 43.416000000000004,
            "McI": 108.216,
            "PlacI": -10.0,
            "PtetR": -5.0,
            "PcI": -15.0,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    check = get_elowitz2000_repressilator().check_units(time_unit=dimensionless)
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
