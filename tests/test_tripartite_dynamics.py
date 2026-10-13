import pandas as pd
from mxlpy.units import second

from mxlmodels import get_tripartite_dynamics


def test_rhs() -> None:
    model = get_tripartite_dynamics()
    expected = pd.Series(
        {"Public": 0.3996, "Cheater": 0.00019, "Private": 0.19980000000000003}
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_tripartite_dynamics()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
