import pandas as pd

from mxlmodels import get_lotka_volterra_v1


def test_rhs() -> None:
    model = get_lotka_volterra_v1()
    expected = pd.Series({"Prey": -1.0, "Predator": -2.0})
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
