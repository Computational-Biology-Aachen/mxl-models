import pandas as pd

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
