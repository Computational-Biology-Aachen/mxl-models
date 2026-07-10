import pandas as pd

from mxlmodels import get_salvatori2022


def test_rhs() -> None:
    model = get_salvatori2022()
    expected = pd.Series(
        {
            "E_PSII": 0.0,
            "Q": 0.0,
            "P_NPQ": 0.07,
            "NADP": 0.2978625,
            "NADPH": -0.2978625,
            "R": 0.0008891099999999999,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
