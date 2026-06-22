import pandas as pd

from mxlmodels import get_zhu_2009


def test_rhs() -> None:
    model = get_zhu_2009()
    expected = pd.Series(
        {
            "RuBP": 2.851831458787981,
            "PGA": 0.45663866909183604,
            "DPGA": 0.2609553158705702,
            "Ru5P": -4.377266241396677,
            "GAP": 1.6857246376811594,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
