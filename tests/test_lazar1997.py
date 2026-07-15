import pandas as pd

from mxlmodels import get_lazar1997


def test_rhs() -> None:
    model = get_lazar1997()
    expected = pd.Series(
        {
            "y1": -1165.56,
            "y2": 1099.56,
            "y3": 0.0,
            "y4": 0.0,
            "y5": 0.0,
            "y6": 0.0,
            "y7": 0.0,
            "y8": 0.0,
            "y9": 59.0,
            "y10": 7.0,
            "y11": 66.0,
            "y12": 0.0,
            "y13": -770.0,
            "y14": 770.0,
            "y15": 0.0,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
