import pandas as pd

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
