import pandas as pd

from mxlmodels import get_lam2026


def test_rhs() -> None:
    model = get_lam2026()
    expected = pd.Series(
        {
            "V": -3.979039320256561e-13,
            "A": -2.8421709430404007e-13,
            "Z": -2.842170943040401e-14,
            "PV": 3.979039320256561e-13,
            "PA": 2.8421709430404007e-13,
            "PZ": 2.842170943040401e-14,
            "QV": 0.0,
            "QA": 0.0,
            "QZ": 0.0,
            "QL": 0.0,
            "PL": 0.0,
            "PSIId": 0.0278673417351,
            "alpha_VDE": 0.0,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
