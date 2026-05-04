import pandas as pd

from mxlmodels import get_dynamic_enterobactin


def test_rhs() -> None:
    model = get_dynamic_enterobactin()
    expected = pd.Series(
        {"e_coli": 8.0, "c_gluta": 3.975, "enterobactin": -2.9794871794871796}
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
