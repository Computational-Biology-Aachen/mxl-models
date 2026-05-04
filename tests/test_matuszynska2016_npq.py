import pandas as pd

from mxlmodels import get_matuszynska2016_npq


def test_rhs() -> None:
    model = get_matuszynska2016_npq()
    expected = pd.Series(
        {
            "pq_red": 100.40160466945876,
            "protons": 2.0124707758530502,
            "vmax_atp_synthase": 0.01,
            "atp": -250.0,
            "psbs_de": -1.8663045611515098e-11,
            "vx": -7.267695401042834e-18,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
