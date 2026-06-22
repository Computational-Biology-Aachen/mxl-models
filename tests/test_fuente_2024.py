import pandas as pd

from mxlmodels import get_fuente_2024


def test_rhs() -> None:
    model = get_fuente_2024()
    expected = pd.Series(
        {
            "Q_active": 0.0004512110259626644,
            "PQ": 2.6602195309753034,
            "PSI_ox": 2.50197729201318e-09,
            "H_lumen": -196.12946164019957,
            "ATP_st": 2.02817318495363e-10,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
