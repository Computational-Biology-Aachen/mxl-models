import pandas as pd

from mxlmodels import get_population_dynamics


def test_rhs() -> None:
    model = get_population_dynamics()
    expected = pd.Series({"e_coli": 12.0, "c_gluta": 5.975})
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
