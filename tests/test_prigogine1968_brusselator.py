import pandas as pd

from mxlmodels import get_prigogine1968_brusselator


def test_rhs() -> None:
    model = get_prigogine1968_brusselator()
    expected = pd.Series({"X": 1.75, "Y": -2.25})
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
