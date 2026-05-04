import pandas as pd

from mxlmodels import get_yokota1985


def test_rhs() -> None:
    model = get_yokota1985()
    expected = pd.Series(
        {
            "glycolate": 0.0,
            "glyoxylate": -1.6927259594012867e-10,
            "glycine": 1.7036398958225618e-09,
            "serine": 7.657270373329084e-10,
            "hydroxypyruvate": 1.566746732351021e-12,
            "H2O2": -1.4210854715202004e-14,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
