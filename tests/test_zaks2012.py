import pandas as pd

from mxlmodels import get_zaks2012


def test_rhs() -> None:
    model = get_zaks2012()
    expected = pd.Series(
        {
            "ATP": 0.0,
            "ActiveATPs": -0.00014999999999762501,
            "Antheraxanthin": 2.5238292186894587e-09,
            "Fdxox": 1e-11,
            "Fdxr": -1e-11,
            "LumenCl": -6.717650131763981e-11,
            "LumenK": 6.717650131763981e-11,
            "LumenMg": 2.6870600527055923e-10,
            "LumenProtons": 1.007074737010375e-05,
            "P680ex": -0.030051003582746653,
            "P680plus": 0.029999703582737697,
            "P700ox": -1.2e-11,
            "P700r": 1.2e-11,
            "PCr": 0.020316273610021727,
            "PQ": 0.010248126811010864,
            "PQH2": -0.018078136811010864,
            "PSIIChlEx": 4.429999999999992e-05,
            "PheAnion": 0.029970003579163906,
            "PsbSQ": 0.0003965285619151292,
            "QAox": -0.00022500000358274694,
            "QBneut": -5.499e-05,
            "QBred1": 0.000125,
            "QBred2": 0.00776,
            "Thrdx": 9e-12,
            "TotalLEF": 0.0,
            "Zeaxanthin": -3.9999873808539065e-18,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
