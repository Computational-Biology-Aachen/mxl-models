import pandas as pd

from mxlmodels import get_li_2021


def test_rhs() -> None:
    model = get_li_2021()
    expected = pd.Series(
        {
            "QA_red": 24.897858520864595,
            "PQH_2": 0.0009321838879450626,
            "pH_lumen": -1.0396059627079977,
            "Dpsi": 2.1059080911489465,
            "K_lu": 0.0,
            "PC_ox": 0.0018643677758901252,
            "Y2": 20.010000000000005,
            "Zx": 2.0095091401600154e-10,
            "singO2": 0.0,
            "Fd_red": 20.010000000000005,
            "NADPH_st": 0.0,
            "Cl_lu": 0.0,
            "Cl_st": 0.0,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
