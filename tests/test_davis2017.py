import pandas as pd
from mxlpy.units import second

from mxlmodels import get_davis2017


def test_rhs() -> None:
    model = get_davis2017()
    expected = pd.Series(
        {
            "QA_red": 82.99319727891155,
            "PQH_2": 5.163834200946098,
            "pH_lumen": 0.08690432008236756,
            "Dpsi": -3.2965638628478366,
            "K_lu": -0.000336,
            "PC_ox": 10.327668401892195,
            "Zx": 1.5848680739893872e-05,
            "singO2": 0.0,
            "P700_ox": 100.0,
            "Fd_red": 100.0,
            "NADPH_st": 0.0,
            "LEF": 0.0,
            "ATP_made": 53.26315789473684,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_davis2017()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
