import pandas as pd

from mxlmodels import get_ebenhoeh2014


def test_rhs() -> None:
    model = get_ebenhoeh2014()
    expected = pd.Series(
        {
            "Plastoquinone (oxidised)": -20.419527078707603,
            "Plastocyanine (oxidised)": 23.308901298571623,
            "Ferredoxine (oxidised)": -23.30890129856012,
            "ATP": 1200.0,
            "NADPH": 0.0,
            "protons_lumen": -55.59300947732491,
            "Light-harvesting complex": 8.846153846153783e-06,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
