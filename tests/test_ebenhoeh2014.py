import pandas as pd

from mxlmodels import get_ebenhoeh2014


def test_rhs() -> None:
    model = get_ebenhoeh2014()
    expected = pd.Series(
        {
            "Plastoquinone (oxidised)": -79.75458834180698,
            "Plastocyanine (oxidised)": 69.79485482677616,
            "Ferredoxine (oxidised)": -69.79485482676466,
            "ATP": 1200.0,
            "NADPH": 0.0,
            "protons_lumen": -54.405608252062926,
            "Light-harvesting complex": 8.846153846153783e-06,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
