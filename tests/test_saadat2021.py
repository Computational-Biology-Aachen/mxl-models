import pandas as pd

from mxlmodels import get_saadat2021


def test_rhs() -> None:
    model = get_saadat2021()
    expected = pd.Series(
        {
            "3PGA": 8.107348167904482e-06,
            "BPGA": -8.107349318109414e-06,
            "GAP": 1.6152970211402717e-09,
            "DHAP": -3.9583881311644475e-10,
            "FBP": 2.097089168984212e-11,
            "F6P": -2.3353001643577898e-08,
            "G6P": 4.440892098500626e-08,
            "G1P": -2.2949089760471164e-08,
            "SBP": -1.8912924004688136e-10,
            "S7P": 2.9646868016097727e-09,
            "E4P": 2.0816681711721685e-09,
            "X5P": -2.42861286636753e-09,
            "R5P": 5.204170427930421e-10,
            "RUBP": -1.156852391659413e-13,
            "RU5P": -2.129048604082584e-09,
            "ATP": 8.107349467545433e-06,
            "Ferredoxine (oxidised)": -2.2453150450019166e-12,
            "protons_lumen": 1.5614964833307599e-12,
            "Light-harvesting complex": 7.486892507867215e-13,
            "NADPH": -2.2322421688869554e-13,
            "Plastocyanine (oxidised)": 1.34718902700115e-11,
            "Plastoquinone (oxidised)": -3.0343949575240003e-11,
            "PsbS (de-protonated)": -1.7778150913222435e-14,
            "Violaxanthin": -4.130309319861909e-12,
            "MDA": -6.505213034913027e-19,
            "H2O2": -1.3552527156068805e-20,
            "DHA": -4.506583374765464e-22,
            "GSSG": -7.541560741010807e-24,
            "Thioredoxin (oxidised)": 6.661338147750939e-14,
            "E_inactive": 1.9027002196025933e-12,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
