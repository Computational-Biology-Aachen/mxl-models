import pandas as pd

from mxlmodels import get_poolman2000


def test_rhs() -> None:
    model = get_poolman2000()
    expected = pd.Series(
        {
            "3PGA": -1.4540407168583958e-07,
            "BPGA": 1.4540408188601361e-07,
            "GAP": -6.631698402231878e-11,
            "DHAP": -2.3197208043335138e-10,
            "FBP": 2.1002605210540537e-10,
            "F6P": -8.070119533742925e-08,
            "G6P": 0.0,
            "G1P": 8.016358116202937e-08,
            "SBP": -1.5623535798425792e-10,
            "S7P": 5.031800531796193e-10,
            "E4P": 6.938893903907228e-10,
            "X5P": -3.816391647148976e-09,
            "R5P": -5.204170427930421e-09,
            "RUBP": 2.7755575615628914e-15,
            "RU5P": 8.551856556238135e-09,
            "ATP": -1.4540408255214743e-07,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
