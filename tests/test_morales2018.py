import pandas as pd

from mxlmodels import get_morales2018


def test_rhs() -> None:
    model = get_morales2018()
    expected = pd.Series(
        {
            "PGA": 1.533181942427158e-05,
            "RuBP": -8.02877447903045e-06,
            "fRB": 0.006576145952370591,
            "fP": 0.0187,
            "fZ": 0.00187,
            "alphar": -0.0003719998864527635,
            "PSIId": 83.9,
            "fR": 0.006279999888823415,
            "PR": 1.4514590675786333e-06,
            "Cc": -0.0010414548111549737,
            "Ccyt": 0.0001567570047269243,
            "Ci": 0.0,
            "Ca": 0.0,
            "H2OS": -0.0006772027431094089,
            "gsw": 6.665792514736718e-05,
            "sumA": 0.0,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
