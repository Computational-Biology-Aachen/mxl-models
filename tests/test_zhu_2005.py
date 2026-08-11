import pandas as pd

from mxlmodels import get_zhu_2005


def test_rhs() -> None:
    model = get_zhu_2005()
    expected = pd.Series(
        {
            "Ap": 1346.938775510204,
            "U": 428.57142857142856,
            "P680plus_Pheominus": 0.0,
            "P680plus_Pheo": 0.0,
            "P680_Pheominus": 0.0,
            "S0T": 0.0,
            "S1T": 0.0,
            "S2T": 0.0,
            "S3T": 0.0,
            "S0Tp": 0.0,
            "S1Tp": 0.0,
            "S2Tp": 0.0,
            "S3Tp": 0.0,
            "QA_QB": -40.0,
            "QAred_QB": 0.0,
            "QA_QBred": 0.0,
            "QAred_QBred": 0.0,
            "QA_QB2red": 40.0,
            "QAred_QB2red": 0.0,
            "PQH2": -790.0,
            "Aip": 0.0,
            "Ui": 0.0,
            "Uifc": 0.0,
            "P680plus_Pheominus_i": 0.0,
            "P680plus_Pheo_i": 0.0,
            "P680_Pheominus_i": 0.0,
            "S0T_i": 0.0,
            "S1T_i": 0.0,
            "S2T_i": 0.0,
            "S3T_i": 0.0,
            "S0Tp_i": 0.0,
            "S1Tp_i": 0.0,
            "S2Tp_i": 0.0,
            "S3Tp_i": 0.0,
            "QA_QB_i": 0.0,
            "QAred_QB_i": 0.0,
            "QA_QBred_i": 0.0,
            "QAred_QBred_i": 0.0,
            "QA_QB2red_i": 0.0,
            "QAred_QB2red_i": 0.0,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
