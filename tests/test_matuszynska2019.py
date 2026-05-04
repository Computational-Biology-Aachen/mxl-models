import pandas as pd

from mxlmodels import get_matuszynska2019


def test_rhs() -> None:
    model = get_matuszynska2019()
    expected = pd.Series(
        {
            "3PGA": -0.8849628764200442,
            "BPGA": -9.000975120798671e-08,
            "GAP": -9.561784311377941e-10,
            "DHAP": 7.390768622106414e-10,
            "FBP": 0.16911741246662448,
            "F6P": -0.16911742478316116,
            "G6P": 0.0,
            "G1P": 0.02162359679148758,
            "SBP": 0.14749382781820108,
            "S7P": -0.14749382781820108,
            "E4P": 1.734723475976807e-10,
            "X5P": 3.2959746043559335e-09,
            "R5P": 1.5612511283791264e-09,
            "RUBP": 0.0,
            "RU5P": 0.4424814787708498,
            "ATP": 0.46410515805101193,
            "Ferredoxine (oxidised)": 1.4210854715202004e-14,
            "protons_lumen": -8.062669455688454e-16,
            "Light-harvesting complex": 7.32920668600201e-17,
            "NADPH": -8.881784197001252e-16,
            "Plastocyanine (oxidised)": -4.263256414560601e-14,
            "Plastoquinone (oxidised)": 3.0753177782116836e-14,
            "PsbS (de-protonated)": -5.366800753803247e-18,
            "Violaxanthin": -5.919218701343557e-16,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
