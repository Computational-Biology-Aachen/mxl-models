import pandas as pd

import mxlmodels
from mxlmodels import get_pfennig2024_synechocystis

data = mxlmodels.data.pfennig2024.load()


def test_rhs() -> None:
    model = get_pfennig2024_synechocystis(
        light_spectrum=data.light_spectrum,
        light_spectrum_measure=data.light_spectrum_measure,
        ocp_absorption=data.ocp_absorption_per_wavelength,
        abs_coef=data.pigment_abs_coef_per_wavelength,
        molar_masses=data.molar_masses,
        ps_comp=data.ps_comp,
        pigment_content=data.pigment_content,
    )

    expected = pd.Series(
        {
            "3PGA": -0.05632792656410103,
            "ATP": 2659.4155535402397,
            "CBBa": 0.02051051878163169,
            "CO2": 28111398.028020833,
            "Fd_ox": -206.98621660794998,
            "Hi": -125.08619451284859,
            "Ho": 11.25616888983532,
            "NADH": -0.5072912585264113,
            "NADPH": 5.816998732143536,
            "O2": 20.944935296383623,
            "OCP": 5.4231669070975316e-05,
            "PC_ox": 215.31173875938464,
            "PG": -0.07154017823521738,
            "PQ_ox": -47.15469510313709,
            "PSII": -0.0035817377846053583,
            "fumarate": 6.1824375536480165,
            "succinate": -6.1824375536480165,
        }
    )

    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
