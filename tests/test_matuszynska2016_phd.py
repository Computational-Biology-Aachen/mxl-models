import pandas as pd
from mxlpy.units import second

from mxlmodels import get_matuszynska2016_phd


def test_rhs() -> None:
    model = get_matuszynska2016_phd()
    expected = pd.Series(
        {
            "ATP": 3.552713678800501e-15,
            "Plastoquinone (oxidised)": -3.4594549447319878e-12,
            "Plastocyanine (oxidised)": 6.998845947236987e-12,
            "Ferredoxine (oxidised)": -6.394884621840902e-14,
            "protons_lumen": -1.3911094498553211e-13,
            "Light-harvesting complex": 2.3852447794681098e-18,
            "PsbS (de-protonated)": -8.673617379884035e-19,
            "Violaxanthin": 4.483175983227561e-17,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    check = get_matuszynska2016_phd().check_units(time_unit=second)
    # The b6f clamp max(rate, -kcat_b6f) is dimensionally inconsistent in the original model
    known_inconsistent = {"b6f"}
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True and rxn not in known_inconsistent
    } == {}
