import pandas as pd
from mxlpy.units import second

from mxlmodels import get_yokota1985


def test_rhs() -> None:
    model = get_yokota1985()
    expected = pd.Series(
        {
            "glycolate": 0.0,
            "glyoxylate": -1.6927259594012867e-10,
            "glycine": 1.7036398958225618e-09,
            "serine": 7.657270373329084e-10,
            "hydroxypyruvate": 1.566746732351021e-12,
            "H2O2": -1.4210854715202004e-14,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_yokota1985()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    # The ping-pong rate law is dimensionally inconsistent, see FIXME in _ping_pong_bi_bi
    known_inconsistent = {"serine_glyoxylate_transaminase"}
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True and rxn not in known_inconsistent
    } == {}
