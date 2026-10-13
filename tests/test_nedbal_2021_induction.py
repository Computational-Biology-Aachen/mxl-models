from mxlpy.units import second

from mxlmodels import get_nedbal_2021_induction


def test_units() -> None:
    check = get_nedbal_2021_induction(
        baseline=0.145, rise=0.047, tau_rise=0.02, decline=0.017, tau_decline=180.0
    ).check_units(time_unit=second)
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
