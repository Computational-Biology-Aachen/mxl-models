from mxlpy.units import second

from mxlmodels import get_nedbal_2021_ojip


def test_units() -> None:
    check = get_nedbal_2021_ojip(
        fluorescence_0=0.2, j=0.3, i=0.2, p=0.3, tau_j=0.001, tau_i=0.03, tau_p=0.3
    ).check_units(time_unit=second)
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
