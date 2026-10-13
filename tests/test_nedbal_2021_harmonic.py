from mxlpy.units import second

from mxlmodels import get_nedbal_2021_harmonic


def test_units() -> None:
    model = get_nedbal_2021_harmonic(
        period=10.0,
        offset=1.0,
        amplitudes=[0.1, 0.05, 0.02, 0.01],
        lags=[0.1, 0.2, 0.3, 0.4],
        light_min=10.0,
        light_max=100.0,
    )
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True
    } == {}
