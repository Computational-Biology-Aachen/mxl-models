import pandas as pd
from mxlpy.units import second

from mxlmodels import get_bellasio_2019


def test_rhs() -> None:
    model = get_bellasio_2019()
    expected = pd.Series(
        {
            "CO2": 21.712575115278078,
            "HCO3": -19.925313194837287,
            "RUBP": 1.512551825220756,
            "PGA": 0.910094929515153,
            "DHAP": -3.9166261480705904,
            "ATP_st": 0.39323048248969883,
            "NADPH_st": 1.7527713361761308,
            "RU5P": -3.2418573259002232,
            "Ract": -8.824713986233144e-05,
            "J_NADPH": -0.0006477811097880364,
            "J_ATP": -0.0002516416039256633,
            "Ci": -0.02022057926870556,
            "gs": 4.934324553889585e-19,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_bellasio_2019()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    # Hidden constants: DHAP - 0.4 (mM) in v_carbohydrate_synthesis and the factor 0.001
    # (umol -> mmol) in CO2_dissolution. Ci is in umol/mol but integrates leaf-level
    # fluxes without an airspace capacity.
    known_inconsistent = {
        ("DHAP", "v_carbohydrate_synthesis"),
        ("ATP_st", "v_carbohydrate_synthesis"),
        ("CO2", "CO2_dissolution"),
        ("Ci", "CO2_dissolution"),
        ("Ci", "CO2_stomatal_diffusion"),
    }
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True and (var, rxn) not in known_inconsistent
    } == {}
