import pandas as pd
from mxlpy.units import second

from mxlmodels import get_ebeling_2026


def test_rhs() -> None:
    model = get_ebeling_2026()
    expected = pd.Series(
        {
            "3PGA": -0.557902801556456,
            "BPGA": -166.08994793819076,
            "GAP": 166.08993983245674,
            "DHAP": -3.9583881311644475e-10,
            "FBP": 2.097089168984212e-11,
            "F6P": -2.3353001643577898e-08,
            "G6P": 4.440892098500626e-08,
            "G1P": -2.2949089760471164e-08,
            "SBP": -1.8912924004688136e-10,
            "S7P": 2.9646868016097727e-09,
            "E4P": 2.0816681711721685e-09,
            "X5P": -2.42861286636753e-09,
            "R5P": 5.204170427930421e-10,
            "RUBP": 0.27895545445219627,
            "RU5P": -2.129048604082584e-09,
            "ATP": -2.107909039712422,
            "Ferredoxine (oxidised)": -36471.29976546911,
            "Light-harvesting complex": 7.486892507867215e-13,
            "NADPH": -166.11322280600317,
            "Plastocyanine (oxidised)": 4858.529023577956,
            "Plastoquinone (oxidised)": -603.0232488078858,
            "PsbS (de-protonated)": 0.0005280869482712752,
            "Violaxanthin": 1.7396112061595006e-06,
            "MDA": -6.505213034913027e-19,
            "H2O2": 0.012035780421210564,
            "DHA": -4.506583374765464e-22,
            "GSSG": -7.541560741010807e-24,
            "Thioredoxin (oxidised)": 0.2397636431956064,
            "E_inactive": 1.9027002196025933e-12,
            "P700FA": 26633.33656475633,
            "P700+FA-": -15047.215976261059,
            "P700FA-": -21558.286305941754,
            "B0": 1664.5296722286862,
            "B1": -777.6944217057198,
            "B2": -552.0806710467853,
            "pH_lumen": -12.428539967197393,
            "pH": 5.324223118106764,
            "ATPactivity": 0.001,
            "delta_psi": 0.9563536878473002,
            "K_stroma": 2.1156656248373695e-17,
            "Cl_stroma": -2.8242303100570773e-17,
        }
    )
    pd.testing.assert_series_equal(
        model.get_right_hand_side().loc[expected.index],
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_units() -> None:
    model = get_ebeling_2026()
    check = model.check_units(time_unit=second)
    assert set(check.per_variable) == set(model.get_raw_variables())
    # The proton-pumping reactions use bare numbers as pH stoichiometries, which hide
    # the mmol (mol Chl)^-1 -> pH unit buffer conversion
    known_inconsistent = {
        (ph, rxn)
        for ph, rxns in {
            "pH_lumen": ["atp_synthase", "proton_leak", "B12", "b6f", "KEA3", "NDH1"],
            "pH": ["atp_synthase", "proton_leak", "B20", "b6f", "KEA3", "NDH1"],
        }.items()
        for rxn in rxns
    }
    # gadph uses the stromal protons in mmol (mol Chl)^-1 with constants in mM
    known_inconsistent |= {(var, "gadph") for var in ["BPGA", "GAP", "NADPH"]}
    assert {
        (var, rxn): result
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if result is not True and (var, rxn) not in known_inconsistent
    } == {}
