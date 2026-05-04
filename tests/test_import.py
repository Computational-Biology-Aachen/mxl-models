import mxlmodels


def test_import() -> None:
    assert mxlmodels.get_dynamic_enterobactin()
    assert mxlmodels.get_ebeling_2026()
    assert mxlmodels.get_elowitz2000_repressilator()
    assert mxlmodels.get_lotka_volterra_v1()
    assert mxlmodels.get_lotka_volterra_v2()
    assert mxlmodels.get_matuszynska2016_npq()
    assert mxlmodels.get_matuszynska2016_phd()
    assert mxlmodels.get_matuszynska2019()
    assert mxlmodels.get_poolman2000()
    assert mxlmodels.get_population_dynamics()
    assert mxlmodels.get_prigogine1968_brusselator()
    assert mxlmodels.get_saadat2021()
    assert mxlmodels.get_selkov1968_glycolysis_oscillator()
    assert mxlmodels.get_tripartite_dynamics()
    assert mxlmodels.get_yokota1985()
