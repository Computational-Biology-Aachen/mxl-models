import numpy as np
import pytest

from mxlmodels import get_bernacchi_2013
from mxlmodels.ss.bernacchi2013 import (
    CI_FIG34,
    DHA_VCMAX,
    JMAX_OMEGA,
    JMAX_TOPT,
    T25,
    arrhenius,
    june_jmax,
)


def test_temperature_functions_are_one_at_25c() -> None:
    assert arrhenius(1.0, DHA_VCMAX, np.array(T25)) == pytest.approx(1.0)
    assert june_jmax(1.0, np.array(25.0), JMAX_TOPT, JMAX_OMEGA) == pytest.approx(1.0)


def test_fig2_co2_response() -> None:
    Ci = np.array([10.0, 280.0, 700.0, 1000.0])
    rubisco, rubp, tpu, an = get_bernacchi_2013(Ci=Ci, PPFD=1500.0)
    # Below Gamma* A follows Rubisco (Eqn 2), not the lower RuBP rate
    assert an[0] == pytest.approx(rubisco[0])
    assert rubp[0] < rubisco[0]
    # Rubisco/RuBP crossover near Ci 280 at A 22.4, RuBP/TPU near Ci 700
    assert rubisco[1] == pytest.approx(22.4, abs=0.3)
    assert rubp[1] == pytest.approx(22.4, abs=0.3)
    assert rubp[2] == pytest.approx(tpu[2], abs=0.3)
    assert tpu[3] == pytest.approx(29.5, abs=0.3)
    assert rubp[3] == pytest.approx(31.3, abs=0.3)


def test_fig3_temperature_response() -> None:
    T = np.linspace(5, 40, 351)
    _, rubp, tpu, an = get_bernacchi_2013(Ci=CI_FIG34, PPFD=1500.0, T_leaf=T)
    assert T[np.argmax(tpu)] == pytest.approx(30, abs=1)
    assert tpu.max() == pytest.approx(34.2, abs=0.5)
    assert T[np.argmax(rubp)] == pytest.approx(33.5, abs=1)
    assert an.max() == pytest.approx(20.1, abs=0.3)
    assert an[-1] == pytest.approx(8.3, abs=0.8)


def test_fig4_ppfd_response() -> None:
    PPFD = np.array([0.0, 1000.0, 2000.0])
    rubisco, rubp, _, an = get_bernacchi_2013(Ci=CI_FIG34, PPFD=PPFD)
    assert rubisco[0] == pytest.approx(19.8, abs=0.2)
    assert an[0] == pytest.approx(-1.1)
    assert rubp[1] == pytest.approx(rubisco[1], abs=0.3)
    assert rubp[2] == pytest.approx(21.5, abs=0.2)
