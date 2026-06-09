"""MxlModels is a Python package of reference mechanistic models.

It contains the same models as in the [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) repo,
but written as single, flat files to make inspection easier.

"""

# Re-export mxlpy modules for easy access
from mxlpy import Simulator, fit, mc, mca, plot, scan

from . import data
from ._dynamic_enterobactin import get_dynamic_enterobactin
from ._population_dynamics import get_population_dynamics
from ._tripartite_dynamics import get_tripartite_dynamics
from .bellasio2019 import get_bellasio_2019
from .davis2017 import get_davis2017
from .ebeling2026 import get_ebeling_2026
from .ebenhoeh2014 import get_ebenhoeh2014
from .elowitz2000_repressilator import get_elowitz2000_repressilator
from .fuente2024 import get_fuente_2024
from .hahn1987 import get_hahn1987
from .lazar1997 import get_lazar1997
from .li2021 import get_li_2021
from .lotka_volterra_v1 import get_lotka_volterra_v1
from .lotka_volterra_v2 import get_lotka_volterra_v2
from .matuszynska2016_npq import get_matuszynska2016_npq
from .matuszynska2016_phd import get_matuszynska2016_phd
from .matuszynska2019 import get_matuszynska2019
from .nguyen2026_tomato import get_nguyen2026_tomato
from .pfennig2024_synechocystis import get_pfennig2024_synechocystis
from .poolman2000 import get_poolman2000
from .prigogine1968_brusselator import get_prigogine1968_brusselator
from .saadat2021 import get_saadat2021 as get_saadat2021
from .selkov1968_oscillator import get_selkov1968_glycolysis_oscillator
from .sir import get_sir, get_sird
from .yokota1985 import get_yokota1985
from .zhu2009 import get_zhu_2009

__all__ = [
    "Simulator",
    "data",
    "fit",
    "get_bellasio_2019",
    "get_davis2017",
    "get_dynamic_enterobactin",
    "get_ebeling_2026",
    "get_ebenhoeh2014",
    "get_elowitz2000_repressilator",
    "get_fuente_2024",
    "get_hahn1987",
    "get_lazar1997",
    "get_li_2021",
    "get_lotka_volterra_v1",
    "get_lotka_volterra_v2",
    "get_matuszynska2016_npq",
    "get_matuszynska2016_phd",
    "get_matuszynska2019",
    "get_nguyen2026_tomato",
    "get_pfennig2024_synechocystis",
    "get_poolman2000",
    "get_population_dynamics",
    "get_prigogine1968_brusselator",
    "get_saadat2021",
    "get_selkov1968_glycolysis_oscillator",
    "get_sir",
    "get_sird",
    "get_tripartite_dynamics",
    "get_yokota1985",
    "get_zhu_2009",
    "mc",
    "mca",
    "plot",
    "scan",
]
