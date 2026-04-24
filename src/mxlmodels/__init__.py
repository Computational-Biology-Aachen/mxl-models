"""MxlModels is a Python package of reference mechanistic models.

It contains the same models as in the [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) repo,
but written as single, flat files to make inspection easier.

"""

from .dyn_entro import create_model as get_dynamic_enterobactin
from .ebeling2026 import create_model as get_ebeling_2026
from .lotka_volterra_v1 import create_model as get_lotka_volterra_v1
from .lotka_volterra_v2 import create_model as get_lotka_volterra_v2
from .matuszynska2016_npq import create_model as get_matuszynska2016_npq
from .matuszynska2016_phd import create_model as get_matuszynska2016_phd
from .matuszynska2019 import create_model as get_matuszynska2019
from .poolman2000 import create_model as get_poolman2000
from .pop_dyn import create_model as get_population_dynamics
from .saadat2021 import create_model as get_saadat2021
from .trip_dyn import create_model as get_tripartite_dynamics
from .yokota1985 import create_model as get_yokota1985

__all__ = [
    "get_dynamic_enterobactin",
    "get_ebeling_2026",
    "get_lotka_volterra_v1",
    "get_lotka_volterra_v2",
    "get_matuszynska2016_npq",
    "get_matuszynska2016_phd",
    "get_matuszynska2019",
    "get_poolman2000",
    "get_population_dynamics",
    "get_saadat2021",
    "get_tripartite_dynamics",
    "get_yokota1985",
]
