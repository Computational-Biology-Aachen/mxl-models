r"""Simple Calvin Cycle model from Zhu et al. (2009).

|             |                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------ |
| doi         | 10.1016/j.nonrwa.2008.01.021                                                                                             |
| main author | Xin-Guang Zhu                                                                                                            |
| paper title | A simple model of the Calvin cycle has only one physiologically feasible steady state under the same external conditions |
| published   | June 2009                                                                                                                |
| journal     | Nonlinear Analysis: Real World Applications                                                                              |
| organism    | C3 plant (Calvin cycle)                                                                                                  |
| Ported by   | ElouenCorvest ( @ElouenCorvest )                                                                                         |

Note: dynamics do not yet perfectly reproduce the published figures.

The [Zhu 2009](https://doi.org/10.1016/j.nonrwa.2008.01.021) model is a
deliberately simplified kinetic model of the Calvin-Benson-Bassham (CBB) cycle,
the carbon-fixing dark reactions of photosynthesis. It tracks only five
metabolites - ribulose-1,5-bisphosphate (RuBP), 3-phosphoglycerate (PGA),
1,3-bisphosphoglycerate (DPGA), glyceraldehyde-3-phosphate (GAP), and
ribulose-5-phosphate (Ru5P) - and lumps the many intermediate steps of the
cycle into a handful of Michaelis-Menten reactions, with ATP supplied as a
fixed external parameter rather than a dynamic variable. This reduction keeps
the system small enough to be analysed mathematically while still capturing the
essential autocatalytic structure of carbon fixation, where RuBP is both
consumed by RuBisCO and regenerated downstream.
"""

from mxlpy import Model, Parameter, Slider, Variable
from mxlpy.fns import michaelis_menten_1s
from mxlpy.units import liter, mmol, mumol, second, sqm

_mm = mmol / liter
_mm_s = _mm / second

__all__ = ["get_zhu_2009"]


def _enzyme_atp_dependent(
    s1: float,
    atp: float,
    vmax: float,
    km_s1: float,
    km_atp: float,
) -> float:
    return vmax * s1 * atp / ((s1 + km_s1) * (atp + km_atp))


def _calculate_a(v1: float) -> float:
    return v1 * 33.33


def get_zhu_2009() -> Model:
    r"""Simple Calvin Cycle Model developed by Zhu et al. (2009)."""
    m = Model()
    m.add_parameters(
        {
            # Vmax:
            "V1_max": Parameter(
                3.78, unit=_mm_s, slider=Slider(min="1.9", max="7.6", step="0.1")
            ),  # [mM/S] Zhu et al 2009
            "V2_max": Parameter(
                11.75, unit=_mm_s, slider=Slider(min="5.9", max="23.5", step="0.1")
            ),  # [mM/S] Zhu et al 2009
            "V3_max": Parameter(
                5.04, unit=_mm_s, slider=Slider(min="2.5", max="10.1", step="0.1")
            ),  # [mM/S] Zhu et al 2009
            "V4_max": Parameter(
                3.05, unit=_mm_s, slider=Slider(min="1.5", max="6.1", step="0.1")
            ),  # [mM/S] estimate by Zhu et al.
            "V5_max": Parameter(
                3.0, unit=_mm_s, slider=Slider(min="1.5", max="6", step="0.1")
            ),  # [mM/S] Zhu et al 2009
            "V6_max": Parameter(
                0.1, unit=_mm_s, slider=Slider(min="0.1", max="0.2", step="0.1")
            ),  # [mM/S] estimate by Zhu et al.
            "V13_max": Parameter(
                8.0, unit=_mm_s, slider=Slider(min="4.0", max="16", step="0.1")
            ),
            # Km:
            "K_m1": Parameter(1.0, unit=_mm),  # [mM] Zhu et al 2009
            "K_m21": Parameter(0.24, unit=_mm),  # [mM] Zhu et al 2009
            "K_m22": Parameter(0.39, unit=_mm),  # [mM] Zhu et al 2009
            "K_m3": Parameter(0.5, unit=_mm),  # [mM] Zhu et al 2009
            "K_m4": Parameter(0.84, unit=_mm),  # [mM] Zhu et al 2009
            "K_m51": Parameter(0.75, unit=_mm),  # [mM] Zhu et al 2009
            "K_m52": Parameter(
                0.275, unit=_mm
            ),  # [mM] not specified in the paper but fitted by us to match the figure
            "K_m6": Parameter(5, unit=_mm),  # [mM] Zhu et al 2009
            "K_m131": Parameter(0.15, unit=_mm),  # [mM] Zhu et al 2009
            "K_m132": Parameter(0.059, unit=_mm),  # [mM] Zhu et al 2009
            "ATP": Parameter(
                0.2, unit=_mm, slider=Slider(min="0.1", max="0.4", step="0.1")
            ),  # [mM]Not in Zhu et al 2009 but in the reference [9] in the paper
        }
    )
    m.add_variables(
        {
            "RuBP": Variable(2, unit=_mm, display_name="RUBP"),  # Zhu et al 2009
            "PGA": Variable(2.4, unit=_mm),  # Zhu et al 2009
            "DPGA": Variable(1, unit=_mm, display_name="BPGA"),  # Zhu et al 2009
            "Ru5P": Variable(1, unit=_mm, display_name="RU5P"),  # Zhu et al 2009
            "GAP": Variable(1, unit=_mm),  # Zhu et al 2009
        }
    )
    m.add_reaction(
        "v1",
        fn=michaelis_menten_1s,
        stoichiometry={"RuBP": -1, "PGA": 2},
        args=[
            "RuBP",
            "V1_max",
            "K_m1",
        ],
    )
    m.add_reaction(
        "v2",
        fn=_enzyme_atp_dependent,
        stoichiometry={"PGA": -1, "DPGA": 1},
        args=["PGA", "ATP", "V2_max", "K_m21", "K_m22"],
    )
    m.add_reaction(
        "v3",
        fn=michaelis_menten_1s,
        stoichiometry={"DPGA": -1, "GAP": 1},
        args=["DPGA", "V3_max", "K_m3"],
    )
    m.add_reaction(
        "v4",
        fn=michaelis_menten_1s,
        stoichiometry={"GAP": -1, "Ru5P": 0.6},
        args=["GAP", "V4_max", "K_m4"],
    )
    m.add_reaction(
        "v5",
        fn=_enzyme_atp_dependent,
        stoichiometry={
            "PGA": -1,
        },
        args=["PGA", "ATP", "V5_max", "K_m51", "K_m52"],
    )
    m.add_reaction(
        "v6",
        fn=michaelis_menten_1s,
        stoichiometry={
            "GAP": -1,
        },
        args=["GAP", "V6_max", "K_m6"],
    )
    m.add_reaction(
        "v13",
        fn=_enzyme_atp_dependent,
        stoichiometry={"Ru5P": -1, "RuBP": 1},
        args=["Ru5P", "ATP", "V13_max", "K_m131", "K_m132"],
    )
    # 1 mmol l^-1 s^-1 on stroma volume basis corresponds to 33.3 umol m^-2 s^-1 on leaf area basis
    m.add_derived("A", fn=_calculate_a, args=["v1"], unit=mumol / sqm / second)
    return m
