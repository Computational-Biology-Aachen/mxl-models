r"""Fuente 2024 model.

|             |                                                                                                                      |
| ----------- | -------------------------------------------------------------------------------------------------------------------- |
| doi         | 10.1016/j.plaphy.2024.109138                                                                                         |
| main author | David Fuente                                                                                                         |
| paper title | A mathematical model to simulate the dynamics of photosynthetic light reactions under harmonically oscillating light |
| published   | September 2024                                                                                                       |
| journal     | Plant Physiology and Biochemistry                                                                                    |
| organism    | Chlamydomonas reinhardtii                                                                                            |
| Ported by   | Marvin van Aalst ( @marvinvanaalst )                                                                                 |

The Fuente2024 model is a kinetic model of photosynthesis designed according to
Occam's razor, aiming to capture the core processes of photosynthesis with
minimal complexity. The model focuses on responses of the photosynthetic
machinery to dynamic light oscillations and includes only light-dependent
reactions. It contains simplified representations of photosystem II,
photosystem I, the plastoquinone pool, and proton and ATP concentrations in the
lumen and stroma. The model also describes activation of non-photochemical
quenching (NPQ), chlorophyll fluorescence dynamics, and oxygen evolution rates.

Light intensity is represented as a sinusoidal function with adjustable
amplitude and frequency. To facilitate comparison with models that assume
constant irradiance, the oscillation is defined around a base light intensity.
The use of frequency-dependent light inputs enables additional analyses that
are not possible under steady-state conditions. In the original study, the
model was used to generate Bode plots of fluorescence responses to light
oscillations and to compare them with experimental data from Chlamydomonas
reinhardtii.

Consistent with its minimalist design, the model serves as a foundation for
further extension while demonstrating a novel approach to photosynthesis
modeling. The authors show that dynamic light protocols can reveal insights
that may be missed by traditional steady-state models. To support reuse and
extension, they provide a detailed Wolfram Language notebook that reproduces
several figures from the publication.
"""

import numpy as np
from mxlpy import Derived, Model, Parameter, Variable
from mxlpy.units import dimensionless, liter, mumol, second

# Pools are per RCII (dimensionless), H+ and ATP in uM. PPFD acts as an excitation
# rate: the effective antenna size sigma_II = 1 s^-1/(umol photons m^-2 s^-1) is folded in
_um = mumol / liter
_per_s = 1 / second


def _mass_action_1s(
    s1: float,
    k_fwd: float,
) -> float:
    return k_fwd * s1


def _moiety_1(
    concentration: float,
    total: float,
) -> float:
    return total - concentration


def _osc_light(
    pfd: float,
    pfd_add: float,
    f: float,
    time: float,
) -> float:
    return pfd + pfd_add * np.cos(2 * np.pi * f * time)


def _sigma_psii(
    npq_max: float,
    q_active: float,
) -> float:
    return 1 - npq_max * q_active


def _rcii_closed(
    k1p: float,
    pq_ox: float,
    sigma_psii: float,
    light: float,
    k1m: float,
    pq_red: float,
) -> float:
    top = 1
    bottom = 1 + k1p * pq_ox / (sigma_psii * light + k1m * pq_red)
    return top / bottom


def _rcii_open(
    k1p: float,
    pq_ox: float,
    sigma_psii: float,
    k1m: float,
    pq_red: float,
) -> float:
    # FIXME: the paper (Eq. 4-5) defines RCII_open = 1 - RCII_closed
    #   = k1p * pq_ox / (sigma_psii * light + k1m * pq_red + k1p * pq_ox)
    # This port drops the light term, which makes the expression dimensionally
    # inconsistent (sigma_psii is dimensionless, k1m * pq_red is a rate) and
    # changes the v_PSII_PQ flux. Check against the authors' Wolfram notebook,
    # then fix it, update the expected right hand side in test_rhs and remove
    # v_PSII_PQ from the known inconsistencies in test_units.
    return k1p * pq_ox / ((sigma_psii + k1m * pq_red) + k1p * pq_ox)


def _flourescence(
    fluo_0: float,
    rcii_closed: float,
    sigma_psii: float,
) -> float:
    return (fluo_0 + rcii_closed) * sigma_psii


def _npq(
    npq_max: float,
    q_active: float,
) -> float:
    return npq_max * q_active / (1 - npq_max * q_active)


def _o2(
    n_psii: float,
    k1p: float,
    rci_iclosed: float,
    pq: float,
    k1m: float,
    pq_red: float,
) -> float:
    return (n_psii * (k1p * rci_iclosed * pq - k1m * (1 - rci_iclosed) * pq_red)) / 4


def _v_psii_o2(
    stoic_psii: float,
    sigma_psii: float,
    light: float,
    rcii_closed: float,
) -> float:
    return stoic_psii * sigma_psii * light * (1 - rcii_closed)


def _x_div_yz(
    x: float,
    y: float,
    z: float,
) -> float:
    return x / (y * z)


def _v_psi(
    stoic_psi: float,
    sigma_psi: float,
    light: float,
    psi_ox: float,
    l_psi: float,
) -> float:
    return (
        stoic_psi * sigma_psi * l_psi * light / (l_psi + light) * (stoic_psi - psi_ox)
    )


def _v_psii_pq(
    k1p: float,
    rcii_closed: float,
    pq_ox: float,
    k1m: float,
    rcii_open: float,
    pq_red: float,
) -> float:
    return k1p * rcii_closed * pq_ox - (k1m * rcii_open * pq_red)


def _v_pqh2_psi(
    k2p: float,
    pq_red: float,
    psi_ox: float,
    k2m: float,
    pq_ox: float,
    psi_red: float,
) -> float:
    return k2p * pq_red * psi_ox - k2m * pq_ox * psi_red


def _v3(
    k3: float,
    h_lumen: float,
    q_active: float,
    k_npq: float,
    n: float,
) -> float:
    return k3 * (1 - q_active) / (1 + (k_npq / h_lumen) ** n)


def _v_at_psynth(
    k5: float,
    adp: float,
    atp: float,
    h_stroma: float,
    h_lumen: float,
    c_eq_p: float,
) -> float:
    return k5 * (adp - atp * (h_stroma / h_lumen) ** (14 / 3) / c_eq_p)


def _proton_generation(
    v_stroma: float,
    v_lumen: float,
    b_h: float,
) -> float:
    return -14 / 3 * v_stroma / v_lumen * b_h


def _v_leak(
    k7: float,
    h_lumen: float,
    h_stroma: float,
) -> float:
    return k7 * (h_lumen - h_stroma)


def get_fuente_2024() -> Model:
    r"""Fuente 2024 model

    The Fuente2024 model is a kinetic model of photosynthesis designed according to
    Occam's razor, aiming to capture the core processes of photosynthesis with
    minimal complexity. The model focuses on responses of the photosynthetic
    machinery to dynamic light oscillations and includes only light-dependent
    reactions. It contains simplified representations of photosystem II,
    photosystem I, the plastoquinone pool, and proton and ATP concentrations in the
    lumen and stroma. The model also describes activation of non-photochemical
    quenching (NPQ), chlorophyll fluorescence dynamics, and oxygen evolution rates.

    Light intensity is represented as a sinusoidal function with adjustable
    amplitude and frequency. To facilitate comparison with models that assume
    constant irradiance, the oscillation is defined around a base light intensity.
    The use of frequency-dependent light inputs enables additional analyses that
    are not possible under steady-state conditions. In the original study, the
    model was used to generate Bode plots of fluorescence responses to light
    oscillations and to compare them with experimental data from Chlamydomonas
    reinhardtii.

    Consistent with its minimalist design, the model serves as a foundation for
    further extension while demonstrating a novel approach to photosynthesis
    modeling. The authors show that dynamic light protocols can reveal insights
    that may be missed by traditional steady-state models. To support reuse and
    extension, they provide a detailed Wolfram Language notebook that reproduces
    several figures from the publication.

    The initial state is a fixed reference state. The paper starts every
    oscillating-light simulation from the steady state in the constant light
    u0 (Section 3.2): to reproduce it, simulate to steady state with
    `PPFD_add = 0` first and continue from that state.
    """
    m = Model()

    m.add_parameters(
        {
            "stoic_PSII": Parameter(1, unit=dimensionless),
            "stoic_PSI": Parameter(1, unit=dimensionless),
            "PQ_tot": Parameter(7, unit=dimensionless, display_name="PQ_tot"),
            "H_stroma": Parameter(0.015848931924611134, unit=_um),
            "AP_tot": Parameter(1000, unit=_um, display_name="AP_tot"),
            "V_lumen": Parameter(2.62e-21, unit=liter),
            "V_stroma": Parameter(2.09e-20, unit=liter),
            "sigma_PSI_0": Parameter(1, unit=dimensionless),
            "k1p": Parameter(25000, unit=_per_s),
            "k1m": Parameter(2500, unit=_per_s),
            "k2p": Parameter(100, unit=_per_s),
            "k2m": Parameter(10, unit=_per_s),
            "k3": Parameter(0.05, unit=_per_s),
            "k4": Parameter(0.004, unit=_per_s),
            "k5": Parameter(100, unit=_per_s),
            "k6": Parameter(10, unit=_per_s),
            "k7": Parameter(500, unit=_per_s),
            "k_X": Parameter(1, unit=_per_s),
            "L_PSI": Parameter(10000, unit=_per_s),
            "bH": Parameter(0.01, unit=dimensionless, display_name="bH"),
            "NPQ_max": Parameter(0.6, unit=dimensionless, display_name="NPQ_max"),
            "cEqP": Parameter(4.3e-08, unit=dimensionless),
            "keq_NPQ": Parameter(1, unit=_um),
            "n_NPQ": Parameter(5.3, unit=dimensionless),
            "N_A": Parameter(6.02214076e17, unit=1 / mumol),
            "PPFD": Parameter(50, unit=_per_s, display_name="PPFD"),
            "PPFD_add": Parameter(0, unit=_per_s),
            "f": Parameter(1, unit=_per_s),
            "PSI_total": Parameter(1, unit=dimensionless, display_name="PSI_total"),
            "Fluo_0": Parameter(0.25, unit=dimensionless),
            "Q_total": Parameter(1, unit=dimensionless),
        }
    )
    m.add_variables(
        {
            "Q_active": Variable(
                np.float64(1.6044277821745498e-23), unit=dimensionless
            ),
            "PQ": Variable(
                np.float64(3.537090541057567), unit=dimensionless, display_name="PQ"
            ),
            "PSI_ox": Variable(np.float64(0.197310072778891), unit=dimensionless),
            "H_lumen": Variable(np.float64(0.4120700665522831), unit=_um),
            "ATP_st": Variable(
                np.float64(144.95412072145785), unit=_um, display_name="ATP"
            ),
        }
    )

    m.add_derived(
        name="Q_inactive",
        fn=_moiety_1,
        args=["Q_active", "Q_total"],
    )

    m.add_derived(
        name="PQH_2",
        display_name="PQH2",
        fn=_moiety_1,
        args=["PQ", "PQ_tot"],
    )

    m.add_derived(
        name="PSI_red",
        fn=_moiety_1,
        args=["PSI_ox", "PSI_total"],
    )

    m.add_derived(
        name="ADP_st",
        display_name="ADP",
        fn=_moiety_1,
        args=["ATP_st", "AP_tot"],
    )

    m.add_derived(
        name="osc_light",
        fn=_osc_light,
        args=["PPFD", "PPFD_add", "f", "time"],
    )

    m.add_derived(
        name="sigma_PSII",
        fn=_sigma_psii,
        args=["NPQ_max", "Q_active"],
    )

    m.add_derived(
        name="RCII_closed",
        fn=_rcii_closed,
        args=["k1p", "PQ", "sigma_PSII", "osc_light", "k1m", "PQH_2"],
    )

    m.add_derived(
        name="RCII_open",
        fn=_rcii_open,
        args=["k1p", "PQ", "sigma_PSII", "k1m", "PQH_2"],
    )

    m.add_derived(
        name="Fluo",
        display_name="Fluo",
        fn=_flourescence,
        args=["Fluo_0", "RCII_closed", "sigma_PSII"],
    )

    m.add_derived(
        name="NPQ",
        display_name="NPQ",
        fn=_npq,
        args=["NPQ_max", "Q_active"],
    )

    m.add_derived(
        name="O2",
        display_name="O2",
        fn=_o2,
        args=["PSI_total", "k1p", "RCII_closed", "PQ", "k1m", "PQH_2"],
    )

    m.add_reaction(
        name="v_PSII_O2",
        fn=_v_psii_o2,
        args=["stoic_PSII", "sigma_PSII", "osc_light", "RCII_closed"],
        stoichiometry={
            "H_lumen": Derived(
                fn=_x_div_yz,
                args=["bH", "V_lumen", "N_A"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_PSI",
        fn=_v_psi,
        args=["stoic_PSI", "sigma_PSI_0", "osc_light", "PSI_ox", "L_PSI"],
        stoichiometry={
            "PSI_ox": 1,
        },
    )
    m.add_reaction(
        name="v_PSII_PQ",
        fn=_v_psii_pq,
        args=["k1p", "RCII_closed", "PQ", "k1m", "RCII_open", "PQH_2"],
        stoichiometry={
            "PQ": -0.5,
        },
    )
    m.add_reaction(
        name="v_PQH2_PSI",
        fn=_v_pqh2_psi,
        args=["k2p", "PQH_2", "PSI_ox", "k2m", "PQ", "PSI_red"],
        stoichiometry={
            "PQ": 0.5,
            "PSI_ox": -1,
            "H_lumen": Derived(
                fn=_x_div_yz,
                args=["bH", "V_lumen", "N_A"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v3",
        fn=_v3,
        args=["k3", "H_lumen", "Q_active", "keq_NPQ", "n_NPQ"],
        stoichiometry={
            "Q_active": 1,
        },
    )
    m.add_reaction(
        name="v4",
        fn=_mass_action_1s,
        args=["Q_active", "k4"],
        stoichiometry={
            "Q_active": -1,
        },
    )
    m.add_reaction(
        name="v_ATPsynth",
        fn=_v_at_psynth,
        args=["k5", "ADP_st", "ATP_st", "H_stroma", "H_lumen", "cEqP"],
        stoichiometry={
            "ATP_st": 1,
            "H_lumen": Derived(
                fn=_proton_generation,
                args=["V_stroma", "V_lumen", "bH"],
                unit=None,
            ),
        },
    )
    m.add_reaction(
        name="v_ATPcons",
        fn=_mass_action_1s,
        args=["ATP_st", "k6"],
        stoichiometry={
            "ATP_st": -1,
        },
    )
    m.add_reaction(
        name="v_Leak",
        display_name="proton leak",
        fn=_v_leak,
        args=["k7", "H_lumen", "H_stroma"],
        stoichiometry={
            "H_lumen": -1,
        },
    )
    m.add_reaction(
        name="v_PQ",
        display_name="PQ reduction",
        fn=_mass_action_1s,
        args=["PQH_2", "k_X"],
        stoichiometry={
            "PQ": 1,
        },
    )

    return m
