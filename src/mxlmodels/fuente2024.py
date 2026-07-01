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

from collections.abc import Iterable
from functools import partial

import numpy as np
from mxlpy import Derived, InitialAssignment, Model
from scipy.optimize import least_squares


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


def _initial_equations(
    initial_guess: Iterable,
    pq_tot: float,
    h_stroma: float,
    atot: float,
    n_psi: float,
    stoic_psii: float,
    npq_max: float,
    b_h: float,
    v_stroma: float,
    v_lumen: float,
    k1p: float,
    k1m: float,
    k2p: float,
    k2m: float,
    k3: float,
    k_npq: float,
    n_npq: float,
    k4: float,
    k5: float,
    c_eq_p: float,
    k6: float,
    k7: float,
    pfd: float,
    k_x: float,
    l_psi: float,
    stoic_psi: float,
    sigma_psi: float,
    n_a: float,
) -> list[float]:

    q_active, pq_ox, psi_ox, h_lumen, atp = initial_guess

    light = _osc_light(
        pfd=pfd,
        pfd_add=0,
        f=0,
        time=0,
    )
    pq_red = pq_tot - pq_ox
    psi_red = n_psi - psi_ox
    adp = atot - atp
    sigma_psii = _sigma_psii(npq_max=npq_max, q_active=q_active)
    rcii_closed = _rcii_closed(
        k1p=k1p,
        pq_ox=pq_ox,
        sigma_psii=sigma_psii,
        light=light,
        k1m=k1m,
        pq_red=pq_red,
    )
    rcii_open = _rcii_open(
        k1p=k1p,
        pq_ox=pq_ox,
        sigma_psii=sigma_psii,
        k1m=k1m,
        pq_red=pq_red,
    )

    v_PSII_O2 = _v_psii_o2(
        stoic_psii=stoic_psii,
        sigma_psii=sigma_psii,
        light=light,
        rcii_closed=rcii_closed,
    )
    v_ps1 = _v_psi(
        stoic_psi=stoic_psi,
        sigma_psi=sigma_psi,
        light=light,
        psi_ox=psi_ox,
        l_psi=l_psi,
    )
    v_PSII_PQ = _v_psii_pq(
        k1p=k1p,
        rcii_closed=rcii_closed,
        pq_ox=pq_ox,
        k1m=k1m,
        rcii_open=rcii_open,
        pq_red=pq_red,
    )
    v_PQH2_PSI = _v_pqh2_psi(
        k2p=k2p,
        pq_red=pq_red,
        psi_ox=psi_ox,
        k2m=k2m,
        pq_ox=pq_ox,
        psi_red=psi_red,
    )
    v3 = _v3(
        k3=k3,
        h_lumen=h_lumen,
        q_active=q_active,
        k_npq=k_npq,
        n=n_npq,
    )
    v4 = _mass_action_1s(q_active, k4)
    v_ATPsynth = _v_at_psynth(
        k5=k5,
        adp=adp,
        atp=atp,
        h_stroma=h_stroma,
        h_lumen=h_lumen,
        c_eq_p=c_eq_p,
    )
    v_ATPcons = _mass_action_1s(atp, k6)
    v_Leak = _v_leak(
        k7=k7,
        h_lumen=h_lumen,
        h_stroma=h_stroma,
    )
    v_PQ = _mass_action_1s(psi_red, k_x)

    alpha = b_h / (n_a * v_lumen)
    beta = 14 / 3 * b_h * v_stroma / v_lumen

    dqactive_dt = v3 - v4
    dpqox_dt = 1 / 2 * (v_PQH2_PSI - v_PSII_PQ) + v_PQ
    dpsiox_dt = v_ps1 - v_PQH2_PSI
    dhlumen_dt = (
        alpha * v_PSII_O2 + alpha * v_PQH2_PSI - beta * v_ATPsynth - b_h * v_Leak
    )
    datp_dt = v_ATPsynth - v_ATPcons
    return [dqactive_dt, dpqox_dt, dpsiox_dt, dhlumen_dt, datp_dt]


def _x0_guess(
    pq_tot: float,
    h_stroma: float,
    atot: float,
    n_psi: float,
) -> tuple[float, float, float, float, float]:
    PQ_ox_stst = pq_tot / 2
    h_lumen_stst = h_stroma * 10
    qactive_stst = 0.5
    atp_stst = atot / 2
    psi_ox_stst = n_psi / 2

    return qactive_stst, PQ_ox_stst, psi_ox_stst, h_lumen_stst, atp_stst


def _initial_combined(
    pq_tot: float,
    h_stroma: float,
    atot: float,
    n_psi: float,
    stoic_psii: float,
    npq_max: float,
    b_h: float,
    v_stroma: float,
    v_lumen: float,
    k1p: float,
    k1m: float,
    k2p: float,
    k2m: float,
    k3: float,
    k_npq: float,
    n_npq: float,
    k4: float,
    k5: float,
    c_eq_p: float,
    k6: float,
    k7: float,
    pfd: float,
    k_x: float,
    l_psi: float,
    stoic_psi: float,
    sigma_psi: float,
    n_a: float,
    extract_str: str,
) -> float:

    bounds = [
        (0, 1),  # Q_active
        (0, pq_tot),  # PQ_ox
        (0, n_psi),  # PSI_ox
        (h_stroma, h_stroma * 100),  # h_lumen
        (0, atot),  # ATP
    ]

    bounds_low = [b[0] for b in bounds]
    bounds_high = [b[1] for b in bounds]

    res = least_squares(
        fun=_initial_equations,
        x0=_x0_guess(
            pq_tot=pq_tot,
            h_stroma=h_stroma,
            atot=atot,
            n_psi=n_psi,
        ),
        args=(
            pq_tot,
            h_stroma,
            atot,
            n_psi,
            stoic_psii,
            npq_max,
            b_h,
            v_stroma,
            v_lumen,
            k1p,
            k1m,
            k2p,
            k2m,
            k3,
            k_npq,
            n_npq,
            k4,
            k5,
            c_eq_p,
            k6,
            k7,
            pfd,
            k_x,
            l_psi,
            stoic_psi,
            sigma_psi,
            n_a,
        ),
        bounds=(bounds_low, bounds_high),
        method="trf",
        xtol=1e-8,
    )
    pointer = {
        "qactive": 0,
        "pqox": 1,
        "psiox": 2,
        "hlumen": 3,
        "atp": 4,
    }
    return np.real(res.x[pointer[extract_str]])


def get_fuente_2024(*, at_reference: bool = False) -> Model:
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
    """
    m = Model()

    m.add_parameters(
        {
            "stoic_PSII": 1,
            "stoic_PSI": 1,
            "PQ_tot": 7,
            "H_stroma": 0.015848931924611134,
            "AP_tot": 1000,
            "V_lumen": 2.62e-21,
            "V_stroma": 2.09e-20,
            "sigma_PSI_0": 1,
            "k1p": 25000,
            "k1m": 2500,
            "k2p": 100,
            "k2m": 10,
            "k3": 0.05,
            "k4": 0.004,
            "k5": 100,
            "k6": 10,
            "k7": 500,
            "k_X": 1,
            "L_PSI": 10000,
            "bH": 0.01,
            "NPQ_max": 0.6,
            "cEqP": 4.3e-08,
            "keq_NPQ": 1,
            "n_NPQ": 5.3,
            "N_A": 6.02214076e17,
            "PPFD": 50,
            "PPFD_add": 0,
            "f": 1,
            "PSI_total": 1,
            "Fluo_0": 0.25,
            "Q_total": 1,
        }
    )
    if at_reference:
        m.add_variables(
            {
                "Q_active": np.float64(1.6044277821745498e-23),
                "PQ": np.float64(3.537090541057567),
                "PSI_ox": np.float64(0.197310072778891),
                "H_lumen": np.float64(0.4120700665522831),
                "ATP_st": np.float64(144.95412072145785),
            }
        )
    else:
        args = [
            "PQ_tot",
            "H_stroma",
            "AP_tot",
            "PSI_total",
            "stoic_PSII",
            "NPQ_max",
            "bH",
            "V_stroma",
            "V_lumen",
            "k1p",
            "k1m",
            "k2p",
            "k2m",
            "k3",
            "keq_NPQ",
            "n_NPQ",
            "k4",
            "k5",
            "cEqP",
            "k6",
            "k7",
            "PPFD",
            "k_X",
            "L_PSI",
            "stoic_PSI",
            "sigma_PSI_0",
            "N_A",
        ]
        m.add_variables(
            {
                "Q_active": InitialAssignment(
                    fn=partial(_initial_combined, extract_str="qactive"),
                    args=args,
                ),
                "PQ": InitialAssignment(
                    fn=partial(_initial_combined, extract_str="pqox"),
                    args=args,
                ),
                "PSI_ox": InitialAssignment(
                    fn=partial(_initial_combined, extract_str="psiox"),
                    args=args,
                ),
                "H_lumen": InitialAssignment(
                    fn=partial(_initial_combined, extract_str="hlumen"),
                    args=args,
                ),
                "ATP_st": InitialAssignment(
                    fn=partial(_initial_combined, extract_str="atp"),
                    args=args,
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
        fn=_flourescence,
        args=["Fluo_0", "RCII_closed", "sigma_PSII"],
    )

    m.add_derived(
        name="NPQ",
        fn=_npq,
        args=["NPQ_max", "Q_active"],
    )

    m.add_derived(
        name="O2",
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
        fn=_v_leak,
        args=["k7", "H_lumen", "H_stroma"],
        stoichiometry={
            "H_lumen": -1,
        },
    )
    m.add_reaction(
        name="v_PQ",
        fn=_mass_action_1s,
        args=["PQH_2", "k_X"],
        stoichiometry={
            "PQ": 1,
        },
    )

    return m
