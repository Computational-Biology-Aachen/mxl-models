r"""Nedbal and Lazár (2021) photosynthesis dynamics model.

|             |                                 |
| ----------- | ------------------------------- |
| doi         | FIXME                           |
| main author | FIXME                           |
| paper title | FIXME                           |
| published   | FIXME                           |
| journal     | FIXME                           |
| organism    | FIXME                           |
| Ported by   | Tanvir Hassan ( @Tanvir96rwth ) |

The model represents chlorophyll-fluorescence responses to constant and
sinusoidally modulated light using harmonic and induction kinetics. Figures F1,
F2, F4, F5, and F6A reproduce the reported time- and frequency-domain response
patterns.

Figure F2 is approximate because the original fluorescence traces, harmonic
amplitudes, phase shifts, and fitted values are unavailable. Figure F6B-C is
omitted because the complete validated 42-state model, direct-CET coefficient,
and light-scaling parameters are not reported.
"""

import numpy as np
from mxlpy import Model

# ------------------------------------------------------------------
# Harmonic model
# ------------------------------------------------------------------


def _harmonic_value(
    time: float,
    offset: float,
    amplitude: float,
    period: float,
    lag: float,
    harmonic: float,
) -> float:
    angle = harmonic * 2.0 * np.pi * time / period - lag

    return offset - amplitude * np.cos(angle)


def _harmonic_sum(
    time: float,
    offset: float,
    period: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
) -> float:
    return offset - sum(
        amplitude * np.cos(harmonic * 2.0 * np.pi * time / period - lag)
        for harmonic, amplitude, lag in zip(
            [1, 2, 3, 4],
            [a1, a2, a3, a4],
            [p1, p2, p3, p4],
            strict=True,
        )
    )


def _light_rate(
    time: float,
    amplitude: float,
    period: float,
) -> float:
    omega = 2.0 * np.pi / period

    return amplitude * omega * np.sin(omega * time)


def _harmonic_rate(
    time: float,
    amplitude: float,
    period: float,
    lag: float,
    harmonic: float,
) -> float:
    omega = 2.0 * np.pi / period

    return amplitude * harmonic * omega * np.sin(harmonic * omega * time - lag)


def _harmonic_sum_rate(
    time: float,
    period: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
) -> float:
    # return sum(
    #     harmonic_rate(
    #         time,
    #         amplitude,
    #         period,
    #         lag,
    #         harmonic,
    #     )
    #     for harmonic, amplitude, lag in zip(
    #         [1, 2, 3, 4],
    #         [a1, a2, a3, a4],
    #         [p1, p2, p3, p4],
    #     )
    # )
    return (
        _harmonic_rate(time, 1, period, a1, p1)
        + _harmonic_rate(time, 2, period, a2, p2)
        + _harmonic_rate(time, 3, period, a3, p3)
        + _harmonic_rate(time, 4, period, a4, p4)
    )


def get_harmonic_model(
    period: float,
    offset: float,
    amplitudes: list[float],
    lags: list[float],
    light_min: float,
    light_max: float,
) -> Model:
    r"""Get the harmonic model."""
    light_offset = (light_min + light_max) / 2.0
    light_amplitude = (light_max - light_min) / 2.0
    variables = {
        "Light": light_offset - light_amplitude,
        "Fit": _harmonic_sum(
            0.0,
            offset,
            period,
            *amplitudes,
            *lags,
        ),
        **{
            f"H{i}": _harmonic_value(
                0.0,
                offset,
                amplitudes[i - 1],
                period,
                lags[i - 1],
                i,
            )
            for i in range(1, 5)
        },
    }

    parameters = {
        "period": period,
        "light_amplitude": light_amplitude,
        **{f"a{i}": amplitudes[i - 1] for i in range(1, 5)},
        **{f"p{i}": lags[i - 1] for i in range(1, 5)},
        **{f"n{i}": float(i) for i in range(1, 5)},
    }

    model = (
        Model()
        .add_variables(variables)
        .add_parameters(parameters)
        .add_reaction(
            "light",
            fn=_light_rate,
            args=[
                "time",
                "light_amplitude",
                "period",
            ],
            stoichiometry={"Light": 1.0},
        )
        .add_reaction(
            "fluorescence",
            fn=_harmonic_sum_rate,
            args=[
                "time",
                "period",
                "a1",
                "a2",
                "a3",
                "a4",
                "p1",
                "p2",
                "p3",
                "p4",
            ],
            stoichiometry={"Fit": 1.0},
        )
    )

    for i in range(1, 5):
        model.add_reaction(
            f"harmonic_{i}",
            fn=_harmonic_rate,
            args=[
                "time",
                f"a{i}",
                "period",
                f"p{i}",
                f"n{i}",
            ],
            stoichiometry={f"H{i}": 1.0},
        )

    return model


# ------------------------------------------------------------------
# Fluorescence-induction models
# ------------------------------------------------------------------


def _induction_rate(
    time: float,
    rise: float,
    tau_rise: float,
    decline: float,
    tau_decline: float,
) -> float:
    return rise / tau_rise * np.exp(-time / tau_rise) - decline / tau_decline * np.exp(
        -time / tau_decline
    )


def get_induction_model(
    baseline: float,
    rise: float,
    tau_rise: float,
    decline: float,
    tau_decline: float,
) -> Model:
    r"""Get the induction model."""
    return (
        Model()
        .add_variables(
            {
                "Signal": baseline,
            }
        )
        .add_parameters(
            {
                "rise": rise,
                "tau_rise": tau_rise,
                "decline": decline,
                "tau_decline": tau_decline,
            }
        )
        .add_reaction(
            "induction",
            fn=_induction_rate,
            args=[
                "time",
                "rise",
                "tau_rise",
                "decline",
                "tau_decline",
            ],
            stoichiometry={"Signal": 1.0},
        )
    )


def _ojip_rate(
    time: float,
    j: float,
    i: float,
    p: float,
    tau_j: float,
    tau_i: float,
    tau_p: float,
) -> float:
    return (
        j / tau_j * np.exp(-time / tau_j)
        + i / tau_i * np.exp(-time / tau_i)
        + p / tau_p * np.exp(-time / tau_p)
    )


def get_ojip_model(
    fluorescence_0: float,
    j: float,
    i: float,
    p: float,
    tau_j: float,
    tau_i: float,
    tau_p: float,
) -> Model:
    r"""Get the ojip model."""
    return (
        Model()
        .add_variables(
            {
                "ChlF": fluorescence_0,
            }
        )
        .add_parameters(
            {
                "j": j,
                "i": i,
                "p": p,
                "tau_j": tau_j,
                "tau_i": tau_i,
                "tau_p": tau_p,
            }
        )
        .add_reaction(
            "ojip",
            fn=_ojip_rate,
            args=[
                "time",
                "j",
                "i",
                "p",
                "tau_j",
                "tau_i",
                "tau_p",
            ],
            stoichiometry={"ChlF": 1.0},
        )
    )
