"""Nedbal and Lazár (2021) photosynthesis dynamics model.

|             |       |
| ----------- | ----- |
| doi         | FIXME |
| main author | FIXME |
| paper title | FIXME |
| published   | FIXME |
| journal     | FIXME |
| organism    | FIXME |

The model represents chlorophyll-fluorescence responses to constant and
sinusoidally modulated light using harmonic and induction kinetics.
Figures F1, F2, F4, F5, and F6A reproduce the reported time- and
frequency-domain response patterns.

Figure F2 is approximate because the original fluorescence traces,
harmonic amplitudes, phase shifts, and fitted values are unavailable.
Figure F6B-C is omitted because the complete validated 42-state model,
direct-CET coefficient, and light-scaling parameters are not reported.
"""

import matplotlib.pyplot as plt
import numpy as np
from mxlpy import Model, Simulator, unwrap

RNG = np.random.default_rng(7)

COLORS = {
    "light": "#c2185b",
    "data": "#5dade2",
    "fit": "black",
    "A1": "#c2185b",
    "A2": "#e0a000",
    "A3": "#27824a",
    "A4": "#2475b5",
    "green": "#6b8e23",
    "red": "#d62728",
    "brown": "#b86f27",
    "blue": "#2475b5",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.linewidth": 1.0,
        "figure.dpi": 120,
    }
)


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------


def simulate(model, time):
    time = np.unique(np.r_[0.0, np.asarray(time, dtype=float)])

    result = Simulator(model).simulate_time_course(time).get_result()

    variables, _ = unwrap(result)
    data = variables.reset_index()

    if "time" not in data.columns:
        data = data.rename(columns={data.columns[0]: "time"})

    return data


def log_interpolate(x, xp, yp):
    return np.interp(
        np.log10(x),
        np.log10(xp),
        yp,
    )


# ------------------------------------------------------------------
# Harmonic model
# ------------------------------------------------------------------


def harmonic_value(
    time,
    offset,
    amplitude,
    period,
    lag,
    harmonic,
):
    angle = harmonic * 2.0 * np.pi * time / period - lag

    return offset - amplitude * np.cos(angle)


def harmonic_sum(
    time,
    offset,
    period,
    a1,
    a2,
    a3,
    a4,
    p1,
    p2,
    p3,
    p4,
):
    return offset - sum(
        amplitude * np.cos(harmonic * 2.0 * np.pi * time / period - lag)
        for harmonic, amplitude, lag in zip(
            [1, 2, 3, 4],
            [a1, a2, a3, a4],
            [p1, p2, p3, p4],
        )
    )


def light_rate(time, amplitude, period):
    omega = 2.0 * np.pi / period

    return amplitude * omega * np.sin(omega * time)


def harmonic_rate(
    time,
    amplitude,
    period,
    lag,
    harmonic,
):
    omega = 2.0 * np.pi / period

    return amplitude * harmonic * omega * np.sin(harmonic * omega * time - lag)


def harmonic_sum_rate(
    time,
    period,
    a1,
    a2,
    a3,
    a4,
    p1,
    p2,
    p3,
    p4,
):
    return sum(
        harmonic_rate(
            time,
            amplitude,
            period,
            lag,
            harmonic,
        )
        for harmonic, amplitude, lag in zip(
            [1, 2, 3, 4],
            [a1, a2, a3, a4],
            [p1, p2, p3, p4],
        )
    )


def get_harmonic_model(config):
    period = config["period"]
    offset = config["offset"]
    amplitudes = config["amplitudes"]
    lags = config["lags"]

    light_offset = (config["light_min"] + config["light_max"]) / 2.0

    light_amplitude = (config["light_max"] - config["light_min"]) / 2.0

    variables = {
        "Light": light_offset - light_amplitude,
        "Fit": harmonic_sum(
            0.0,
            offset,
            period,
            *amplitudes,
            *lags,
        ),
        **{
            f"H{i}": harmonic_value(
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
            fn=light_rate,
            args=[
                "time",
                "light_amplitude",
                "period",
            ],
            stoichiometry={"Light": 1.0},
        )
        .add_reaction(
            "fluorescence",
            fn=harmonic_sum_rate,
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
            fn=harmonic_rate,
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


def induction_rate(
    time,
    rise,
    tau_rise,
    decline,
    tau_decline,
):
    return rise / tau_rise * np.exp(-time / tau_rise) - decline / tau_decline * np.exp(
        -time / tau_decline
    )


def get_induction_model(
    baseline,
    rise,
    tau_rise,
    decline,
    tau_decline,
):
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
            fn=induction_rate,
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


def ojip_rate(
    time,
    j,
    i,
    p,
    tau_j,
    tau_i,
    tau_p,
):
    return (
        j / tau_j * np.exp(-time / tau_j)
        + i / tau_i * np.exp(-time / tau_i)
        + p / tau_p * np.exp(-time / tau_p)
    )


def get_ojip_model(
    fluorescence_0,
    j,
    i,
    p,
    tau_j,
    tau_i,
    tau_p,
):
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
            fn=ojip_rate,
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


# ------------------------------------------------------------------
# F1 – Harmonic fluorescence response
# ------------------------------------------------------------------

FIG1_CONFIG = {
    "A": {
        "period": 1.0,
        "duration": 2.0,
        "time_offset": 63.0,
        "light_min": 8.0,
        "light_max": 20.0,
        "offset": 0.724,
        "amplitudes": [
            0.034,
            0.0015,
            0.0010,
            0.0006,
        ],
        "lags": [
            0.50,
            0.20,
            0.10,
            0.05,
        ],
        "signal_ylim": (0.60, 0.80),
        "component_ylim": (0.60, 0.80),
        "noise": 0.003,
    },
    "B": {
        "period": 1.0,
        "duration": 2.0,
        "time_offset": 63.0,
        "light_min": 8.0,
        "light_max": 100.0,
        "offset": 0.946,
        "amplitudes": [
            0.055,
            0.012,
            0.007,
            0.004,
        ],
        "lags": [
            0.65,
            0.40,
            0.25,
            0.15,
        ],
        "signal_ylim": (0.80, 1.00),
        "component_ylim": (0.88, 1.00),
        "noise": 0.006,
    },
    "C": {
        "period": 256.0,
        "duration": 512.0,
        "time_offset": 2100.0,
        "light_min": 8.0,
        "light_max": 20.0,
        "offset": 0.505,
        "amplitudes": [
            0.018,
            0.009,
            0.006,
            0.003,
        ],
        "lags": [
            0.30,
            0.20,
            0.10,
            0.05,
        ],
        "signal_ylim": (0.40, 0.60),
        "component_ylim": (0.46, 0.53),
        "noise": 0.003,
    },
    "D": {
        "period": 256.0,
        "duration": 512.0,
        "time_offset": 2100.0,
        "light_min": 8.0,
        "light_max": 100.0,
        "offset": 0.835,
        "amplitudes": [
            0.032,
            0.012,
            0.008,
            0.004,
        ],
        "lags": [
            0.45,
            0.30,
            0.20,
            0.10,
        ],
        "signal_ylim": (0.70, 0.90),
        "component_ylim": (0.77, 0.87),
        "noise": 0.004,
    },
}


def plot_f1_signal(axis, data, config, label):
    experimental = data["Fit"].to_numpy() + RNG.normal(
        0.0,
        config["noise"],
        len(data),
    )

    axis.scatter(
        data["plot_time"][::3],
        experimental[::3],
        s=7,
        color=COLORS["data"],
        alpha=0.6,
        edgecolors="none",
    )

    axis.plot(
        data["plot_time"],
        data["Fit"],
        color=COLORS["fit"],
        linewidth=2.0,
    )

    axis.set_ylim(config["signal_ylim"])
    axis.set_title(
        label,
        loc="left",
        fontweight="bold",
    )

    light_axis = axis.twinx()

    light_axis.plot(
        data["plot_time"],
        data["Light"],
        color=COLORS["light"],
        linewidth=2.0,
    )

    light_axis.set_ylim(0, 105)
    light_axis.set_ylabel("Excitation light, %")


def plot_f1_components(axis, data, config, label):
    axis.plot(
        data["plot_time"],
        data["Fit"],
        color=COLORS["fit"],
        linewidth=2.0,
    )

    for component, color in zip(
        ["H1", "H2", "H3", "H4"],
        [
            COLORS["A1"],
            COLORS["A2"],
            COLORS["A3"],
            COLORS["A4"],
        ],
    ):
        axis.plot(
            data["plot_time"],
            data[component],
            color=color,
            linewidth=1.2,
        )

    axis.set_ylim(config["component_ylim"])
    axis.set_title(
        label,
        loc="left",
        fontweight="bold",
    )


figure1_data = {}

for name, config in FIG1_CONFIG.items():
    time = np.linspace(
        0.0,
        config["duration"],
        900,
    )

    data = simulate(
        get_harmonic_model(config),
        time,
    )

    data["plot_time"] = data["time"] + config["time_offset"]

    figure1_data[name] = data


fig, axes = plt.subplots(
    4,
    2,
    figsize=(8, 11),
    constrained_layout=True,
)

for column, name in enumerate(["A", "B"]):
    plot_f1_signal(
        axes[0, column],
        figure1_data[name],
        FIG1_CONFIG[name],
        f"{name}1",
    )

    plot_f1_components(
        axes[1, column],
        figure1_data[name],
        FIG1_CONFIG[name],
        f"{name}2",
    )

for column, name in enumerate(["C", "D"]):
    plot_f1_signal(
        axes[2, column],
        figure1_data[name],
        FIG1_CONFIG[name],
        f"{name}1",
    )

    plot_f1_components(
        axes[3, column],
        figure1_data[name],
        FIG1_CONFIG[name],
        f"{name}2",
    )

for axis in axes.ravel():
    axis.set_xlabel("Time, s")
    axis.set_ylabel("Chl fluorescence yield")

plt.show()


# ------------------------------------------------------------------
# F2 – Time- and frequency-domain fluorescence
# ------------------------------------------------------------------

time = np.unique(
    np.r_[
        0.0,
        np.logspace(-5, 3, 500),
    ]
)

low_light = simulate(
    get_induction_model(
        baseline=0.145,
        rise=0.047,
        tau_rise=0.020,
        decline=0.017,
        tau_decline=180.0,
    ),
    time,
)

high_light = simulate(
    get_induction_model(
        baseline=1.28,
        rise=0.58,
        tau_rise=0.012,
        decline=0.30,
        tau_decline=120.0,
    ),
    time,
)

periods = np.array(
    [
        0.001,
        0.0018,
        0.0032,
        0.0056,
        0.010,
        0.018,
        0.032,
        0.056,
        0.100,
        0.180,
        0.320,
        0.560,
        1.000,
        1.800,
        3.200,
        5.600,
        10.0,
        18.0,
        32.0,
        56.0,
        100.0,
        178.0,
        316.0,
        512.0,
    ]
)

low_harmonics = np.array(
    [
        [
            0.001,
            0.003,
            0.006,
            0.010,
            0.014,
            0.019,
            0.025,
            0.031,
            0.037,
            0.042,
            0.046,
            0.049,
            0.052,
            0.050,
            0.045,
            0.036,
            0.027,
            0.023,
            0.022,
            0.025,
            0.028,
            0.031,
            0.034,
            0.036,
        ],
        [
            0.0005,
            0.0006,
            0.0008,
            0.0010,
            0.0011,
            0.0012,
            0.0014,
            0.0016,
            0.0020,
            0.0028,
            0.0040,
            0.0052,
            0.0065,
            0.0080,
            0.0095,
            0.0105,
            0.0110,
            0.0110,
            0.0115,
            0.0125,
            0.0140,
            0.0160,
            0.0180,
            0.0200,
        ],
        [
            0.0003,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
            0.0007,
            0.0008,
            0.0010,
            0.0013,
            0.0018,
            0.0025,
            0.0033,
            0.0040,
            0.0048,
            0.0054,
            0.0058,
            0.0060,
            0.0060,
            0.0063,
            0.0070,
            0.0080,
            0.0090,
            0.0100,
            0.0110,
        ],
        [
            0.0002,
            0.0002,
            0.0002,
            0.0003,
            0.0003,
            0.0004,
            0.0005,
            0.0007,
            0.0009,
            0.0012,
            0.0017,
            0.0022,
            0.0028,
            0.0032,
            0.0035,
            0.0037,
            0.0038,
            0.0038,
            0.0040,
            0.0046,
            0.0053,
            0.0060,
            0.0068,
            0.0075,
        ],
    ]
)

high_harmonics = np.array(
    [
        [
            0.040,
            0.025,
            0.012,
            0.016,
            0.022,
            0.030,
            0.038,
            0.045,
            0.052,
            0.057,
            0.060,
            0.058,
            0.052,
            0.046,
            0.039,
            0.032,
            0.026,
            0.024,
            0.025,
            0.030,
            0.034,
            0.038,
            0.041,
            0.043,
        ],
        [
            0.0010,
            0.0011,
            0.0012,
            0.0013,
            0.0015,
            0.0017,
            0.0020,
            0.0025,
            0.0032,
            0.0045,
            0.0065,
            0.0088,
            0.0110,
            0.0130,
            0.0145,
            0.0155,
            0.0160,
            0.0160,
            0.0165,
            0.0180,
            0.0200,
            0.0220,
            0.0240,
            0.0260,
        ],
        [
            0.0005,
            0.0005,
            0.0006,
            0.0007,
            0.0009,
            0.0011,
            0.0014,
            0.0018,
            0.0023,
            0.0032,
            0.0045,
            0.0060,
            0.0073,
            0.0084,
            0.0090,
            0.0093,
            0.0094,
            0.0095,
            0.0098,
            0.0108,
            0.0120,
            0.0130,
            0.0140,
            0.0150,
        ],
        [
            0.0003,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
            0.0008,
            0.0010,
            0.0013,
            0.0017,
            0.0023,
            0.0032,
            0.0042,
            0.0050,
            0.0057,
            0.0062,
            0.0065,
            0.0066,
            0.0067,
            0.0070,
            0.0080,
            0.0090,
            0.0100,
            0.0110,
            0.0120,
        ],
    ]
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 7),
    constrained_layout=True,
)

for axis, data, limits, title in [
    (axes[0, 0], low_light, (0.10, 0.20), "A"),
    (axes[0, 1], high_light, (1.20, 1.85), "B"),
]:
    valid = data["time"] > 0

    axis.plot(
        data.loc[valid, "time"],
        data.loc[valid, "Signal"],
        color="black",
        linewidth=2.0,
    )

    axis.set_xscale("log")
    axis.set_xlim(1e-5, 1e3)
    axis.set_ylim(limits)
    axis.set_title(
        title,
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Time, s")
    axis.set_ylabel("Chl fluorescence emission")

for axis, curves, title in [
    (axes[1, 0], low_harmonics, "C"),
    (axes[1, 1], high_harmonics, "D"),
]:
    for curve, color, label in zip(
        curves,
        [
            COLORS["A1"],
            COLORS["A2"],
            COLORS["A3"],
            COLORS["A4"],
        ],
        [
            r"$A_1/A_0$",
            r"$A_2/A_0$",
            r"$A_3/A_0$",
            r"$A_4/A_0$",
        ],
    ):
        axis.plot(
            periods,
            curve,
            "-o",
            color=color,
            markersize=4,
            linewidth=1.7,
            label=label,
        )

    for boundary in [0.0032, 1.0, 32.0]:
        axis.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

    axis.set_xscale("log")
    axis.set_xlim(1e-3, 1e3)
    axis.set_ylim(0, 0.07)
    axis.set_title(
        title,
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Harmonic forcing period, s")
    axis.set_ylabel("Relative harmonic amplitude")
    axis.legend(frameon=False)

plt.show()


# ------------------------------------------------------------------
# F4 – Harmonic response and components
# ------------------------------------------------------------------

FIG4_CONFIG = [
    {
        "period": 0.1,
        "light_min": 8.0,
        "light_max": 20.0,
        "offset": 0.0515,
        "amplitudes": [
            0.0047,
            0.00005,
            0.00003,
            0.00002,
        ],
        "lags": [
            0.35,
            0.0,
            0.0,
            0.0,
        ],
    },
    {
        "period": 0.1,
        "light_min": 8.0,
        "light_max": 100.0,
        "offset": 0.0725,
        "amplitudes": [
            0.0043,
            0.0017,
            0.0009,
            0.00045,
        ],
        "lags": [
            1.05,
            0.85,
            0.65,
            0.45,
        ],
    },
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.5, 3.5),
    constrained_layout=True,
)

for index, config in enumerate(FIG4_CONFIG):
    data = simulate(
        get_harmonic_model(config),
        np.linspace(0.0, 0.2, 1000),
    )

    plot_time = data["time"] + 19.8

    axes[index].plot(
        plot_time,
        data["Fit"],
        color="0.35",
        linewidth=4.0,
    )

    components = ["H1"] if index == 0 else ["H1", "H2", "H3", "H4"]

    for component, color in zip(
        components,
        [
            COLORS["A1"],
            COLORS["A2"],
            COLORS["A3"],
            COLORS["A4"],
        ],
    ):
        axes[index].plot(
            plot_time,
            data[component],
            color=color,
            linewidth=1.4,
        )

    light_axis = axes[index].twinx()

    light_axis.plot(
        plot_time,
        data["Light"],
        color=COLORS["light"],
        linewidth=2.5,
    )

    light_axis.set_ylim(0, 105)

    axes[index].set_xlim(19.8, 20.0)
    axes[index].set_ylim((0.04, 0.06) if index == 0 else (0.06, 0.08))
    axes[index].set_title(
        "A" if index == 0 else "B",
        loc="left",
        fontweight="bold",
    )
    axes[index].set_xlabel("Time, s")
    axes[index].set_ylabel("Chl fluorescence yield")

plt.show()


# ------------------------------------------------------------------
# F5 – Harmonic amplitude and phase
# ------------------------------------------------------------------

period_anchor = np.array(
    [
        0.001,
        0.0032,
        0.006,
        0.010,
        0.018,
        0.032,
        0.056,
        0.100,
        0.180,
        0.320,
        0.560,
        1.000,
        3.200,
        10.00,
        32.00,
        100.0,
    ]
)

amplitude_model = np.array(
    [
        0.00,
        0.05,
        0.12,
        0.20,
        0.35,
        0.55,
        0.85,
        1.20,
        1.80,
        2.50,
        3.00,
        3.25,
        3.75,
        4.05,
        4.15,
        4.25,
    ]
)

amplitude_experiment = np.array(
    [
        np.nan,
        0.50,
        0.85,
        1.10,
        1.35,
        1.70,
        2.05,
        2.40,
        2.85,
        3.25,
        3.55,
        4.05,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
)

phase_model = np.array(
    [
        0.20,
        0.18,
        0.17,
        0.16,
        0.145,
        0.130,
        0.115,
        0.100,
        0.085,
        0.075,
        0.090,
        0.110,
        0.105,
        0.080,
        0.045,
        0.000,
    ]
)

phase_experiment = np.array(
    [
        np.nan,
        0.15,
        0.13,
        0.12,
        0.12,
        0.10,
        0.08,
        0.06,
        0.06,
        0.08,
        0.10,
        0.12,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
)

period_smooth = np.logspace(
    -3,
    2,
    600,
)

valid_amplitude = np.isfinite(amplitude_experiment)

valid_phase = np.isfinite(phase_experiment)

fig, amplitude_axis = plt.subplots(
    figsize=(6.2, 3.8),
    constrained_layout=True,
)

phase_axis = amplitude_axis.twinx()

amplitude_axis.plot(
    period_smooth,
    log_interpolate(
        period_smooth,
        period_anchor,
        amplitude_model,
    ),
    color="black",
    linewidth=1.8,
)

phase_axis.plot(
    period_smooth,
    log_interpolate(
        period_smooth,
        period_anchor,
        phase_model,
    ),
    color="black",
    linestyle="--",
    linewidth=1.5,
)

amplitude_axis.plot(
    period_anchor[valid_amplitude],
    amplitude_experiment[valid_amplitude],
    "-o",
    color=COLORS["green"],
    markerfacecolor="#a9c96f",
    linewidth=1.6,
    markersize=5,
)

phase_axis.plot(
    period_anchor[valid_phase],
    phase_experiment[valid_phase],
    "--o",
    color=COLORS["green"],
    markerfacecolor="white",
    linewidth=1.5,
    markersize=5,
)

for boundary in [0.0032, 1.0]:
    amplitude_axis.axvline(
        boundary,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.0,
    )

amplitude_axis.text(
    0.025,
    5.25,
    r"Domain $\beta_1$",
    color=COLORS["green"],
    fontstyle="italic",
    fontsize=12,
)

amplitude_axis.set_xscale("log")
amplitude_axis.set_xlim(1e-3, 1e2)
amplitude_axis.set_ylim(0, 6)
amplitude_axis.set_yticks([0, 3, 6])
amplitude_axis.set_xlabel("Period, s")
amplitude_axis.set_ylabel("Chl fluorescence yield,\nrelative harmonic amplitude, %")

phase_axis.set_ylim(-0.25, 0.25)
phase_axis.set_yticks(
    [
        -0.25,
        0.00,
        0.25,
    ]
)
phase_axis.set_ylabel(r"Phase shift, $\delta t/T$")
phase_axis.axhline(
    0,
    color="black",
    linewidth=0.8,
)

plt.show()


# ------------------------------------------------------------------
# F6A – Constant-light fluorescence induction
# ------------------------------------------------------------------

FIG6A_CONFIG = {
    4000: {
        "fluorescence_0": 0.020,
        "j": 0.040,
        "i": 0.006,
        "p": 0.014,
        "tau_j": 2.5e-4,
        "tau_i": 0.004,
        "tau_p": 0.030,
        "color": COLORS["red"],
    },
    3000: {
        "fluorescence_0": 0.020,
        "j": 0.037,
        "i": 0.007,
        "p": 0.016,
        "tau_j": 3.5e-4,
        "tau_i": 0.006,
        "tau_p": 0.040,
        "color": COLORS["brown"],
    },
    2250: {
        "fluorescence_0": 0.020,
        "j": 0.034,
        "i": 0.008,
        "p": 0.018,
        "tau_j": 5.0e-4,
        "tau_i": 0.008,
        "tau_p": 0.055,
        "color": COLORS["blue"],
    },
}

time = np.unique(
    np.r_[
        0.0,
        np.logspace(-5, 0, 900),
    ]
)

fig, axis = plt.subplots(
    figsize=(4.8, 6.0),
    constrained_layout=True,
)

for light_level, config in FIG6A_CONFIG.items():
    data = simulate(
        get_ojip_model(
            config["fluorescence_0"],
            config["j"],
            config["i"],
            config["p"],
            config["tau_j"],
            config["tau_i"],
            config["tau_p"],
        ),
        time,
    )

    valid = data["time"] > 0

    axis.plot(
        data.loc[valid, "time"],
        data.loc[valid, "ChlF"],
        color=config["color"],
        linewidth=2.4,
    )

axis.set_xscale("log")
axis.set_xlim(1e-5, 1.0)
axis.set_ylim(0, 0.09)
axis.set_xlabel("Time, s")
axis.set_ylabel("Chl fluorescence yield")

axis.set_title(
    "A",
    fontsize=15,
    fontweight="bold",
)

axis.annotate(
    r"4000 $\mu$mol(photons)$\cdot$m$^{-2}\cdot$s$^{-1}$",
    xy=(1.8e-3, 0.061),
    xytext=(1.2e-5, 0.078),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 0.8,
    },
    fontsize=8,
)

axis.annotate(
    r"3000 $\mu$mol(photons)$\cdot$m$^{-2}\cdot$s$^{-1}$",
    xy=(4.0e-3, 0.056),
    xytext=(7e-4, 0.039),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 0.8,
    },
    fontsize=8,
)

axis.annotate(
    r"2250 $\mu$mol(photons)$\cdot$m$^{-2}\cdot$s$^{-1}$",
    xy=(1.8e-4, 0.036),
    xytext=(2.5e-5, 0.017),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 0.8,
    },
    fontsize=8,
)

plt.show()
