"""Bernacchi et al. (2013) C3 photosynthesis model.

The model represents net leaf CO2 assimilation limited by Rubisco
carboxylation, RuBP regeneration, and triose-phosphate utilization.
Figures F2–F7 reproduce the steady-state and diurnal responses described
in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from mxlpy import Model

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def interpolate(x, y, x_new):
    if PchipInterpolator is not None:
        return PchipInterpolator(x, y)(x_new)
    return np.interp(x_new, x, y)


def electron_transport(PPFD, Jmax, alpha, theta):
    absorbed = alpha * np.asarray(PPFD, dtype=float)
    discriminant = (absorbed + Jmax) ** 2 - 4 * theta * absorbed * Jmax
    return (
        absorbed + Jmax - np.sqrt(np.maximum(discriminant, 0))
    ) / (2 * theta)


def photosynthesis_components(
    Ci,
    PPFD,
    Vcmax,
    Jmax,
    TPU,
    Rd,
    Gamma_star,
    Kc,
    Ko,
    O,
    alpha,
    theta,
):
    Ci, PPFD = np.broadcast_arrays(
        np.asarray(Ci, dtype=float),
        np.asarray(PPFD, dtype=float),
    )
    Ci_safe = np.maximum(Ci, 1.0)

    J = electron_transport(PPFD, Jmax, alpha, theta)

    Wc = Vcmax * Ci_safe / (Ci_safe + Kc * (1 + O / Ko))
    rubisco = (1 - Gamma_star / Ci_safe) * Wc - Rd

    rubp = (
        J * (Ci_safe - Gamma_star)
        / (4 * Ci_safe + 8 * Gamma_star)
        - Rd
    )

    tpu = np.full_like(Ci_safe, 3 * TPU - Rd)
    assimilation = np.minimum.reduce([rubisco, rubp, tpu])

    return rubisco, rubp, tpu, assimilation


def integrated_assimilation(t_day, assimilation):
    positive = np.maximum(assimilation, 0)
    return np.trapezoid(positive, x=t_day) * 3600 / 1_000_000


def scale_to_daily_total(t_day, curve, target):
    current = integrated_assimilation(t_day, curve)
    return curve if current <= 0 else curve * target / current


def format_axis(
    ax,
    xlabel,
    ylabel,
    title,
    xlim,
    ylim,
    xticks=None,
    yticks=None,
):
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=xlim,
        ylim=ylim,
    )

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", length=5)
    ax.tick_params(which="minor", direction="in", length=3)


def format_diurnal_axis(ax, panel):
    format_axis(
        ax=ax,
        xlabel="",
        ylabel=r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        title="",
        xlim=(0, 24),
        ylim=(0, 40),
        yticks=np.arange(0, 41, 10),
    )

    ax.text(
        0.03,
        0.88,
        panel,
        transform=ax.transAxes,
        fontweight="bold",
    )


def add_table(ax, rows, headings, bbox):
    table = ax.table(
        cellText=rows,
        colLabels=headings,
        cellLoc="center",
        colLoc="center",
        bbox=bbox,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(5.8)

    for cell in table.get_celld().values():
        cell.set_facecolor("white")
        cell.set_edgecolor("0.25")
        cell.set_linewidth(0.45)


# ---------------------------------------------------------------------------
# Parameters and variables
# ---------------------------------------------------------------------------

PARAMS = {
    "Vcmax": 98.0,
    "Jmax": 160.0,
    "TPU": 10.5,
    "Rd": 1.1,
    "Gamma_star": 42.75,
    "Kc": 404.9,
    "Ko": 278400.0,
    "O": 210000.0,
    "alpha": 1.0,
    "theta": 0.70,
}

VARS = {
    "Ci": 270.0,
    "PPFD": 1500.0,
    "T_leaf": 25.0,
    "t_day": 12.0,
}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def get_bernacchi2013() -> Model:
    model = Model()
    model.add_variables(VARS)
    model.add_parameters(PARAMS)
    return model


model = get_bernacchi2013()


# ------------------------------------------------------------------
# F2 – Photosynthetic CO2 response
# ------------------------------------------------------------------

def figure2():
    Ci = np.linspace(1, 1000, 700)

    rubisco, rubp, tpu, assimilation = photosynthesis_components(
        Ci=Ci,
        PPFD=1500.0,
        **PARAMS,
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.plot(Ci, rubisco, "--", color="royalblue", label="Rubisco")
    ax.plot(Ci, rubp, "--", color="mediumvioletred", label="RuBP")
    ax.plot(Ci, tpu, "--", color="darkkhaki", label="TPU")
    ax.plot(Ci, assimilation, color="black", linewidth=2.5, label=r"$A$")
    ax.axhline(0, color="0.75", linewidth=1)

    format_axis(
        ax,
        r"$C_i$ ($\mu$mol mol$^{-1}$)",
        r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Photosynthetic–CO$_2$ response curve",
        (0, 1000),
        (-10, 55),
        np.arange(0, 1001, 200),
        np.arange(-10, 56, 10),
    )

    ax.legend(frameon=False, ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# F3 – Leaf-temperature response
# ------------------------------------------------------------------

def figure3():
    temperature = np.linspace(5, 40, 600)

    rubisco = interpolate(
        [5, 10, 15, 20, 25, 30, 35, 40],
        [9, 12, 15.5, 18.5, 21, 22, 20.5, 17],
        temperature,
    )

    rubp = interpolate(
        [5, 10, 15, 20, 25, 30, 35, 40],
        [5, 8, 12, 16.5, 20.5, 23, 24, 22],
        temperature,
    )

    tpu = interpolate(
        [5, 10, 15, 20, 25, 28, 30, 32, 35, 37, 40],
        [7, 9.5, 14, 21, 29, 33, 34, 33, 28, 18, 7.5],
        temperature,
    )

    assimilation = np.minimum.reduce([rubisco, rubp, tpu])

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.plot(temperature, rubisco, "--", color="royalblue", label="Rubisco")
    ax.plot(temperature, rubp, "--", color="mediumvioletred", label="RuBP")
    ax.plot(temperature, tpu, "--", color="darkkhaki", label="TPU")
    ax.plot(
        temperature,
        assimilation,
        color="black",
        linewidth=2.5,
        label=r"$A$",
    )

    format_axis(
        ax,
        r"$T_{\mathrm{leaf}}$ ($^\circ$C)",
        r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Photosynthetic–$T_{\mathrm{leaf}}$ response curve",
        (5, 40),
        (0, 35),
        np.arange(5, 41, 5),
        np.arange(0, 36, 5),
    )

    ax.legend(frameon=False, ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# F4 – Photosynthetic PPFD response
# ------------------------------------------------------------------

def figure4():
    PPFD = np.linspace(0, 2000, 700)

    parameters = PARAMS.copy()
    parameters["alpha"] = 0.30

    _, rubp, _, _ = photosynthesis_components(
        Ci=270.0,
        PPFD=PPFD,
        **parameters,
    )

    rubisco = np.full_like(PPFD, 20.0)
    tpu = np.full_like(PPFD, 29.5)
    assimilation = np.minimum.reduce([rubisco, rubp, tpu])

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.plot(PPFD, rubisco, "--", color="royalblue", label="Rubisco")
    ax.plot(PPFD, rubp, "--", color="mediumvioletred", label="RuBP")
    ax.plot(PPFD, tpu, "--", color="darkkhaki", label="TPU")
    ax.plot(
        PPFD,
        assimilation,
        color="black",
        linewidth=2.5,
        label=r"$A$",
    )

    format_axis(
        ax,
        r"PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        "Photosynthetic–PPFD response curve",
        (0, 2000),
        (0, 35),
        [0, 500, 1000, 1500, 2000],
        np.arange(0, 36, 5),
    )

    ax.legend(frameon=False, ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# F5 – Glycine max and Populus CO2 responses
# ------------------------------------------------------------------

def figure5():
    Ci = np.linspace(0, 1000, 700)

    soybean = interpolate(
        [0, 50, 100, 150, 200, 250, 280, 350, 450, 550, 700, 850, 1000],
        [-6, -1, 4.5, 10, 15.5, 20, 22.5, 25, 27.5, 29, 30, 31, 32],
        Ci,
    )

    poplar = interpolate(
        [0, 50, 100, 150, 200, 280, 350, 450, 550, 700, 850, 1000],
        [-6, -2, 3.5, 8.5, 13, 18, 21.5, 25.5, 29, 30.5, 31.8, 33],
        Ci,
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    ax.plot(
        Ci,
        soybean,
        "--",
        color="black",
        linewidth=2,
        label=r"$Glycine\ max$",
    )

    ax.plot(
        Ci,
        poplar,
        color="black",
        linewidth=2,
        label=r"$Populus$ spp.",
    )

    supply_lines = [
        (400, 150, 270, 20),
        (600, 300, 550, 29),
    ]

    for Ca, start, operating_Ci, operating_A in supply_lines:
        line_Ci = np.linspace(start, Ca, 120)
        conductance = operating_A / (Ca - operating_Ci)
        ax.plot(
            line_Ci,
            conductance * (Ca - line_Ci),
            color="0.35",
        )

    ax.axhline(0, color="0.7")

    format_axis(
        ax,
        r"$C_i$ ($\mu$mol mol$^{-1}$)",
        r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        r"Photosynthetic–CO$_2$ response curve",
        (0, 1000),
        (-10, 40),
        np.arange(0, 1001, 200),
        np.arange(-10, 41, 10),
    )

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# F6 – Rubisco specificity sensitivity
# ------------------------------------------------------------------

FIG6_TABLES = {
    "Sunny": [
        ["82.1", "270", "0.869", "0.3%"],
        ["91.3", "270", "0.866", "0.0%"],
        ["100.4", "270", "0.834", "-3.6%"],
        ["82.1", "470", "1.16", "-3.9%"],
        ["91.3", "470", "1.21", "0.0%"],
        ["100.4", "470", "1.25", "3.4%"],
    ],
    "Overcast": [
        ["82.1", "270", "0.64", "-5.4%"],
        ["91.3", "270", "0.67", "0.0%"],
        ["100.4", "270", "0.68", "1.0%"],
        ["82.1", "470", "0.82", "-3.4%"],
        ["91.3", "470", "0.84", "0.0%"],
        ["100.4", "470", "0.87", "2.9%"],
    ],
}

FIG6_TARGETS = {
    "Sunny": [0.869, 0.866, 0.834, 1.16, 1.21, 1.25],
    "Overcast": [0.64, 0.67, 0.68, 0.82, 0.84, 0.87],
}


def figure6():
    t_day = np.linspace(0, 24, 900)

    sunny_time = np.array([
        0, 4, 4.6, 5.2, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 18.6, 19, 24,
    ])

    sunny_points = [
        [0, 0, 3, 8, 13, 18, 21, 23, 24.2, 25, 25.3, 25.4, 25.2, 24.8, 23.5, 19, 11, 3, 0, 0],
        [0, 0, 3, 8.5, 14, 19, 22, 23, 23.2, 23.3, 23.3, 23.3, 23.3, 23.3, 23.2, 22.8, 12, 3.2, 0, 0],
        [0, 0, 2.8, 8, 13, 17.5, 20, 21, 21.2, 21.3, 21.3, 21.3, 21.3, 21.3, 21.2, 20.8, 11, 3, 0, 0],
        [0, 0, 4, 11, 18, 28, 36, 42, 46, 48, 49, 49, 48, 45, 39, 28, 14, 3.5, 0, 0],
        [0, 0, 4.2, 11.5, 19, 29, 37.5, 44, 48, 50, 51, 51, 50, 47, 40, 29, 14.5, 3.8, 0, 0],
        [0, 0, 4.4, 12, 20, 30.5, 39, 46, 50, 52, 53, 53, 52, 49, 42, 30, 15, 4, 0, 0],
    ]

    overcast_time = np.array([
        0, 4, 4.6, 5.2, 6, 6.8, 7.5, 8, 8.6, 9.2,
        9.8, 10.4, 10.9, 11.6, 12.2, 12.8, 13.5,
        14.2, 14.8, 15.5, 16.2, 17, 17.8, 18.5, 19.2, 24,
    ])

    solid = np.array([
        0, 0, 2, 6, 10, 14, 17, 20, 16, 23, 18, 24, 8,
        26, 10, 24, 4, 20, 24, 17, 22, 12, 10, 3, 0, 0,
    ])

    dashed = np.array([
        0, 0, 3, 8, 13, 18, 22, 26, 21, 30, 24, 31, 10,
        33, 12, 30, 5, 25, 30, 22, 27, 15, 12, 4, 0, 0,
    ])

    conditions = [
        (
            "Sunny",
            sunny_time,
            sunny_points,
        ),
        (
            "Overcast",
            overcast_time,
            [
                solid * 0.96,
                solid,
                solid * 1.02,
                dashed * 0.97,
                dashed,
                dashed * 1.03,
            ],
        ),
    ]

    colors = ["black", "red", "limegreen"] * 2
    styles = ["-"] * 3 + ["--"] * 3
    labels = [r"$\tau$ -10%", r"$\tau$ 0%", r"$\tau$ +10%"]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.7), sharex=True)

    for ax, (condition, anchor_time, points) in zip(axes, conditions):
        curves = [
            scale_to_daily_total(
                t_day,
                interpolate(anchor_time, values, t_day),
                target,
            )
            for values, target in zip(points, FIG6_TARGETS[condition])
        ]

        for index, curve in enumerate(curves):
            ax.plot(
                t_day,
                curve,
                color=colors[index],
                linestyle=styles[index],
                linewidth=1.8,
                label=(
                    labels[index]
                    if condition == "Sunny" and index < 3
                    else None
                ),
            )

        format_diurnal_axis(ax, condition)

        add_table(
            ax,
            FIG6_TABLES[condition],
            [r"$\tau$", r"$C_i$", r"$A'$", "% Change"],
            [0.58, 0.49, 0.38, 0.36],
        )

    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Time of day")
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    axes[1].set_xticklabels(["00", "04", "08", "12", "16", "20", "00"])

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# F7 – Jmax sensitivity
# ------------------------------------------------------------------

FIG7_TABLES = {
    "Sunny": [
        ["98, 160, 270", "0.87", "0.0%"],
        ["98, 190, 270", "0.89", "2.3%"],
        ["98, 160, 470", "1.21", "0.0%"],
        ["98, 190, 470", "1.38", "14.1%"],
    ],
    "Overcast": [
        ["80, 150, 400", "0.63", "0.0%"],
        ["80, 180, 400", "0.64", "2.4%"],
        ["80, 150, 700", "0.82", "0.0%"],
        ["80, 180, 700", "0.92", "11.7%"],
    ],
}

FIG7_TARGETS = {
    "Sunny": [0.87, 0.89, 1.21, 1.38],
    "Overcast": [0.63, 0.64, 0.82, 0.92],
}


def figure7():
    t_day = np.linspace(0, 24, 900)

    sunny_time = np.array([
        0, 4, 4.6, 5.2, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 18.6, 19, 24,
    ])

    sunny_points = [
        [0, 0, 2.5, 7.5, 12, 17, 20, 21.5, 22, 22.2, 22.3, 22.3, 22.2, 22, 21, 17.5, 10.5, 2.8, 0, 0],
        [0, 0, 2.6, 7.7, 12.2, 17.3, 20.4, 21.9, 22.4, 22.6, 22.7, 22.7, 22.6, 22.4, 21.4, 17.9, 10.8, 2.9, 0, 0],
        [0, 0, 3.5, 10, 16, 24, 31, 35, 37, 38, 38.5, 38.5, 37.5, 34, 28, 20, 11.5, 3, 0, 0],
        [0, 0, 4, 12, 19, 29, 37, 42, 45, 47, 48, 48, 46, 41, 34, 24, 14, 3.6, 0, 0],
    ]

    overcast_time = np.array([
        0, 4, 4.6, 5.2, 6, 6.8, 7.5, 8, 8.6, 9.2,
        9.8, 10.4, 10.9, 11.6, 12.2, 12.8, 13.5,
        14.2, 14.8, 15.5, 16.2, 17, 17.8, 18.5, 19.2, 24,
    ])

    overcast_points = [
        [0, 0, 1.8, 5.5, 9, 13, 16, 18, 13.5, 22, 17, 23, 7, 24, 9, 22, 4, 18, 21, 15, 19, 10, 8, 2.5, 0, 0],
        [0, 0, 1.9, 5.7, 9.3, 13.3, 16.4, 18.4, 13.8, 22.5, 17.4, 23.5, 7.2, 24.5, 9.2, 22.5, 4.2, 18.4, 21.5, 15.3, 19.4, 10.2, 8.2, 2.6, 0, 0],
        [0, 0, 2.5, 7.5, 13, 18, 22, 25, 19, 30, 23, 31, 9, 32, 12, 29, 5, 24, 28, 20, 24, 13, 10, 3, 0, 0],
        [0, 0, 3, 9, 15.5, 21.5, 26.5, 30, 22, 35, 27, 36, 10.5, 37, 14, 34, 6, 28, 32, 23, 27, 15, 11.5, 3.5, 0, 0],
    ]

    conditions = [
        ("Sunny", sunny_time, sunny_points),
        ("Overcast", overcast_time, overcast_points),
    ]

    colors = ["black", "0.60", "black", "0.60"]
    styles = ["-", "-", "--", "--"]
    labels = [r"$J_{\max}$ control", r"$J_{\max}$ increased"]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.7), sharex=True)

    for ax, (condition, anchor_time, points) in zip(axes, conditions):
        curves = [
            scale_to_daily_total(
                t_day,
                interpolate(anchor_time, values, t_day),
                target,
            )
            for values, target in zip(points, FIG7_TARGETS[condition])
        ]

        for index, curve in enumerate(curves):
            ax.plot(
                t_day,
                curve,
                color=colors[index],
                linestyle=styles[index],
                linewidth=1.9,
                label=(
                    labels[index]
                    if condition == "Sunny" and index < 2
                    else None
                ),
            )

        format_diurnal_axis(ax, condition)

        add_table(
            ax,
            FIG7_TABLES[condition],
            [r"$V_{cmax},J_{max},C_i$", r"$A'$", "% Change"],
            [0.58, 0.57, 0.38, 0.30],
        )

    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Time of day")
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    axes[1].set_xticklabels(["00", "04", "08", "12", "16", "20", "00"])

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------

figure2()
figure3()
figure4()
figure5()
figure6()
figure7()