import numpy as np


def electron_transport(PPFD, Jmax, alpha, theta):
    absorbed = alpha * np.asarray(PPFD, dtype=float)
    discriminant = (absorbed + Jmax) ** 2 - 4 * theta * absorbed * Jmax
    return (absorbed + Jmax - np.sqrt(np.maximum(discriminant, 0))) / (2 * theta)


def get_bernacchi_2023(
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

    rubp = J * (Ci_safe - Gamma_star) / (4 * Ci_safe + 8 * Gamma_star) - Rd

    tpu = np.full_like(Ci_safe, 3 * TPU - Rd)
    assimilation = np.minimum.reduce([rubisco, rubp, tpu])

    return rubisco, rubp, tpu, assimilation
