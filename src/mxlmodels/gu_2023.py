import math

from mxlpy import Model


def f_q(q: float, a_q: float) -> float:
    """Redox poise balance between Cyt b6f and PSII."""
    return (1.0 + a_q) / (1.0 + a_q * q)


def f_T(T: float, T0: float, E_T: float) -> float:
    """
    Placeholder standardized temperature response.
    Set E_T=0 or f_T=1 if temperature response is not fitted.
    """
    if E_T == 0:
        return 1.0
    return math.exp(E_T * (T - T0) / (T * T0))


def f_s(Jg: float, b_s: float, c_s: float) -> float:
    """
    Placeholder swelling/crowding function.
    Set b_s=0 or c_s=0 to collapse to 1.
    The exact functional form should be checked against the paper text.
    """
    return 1.0 / (1.0 + c_s * (1.0 - math.exp(-b_s * Jg)))


def j_psii_gu(
    q: float,
    U: float,
    R1: float,
    R2: float,
    q_r: float,
    a_q: float,
    b_s: float,
    c_s: float,
    Jg: float,
    T: float,
    T0: float,
    E_T: float,
) -> float:
    """
    Gu et al. steady-state redox PET model.

    J_PSII = 2 U f_T f_s f_q (q_r - q) q /
             [ (R1 + 2 R2 f_s f_q - 1) q + q_r ]
    """
    fq = f_q(q, a_q)
    fs = f_s(Jg, b_s, c_s)
    ft = f_T(T, T0, E_T)

    numerator = 2.0 * U * ft * fs * fq * (q_r - q) * q
    denominator = (R1 + 2.0 * R2 * fs * fq - 1.0) * q + q_r

    return numerator / denominator


def get_gu_2023() -> Model:
    return (
        Model()
        .add_variables(
            {
                # q is treated as externally supplied measured open PSII fraction
                "q": 0.7,
            }
        )
        .add_parameters(
            {
                # composite PET parameters
                "U": 250.0,  # µmol m-2 s-1, max PQ/PQH2 oxidation potential
                "R1": 0.2,  # first resistance
                "R2": 0.5,  # second resistance
                "q_r": 1.0,  # reversible PSII fraction
                # regulatory functions
                "a_q": 0.0,  # Cyt/PSII redox-poise parameter
                "b_s": 0.0,  # swelling/crowding parameter
                "c_s": 0.0,  # max crowding impact
                "Jg": 500.0,  # gross excitation flux
                # temperature response
                "T": 298.15,
                "T0": 298.15,
                "E_T": 0.0,
            }
        )
        .add_derived(
            "J_PSII",
            j_psii_gu,
            args=[
                "q",
                "U",
                "R1",
                "R2",
                "q_r",
                "a_q",
                "b_s",
                "c_s",
                "Jg",
                "T",
                "T0",
                "E_T",
            ],
        )
    )
