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


def get_gu2023(
    q: float = 0.7,
    U: float = 250.0,
    R1: float = 0.2,
    R2: float = 0.5,
    q_r: float = 1.0,
    a_q: float = 0.0,
    b_s: float = 0.0,
    c_s: float = 0.0,
    Jg: float = 500.0,
    T: float = 298.15,
    T0: float = 298.15,
    E_T: float = 0.0,
) -> Model:
    
    ft = 1.0 if E_T == 0 else math.exp(E_T * (T - T0) / (T * T0))
    fq = (1.0 + a_q) / (1.0 + a_q * q)
    fs = 1.0 / (1.0 + c_s * (1.0 - math.exp(-b_s * Jg)))


    numerator = 2.0 * U * ft * fs * fq * (q_r - q) * q
    denominator = (R1 + 2.0 * R2 * fs * fq - 1.0) * q + q_r
    
    j_psii = numerator / denominator
    h_cyt = q * (1.0 + a_q) / (1.0 + a_q * q)
    
    return {"J_PSII": j_psii, "h_cyt": h_cyt}