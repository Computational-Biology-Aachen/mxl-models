import math

# from mxlpy import SteadyStateModelBuilder
# from mxlpy.meta import generate_model_code_mxlweb


def _derived_ft(E_T: float, T: float, T0: float) -> float:
    return 1.0 if E_T == 0 else math.sqrt(T0 / T) * math.exp(E_T * (1.0 / T0 - 1.0 / T))

def _derived_fq(q: float, a_q: float) -> float:
    return (1.0 + a_q) / (1.0 + a_q * q)

def _derived_fs(alpha: float, PAR: float, b_s: float, c_s: float) -> float:
    return 1.0 / (1.0 + c_s * (math.exp(-b_s * alpha * PAR)))

def _derived_j_psii(U: float, R1: float, R2: float, q_r: float, q: float, ft: float, fs: float, fq: float) -> float:
    numerator = 2.0 * U * ft * fs * fq * (q_r - q) * q
    denominator = (R1 + 2.0 * R2 * fs * fq - 1.0) * q + q_r
    
    return numerator / denominator

def _derived_h_cyt(q: float, a_q: float) -> float:
    return q * (1.0 + a_q) / (1.0 + a_q * q)

def _derived_h_pqh2(j_psii: float, U: float, ft: float, fs: float, fq: float, q: float) -> float:
    return j_psii / (2 * U * ft * fs * fq * q)

def _derived_h_pq(h_pqh2: float) -> float:
    return 1.0 - h_pqh2

def get_gu2023(
    q: float = 0.7,
    U: float = 250.0,
    R1: float = 0.2,
    R2: float = 0.5,
    q_r: float = 1.0,
    a_q: float = 0.0,
    b_s: float = 0.0,
    c_s: float = 0.0,
    alpha: float = 0.85,
    PAR: float = 500 / 0.85,
    T: float = 298.15,
    T0: float = 298.15,
    E_T: float = 0.0
):
    ft = _derived_ft(E_T, T, T0)
    fq = _derived_fq(q, a_q)
    fs = _derived_fs(alpha, PAR, b_s, c_s)
    J_PSII = _derived_j_psii(U, R1, R2, q_r, q, ft, fs, fq)
    h_cyt = _derived_h_cyt(q, a_q)
    h_pqh2 = _derived_h_pqh2(J_PSII, U, ft, fs, fq, q)
    h_pq = _derived_h_pq(h_pqh2)
    return {
        "J_PSII": J_PSII,
        "h_cyt": h_cyt,
        "h_pqh2": h_pqh2,
        "h_pq": h_pq,
        "ft": ft,
        "fq": fq,
        "fs": fs,
    }


# def get_gu2023() -> SteadyStateModelBuilder:
#     """
#     Returns a SteadyStateModelBuilder for the Gu et al. 2023 model.
#     """
#     model = SteadyStateModelBuilder()
#     model.add_parameter("q", 0.7)
#     model.add_parameter("U", 250.0)
#     model.add_parameter("R1", 0.2)
#     model.add_parameter("R2", 0.5)
#     model.add_parameter("q_r", 1.0)
#     model.add_parameter("a_q", 0.0)
#     model.add_parameter("b_s", 0.0)
#     model.add_parameter("c_s", 0.0)
#     model.add_parameter("alpha", 0.85)
#     model.add_parameter("PAR", 500 / 0.85)
#     model.add_parameter("T", 298.15)
#     model.add_parameter("T0", 298.15)
#     model.add_parameter("E_T", 0.0)
    
#     model.add_derived("ft", _derived_ft, args=["E_T", "T", "T0"])
#     model.add_derived("fq", _derived_fq, args=["q", "a_q"])
#     model.add_derived("fs", _derived_fs, args=["alpha", "PAR", "b_s", "c_s"])
#     model.add_derived("J_PSII", _derived_j_psii, args=["U", "R1", "R2", "q_r", "q", "ft", "fs", "fq"])
#     model.add_derived("h_cyt", _derived_h_cyt, args=["q", "a_q"])
#     model.add_derived("h_pqh2", _derived_h_pqh2, args=["J_PSII", "U", "ft", "fs", "fq", "q"])
#     model.add_derived("h_pq", _derived_h_pq, args=["h_pqh2"])

#     return model
