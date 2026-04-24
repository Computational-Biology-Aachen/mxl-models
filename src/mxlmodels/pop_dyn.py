from mxlpy import Model


def dEdt(mu_e: float, a_e: float, e_coli: float) -> float:
    return a_e * e_coli * mu_e


def dCdt(mu_c: float, a_c: float, c_gluta: float, theta: float) -> float:
    return a_c * c_gluta * mu_c - c_gluta**2.0 * theta


def create_model() -> Model:
    return (
        Model()
        .add_variable("e_coli", initial_value=5.0)
        .add_variable("c_gluta", initial_value=5.0)
        .add_parameter("mu_e", value=0.4)
        .add_parameter("mu_c", value=0.3)
        .add_parameter("a_e", value=6.0)
        .add_parameter("a_c", value=4.0)
        .add_parameter("theta", value=0.001)
        .add_reaction(
            "dEdt",
            fn=dEdt,
            args=["mu_e", "a_e", "e_coli"],
            stoichiometry={"e_coli": 1.0},
        )
        .add_reaction(
            "dCdt",
            fn=dCdt,
            args=["mu_c", "a_c", "c_gluta", "theta"],
            stoichiometry={"c_gluta": 1.0},
        )
    )
