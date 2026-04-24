from mxlpy import Derived, InitialAssignment, Model


def a_c(a_e: float) -> float:
    return 10.0 - a_e


def uptake_E_growth(a_e: float, enterobactin: float, K_e: float) -> float:
    return a_e * enterobactin / (K_e + enterobactin)


def uptake_C_growth(enterobactin: float, a_c: float, K_c: float) -> float:
    return a_c * enterobactin / (K_c + enterobactin)


def cons_term_E(a_e: float, e_coli: float, K_e: float, mu_e: float) -> float:
    return a_e * e_coli * mu_e / (K_e + a_e)


def cons_term_C(mu_c: float, a_c: float, K_c: float, c_gluta: float) -> float:
    return a_c * c_gluta * mu_c / (K_c + a_c)


def dEdt(mu_e: float, e_coli: float, uptake_E_growth: float) -> float:
    return e_coli * mu_e * uptake_E_growth


def dCdt(mu_c: float, c_gluta: float, uptake_C_growth: float, theta: float) -> float:
    return c_gluta * mu_c * uptake_C_growth - c_gluta**2.0 * theta


def dBdt(
    enterobactin: float,
    r_cons_e: float,
    cons_term_E: float,
    r_prod: float,
    cons_term_C: float,
    r_cons_c: float,
) -> float:
    return -cons_term_C * r_cons_c - cons_term_E * r_cons_e + enterobactin * r_prod


def create_model() -> Model:
    return (
        Model()
        .add_variable("e_coli", initial_value=5.0)
        .add_variable("c_gluta", initial_value=5.0)
        .add_variable("enterobactin", initial_value=1.0)
        .add_parameter("mu_e", value=0.4)
        .add_parameter("mu_c", value=0.3)
        .add_parameter("a_e", value=6.0)
        .add_parameter("K_e", value=0.5)
        .add_parameter("K_c", value=0.5)
        .add_parameter("theta", value=0.001)
        .add_parameter("r_prod", value=0.2)
        .add_parameter("r_cons_e", value=1.0)
        .add_parameter("r_cons_c", value=1.0)
        .add_derived(
            "a_c",
            fn=a_c,
            args=["a_e"],
        )
        .add_derived(
            "uptake_E_growth",
            fn=uptake_E_growth,
            args=["a_e", "enterobactin", "K_e"],
        )
        .add_derived(
            "uptake_C_growth",
            fn=uptake_C_growth,
            args=["enterobactin", "a_c", "K_c"],
        )
        .add_derived(
            "cons_term_E",
            fn=cons_term_E,
            args=["a_e", "e_coli", "K_e", "mu_e"],
        )
        .add_derived(
            "cons_term_C",
            fn=cons_term_C,
            args=["mu_c", "a_c", "K_c", "c_gluta"],
        )
        .add_reaction(
            "dEdt",
            fn=dEdt,
            args=["mu_e", "e_coli", "uptake_E_growth"],
            stoichiometry={"e_coli": 1.0},
        )
        .add_reaction(
            "dCdt",
            fn=dCdt,
            args=["mu_c", "c_gluta", "uptake_C_growth", "theta"],
            stoichiometry={"c_gluta": 1.0},
        )
        .add_reaction(
            "dBdt",
            fn=dBdt,
            args=[
                "enterobactin",
                "r_cons_e",
                "cons_term_E",
                "r_prod",
                "cons_term_C",
                "r_cons_c",
            ],
            stoichiometry={"enterobactin": 1.0},
        )
    )
