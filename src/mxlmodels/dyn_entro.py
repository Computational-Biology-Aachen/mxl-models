"""Dynamic enterobactin competition model: E. coli vs C. glutamicum siderophore cross-feeding."""

from mxlpy import Model


def _a_c(
    a_e: float,
) -> float:
    """Affinity of C. glutamicum for enterobactin: conservation with E. coli affinity."""
    return 10.0 - a_e


def _uptake_e_growth(
    a_e: float,
    enterobactin: float,
    k_e: float,
) -> float:
    """Michaelis-Menten uptake of enterobactin by E. coli, scaled by E. coli affinity."""
    return a_e * enterobactin / (k_e + enterobactin)


def _uptake_c_growth(
    enterobactin: float,
    _a_c: float,
    k_c: float,
) -> float:
    """Michaelis-Menten uptake of enterobactin by C. glutamicum, scaled by its affinity."""
    return _a_c * enterobactin / (k_c + enterobactin)


def _cons_term_e(
    a_e: float,
    e_coli: float,
    k_e: float,
    mu_e: float,
) -> float:
    """Enterobactin consumption term for E. coli: affinity * population * growth rate."""
    return a_e * e_coli * mu_e / (k_e + a_e)


def _cons_term_c(
    mu_c: float,
    _a_c: float,
    k_c: float,
    c_gluta: float,
) -> float:
    """Enterobactin consumption term for C. glutamicum: affinity * population * growth rate."""
    return _a_c * c_gluta * mu_c / (k_c + _a_c)


def _d_edt(
    mu_e: float,
    e_coli: float,
    _uptake_e_growth: float,
) -> float:
    """Net growth rate of E. coli population."""
    return e_coli * mu_e * _uptake_e_growth


def _d_cdt(
    mu_c: float,
    c_gluta: float,
    _uptake_c_growth: float,
    theta: float,
) -> float:
    """Net growth rate of C. glutamicum minus density-dependent death."""
    return c_gluta * mu_c * _uptake_c_growth - c_gluta**2.0 * theta


def _d_bdt(
    enterobactin: float,
    r_cons_e: float,
    _cons_term_e: float,
    r_prod: float,
    _cons_term_c: float,
    r_cons_c: float,
) -> float:
    """Net rate of change of enterobactin: production by E. coli minus consumption by both species."""
    return -_cons_term_c * r_cons_c - _cons_term_e * r_cons_e + enterobactin * r_prod


def create_model() -> Model:
    """Build the dynamic enterobactin cross-feeding model (E. coli / C. glutamicum)."""
    return (
        Model()
        .add_variable(
            "e_coli",
            initial_value=5.0,
        )
        .add_variable(
            "c_gluta",
            initial_value=5.0,
        )
        .add_variable(
            "enterobactin",
            initial_value=1.0,
        )
        .add_parameter(
            "mu_e",
            value=0.4,
        )
        .add_parameter(
            "mu_c",
            value=0.3,
        )
        .add_parameter(
            "a_e",
            value=6.0,
        )
        .add_parameter(
            "K_e",
            value=0.5,
        )
        .add_parameter(
            "K_c",
            value=0.5,
        )
        .add_parameter(
            "theta",
            value=0.001,
        )
        .add_parameter(
            "r_prod",
            value=0.2,
        )
        .add_parameter(
            "r_cons_e",
            value=1.0,
        )
        .add_parameter(
            "r_cons_c",
            value=1.0,
        )
        .add_derived(
            "a_c",
            fn=_a_c,
            args=["a_e"],
        )
        .add_derived(
            "uptake_E_growth",
            fn=_uptake_e_growth,
            args=["a_e", "enterobactin", "K_e"],
        )
        .add_derived(
            "uptake_C_growth",
            fn=_uptake_c_growth,
            args=["enterobactin", "a_c", "K_c"],
        )
        .add_derived(
            "cons_term_E",
            fn=_cons_term_e,
            args=["a_e", "e_coli", "K_e", "mu_e"],
        )
        .add_derived(
            "cons_term_C",
            fn=_cons_term_c,
            args=["mu_c", "a_c", "K_c", "c_gluta"],
        )
        .add_reaction(
            "dEdt",
            fn=_d_edt,
            args=["mu_e", "e_coli", "uptake_E_growth"],
            stoichiometry={"e_coli": 1.0},
        )
        .add_reaction(
            "dCdt",
            fn=_d_cdt,
            args=["mu_c", "c_gluta", "uptake_C_growth", "theta"],
            stoichiometry={"c_gluta": 1.0},
        )
        .add_reaction(
            "dBdt",
            fn=_d_bdt,
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
