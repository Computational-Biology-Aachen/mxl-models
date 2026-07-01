def get_fvcb_1980(
    Ci: float = 500,
    vcmax: float = 80,
    kc: float = 259,
    ko: float = 179,
    ccp: float = 38.6,
    rl: float = 1,
    alpha: float = 0,
    O2: float = 210,
    J: float = 124,
    Tp: float = 15,
):
    Wc = (Ci * vcmax) / (Ci + kc * (1 + (O2) / (ko)))
    Wj = (Ci * J) / (4 * Ci + 8 * ccp)
    Wp = (
        100
        if Ci <= ccp * (1 + 3 * alpha)
        else (3 * Ci * Tp) / ((Ci) - (ccp * (1 + 3 * alpha)))
    )
    Vc = min(Wc, Wj, Wp)
    An = (Vc * ((1) - ((ccp) / (Ci)))) - (rl)
    return Wc, Wj, Wp, Vc, An
