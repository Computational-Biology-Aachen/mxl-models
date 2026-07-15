import math
from typing import Literal


def min_solve(a: float, b: float, c: float) -> float:
    """Minimum of quadratic solutions of a * x**2 + b * x + c"""
    discriminant = max(0.0, (b**2) - (4 * a * c))
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    return min(root1, root2)

def arrhenius(param_298: float, Ea: float, T: float) -> float:
    return param_298 * math.exp(Ea * (T - 298) / (298 * 8.314 * T))

def jmax_tempscaling(jmax_298: float, Ea_jmax: float, S: float, H: float, T: float) -> float:
    R = 8.314
    den_298 = 1 + math.exp((S * 298 - H) / (R * 298))
    den_T = 1 + math.exp((S * T - H) / (R * T))
    return arrhenius(jmax_298, Ea_jmax, T) * (den_298 / den_T)

def get_fvcb(
    pco2: float,
    po2: float = 210.0,
    T: float = 298.0,
    I: float = 1000.0,
    vc_max: float = 98.0,
    j_max: float = 210.0,
    tp: float = 11.8,
    alpha_old: float = 0.0,
    km_co2: float = 460.0,
    km_o2: float = 330.0,
    kc: float = 2.5,
    f: float = 0.23,
    z: float = 0.0,
    r_light: float = 1.1,
    j_infinite: bool = False,
    model_version: Literal["1980", "2025"] = "1980",
    use_2025_default: bool = False
):
    
    if use_2025_default:
        # Override parameters with 2025 defaults
        vc_max = 100.0
        j_max = 170.0
        tp = 11.8
        alpha_old = 0.0
        km_co2 = 259.0
        km_o2 = 179.0
        kc = 2.5
        f = 0.23
        z = 0.0
        r_light = 1.0
        gammastar = 38.6
    
    # ----------------------------------------
    # A. Shared Temperature Scaling
    # ----------------------------------------
    if T != 298.0:
        km_co2 = arrhenius(km_co2, 59356, T)
        km_o2 = arrhenius(km_o2, 35948, T)
        kc = arrhenius(kc, 58520, T)
        vc_max = arrhenius(vc_max, 58520, T)
        r_light = arrhenius(r_light, 66405, T)
        j_max = jmax_tempscaling(j_max, 37000, 710, 220000, T)
        
    # ----------------------------------------
    # B. Shared Base Kinetics
    # ----------------------------------------
    ko = 0.21 * kc
    phi = (ko / kc) * ((po2 / km_o2) / (pco2 / km_co2))
    
    
    # Wc (Rubisco limitation)
    Wc = vc_max * (pco2) / (pco2 + km_co2 * (1 + po2 / km_o2))
    
    # Potential linear electron transport (J) based on light (I)
    if model_version == "1980":
        if j_infinite:
            J = math.inf
        else:
            J = min_solve(
                a=1.0,
                b=-(0.5 * (1 - f) * I + j_max + z),
                c=0.5 * (1 - f) * I * j_max
            )
            
        gammastar = km_co2 * po2 * ko / (2 * km_o2 * kc)
    elif model_version == "2025":
        J = j_max

    # ----------------------------------------
    # C. Version-Specific Logic
    # ----------------------------------------
    if model_version == "1980":
        # The 1980 paper explicitly models the RuBP pool size
        J_prime = J / (2 * (2 + 2 * phi))
        P = Wc * (300.0 / 87.0)
        M = 2.0 * vc_max
        M_prime = M / (2.0 + 1.5 * phi)
        
        B = J_prime + P + (J_prime * P) / M_prime
        C_quad = J_prime * P
        
        Wj = min_solve(a=1.0, b=-B, c=C_quad) if not j_infinite else math.inf
        Wp = math.inf  # TPU limitation was not modeled in the 1980 paper
        
        vc = Wc if j_infinite else min(Wc, Wj)

    elif model_version == "2025":
        # The modern "min-W" approach simplifies Wj and adds TPU limitation
        Wj = (pco2 * J) / (4 * pco2 + 8 * gammastar) if not j_infinite else math.inf
        
        # TPU (Triose Phosphate Utilization) limited rate
        if pco2 <= gammastar * (1 + 3 * alpha_old):
            Wp = math.inf
        else:
            Wp = 3 * pco2 * tp / (pco2 - gammastar * (1 + 3 * alpha_old))
            
        vc = min(Wc, Wj, Wp)

    else:
        raise ValueError("model_version must be either '1980' or '2025'")
    
    # ----------------------------------------
    # D. Final Assimilation
    # ----------------------------------------
    A = vc * (1 - gammastar / pco2) - r_light
    Vo = phi * vc
    
    return {
        "A": A, 
        "Vc": vc,
        "Wc": Wc, 
        "Wj": Wj, 
        "Wp": Wp, 
        "Vo": Vo, 
        "gammastar": gammastar
    }
