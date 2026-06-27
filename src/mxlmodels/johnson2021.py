"""Johnson 2021 model.

|             |                                                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| doi         | 10.1007/s11120-021-00840-4                                                                                              |
| main author | Chandra Bellasio                                                                                                       |
| paper title | A generalised dynamic model of leaf-level C3 photosynthesis combining light and dark reactions with stomatal behaviour |
| published   | January 2019                                                                                                           |
| journal     | Photosynthesis Research                                                                                                |
| organism    | C3 leaf                                                                                                                |

The [Johnson2021] (https://doi.org/10.1007/s11120-021-00840-4) model is a
generalized C3 leaf-photosynthesis model, that operates in steady-state. Therefore, it
is not an ODE model and is made from two functions, a solving function and the function,
calling the model. The model function takes, so called environmental variables as input. 
They are Light, Temperature, CO2 and O2 concentration. Based on those input, the outputs 
of the model are calculated. For the publication the most important outputs are 
Linear and Cyclic Electron Flow, NPQ and Net carbon assimilation. The goal of the model is 
to investigate how photosynthesis is mainly controlled in steady-state conditions. It is built 
around the idea that control is shared between CBB and b6f, but with a special focus on b6f as 
the main regulatory bottleneck. For the analysis, the simulations are mainly performed under light 
intensities from 0-2400 and an additional perturbation of a model parameter to investigate 
scenarios of cytb6f and rubisco limitation.

The reproduction of the model was straightforward, but their github only included two example
simulation, which had no connection to the respective publication. It is not disclosed how the 
model perturbations are performed and one has to derive it from the publication, which for some of 
the cases can be difficult as the model to publication connection is not clear from time to time. 
"""





import numpy as np


def solve_xcs(Abs, CB6F, Kd, Kf, Kp2, Ku2, Q, eta, kq, phi1P_max):

    Q = np.array(Q, dtype=float)

    num_sqrt = (
        (Kd + Kf + Ku2)
        * (
            CB6F**2 * Kd**3 * kq**2 * phi1P_max**2
            + CB6F**2 * Kf**3 * kq**2 * phi1P_max**2
            + CB6F**2 * Kd * Kp2**2 * eta**2 * kq**2
            + CB6F**2 * Kf * Kp2**2 * eta**2 * kq**2
            + CB6F**2 * Kp2**2 * Ku2 * eta**2 * kq**2
            + CB6F**2 * Kd * Kf**2 * kq**2 * phi1P_max**2 * 3.0
            + CB6F**2 * Kd**2 * Kf * kq**2 * phi1P_max**2 * 3.0
            + CB6F**2 * Kd * Kp2**2 * kq**2 * phi1P_max**2
            + CB6F**2 * Kd**2 * Kp2 * kq**2 * phi1P_max**2 * 2.0
            + CB6F**2 * Kf * Kp2**2 * kq**2 * phi1P_max**2
            + CB6F**2 * Kf**2 * Kp2 * kq**2 * phi1P_max**2 * 2.0
            + CB6F**2 * Kd**2 * Ku2 * kq**2 * phi1P_max**2
            + CB6F**2 * Kf**2 * Ku2 * kq**2 * phi1P_max**2
            + CB6F**2 * Kp2**2 * Ku2 * kq**2 * phi1P_max**2
            + Abs**2 * Kd * Kp2**2 * Q**2 * eta**2 * phi1P_max**2
            + Abs**2 * Kf * Kp2**2 * Q**2 * eta**2 * phi1P_max**2
            + Abs**2 * Kp2**2 * Ku2 * Q**2 * eta**2 * phi1P_max**2
            + CB6F**2 * Kd * Kf * Kp2 * kq**2 * phi1P_max**2 * 4.0
            + CB6F**2 * Kd * Kf * Ku2 * kq**2 * phi1P_max**2 * 2.0
            + CB6F**2 * Kd * Kp2 * Ku2 * kq**2 * phi1P_max**2 * 2.0
            + CB6F**2 * Kf * Kp2 * Ku2 * kq**2 * phi1P_max**2 * 2.0
            + CB6F**2 * Kd * Kp2**2 * eta * kq**2 * phi1P_max * 2.0
            + CB6F**2 * Kd**2 * Kp2 * eta * kq**2 * phi1P_max * 2.0
            + CB6F**2 * Kf * Kp2**2 * eta * kq**2 * phi1P_max * 2.0
            + CB6F**2 * Kf**2 * Kp2 * eta * kq**2 * phi1P_max * 2.0
            + CB6F**2 * Kp2**2 * Ku2 * eta * kq**2 * phi1P_max * 2.0
            + CB6F**2 * Kd * Kf * Kp2 * eta * kq**2 * phi1P_max * 4.0
            + CB6F**2 * Kd * Kp2 * Ku2 * eta * kq**2 * phi1P_max * 2.0
            + CB6F**2 * Kf * Kp2 * Ku2 * eta * kq**2 * phi1P_max * 2.0
            + Abs * CB6F * Kd * Kp2**2 * Q * eta * kq * phi1P_max**2 * 2.0
            + Abs * CB6F * Kd * Kp2**2 * Q * eta**2 * kq * phi1P_max * 2.0
            + Abs * CB6F * Kd**2 * Kp2 * Q * eta * kq * phi1P_max**2 * 2.0
            + Abs * CB6F * Kf * Kp2**2 * Q * eta * kq * phi1P_max**2 * 2.0
            + Abs * CB6F * Kf * Kp2**2 * Q * eta**2 * kq * phi1P_max * 2.0
            + Abs * CB6F * Kf**2 * Kp2 * Q * eta * kq * phi1P_max**2 * 2.0
            - Abs * CB6F * Kp2**2 * Ku2 * Q * eta * kq * phi1P_max**2 * 2.0
            + Abs * CB6F * Kp2**2 * Ku2 * Q * eta**2 * kq * phi1P_max * 2.0
            + Abs * CB6F * Kd * Kf * Kp2 * Q * eta * kq * phi1P_max**2 * 4.0
            + Abs * CB6F * Kd * Kp2 * Ku2 * Q * eta * kq * phi1P_max**2 * 2.0
            + Abs * CB6F * Kf * Kp2 * Ku2 * Q * eta * kq * phi1P_max**2 * 2.0
        )
    )

    num_linear = (
        CB6F * Kd**2 * kq * phi1P_max
        + CB6F * Kf**2 * kq * phi1P_max
        + Abs * Kd**2 * Q * phi1P_max**2 * 2.0
        + Abs * Kf**2 * Q * phi1P_max**2 * 2.0
        + CB6F * Kd * Kp2 * eta * kq
        + CB6F * Kf * Kp2 * eta * kq
        + CB6F * Kp2 * Ku2 * eta * kq
        + CB6F * Kd * Kf * kq * phi1P_max * 2.0
        + CB6F * Kd * Kp2 * kq * phi1P_max
        + CB6F * Kf * Kp2 * kq * phi1P_max
        + CB6F * Kd * Ku2 * kq * phi1P_max
        + CB6F * Kf * Ku2 * kq * phi1P_max
        + CB6F * Kp2 * Ku2 * kq * phi1P_max
        + Abs * Kd * Kf * Q * phi1P_max**2 * 4.0
        + Abs * Kd * Kp2 * Q * phi1P_max**2 * 2.0
        + Abs * Kf * Kp2 * Q * phi1P_max**2 * 2.0
        + Abs * Kd * Ku2 * Q * phi1P_max**2 * 2.0
        + Abs * Kf * Ku2 * Q * phi1P_max**2 * 2.0
        + Abs * Kd * Kp2 * Q * eta * phi1P_max
        + Abs * Kf * Kp2 * Q * eta * phi1P_max
        + Abs * Kp2 * Ku2 * Q * eta * phi1P_max
    )

    denom = (
        Q
        * phi1P_max
        * (
            Kd**2 * phi1P_max
            + Kf**2 * phi1P_max
            + Kd * Kp2 * eta
            + Kf * Kp2 * eta
            + Kp2 * Ku2 * eta
            + Kd * Kf * phi1P_max * 2.0
            + Kd * Kp2 * phi1P_max
            + Kf * Kp2 * phi1P_max
            + Kd * Ku2 * phi1P_max
            + Kf * Ku2 * phi1P_max
        )
        * 2.0
    )

    xcs = (-np.sqrt(num_sqrt) + num_linear) / denom
    return xcs


def get_johnson2021(
    PAR=800,
    Temp=25,
    CO2=200,
    O2=209,
    Abs=0.85,
    beta=0.52,
    CB6F=(350 / 300) / 1e6,
    RUB=(100 / 3.6) / 1e6,
    Rds=0.01,
    Ku2=0e09,
    theta1=1,
    eps1=0,
    eps2=1,
    alpha_opt="static",
    nl=0.75,
    nc=1.00,
    case_id=0
):

        
    # Ensure arrays
    PAR = np.atleast_1d(np.array(PAR, dtype=float))
    Temp = np.full_like(PAR, Temp, dtype=float)
    CO2  = np.full_like(PAR, CO2, dtype=float)
    O2   = np.full_like(PAR, O2, dtype=float)


    # Broadcast to common shape
    Q = PAR / 1e6
    Tc = Temp / 1e6
    C = CO2 / 1e6
    O = O2 / 1e3

    # Photochemical constants
    Kf = 0.05e09
    Kd = 0.55e09
    Kp1 = 14.5e09
    Kn1 = 14.5e09
    Kp2 = 4.5e09

    # Biochemical constants
    kq = 300
    kc = 3.6
    ko = 3.6 * 0.27
    Kc = 260 / 1e6
    Ko = 179000 / 1e6

    # Derived variables
    Vqmax = CB6F * kq
    Vcmax = RUB * kc
    Rd = Vcmax * Rds
    S = (kc / Kc) * (Ko / ko)
    gammas = O / (2 * S)
    eta = 1 - (nl / nc) + (3 + 7 * gammas / C) / ((4 + 8 * gammas / C) * nc)
    phi1P_max = Kp1 / (Kp1 + Kd + Kf)

    # Cross-sections
    if alpha_opt == "static":
        a2 = Abs * beta
        a1 = Abs - a2
        a2 = np.full_like(Q, a2, dtype=float)
        a1 = np.full_like(Q, a1, dtype=float)
    elif alpha_opt == "dynamic":
        a2 = solve_xcs(Abs, CB6F, Kd, Kf, Kp2, Ku2, Q, eta, kq, phi1P_max)
        a1 = Abs - a2
    else:
        raise ValueError("argument alpha_opt not identified")

    # Potential Cyt b6f-limited rates (_j)
    JP700_j = (Q * Vqmax) / (Q + Vqmax / (a1 * phi1P_max))
    JP680_j = JP700_j / eta
    Vc_j = JP680_j / (4 * (1 + 2 * gammas / C))
    Vo_j = Vc_j * 2 * gammas / C
    Ag_j = Vc_j - Vo_j / 2

    # Potential Rubisco-limited rates (_c)
    Vc_c = C * Vcmax / (C + Kc * (1 + O / Ko))
    Vo_c = Vc_c * 2 * gammas / C
    Ag_c = Vc_c - Vo_c / 2
    JP680_c = Ag_c * 4 * (1 + 2 * gammas / C) / (1 - gammas / C)
    JP700_c = JP680_c * eta

    # Quadratic smoothing function tr(l1, l2, th)
    def tr(l1, l2, th):
        l1 = np.array(l1, dtype=float)
        l2 = np.array(l2, dtype=float)
        th = float(th)
        disc = (l1 + l2) ** 2 - 4 * th * l1 * l2
        disc = np.maximum(disc, 0.0)
        q1 = ((l1 + l2) + np.sqrt(disc)) / (2 * th)
        q2 = ((l1 + l2) - np.sqrt(disc)) / (2 * th)
        return q1, q2
    
    # Actual rates (minimum of the roots)
    JP680_a = np.minimum(*tr(JP680_j, JP680_c, theta1))
    
    # Case 8 Override: PSI ignores Rubisco limitations
    if case_id == 8:
        JP700_a = JP700_j
    else:
        JP700_a = np.minimum(*tr(JP700_j, JP700_c, theta1))

    # Select minimum Ag_a (gross)
    q1_Ag, q2_Ag = tr(Ag_j, Ag_c, theta1)
    Ag_min = np.minimum(q1_Ag, q2_Ag)
    Ag_max = np.maximum(q1_Ag, q2_Ag)

    Ag_a = np.where(C > gammas, Ag_min, Ag_max)

    # Net assimilation
    An_a = Ag_a - Rd

    # Dynamic alpha_opt update
    if alpha_opt == "dynamic":
        JP700_j_c = np.vstack([JP700_j, JP700_c]).T
        I = np.argmin(JP700_j_c, axis=1)
        I_diff = np.diff(I)
        positions = np.where(I_diff != 0)[0]
        if len(positions) > 0:
            pos = positions[-1]
            a2_new = np.full_like(a2, a2[pos])
            a2_old = a2.copy()
            a2 = np.maximum(a2_old, a2_new)
            a1 = Abs - a2

    # Primary fluorescence parameters
    phi1P_a = JP700_a / (Q * a1)
    q1_a = phi1P_a / phi1P_max
    phi2P_a = JP680_a / (Q * a2)
    
    if case_id == 5:
        # CASE 5: Cyt b6f feedback alone (NO NPQ)
        # PSII closure (q2_a) is driven entirely by the lack of heat dissipation
        q2_a = np.clip(phi2P_a * (Kp2 + Kd + Kf) / Kp2, 0.0, 1.0)
        
        # Cyt b6f active sites match the PQ reduction pool precisely
        CB6F_a = CB6F * (1 - q2_a) 
        
        # Force NPQ to exactly zero to bypass the polynomial and avoid math errors
        Kn2_a = np.zeros_like(Q)
        
    elif case_id == 6:
        # CASE 6: NPQ alone (Cyt b6f at max turnover)
        CB6F_a = JP700_a / kq
        q2_a = np.clip(1 - CB6F_a / CB6F, 0.0, 1.0)
        
    elif case_id == 8:
        # CASE 8: Regulatory CEF
        # Cyt b6f runs at maximum turnover (kq) using the massive actual PSI flow
        CB6F_a = JP700_a / kq
        q2_a = np.clip(1 - CB6F_a / CB6F, 0.0, 1.0)
        
    else:
        # CASE 7 (Default): NPQ and Cyt b6f together
        # PQ pool is poised according to the light-limited potential
        CB6F_a = JP700_j / kq       
        q2_a = np.clip(1 - CB6F_a / CB6F, 0.0, 1.0)

    # Kn2_a from rearranged Eq. 25a
    if case_id == 5:
        Kn2_a = np.zeros_like(Q)
    else:
        num_kn = (
            Kp2**2 * phi2P_a**2 
            - 2 * Kp2**2 * phi2P_a * q2_a 
            + Kp2**2 * q2_a**2 
            - 4 * Kp2 * Ku2 * phi2P_a**2 * q2_a 
            + 2 * Kp2 * Ku2 * phi2P_a**2 
            + 2 * Kp2 * Ku2 * phi2P_a * q2_a 
            + Ku2**2 * phi2P_a**2
        )
        num_kn = np.maximum(num_kn, 0.0)

        Kn2_a = (
            np.sqrt(num_kn)
            - Kp2 * phi2P_a
            + Ku2 * phi2P_a
            + Kp2 * q2_a
        ) / (2 * phi2P_a) - Kf - Ku2 - Kd
    
    # PSII internal yields
    denom2 = (Kp2 + Kn2_a + Kd + Kf + Ku2)
    denom2c = (Kn2_a + Kd + Kf + Ku2)

    phi2p_a = q2_a * Kp2 / denom2
    phi2n_a = q2_a * Kn2_a / denom2 + (1 - q2_a) * Kn2_a / denom2c
    phi2d_a = q2_a * Kd / denom2 + (1 - q2_a) * Kd / denom2c
    phi2f_a = q2_a * Kf / denom2 + (1 - q2_a) * Kf / denom2c
    phi2u_a = q2_a * Ku2 / denom2 + (1 - q2_a) * Ku2 / denom2c

    phi2P_a = phi2p_a / (1 - phi2u_a)
    phi2N_a = phi2n_a / (1 - phi2u_a)
    phi2D_a = phi2d_a / (1 - phi2u_a)
    phi2F_a = phi2f_a / (1 - phi2u_a)

    # PSI yields
    denom1_open = (Kp1 + Kd + Kf)
    denom1_closed = (Kn1 + Kd + Kf)

    phi1P_a = q1_a * Kp1 / denom1_open
    phi1N_a = (1 - q1_a) * Kn1 / denom1_closed
    phi1D_a = q1_a * Kd / denom1_open + (1 - q1_a) * Kd / denom1_closed
    phi1F_a = q1_a * Kf / denom1_open + (1 - q1_a) * Kf / denom1_closed

    # PAM fluorescence levels
    Fm_a = a2[0] * Kf / (Kd + Kf) * eps2 + a1[0] * Kf / (Kn1 + Kd + Kf) * eps1
    Fo_a = a2[0] * Kf / (Kp2 + Kd + Kf) * eps2 + a1[0] * Kf / (Kp1 + Kd + Kf) * eps1
    Fmp_a = a2 * Kf / (Kn2_a + Kd + Kf) * eps2 + a1 * Kf / (Kn1 + Kd + Kf) * eps1
    Fop_a = a2 * Kf / (Kp2 + Kn2_a + Kd + Kf) * eps2 + a1 * Kf / (Kp1 + Kd + Kf) * eps1
    Fs_a = a2 * phi2F_a * eps2 + a1 * phi1F_a * eps1

    PAM1_a = 1 - Fs_a / Fmp_a
    PAM2_a = Fs_a * (1 / Fmp_a - 1 / Fm_a)
    PAM3_a = Fs_a / Fm_a

    #Other PAM indices used in paper
    ETR = Q*0.85/2.*PAM1_a
    qP = (Fmp_a - Fs_a)/(Fmp_a - Fop_a)
    qL = (Fmp_a - Fs_a)*Fop_a/((Fmp_a - Fop_a)*Fs_a)
    kPuddle = ETR/(1-qP) 
    kLake = ETR/(1-qL)
    NPQ = Fm_a/Fmp_a - 1
    
    # Calculate active Cyt b6f sites
    if case_id == 6:
        active_cb6f = np.maximum(CB6F_a, 1e-12)
    else:
        active_cb6f = np.maximum(CB6F * (1 - q2_a), 1e-12)
        
    # Safely calculate k_CB6F. If the PQ pool is effectively empty (at PAR 0), default to kq (300)
    k_CB6F = np.where(JP700_a >= JP700_j * 0.999, kq, JP700_a / active_cb6f)

    # Build result dict (vector outputs)
    return {
        "PAR": PAR,
        "Temp": Temp,
        "CO2": CO2,
        "O2": O2,
        "Q": Q,
        "C": C,
        "O_bar": O,
        "Abs": Abs,
        "beta": beta,
        "CB6F": CB6F,
        "RUB": RUB,
        "Rds": Rds,
        "Ku2": Ku2,
        "theta1": theta1,
        "eps1": eps1,
        "eps2": eps2,
        "Ag_a": Ag_a,
        "Ag_c": Ag_c,
        "Ag_j": Ag_j,
        "An_a": An_a,
        "CB6F_a": CB6F_a,
        "Fm_a": Fm_a,
        "Fmp_a": Fmp_a,
        "Fo_a": Fo_a,
        "Fop_a": Fop_a,
        "Fs_a": Fs_a,
        "JP680_a": JP680_a,
        "JP680_c": JP680_c,
        "JP680_j": JP680_j,
        "JP700_a": JP700_a,
        "JP700_c": JP700_c,
        "JP700_j": JP700_j,
        "Kn2_a": Kn2_a,
        "PAM1_a": PAM1_a,
        "PAM2_a": PAM2_a,
        "PAM3_a": PAM3_a,
        "Rd": Rd,
        "S": S,
        "Vcmax": Vcmax,
        "Vqmax": Vqmax,
        "a1": a1,
        "a2": a2,
        "eta": eta,
        "gammas": gammas,
        "phi1P_max": phi1P_max,
        "phi1D_a": phi1D_a,
        "phi1F_a": phi1F_a,
        "phi1N_a": phi1N_a,
        "phi1P_a": phi1P_a,
        "phi2D_a": phi2D_a,
        "phi2F_a": phi2F_a,
        "phi2N_a": phi2N_a,
        "phi2P_a": phi2P_a,
        "phi2d_a": phi2d_a,
        "phi2f_a": phi2f_a,
        "phi2n_a": phi2n_a,
        "phi2p_a": phi2p_a,
        "phi2u_a": phi2u_a,
        "q1_a": q1_a,
        "q2_a": q2_a,
        "ETR":ETR,
        "qP":qP,
        "qL": qL,
        "kPuddle":kPuddle,
        "kLake": kLake,
        "NPQ":NPQ,
        "k_CB6F": k_CB6F
    }
