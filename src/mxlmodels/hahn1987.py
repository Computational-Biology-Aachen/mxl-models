r"""Hahn 1987 Photorespiration and Photosynthesis model.

|             |                                                             |
| ----------- | ----------------------------------------------------------- |
| doi         | 10.1093/oxfordjournals.aob.a087432                          |
| main author | Brian D. Hahn                                               |
| paper title | A Mathematical Model of Photorespiration and Photosynthesis |
| published   | August 1987                                                 |
| journal     | Annals of Botany                                            |
| organism    | C3 leaf                                                     |
| Ported by   | ElouenCorvest ( @ElouenCorvest )                            |

A comprehensive model of C3 leaf carbon metabolism combining the Calvin cycle
with the glycolate and glycerate pathways of photorespiration, formulated as a
system of non-linear differential equations. It extends Hahn's earlier
Calvin-cycle models (1984, 1986) by adding the competitive inhibition of
ribulose-bisphosphate carboxylase by oxygen. The model reaches an effectively
stable steady state and behaves realistically when external CO2 and O2
concentrations are varied: photosynthesis is inhibited by higher oxygen levels,
while photorespiration is inhibited by higher carbon dioxide levels.
"""

from mxlpy import Model


def moiety_1(concentration: float, total: float) -> float:
    return total - concentration


def PAn(h, k1, co2, rubp, k20, gn, rd, gf):
    return 36000 * 44 * h * (k1 * co2 * rubp - k20 * gn**2 - 12 * rd * gf)


def v1(k1: float, CO2: float, RuBP: float) -> float:
    return k1 * CO2 * RuBP


def v2(k2: float, ADP: float, Pi: float) -> float:
    return k2 * ADP * Pi


def v3(k3: float, PGA: float, ATP: float) -> float:
    return k3 * PGA * ATP


def v4(k4: float, TP: float) -> float:
    return k4 * TP**2


def v5(k5: float, HP: float) -> float:
    return k5 * HP


def v6(k6: float, E4P: float, TP: float) -> float:
    return k6 * E4P * TP


def v7(k7: float, S7P: float) -> float:
    return k7 * S7P


def v8(k8: float, TPGA: float, TP: float) -> float:
    return k8 * TPGA * TP


def v9(k9: float, Ru5P: float, ATP: float) -> float:
    return k9 * ATP * Ru5P


def v10(k10: float, ATP: float, HP: float) -> float:
    return k10 * ATP * HP


def v11(k11: float, GG: float, Pi: float) -> float:
    return k11 * GG * Pi


def v12(k12: float, TP: float, Pio: float) -> float:
    return k12 * TP * Pio


def v13(k13: float, TPo: float) -> float:
    return k13 * TPo**2


def v14(k14: float, UDP: float, Pio: float) -> float:
    return k14 * UDP * Pio


def v15(k15: float, UTP: float, HPo: float) -> float:
    return k15 * UTP * HPo


def v16(k16: float, O2: float, RuBP: float) -> float:
    return k16 * O2 * RuBP


def v17(k17: float, PGl: float) -> float:
    return k17 * PGl


def v18(k18: float, Gl: float, O2: float) -> float:
    return k18 * Gl**2 * O2


def v19(k19: float, Gx: float, Sn: float) -> float:
    return k19 * Gx * Sn


def v20(k20: float, Gn: float) -> float:
    return k20 * Gn**2


def v21(k21: float, ATP: float, GA: float) -> float:
    return k21 * ATP * GA


def v22(k22: float, ATP: float, GmA: float, NH3: float) -> float:
    return k22 * ATP * GmA * NH3


def v23(k23: float, Glm: float, OxA: float) -> float:
    return k23 * Glm * OxA


def v24(k24: float, Gx: float, GmA: float) -> float:
    return k24 * Gx * GmA


def vrd(rd: float, GF: float) -> float:
    return rd * GF


def vphis(phis: float, E: float, GF: float) -> float:
    return phis * (GF - E)


def vD(D: float, GF: float, GFV: float) -> float:
    return D * (GF - GFV)


def vc1(kci: float, Ci: float) -> float:
    return kci * Ci


def vc2(kc2: float, CO2: float) -> float:
    return kc2 * CO2


def vo1(ko1: float, Oi: float) -> float:
    return ko1 * Oi


def vo2(ko2: float, O2: float) -> float:
    return ko2 * O2


def vphic(phic: float, Ca: float, Ci: float) -> float:
    return phic * (Ca - Ci)


def vphio(phio: float, Oa: float, Oi: float) -> float:
    return phio * (Oa - Oi)


def get_hahn1987():
    return (
        Model()
        .add_parameters(
            {
                "k1": 0.344,
                "k2": 0.460e-1,
                "k3": 0.261e-1,
                "k4": 0.455e-1,
                "k5": 0.455e-1,
                "k6": 0.455,
                "k7": 0.455,
                "k8": 0.909,
                "k9": 0.136e-1,
                "k10": 0.400e-2,
                "k11": 0.400e-4,
                "k12": 0.341e-1,
                "k13": 1.70,
                "k14": 0.852e-3,
                "k15": 0.852e-2,
                "k16": 0.928e-1,
                "k17": 0.227e-1,
                "k18": 0.467e-1,
                "k19": 0.114e-2,
                "k20": 0.114e-1,
                "k21": 0.114e-2,
                "k22": 0.114e-2,
                "k23": 0.114e-1,
                "k24": 0.114e-1,
                "kc1": 1,
                "kc2": 0.933,
                "rd": 1.1e-5,  # kc3 # derived from formula in pub
                "ko1": 0.1e-1,
                "ko2": 4.31,
                "phic": 1.84,
                "phio": 0.453e-1,
                "phis": 0.100e-3,
                "D": 0.100e-3,
                "E": 0.500,
                "Oa": 100,
                "Ca": 0.450,
                "PAg": 0.100e-2,
                "PAr": 0.200e-3,
                "h": 0.200e-3,
            }
        )
        .add_variables(
            {
                "RuBP": 1,
                "PGA": 1,
                "ADP": 1,
                "Pi": 10,
                "TP": 1,
                "HP": 1,
                "GG": 100,
                "Pio": 1,
                "TPo": 0.1,
                "HPo": 0.1,
                "UDP": 10,
                "GF": 77.3,
                "GFV": 77.3,
                "TPGA": 0.1,
                "E4P": 0.1,
                "S7P": 0.1,
                "Ru5P": 1,
                "PGl": 1,
                "Gl": 1,
                "Gx": 1,
                "Sn": 10,
                "Gn": 1,
                "GA": 1,
                "GmA": 1,
                "Glm": 1,
                "OxA": 1,
                "NH3": 1,
                "CO2": 0.330,
                "O2": 0.245,
                "Ci": 0.400,
                "Oi": 101.0,
                "AP_tot": 11,
                "UP_tot": 20,
            }
        )
        .add_derived(name="ATP", fn=moiety_1, args=["ADP", "AP_tot"])
        .add_derived(name="UTP", fn=moiety_1, args=["UDP", "UP_tot"])
        .add_derived(
            name="PAn", fn=PAn, args=["h", "k1", "CO2", "RuBP", "k20", "Gn", "rd", "GF"]
        )
        .add_reaction(
            name="v1",
            fn=v1,
            args=["k1", "CO2", "RuBP"],
            stoichiometry={"RuBP": -1, "PGA": 2, "CO2": -1, "O2": -1 / 2},
        )
        .add_reaction(
            name="v2",
            fn=v2,
            args=["k2", "ADP", "Pi"],
            stoichiometry={"ADP": -1, "Pi": -1},
        )
        .add_reaction(
            name="v3",
            fn=v3,
            args=["k3", "PGA", "ATP"],
            stoichiometry={"PGA": -1, "ADP": 1, "Pi": 1, "TP": 1, "O2": 1 / 2},
        )
        .add_reaction(
            name="v4",
            fn=v4,
            args=["k4", "TP"],
            stoichiometry={"TP": -2, "Pi": 1, "HP": 1},
        )
        .add_reaction(
            name="v5",
            fn=v5,
            args=["k5", "HP"],
            stoichiometry={"HP": -1, "TPGA": 1, "E4P": 1},
        )
        .add_reaction(
            name="v6",
            fn=v6,
            args=["k6", "E4P", "TP"],
            stoichiometry={"E4P": -1, "TP": -1, "S7P": 1, "Pi": 1},
        )
        .add_reaction(
            name="v7",
            fn=v7,
            args=["k7", "S7P"],
            stoichiometry={"S7P": -1, "TPGA": 1, "Ru5P": 1},
        )
        .add_reaction(
            name="v8",
            fn=v8,
            args=["k8", "TPGA", "TP"],
            stoichiometry={"TPGA": -1, "TP": -1, "Ru5P": 1},
        )
        .add_reaction(
            name="v9",
            fn=v9,
            args=["k9", "Ru5P", "ATP"],
            stoichiometry={"Ru5P": -1, "ADP": 1, "RuBP": 1},
        )
        .add_reaction(
            name="v10",
            fn=v10,
            args=["k10", "ATP", "HP"],
            stoichiometry={"ADP": 1, "HP": -1, "Pi": 2, "GG": 1},
        )
        .add_reaction(
            name="v11",
            fn=v11,
            args=["k11", "GG", "Pi"],
            stoichiometry={"HP": 1, "GG": -1, "Pi": -1},
        )
        .add_reaction(
            name="v12",
            fn=v12,
            args=["k12", "TP", "Pio"],
            stoichiometry={"TP": -1, "Pio": -1, "Pi": 1, "TPo": 1},
        )
        .add_reaction(
            name="v13",
            fn=v13,
            args=["k13", "TPo"],
            stoichiometry={"TPo": -2, "Pio": 1, "HPo": 1},
        )
        .add_reaction(
            name="v14",
            fn=v14,
            args=["k14", "UDP", "Pio"],
            stoichiometry={"UDP": -1, "Pio": -1},
        )
        .add_reaction(
            name="v15",
            fn=v15,
            args=["k15", "UTP", "HPo"],
            stoichiometry={"UDP": 1, "HPo": -2, "Pio": 3, "GF": 1, "O2": 1 / 2},
        )
        .add_reaction(
            name="v16",
            fn=v16,
            args=["k16", "O2", "RuBP"],
            stoichiometry={"O2": -1, "RuBP": -1, "PGl": 1, "PGA": 1},
        )
        .add_reaction(
            name="v17",
            fn=v17,
            args=["k17", "PGl"],
            stoichiometry={"PGl": -1, "Gl": 1, "Pi": 1},
        )
        .add_reaction(
            name="v18",
            fn=v18,
            args=["k18", "Gl", "O2"],
            stoichiometry={"Gl": -2, "Gx": 2},
        )
        .add_reaction(
            name="v19",
            fn=v19,
            args=["k19", "Gx", "Sn"],
            stoichiometry={"Gx": -1, "Sn": -1, "Gn": 1, "GA": 1},
        )
        .add_reaction(
            name="v20",
            fn=v20,
            args=["k20", "Gn"],
            stoichiometry={"Gn": -2, "Sn": 1, "NH3": 1, "CO2": 1, "O2": -1 / 2},
        )
        .add_reaction(
            name="v21",
            fn=v21,
            args=["k21", "ATP", "GA"],
            stoichiometry={"ADP": 1, "GA": -1, "PGA": 1},
        )
        .add_reaction(
            name="v22",
            fn=v22,
            args=["k22", "ATP", "GmA", "NH3"],
            stoichiometry={
                "ADP": 1,
                "GmA": -1,
                "NH3": -1,
                "Pi": 1,
                "Glm": 1,
                "O2": 1 / 2,
            },
        )
        .add_reaction(
            name="v23",
            fn=v23,
            args=["k23", "Glm", "OxA"],
            stoichiometry={"Glm": -1, "OxA": -1, "GmA": 2},
        )
        .add_reaction(
            name="v24",
            fn=v24,
            args=["k24", "Gx", "GmA"],
            stoichiometry={"Gx": -1, "GmA": -1, "Gn": 1, "OxA": 1},
        )
        .add_reaction(
            name="vrd",
            fn=vrd,
            args=["rd", "GF"],
            stoichiometry={"GF": -1, "CO2": 12, "O2": -12},
        )
        .add_reaction(
            name="vphis", fn=vphis, args=["phis", "E", "GF"], stoichiometry={"GF": -1}
        )
        .add_reaction(
            name="vD",
            fn=vD,
            args=["D", "GF", "GFV"],
            stoichiometry={"GF": -1, "GFV": 1},
        )
        .add_reaction(
            name="vc1", fn=vc1, args=["kc1", "Ci"], stoichiometry={"Ci": -1, "CO2": 1}
        )
        .add_reaction(
            name="vc2", fn=vc2, args=["kc2", "CO2"], stoichiometry={"CO2": -1, "Ci": 1}
        )
        .add_reaction(
            name="vo1", fn=vo1, args=["ko1", "Oi"], stoichiometry={"Oi": -1, "O2": 1}
        )
        .add_reaction(
            name="vo2", fn=vo2, args=["ko2", "O2"], stoichiometry={"O2": -1, "Oi": 1}
        )
        .add_reaction(
            name="vphic", fn=vphic, args=["phic", "Ca", "Ci"], stoichiometry={"Ci": 1}
        )
        .add_reaction(
            name="vphio", fn=vphio, args=["phio", "Oa", "Oi"], stoichiometry={"Oi": 1}
        )
    )
