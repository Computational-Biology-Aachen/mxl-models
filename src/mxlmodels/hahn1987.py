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


def _moiety_1(concentration: float, total: float) -> float:
    return total - concentration


def _p_an(
    h: float,
    k1: float,
    co2: float,
    rubp: float,
    k20: float,
    gn: float,
    rd: float,
    gf: float,
) -> float:
    return 36000 * 44 * h * (k1 * co2 * rubp - k20 * gn**2 - 12 * rd * gf)


def _v1(k1: float, co2: float, ru_bp: float) -> float:
    return k1 * co2 * ru_bp


def _v2(k2: float, adp: float, pi: float) -> float:
    return k2 * adp * pi


def _v3(k3: float, pga: float, atp: float) -> float:
    return k3 * pga * atp


def _v4(k4: float, tp: float) -> float:
    return k4 * tp**2


def _v5(k5: float, hp: float) -> float:
    return k5 * hp


def _v6(k6: float, e4_p: float, tp: float) -> float:
    return k6 * e4_p * tp


def _v7(k7: float, s7_p: float) -> float:
    return k7 * s7_p


def _v8(k8: float, tpga: float, tp: float) -> float:
    return k8 * tpga * tp


def _v9(k9: float, ru5_p: float, atp: float) -> float:
    return k9 * atp * ru5_p


def _v10(k10: float, atp: float, hp: float) -> float:
    return k10 * atp * hp


def _v11(k11: float, gg: float, pi: float) -> float:
    return k11 * gg * pi


def _v12(k12: float, tp: float, pio: float) -> float:
    return k12 * tp * pio


def _v13(k13: float, t_po: float) -> float:
    return k13 * t_po**2


def _v14(k14: float, udp: float, pio: float) -> float:
    return k14 * udp * pio


def _v15(k15: float, utp: float, h_po: float) -> float:
    return k15 * utp * h_po


def _v16(k16: float, o2: float, ru_bp: float) -> float:
    return k16 * o2 * ru_bp


def _v17(k17: float, p_gl: float) -> float:
    return k17 * p_gl


def _v18(k18: float, gl: float, o2: float) -> float:
    return k18 * gl**2 * o2


def _v19(k19: float, gx: float, sn: float) -> float:
    return k19 * gx * sn


def _v20(k20: float, gn: float) -> float:
    return k20 * gn**2


def _v21(k21: float, atp: float, ga: float) -> float:
    return k21 * atp * ga


def _v22(k22: float, atp: float, gm_a: float, nh3: float) -> float:
    return k22 * atp * gm_a * nh3


def _v23(k23: float, glm: float, ox_a: float) -> float:
    return k23 * glm * ox_a


def _v24(k24: float, gx: float, gm_a: float) -> float:
    return k24 * gx * gm_a


def _vrd(rd: float, gf: float) -> float:
    return rd * gf


def _vphis(phis: float, e: float, gf: float) -> float:
    return phis * (gf - e)


def _v_d(d: float, gf: float, gfv: float) -> float:
    return d * (gf - gfv)


def _vc1(kci: float, ci: float) -> float:
    return kci * ci


def _vc2(kc2: float, co2: float) -> float:
    return kc2 * co2


def _vo1(ko1: float, oi: float) -> float:
    return ko1 * oi


def _vo2(ko2: float, o2: float) -> float:
    return ko2 * o2


def _vphic(phic: float, ca: float, ci: float) -> float:
    return phic * (ca - ci)


def _vphio(phio: float, oa: float, oi: float) -> float:
    return phio * (oa - oi)


def get_hahn1987() -> Model:
    r"""Get the Hahn 1987 model."""
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
        .add_derived(name="ATP", fn=_moiety_1, args=["ADP", "AP_tot"])
        .add_derived(name="UTP", fn=_moiety_1, args=["UDP", "UP_tot"])
        .add_derived(
            name="PAn",
            fn=_p_an,
            args=["h", "k1", "CO2", "RuBP", "k20", "Gn", "rd", "GF"],
        )
        .add_reaction(
            name="v1",
            fn=_v1,
            args=["k1", "CO2", "RuBP"],
            stoichiometry={"RuBP": -1, "PGA": 2, "CO2": -1, "O2": -1 / 2},
        )
        .add_reaction(
            name="v2",
            fn=_v2,
            args=["k2", "ADP", "Pi"],
            stoichiometry={"ADP": -1, "Pi": -1},
        )
        .add_reaction(
            name="v3",
            fn=_v3,
            args=["k3", "PGA", "ATP"],
            stoichiometry={"PGA": -1, "ADP": 1, "Pi": 1, "TP": 1, "O2": 1 / 2},
        )
        .add_reaction(
            name="v4",
            fn=_v4,
            args=["k4", "TP"],
            stoichiometry={"TP": -2, "Pi": 1, "HP": 1},
        )
        .add_reaction(
            name="v5",
            fn=_v5,
            args=["k5", "HP"],
            stoichiometry={"HP": -1, "TPGA": 1, "E4P": 1},
        )
        .add_reaction(
            name="v6",
            fn=_v6,
            args=["k6", "E4P", "TP"],
            stoichiometry={"E4P": -1, "TP": -1, "S7P": 1, "Pi": 1},
        )
        .add_reaction(
            name="v7",
            fn=_v7,
            args=["k7", "S7P"],
            stoichiometry={"S7P": -1, "TPGA": 1, "Ru5P": 1},
        )
        .add_reaction(
            name="v8",
            fn=_v8,
            args=["k8", "TPGA", "TP"],
            stoichiometry={"TPGA": -1, "TP": -1, "Ru5P": 1},
        )
        .add_reaction(
            name="v9",
            fn=_v9,
            args=["k9", "Ru5P", "ATP"],
            stoichiometry={"Ru5P": -1, "ADP": 1, "RuBP": 1},
        )
        .add_reaction(
            name="v10",
            fn=_v10,
            args=["k10", "ATP", "HP"],
            stoichiometry={"ADP": 1, "HP": -1, "Pi": 2, "GG": 1},
        )
        .add_reaction(
            name="v11",
            fn=_v11,
            args=["k11", "GG", "Pi"],
            stoichiometry={"HP": 1, "GG": -1, "Pi": -1},
        )
        .add_reaction(
            name="v12",
            fn=_v12,
            args=["k12", "TP", "Pio"],
            stoichiometry={"TP": -1, "Pio": -1, "Pi": 1, "TPo": 1},
        )
        .add_reaction(
            name="v13",
            fn=_v13,
            args=["k13", "TPo"],
            stoichiometry={"TPo": -2, "Pio": 1, "HPo": 1},
        )
        .add_reaction(
            name="v14",
            fn=_v14,
            args=["k14", "UDP", "Pio"],
            stoichiometry={"UDP": -1, "Pio": -1},
        )
        .add_reaction(
            name="v15",
            fn=_v15,
            args=["k15", "UTP", "HPo"],
            stoichiometry={"UDP": 1, "HPo": -2, "Pio": 3, "GF": 1, "O2": 1 / 2},
        )
        .add_reaction(
            name="v16",
            fn=_v16,
            args=["k16", "O2", "RuBP"],
            stoichiometry={"O2": -1, "RuBP": -1, "PGl": 1, "PGA": 1},
        )
        .add_reaction(
            name="v17",
            fn=_v17,
            args=["k17", "PGl"],
            stoichiometry={"PGl": -1, "Gl": 1, "Pi": 1},
        )
        .add_reaction(
            name="v18",
            fn=_v18,
            args=["k18", "Gl", "O2"],
            stoichiometry={"Gl": -2, "Gx": 2},
        )
        .add_reaction(
            name="v19",
            fn=_v19,
            args=["k19", "Gx", "Sn"],
            stoichiometry={"Gx": -1, "Sn": -1, "Gn": 1, "GA": 1},
        )
        .add_reaction(
            name="v20",
            fn=_v20,
            args=["k20", "Gn"],
            stoichiometry={"Gn": -2, "Sn": 1, "NH3": 1, "CO2": 1, "O2": -1 / 2},
        )
        .add_reaction(
            name="v21",
            fn=_v21,
            args=["k21", "ATP", "GA"],
            stoichiometry={"ADP": 1, "GA": -1, "PGA": 1},
        )
        .add_reaction(
            name="v22",
            fn=_v22,
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
            fn=_v23,
            args=["k23", "Glm", "OxA"],
            stoichiometry={"Glm": -1, "OxA": -1, "GmA": 2},
        )
        .add_reaction(
            name="v24",
            fn=_v24,
            args=["k24", "Gx", "GmA"],
            stoichiometry={"Gx": -1, "GmA": -1, "Gn": 1, "OxA": 1},
        )
        .add_reaction(
            name="vrd",
            fn=_vrd,
            args=["rd", "GF"],
            stoichiometry={"GF": -1, "CO2": 12, "O2": -12},
        )
        .add_reaction(
            name="vphis",
            fn=_vphis,
            args=["phis", "E", "GF"],
            stoichiometry={"GF": -1},
        )
        .add_reaction(
            name="vD",
            fn=_v_d,
            args=["D", "GF", "GFV"],
            stoichiometry={"GF": -1, "GFV": 1},
        )
        .add_reaction(
            name="vc1",
            fn=_vc1,
            args=["kc1", "Ci"],
            stoichiometry={"Ci": -1, "CO2": 1},
        )
        .add_reaction(
            name="vc2",
            fn=_vc2,
            args=["kc2", "CO2"],
            stoichiometry={"CO2": -1, "Ci": 1},
        )
        .add_reaction(
            name="vo1",
            fn=_vo1,
            args=["ko1", "Oi"],
            stoichiometry={"Oi": -1, "O2": 1},
        )
        .add_reaction(
            name="vo2",
            fn=_vo2,
            args=["ko2", "O2"],
            stoichiometry={"O2": -1, "Oi": 1},
        )
        .add_reaction(
            name="vphic",
            fn=_vphic,
            args=["phic", "Ca", "Ci"],
            stoichiometry={"Ci": 1},
        )
        .add_reaction(
            name="vphio",
            fn=_vphio,
            args=["phio", "Oa", "Oi"],
            stoichiometry={"Oi": 1},
        )
    )
