"""Lazar 1997 Fluorescence induction model.

|             |                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------- |
| doi         | 10.1006/pest.1997.2277                                                                      |
| main author | Dušan Lazár                                                                                 |
| paper title | Mathematical modeling of changes in chlorophyll fluorescence induction caused by herbicides |
| published   | 1997                                                                                        |
| journal     | Pesticide Biochemistry and Physiology                                                       |
| organism    | plants (chlorophyll fluorescence)                                                           |
| Ported by   | ElouenCorvest ( @ElouenCorvest )                                                            |
"""

from mxlpy import Model


def v1(
    k1: float,
    k17: float,
    k10: float,
    k16: float,
    y1: float,
    y3: float,
    y9: float,
    y11: float,
) -> float:
    return -(k1 + k17) * y1 + k10 * y3 + k16 * y9 * y11


def v2(
    k1: float,
    k23: float,
    k6: float,
    k7: float,
    k11: float,
    k22: float,
    y1: float,
    y2: float,
    y3: float,
    y4: float,
    y9: float,
    y12: float,
) -> float:
    return k1 * y1 - (k6 + k23) * y2 + k7 * y3 + k11 * y4 + k22 * y9 * y12


def v3(k6: float, k2: float, k7: float, k10: float, y2: float, y3: float) -> float:
    return k6 * y2 - (k2 + k7 + k10) * y3


def v4(
    k2: float, k8: float, k11: float, k9: float, y3: float, y4: float, y5: float
) -> float:
    return k2 * y3 - (k8 + k11) * y4 + k9 * y5


def v5(
    k8: float,
    k3: float,
    k9: float,
    k12: float,
    k13: float,
    y4: float,
    y5: float,
    y7: float,
) -> float:
    return k8 * y4 - (k3 + k9 + k12) * y5 + k13 * y7


def v6(k3: float, k18: float, k19: float, y5: float, y6: float, y8: float) -> float:
    return k3 * y5 - k18 * y6 + k19 * y8


def v7(
    k12: float,
    k4: float,
    k13: float,
    k14: float,
    k15: float,
    y5: float,
    y7: float,
    y10: float,
    y11: float,
) -> float:
    return k12 * y5 - (k4 + k13 + k14) * y7 + k15 * y10 * y11


def v8(
    k18: float,
    k4: float,
    k19: float,
    k20: float,
    k21: float,
    y6: float,
    y7: float,
    y8: float,
    y10: float,
    y12: float,
) -> float:
    return k18 * y6 + k4 * y7 - (k19 + k20) * y8 + k21 * y10 * y12


def v9(
    k17: float,
    k23: float,
    k24: float,
    k25: float,
    k16: float,
    k22: float,
    y1: float,
    y2: float,
    y10: float,
    y11: float,
    y12: float,
    y9: float,
) -> float:
    return k17 * y1 + k23 * y2 + k24 * y10 - (k25 + k16 * y11 + k22 * y12) * y9


def v10(
    k14: float,
    k20: float,
    k25: float,
    k24: float,
    k15: float,
    k21: float,
    y7: float,
    y8: float,
    y9: float,
    y11: float,
    y12: float,
    y10: float,
) -> float:
    return k14 * y7 + k20 * y8 + k25 * y9 - (k24 + k15 * y11 + k21 * y12) * y10


def v11(
    k17: float,
    k14: float,
    k16: float,
    k15: float,
    y1: float,
    y7: float,
    y9: float,
    y10: float,
    y11: float,
) -> float:
    return k17 * y1 + k14 * y7 - (k16 * y9 + k15 * y10) * y11


def v12(
    k23: float,
    k20: float,
    k22: float,
    k21: float,
    y2: float,
    y8: float,
    y9: float,
    y10: float,
    y12: float,
) -> float:
    return k23 * y2 + k20 * y8 - (k22 * y9 + k21 * y10) * y12


def v13(k6: float, y13: float) -> float:
    return -k6 * y13


def v14(k5: float, y13: float) -> float:
    return k5 * y13


def v15() -> float:
    return 0


def eprime(y2: float, y4: float, y6: float, y8: float, y12: float, y14: float) -> float:
    return y2 + y4 + y6 + y8 + y12 + y14


def F(eprime: float, p: float, y15: float) -> float:
    nom = (1 - p) * eprime
    denom = 1 - p * eprime
    return nom / denom + y15


def get_lazar1997():
    return (
        Model()
        .add_parameters(
            {
                "k1": 1666,
                "k2": 1250,
                "k3": 500,
                "k4": 20000,
                "k5": 1666,
                "k6": 3500,
                "k7": 175,
                "k8": 1750,
                "k9": 35,
                "k10": 5000,
                "k11": 5000,
                "k12": 150,
                "k13": 100,
                "k14": 150,
                "k15": 100,
                "k16": 150,
                "k17": 100,
                "k18": 150,
                "k19": 100,
                "k20": 150,
                "k21": 100,
                "k22": 150,
                "k23": 100,
                "k24": 1,
                "k25": 1,
                "k6n": 3500,
                "p": 0.55,
            }
        )
        .add_variables(
            {
                "y1": 0.66,
                "y2": 0,
                "y3": 0,
                "y4": 0,
                "y5": 0,
                "y6": 0,
                "y7": 0,
                "y8": 0,
                "y9": 7,
                "y10": 0,
                "y11": 0,
                "y12": 0,
                "y13": 0.22,
                "y14": 0,
                "y15": 0.12,
            }
        )
        .add_reaction(
            "v1",
            fn=v1,
            args=["k1", "k17", "k10", "k16", "y1", "y3", "y9", "y11"],
            stoichiometry={"y1": 1},
        )
        .add_reaction(
            "v2",
            fn=v2,
            args=[
                "k1",
                "k23",
                "k6",
                "k7",
                "k11",
                "k22",
                "y1",
                "y2",
                "y3",
                "y4",
                "y9",
                "y12",
            ],
            stoichiometry={"y2": 1},
        )
        .add_reaction(
            "v3",
            fn=v3,
            args=["k6", "k2", "k7", "k10", "y2", "y3"],
            stoichiometry={"y3": 1},
        )
        .add_reaction(
            "v4",
            fn=v4,
            args=["k2", "k8", "k11", "k9", "y3", "y4", "y5"],
            stoichiometry={"y4": 1},
        )
        .add_reaction(
            "v5",
            fn=v5,
            args=["k8", "k3", "k9", "k12", "k13", "y4", "y5", "y7"],
            stoichiometry={"y5": 1},
        )
        .add_reaction(
            "v6",
            fn=v6,
            args=["k3", "k18", "k19", "y5", "y6", "y8"],
            stoichiometry={"y6": 1},
        )
        .add_reaction(
            "v7",
            fn=v7,
            args=["k12", "k4", "k13", "k14", "k15", "y5", "y7", "y10", "y11"],
            stoichiometry={"y7": 1},
        )
        .add_reaction(
            "v8",
            fn=v8,
            args=["k18", "k4", "k19", "k20", "k21", "y6", "y7", "y8", "y10", "y12"],
            stoichiometry={"y8": 1},
        )
        .add_reaction(
            "v9",
            fn=v9,
            args=[
                "k17",
                "k23",
                "k24",
                "k25",
                "k16",
                "k22",
                "y1",
                "y2",
                "y10",
                "y11",
                "y12",
                "y9",
            ],
            stoichiometry={"y9": 1},
        )
        .add_reaction(
            "v10",
            fn=v10,
            args=[
                "k14",
                "k20",
                "k25",
                "k24",
                "k15",
                "k21",
                "y7",
                "y8",
                "y9",
                "y11",
                "y12",
                "y10",
            ],
            stoichiometry={"y10": 1},
        )
        .add_reaction(
            "v11",
            fn=v11,
            args=["k17", "k14", "k16", "k15", "y1", "y7", "y9", "y10", "y11"],
            stoichiometry={"y11": 1},
        )
        .add_reaction(
            "v12",
            fn=v12,
            args=["k23", "k20", "k22", "k21", "y2", "y8", "y9", "y10", "y12"],
            stoichiometry={"y12": 1},
        )
        .add_reaction("v13", fn=v13, args=["k6n", "y13"], stoichiometry={"y13": 1})
        .add_reaction(
            "v14", fn=v14, args=["k6n", "y13"], stoichiometry={"y14": 1}
        )  # Change in contrast to pub
        .add_reaction("v15", fn=v15, args=[], stoichiometry={"y15": 1})
        .add_derived(
            name="eprime", fn=eprime, args=["y2", "y4", "y6", "y8", "y12", "y14"]
        )
        .add_derived(name="F", fn=F, args=["eprime", "p", "y15"])
    )
