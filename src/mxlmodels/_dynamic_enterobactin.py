r"""Dynamic enterobactin cross-feeding model: *E. coli* vs *C. glutamicum*.

|             |       |
| ----------- | ----- |
| doi         | FIXME |
| main author | FIXME |
| paper title | FIXME |
| published   | FIXME |
| journal     | FIXME |
| organism    | FIXME |

Developed within the SFB MibiNet community to study how siderophore-mediated
iron competition shapes the structure of synthetic microbial communities.

## Biological context

Enterobactin is a catecholate siderophore produced exclusively by *E. coli*
(x1) under iron limitation. Once secreted into the medium, it chelates ferric
iron (Fe³⁺) and is taken back up via specific receptors. *C. glutamicum* (x2)
cannot synthesise enterobactin but expresses uptake machinery for it, enabling
cross-feeding: it benefits from the iron chelated by *E. coli*'s siderophore
without paying the metabolic production cost. Enterobactin therefore acts as a
**public good** in the sense of evolutionary ecology — produced by one
organism, exploitable by both.

## Model structure

Four state variables:

- **x1** (g/L) — *E. coli* biomass
- **x2** (g/L) — *C. glutamicum* biomass
- **s1** (g/L) — shared carbon/energy substrate (e.g. glucose)
- **p1** (g/L) — extracellular enterobactin concentration (siderophore proxy
  for iron availability)

Five reactions:

- **mu1** — *E. coli* growth (double-Monod in s1 and p1)
- **mu2** — *C. glutamicum* growth (double-Monod in s1 and p1)
- **q_p1** — *E. coli* enterobactin production (growth-coupled)
- **q_up1** — *E. coli* enterobactin re-uptake (growth-coupled)
- **q_up2** — *C. glutamicum* enterobactin uptake (growth-coupled)

## ODEs

Growth kinetics follow a **double-Monod** form — growth requires both substrate
and enterobactin simultaneously::

```
mu_i = mu_max_i * s1/(K_s_i + s1) * p1/(K_p_i + p1)
```

Biomass and substrate::

```
dx1/dt = mu1 * x1
dx2/dt = mu2 * x2
ds1/dt = -mu1 * x1 / Y_X1_S  -  mu2 * x2 / Y_X2_S
```

Enterobactin (production minus uptake by both species)::

```
dp1/dt = q_p1_max * (mu1/mu_max1) * x1       # E. coli secretes
       - q_up_X1_max * (mu1/mu_max1) * x1     # E. coli re-uptakes
       - q_up_X2_max * (mu2/mu_max2) * x2     # C. glutamicum uptakes
```

The net specific enterobactin contribution of *E. coli* is
`(q_p1_max - q_up_X1_max) = 0.010 g/(gCDW·h)`, equal in magnitude to C.
glutamicum's uptake rate. At equal biomass the pool is therefore at steady
state; *E. coli* dominance fills it, *C. glutamicum* dominance depletes it.

## Key assumptions

1. **Iron is implicit.** Enterobactin concentration p1 serves as a proxy for
   bioavailable iron. No free Fe³⁺ is tracked; p1 lumps chelation and
   transport.
1. **Growth requires enterobactin.** Both organisms depend on p1 for growth via
   the Monod term. *E. coli* is modelled as highly sensitive (K_s_X1-P =
   0.00001 g/L) — it has evolved tight sensing for its own siderophore. *C.
   glutamicum* is less sensitive (K_s_X2-P = 0.001 g/L), reflecting a looser
   dependence.
1. **Production and uptake are growth-coupled.** Specific rates scale linearly
   with mu/mu_max. No constitutive (growth-independent) production or uptake is
   included.
1. **Only *E. coli* produces enterobactin.** *C. glutamicum* is a pure consumer
   (q_p_X2 = 0). This encodes the public-goods asymmetry.
1. **Batch culture — no dilution.** There is no washout term D·x or D·s, so
   this describes a closed batch experiment, not a chemostat.
1. **Constant, growth-independent yields.** Y_X1_S and Y_X2_S are fixed
   parameters; no maintenance or overflow metabolism.
1. **No enterobactin degradation.** p1 can only increase (production) or
   decrease (uptake); no abiotic hydrolysis or photodegradation is modelled.
1. **Well-mixed, spatially homogeneous.** All concentrations are bulk liquid
   values; no diffusion, gradients, or biofilm structure.

## Parameter notes

- *E. coli* has higher substrate affinity (K_s1 = 0.0005 g/L) but a lower
  maximum growth rate (mu_max1 = 0.22 h⁻¹) than *C. glutamicum* (K_s2 = 0.005
  g/L, mu_max2 = 0.45 h⁻¹). This captures the trade-off between high-affinity
  scavenging and fast growth seen in the two organisms.
- Initial conditions (x1 = x2 = 0.55 g/L) represent equal inoculation; the
  notebook scans across different x1/x2 ratios at fixed total biomass to probe
  community outcome sensitivity to inoculation ratio.
"""

from mxlpy import Derived, Model


def mul(x: float, y: float) -> float:
    return x * y


def mu(s: float, p: float, mu_max: float, kms: float, kmp: float) -> float:
    return mu_max * (s / (kms + s)) * (p / (kmp + p))


def q(q_max: float, mu: float, mu_max: float) -> float:
    return q_max * mu / mu_max


def sub(x: float) -> float:
    return -x


def minus_1_div(x: float) -> float:
    return -1 / x


def minus_div(x: float, y: float) -> float:
    return -x / y


def get_dynamic_enterobactin() -> Model:
    r"""Return the dynamic enterobactin cross-feeding model.

    See module docstring for full biological context, ODEs, and assumptions.
    """
    m = Model()
    m.add_variables({"x1": 0.55, "x2": 0.55, "s1": 10, "p1": 0.02})
    m.add_parameters(
        {
            "Y_X1_S": 0.45,  # gCDW/gSubstrate
            "Y_X2_S": 0.5,  # gCDW/gSubstrate
            "mu_max1": 0.22,  # 1/h
            "mu_max2": 0.45,  # 1/h
            "K_s1": 0.0005,  # g/L
            "K_s2": 0.005,  # g/L
            "q_p1_max": 0.015,  # gProduct/[gCDW*h]
            "q_up_X1_max": 0.005,  # gProduct/[gCDW*h]
            "q_up_X2_max": 0.01,  # gProduct/[gCDW*h]
            "K_s_X1-P": 0.00001,  # g/L
            "K_s_X2-P": 0.001,  # g/L
        }
    )
    m.add_reaction(
        "mu1",
        mu,
        args=["s1", "p1", "mu_max1", "K_s1", "K_s_X1-P"],
        stoichiometry={
            "x1": "x1",
            "s1": Derived(fn=minus_div, args=["x1", "Y_X1_S"]),
        },
    )
    m.add_reaction(
        "mu2",
        mu,
        args=["s1", "p1", "mu_max2", "K_s2", "K_s_X2-P"],
        stoichiometry={
            "x2": "x2",
            "s1": Derived(fn=minus_div, args=["x2", "Y_X2_S"]),
        },
    )

    m.add_reaction(
        "q_p1",
        q,
        args=["q_p1_max", "mu1", "mu_max1"],
        stoichiometry={"p1": "x1"},
    )
    m.add_reaction(
        "q_up1",
        q,
        args=["q_up_X1_max", "mu1", "mu_max1"],
        stoichiometry={"p1": Derived(fn=sub, args=["x1"])},
    )
    m.add_reaction(
        "q_up2",
        q,
        args=["q_up_X2_max", "mu2", "mu_max2"],
        stoichiometry={"p1": Derived(fn=sub, args=["x2"])},
    )
    return m
