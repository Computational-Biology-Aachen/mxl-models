# ADR 0003: A Handful of Non-Photosynthesis Benchmark Models Are Included Deliberately

**Status:** Implemented
**Scope:** `lotka_volterra_v1.py`, `lotka_volterra_v2.py`, `sir.py`,
`prigogine1968_brusselator.py`, `selkov1968_oscillator.py`,
`elowitz2000_repressilator.py`, `_population_dynamics.py`, `_tripartite_dynamics.py`,
`_dynamic_enterobactin.py`

---

## 1. Context

`mxlmodels` is, by intent and by the overwhelming majority of its contents, a curated
collection of published *photosynthesis* models. A small number of files are something
else entirely: classic, non-photosynthesis dynamical-systems benchmarks — Lotka-Volterra
predator-prey, SIR epidemic dynamics, the Brusselator, the Selkov glycolytic oscillator,
the Elowitz-Leibler repressilator — plus a few ecology/microbiology toy models
(population dynamics, tripartite dynamics, dynamic enterobactin).

## 2. Decision

Keep these non-photosynthesis models in `mxlmodels` rather than spinning them into a
separate "generic ODE examples" package, and don't treat their presence as scope creep
to be cleaned up.

## 3. Rationale

These textbook models are small, well-understood ODE systems that are useful as a
lightweight, dependency-light corpus wherever a *diverse* set of `mxlpy` models is needed
for something other than photosynthesis-specific analysis — for example
`schemegen`'s example corpus
(see [schemegen ADR 0004](https://github.com/Computational-Biology-Aachen/schemegen/blob/main/docs/adrs/0004-experimental-status-not-yet-convincing.md))
draws on `get_lotka_volterra_v1`/`get_sir`/`get_prigogine1968_brusselator` alongside the
real photosynthesis models specifically because a layout/rendering tool needs small,
structurally-varied test cases, not another 40-reaction photosynthesis network. Keeping
them here means any `mxlpy`-based tool that needs "a few small, varied example models"
has one package to depend on instead of needing a second one just for toy systems.

## 4. Consequences

- Don't reject a new benchmark/textbook ODE model from `mxlmodels` on "this isn't
  photosynthesis" grounds alone — the bar is closer to "is this a well-known reference
  system useful as a small/varied example," not strict domain match.
- Do keep this category small and clearly a minority — `mxlmodels`' curatorial identity
  and the bulk of maintenance effort is still photosynthesis models; if the non-
  photosynthesis set grew large, splitting it out would become worth reconsidering.
