# ADR 0002: `ss/` Namespace Separates Steady-State Models from Dynamic ODE Models

**Status:** Implemented
**Scope:** `src/mxlmodels/ss/` (`fvcb1980.py`, `bernacchi2013.py`, `johnson2021.py`) vs.
the package's top-level ODE model files

---

## 1. Context

Most models in `mxlmodels` are dynamic ODE systems built with `mxlpy`'s `Model` +
`Simulator`, integrated forward in time. `ss/` holds a different kind of model: steady-
state/algebraic photosynthesis models (Farquhar–von Caemmerer–Berry 1980 C3 assimilation,
Bernacchi 2013 temperature responses, Johnson 2021) that are solved as a system of
algebraic equations at a fixed operating point, not integrated over time.

## 2. Decision

Keep steady-state models in a dedicated `ss/` subpackage, re-exported from the top-level
`mxlmodels` namespace (`get_fvcb`, `get_bernacchi_2013`, `get_johnson2021`) alongside the
dynamic models, rather than either mixing them into the flat top-level files unmarked or
splitting them into a separate package.

## 3. Rationale

Steady-state and dynamic models use a genuinely different slice of the `mxlpy` API
surface (no time integration, no `Simulator` workflow) and answer a different kind of
question (a single operating point vs. a trajectory) — grouping them under `ss/` signals
that difference to a reader browsing the package without requiring a separate
installable package for what is still, curatorially, the same collection of reference
photosynthesis models. Re-exporting them at the top level keeps `mxlmodels`' "one
`import mxlmodels`, all reference models are `get_*` away" convenience intact.

## 4. Consequences

- A new steady-state (non-integrated) reference model belongs in `ss/`, not at the
  top level, even though its import is re-exported at the top level either way.
- Don't assume every model callable as `mxlmodels.get_*()` behaves the same way — check
  whether it's a `ss/` model (call it directly for an operating point) or a dynamic model
  (feed it to `Simulator`) before assuming a time-course workflow applies.
