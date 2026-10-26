# ADR 0006: The Source of GreenSloth's Overlapping Models

**Status:** Implemented (supersedes ADR 0004 for models in both collections)
**Scope:** every model that exists both here and in GreenSloth

---

## 1. Context

ADR 0004 decoupled `mxlmodels` from GreenSloth so GreenSloth could accept models
authored directly in its `model.ts` format. In practice the 17 models present in both
collections drifted apart: different ids, readouts missing on the GreenSloth side
(saadat2021 had none), newer formulas only here, units and display names only there. Comparing
or validating one against the other required hand-made id maps.

## 2. Decision

For every model in both collections, GreenSloth's `model.ts` is *generated* from this
repo (mxlpy's `generate_model_code_mxlweb`, driven by
`python -m e2e.regenerate_greensloth` in mxl-e2e). This repo is the single source,
including what GreenSloth shows: `display_name`s, slider ranges, units and the module
docstring. mxl-e2e's `tests/test_greensloth.py` fails if GreenSloth's committed
`model.ts` differs from what the codegen produces.

Models only in GreenSloth (community contributions, the steady-state models whose
versions here are plain functions) stay authored there, as ADR 0004 intended.

## 3. Consequences

- Add or change information for an overlapping model here, then regenerate
  GreenSloth; never edit its `model.ts` by hand.
- GreenSloth's ids are the canonical identifiers of the ids here (mxl-schemas
  `v1/identifiers.json`), e.g. `"CO2 (dissolved)"` becomes `CO2_dissolved`, with the
  original kept as `display_name`.
