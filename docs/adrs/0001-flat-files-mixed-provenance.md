# ADR 0001: Flat, Single-File Models of Mixed Provenance — Codegen'd or Hand-Written, Indistinguishably

**Status:** Implemented
**Scope:** `src/mxlmodels/*.py`

---

## 1. Context

Every model in `mxlmodels` is one flat, self-contained Python file exporting a
`get_<name>() -> Model` function, explicitly so a reader can inspect a complete model
without following brick composition across many files (`mxlmodels`' whole reason to exist
alongside `mxlbricks`). Some of these files (`yokota1985.py`, `poolman2000.py`,
`matuszynska2016_*.py`, `matuszynska2019.py`, `saadat2021.py`, `ebeling2026.py`) are
generated from `mxlbricks`' brick composition; most others (`bellasio2019.py`,
`davis2017.py`, `lotka_volterra_v1.py`, `sir.py`, `hahn1987.py`, `zhu2005.py`, ...) are
written directly as flat files with no `mxlbricks` counterpart at all — see
[mxlbricks ADR 0004](https://github.com/Computational-Biology-Aachen/mxl-bricks/blob/main/docs/adrs/0004-drifted-relationship-with-mxlmodels.md)
for why that split exists.

## 2. Decision

Do not mark, in the file itself, whether a given model was generated from `mxlbricks` or
hand-written directly in `mxlmodels`. Both kinds live side by side as ordinary flat
`mxlpy` model files with the same `get_<name>()` shape.

## 3. Rationale

From a *consumer's* perspective — someone browsing, running, or citing a model from
`mxlmodels` — provenance doesn't matter: the flat file is the complete, inspectable
definition either way, and that inspectability is the entire value proposition of this
package over reading `mxlbricks`' brick composition. Surfacing provenance in the file
would only matter to a `mxlbricks`/`mxlmodels` maintainer deciding where to make an edit
— and that person can already tell by checking whether `mxlbricks` exports a matching
`get_*` name (see
[mxlbricks ADR 0004](https://github.com/Computational-Biology-Aachen/mxl-bricks/blob/main/docs/adrs/0004-drifted-relationship-with-mxlmodels.md)).

## 4. Consequences

- Before editing a model file here to fix a bug, check whether `mxlbricks` has a matching
  `get_*` export — if it does, the fix likely belongs upstream in the relevant brick(s)
  (and should be regenerated here), not as a direct hand-edit that will be silently
  overwritten by the next regeneration.
- If a model has no `mxlbricks` counterpart, editing it directly here is correct and
  final — there is no upstream to keep in sync with.
- A future contributor should not assume every model here can be "found in `mxlbricks`
  too" — most, by count, cannot.
