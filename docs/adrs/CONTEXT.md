# mxlmodels: Architecture Context

This is the entry point for understanding *why* `mxlmodels` is shaped the way it is —
written down ahead of a maintainer handoff, alongside the equivalent `docs/adrs/`
directories in the sibling `mxlpy`, `mxlbricks`, `absorpig`, `parameteriser`, and
`schemegen` repos.

## What This Package Is

A curated collection of reference mechanistic models — mostly published photosynthesis
models, each as one flat, self-contained, inspectable file exporting `get_<name>() ->
Model`.

→ [ADR 0001 — Flat, single-file models of mixed provenance: codegen'd or hand-written, indistinguishably](0001-flat-files-mixed-provenance.md)
→ [ADR 0002 — `ss/` namespace separates steady-state models from dynamic ODE models](0002-dynamic-vs-steady-state-namespace-split.md)
→ [ADR 0003 — A handful of non-photosynthesis benchmark models are included deliberately](0003-non-photosynthesis-benchmark-models.md)

## Two Relationships That Have Changed Shape

These are the facts most likely to surprise someone who remembers this package's older
role:

→ [ADR 0004 — No longer the source feeding GreenSloth — decoupled so it can accept non-Python contributions](0004-no-longer-source-for-greensloth.md)
→ [ADR 0005 — `schemes/` images are hand-sourced from publications, not generated](0005-schemes-are-hand-sourced-not-generated.md)

`mxlmodels` used to be the pipeline GreenSloth's model catalog was ported from; it no
longer is (ADR 0004). Its diagram images predate, and remain independent of, the
`schemegen` project that later set out to auto-generate them (ADR 0005).

## Convenience Note

`mxlmodels/__init__.py` re-exports `mxlpy`'s `Simulator`, `fit`, `mc`, `mca`, `plot`, and
`scan` alongside every `get_*` model — a single `import mxlmodels` gets both the models
and the core analysis toolkit. This is a convenience choice, not an architectural one;
don't read it as `mxlmodels` owning or wrapping that functionality.

## Threads That Cross Multiple ADRs

- **Curatorial identity vs. convenience inclusions.** ADR 0003's non-photosynthesis
  benchmarks are kept for a real reason (a small, varied `mxlpy` model corpus other tools
  can depend on — see `schemegen`'s use of them) without diluting the package's actual
  photosynthesis-modeling identity, the same way `ss/` (ADR 0002) is kept structurally
  separate rather than blurred into the dynamic-model files.
- **Decoupling as scope discipline.** ADR 0004 and ADR 0005 are the same underlying move
  applied twice: when a downstream consumer (GreenSloth, `schemegen`) hasn't reached the
  quality/independence bar to depend on directly, keep `mxlmodels` self-contained rather
  than building a fragile bridge to something not ready yet.

## See Also

- [`mxlbricks`' `docs/adrs/0004-drifted-relationship-with-mxlmodels.md`](https://github.com/Computational-Biology-Aachen/mxl-bricks/blob/main/docs/adrs/0004-drifted-relationship-with-mxlmodels.md)
  for the inverse view of ADR 0001 — which models actually originate as brick
  compositions.
- [`greensloth`'s `docs/adrs/CONTEXT.md`](https://github.com/Computational-Biology-Aachen/green-sloth/blob/main/docs/adrs/CONTEXT.md)
  for the community-contribution pipeline that replaced the old `mxlmodels`-as-source
  relationship.
- [`schemegen`'s `docs/adrs/CONTEXT.md`](https://github.com/Computational-Biology-Aachen/schemegen/blob/main/docs/adrs/CONTEXT.md)
  for the auto-diagram tool `schemes/` predates and remains independent of.
