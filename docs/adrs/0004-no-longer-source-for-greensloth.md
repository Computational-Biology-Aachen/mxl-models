# ADR 0004: No Longer the Source Feeding GreenSloth — Decoupled So It Can Accept Non-Python Contributions

**Status:** Implemented (supersedes earlier practice)
**Scope:** relationship between `mxlmodels` and the `greensloth` site; no code in this
repo depends on `greensloth`

---

## 1. Context

Several `mxlmodels` files predate this decoupling and say so directly — e.g.
`li2021.py`'s docstring notes the model "was therefore included in GreenSloth." Several
models now shipped by GreenSloth (Matuszyńska 2016 NPQ, Matuszyńska 2019, Davis 2017,
Li 2021) originated as `mxlmodels` Python implementations that were ported into
GreenSloth's own TypeScript model format.

## 2. Decision

`mxlmodels` is no longer treated as GreenSloth's model source. GreenSloth now has its own
independent authoring pipeline (`model.ts` → generated `.mxl.json`, see
[greensloth ADR 0002](https://github.com/Computational-Biology-Aachen/green-sloth/blob/main/docs/adrs/0002-model-ts-generate-mxl-parity-gate.md))
and its own GitHub-issue-driven contribution scaffolding (see
[greensloth ADR 0003](https://github.com/Computational-Biology-Aachen/green-sloth/blob/main/docs/adrs/0003-issue-to-model-contribution-pipeline.md)).
Nothing in `mxlmodels` exports to, or is imported by, GreenSloth going forward.

## 3. Rationale

GreenSloth's actual mission is a **community-contributed** photosynthesis model database
(see [greensloth ADR 0001](https://github.com/Computational-Biology-Aachen/green-sloth/blob/main/docs/adrs/0001-purpose-community-photosynthesis-database.md))
— external labs, students, and paper authors contribute models directly, and many of
those contributors have no reason to know Python or `mxlpy` at all. Requiring every
GreenSloth model to first exist as an `mxlmodels` Python implementation would have made
`mxlmodels`/Python fluency a hard prerequisite for contributing to GreenSloth, directly
contradicting its community-access goal. Decoupling the two lets GreenSloth accept a
model authored straight in its own `model.ts` format, with no `mxlmodels` involvement at
all.

## 4. Consequences

- Docstrings like `li2021.py`'s "included in GreenSloth" note are historical fact about
  how that specific model got there, not a description of an ongoing pipeline — don't
  read them as implying new `mxlmodels` additions will automatically reach GreenSloth.
- Adding a model to `mxlmodels` today has no effect on GreenSloth's catalog; adding a
  model to GreenSloth has no effect on `mxlmodels`' catalog. Treat them as editorially
  independent collections that happen to overlap historically, not as one feeding the
  other.
- If a model is wanted in both places going forward, it needs to be added twice,
  independently, in each project's own native format — there is no shared source of
  truth anymore.
