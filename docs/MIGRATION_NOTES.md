# Migration notes (DDIE v1.0)

This repository is a **minimal, publication-oriented** reconstruction of the original DDIE project for **paper reproduction**.

## Goals

- Keep the repo **small and portable** for GitHub publication.
- Remove **datasets / checkpoints / results / logs** (users download data separately).
- Remove any **user/machine-specific** paths or private files.
- Preserve the **full reproduction chain**: configs → training → experiments → figures.
- Keep **third-party attribution** complete and explicit.

## What is included

- **Core method**: `Models/`, `engine/`
- **Configs**: `Config/*.yaml`
- **Data loaders / utilities** (no data files): `Data/`, `Utils/`
- **Baselines**: `baselines/` (LI/KNN)
- **Entry scripts**: `scripts/` (train / experiments / figures)
- **Docs / compliance**:
  - `README.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/FIGURES.md`
  - `Data/datasets/README.md`
  - `THIRD_PARTY_NOTICES.md`, `docs/REFERENCES.bib`
  - `LICENSE`, `CITATION.cff`, `.gitignore`

## What is excluded (by design)

- Any dataset files (e.g., `*.mat`, `*.csv`)
- Any run outputs: `results*/`, `reports*/`, `Checkpoints_*/`, `log/`, `figures/`
- IDE caches and Python caches (`.vscode/`, `__pycache__/`, etc.)

## Third-party provenance

Parts of the Transformer backbone and related utilities are adapted from **Diffusion-TS** (MIT License).

See:

- `THIRD_PARTY_NOTICES.md`
- `docs/REFERENCES.bib`

## Minimal verification

After installing dependencies:

```bash
python scripts/self_check.py
python make_fig9_applicability_boundary.py
```

