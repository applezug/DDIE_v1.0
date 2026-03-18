# DDIE v1.0 (paper reproduction)

DDI-E is a conditional diffusion model for probabilistic reconstruction (imputation) of electrical degradation time series under random missing data.

This repository contains the **minimal, publication-oriented** code needed to reproduce the paper’s method and figures **without including any datasets, checkpoints, or run outputs**.

## What’s included / excluded

- **Included**: model code, training/evaluation scripts, baseline methods (LI/KNN), figure generation scripts, and experiment configs.
- **Excluded**: datasets, pretrained checkpoints, experiment logs, results, and any files containing machine- or user-specific paths.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Notes for PyTorch:

- If `pip install -r requirements.txt` fails to install `torch` (common on some Windows/CUDA setups), install PyTorch first using the official selector, then re-run the command above.

## Quick Start (no dataset required)

```bash
python scripts/self_check.py
python make_fig9_applicability_boundary.py
```

## Datasets (not included)

Place datasets under `Data/datasets/` using the structure below.

### NASA Battery dataset

- Download: NASA Prognostics Center of Excellence repository (see paper for exact subset).
- Put files (e.g., `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`) in:
  - `Data/datasets/nasa_battery/`

### NASA IGBT dataset

- Prepare the NASA IGBT SMU measurement data (see paper / project notes for the exact folder).
- Default config expects:
  - `Data/datasets/NASA IGBT/Data/SMU Data for new devices/`

If the expected dataset files are not found, some scripts may fall back to **synthetic data** for quick sanity checks (not for paper numbers).

## Quickstart

### Train

```bash
python scripts/train_ddie.py --config Config/nasa_battery.yaml --missing_rate 0.3
```

### Run experiments

```bash
python scripts/run_experiments.py --config Config/nasa_battery.yaml
```

### Generate paper figures

```bash
python scripts/generate_paper_figures.py
```

## Documentation

- `docs/REPRODUCIBILITY.md`: step-by-step reproduction
- `docs/FIGURES.md`: figure generation and FIG1 variants
- `docs/MIGRATION_NOTES.md`: what was migrated/removed and why
- `THIRD_PARTY_NOTICES.md`: third-party attributions

## Repository layout

```
DDIE_v1.0/
├── Config/        # experiment configs (*.yaml)
├── Data/          # dataloader builders (datasets are NOT included)
├── Models/        # DDI-E and backbone
├── Utils/         # dataset loaders + metrics
├── baselines/     # LI/KNN and optional baseline wrappers
├── engine/        # training utilities
└── scripts/       # training / evaluation / figure scripts
```

## Citation

See `CITATION.cff`.

## License

MIT License. See `LICENSE`.
