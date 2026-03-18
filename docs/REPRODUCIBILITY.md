# Reproducibility guide (DDIE v1.0)

This repo is intended for **paper reproduction**. It does not ship datasets, checkpoints, results, or logs.

## 1) Environment

- Python 3.10+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes for PyTorch:

- If `pip install -r requirements.txt` fails to install `torch` (common on some Windows/CUDA setups), install PyTorch first using the official selector, then re-run the command above.

## 2) Dataset setup (not included)

See `Data/datasets/README.md`.

## 3) Training

Example (NASA Battery, missing rate 0.3):

```bash
python scripts/train_ddie.py --config Config/nasa_battery.yaml --missing_rate 0.3
```

## 4) Running experiments

```bash
python scripts/run_experiments.py --config Config/nasa_battery.yaml
```

## 5) Figures

Generate paper figures (scripts will skip parts if corresponding result files are missing):

```bash
python scripts/generate_paper_figures.py
```

Figure 1 variants (see `scripts/generate_paper_figures.py` header comments):

```bash
python scripts/generate_paper_figures.py fig1_redesign          # FIG1-03 (three-column)
python scripts/generate_paper_figures.py fig1_redesign_blocks   # FIG1-02 (blocks composite)
python scripts/generate_paper_figures.py fig1_block B5          # FIG1-00 (block preview)
```

## 6) Notes

- If you need exact paper numbers, please use the same dataset subset and random seeds as the paper.
- The repository intentionally avoids any user/machine-specific paths.
- Third-party attribution: the Transformer backbone is adapted from Diffusion-TS (see `THIRD_PARTY_NOTICES.md` and `docs/REFERENCES.bib`).

