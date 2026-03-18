# Figures

## Generate paper figures (Figure 2–7)

```bash
python scripts/generate_paper_figures_fig2_fig7_final.py
```

Outputs are written under `figures/paper_fig2_fig7_v001/`, `paper_fig2_fig7_v002/`, ... (auto-increment, no overwrite).

Each exported PNG is normalized to journal-ready settings: **RGB**, **min width ≥1100 px**, **300 dpi**, and a `manifest_vNNN.txt` is generated.

## Notes

This GitHub repo is a minimal reproduction package. The paper's final **Figure 1 / Figure 8 / Figure 9** may be maintained as external/vector assets in the authoring workflow; this repo only provides a consolidated generator/exporter for **Figure 2–7**.

