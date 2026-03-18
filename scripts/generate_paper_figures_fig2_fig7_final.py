#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export ONLY the paper figures Figure 2–7 (journal-ready PNGs).

This repo (DDIE v1.0) focuses on reproducibility and does not ship datasets/results.
When results are missing, some plots may be skipped by upstream generators.

Outputs:
  figures/paper_fig2_fig7_v001/ (auto-increment) with:
    Figure2_*.png ... Figure7_*.png
  plus manifest_v001.txt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


@dataclass(frozen=True)
class PaperFig:
    idx: int  # 2..7 (paper numbering)
    title: str
    dst_name: str
    src_path: Path


def _next_out_dir() -> Path:
    k = 1
    while (FIG / f"paper_fig2_fig7_v{k:03d}").exists():
        k += 1
    out = FIG / f"paper_fig2_fig7_v{k:03d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ensure_journal_png(src: Path, dst: Path) -> tuple[int, int]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_journal_mdpi_figures import ensure_mdpi_png

    return ensure_mdpi_png(src, dst)


def _generate_sources() -> Path:
    """
    Generate the required source figures under figures/.
    Returns a temporary Fig4 journal-wide PNG path.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_paper_figures as gpf
    from generate_fig4_downstream_rul import save_journal_copy

    # figures/ is ignored by git and may not exist
    FIG.mkdir(parents=True, exist_ok=True)

    # Fig2
    gpf.fig2_nasa_imputation()
    # Fig3 (paper): LOOCV figure saved as fig3_nasa_loocv_original_scale.png
    gpf.fig8_nasa_loocv_original_scale()
    # Fig5 (paper): NASA IGBT MAPE (alias fig5_*.png)
    gpf.fig3_nasa_igbt_imputation()
    # Fig6 combined
    gpf.fig6_combined_mae()
    # Fig7 example (stored as fig9_imputation_example.png in generator)
    gpf.fig9_imputation_example()

    tmp_fig4 = FIG / "_tmp_Figure4_downstream_RUL.png"
    save_journal_copy(tmp_fig4)
    return tmp_fig4


def main() -> None:
    tmp_fig4 = _generate_sources()

    out_dir = _next_out_dir()
    vid = out_dir.name.replace("paper_fig2_fig7_", "")

    figs = [
        PaperFig(
            2,
            "NASA Battery imputation MAE/RMSE",
            "Figure2_nasa_battery_mae_rmse.png",
            FIG / "fig2_nasa_imputation_mae_rmse.png",
        ),
        PaperFig(
            3,
            "NASA Battery LOOCV (original scale)",
            "Figure3_nasa_battery_loocv.png",
            FIG / "fig3_nasa_loocv_original_scale.png",
        ),
        PaperFig(
            4,
            "Downstream RUL (journal-wide canvas)",
            "Figure4_downstream_rul.png",
            tmp_fig4,
        ),
        PaperFig(
            5,
            "NASA IGBT imputation (MAPE %)",
            "Figure5_nasa_igbt_mape.png",
            FIG / "fig5_nasa_igbt_imputation_mae_rmse.png",
        ),
        PaperFig(
            6,
            "Combined: NASA Battery MAE + NASA IGBT MAPE",
            "Figure6_combined_battery_igbt.png",
            FIG / "fig6_combined_nasa_battery_igbt_mae.png",
        ),
        PaperFig(
            7,
            "Imputation example (synthetic)",
            "Figure7_imputation_example.png",
            FIG / "fig9_imputation_example.png",
        ),
    ]

    manifest_lines: list[str] = []
    for f in figs:
        dst = out_dir / f.dst_name
        if not f.src_path.is_file():
            manifest_lines.append(f"SKIP Figure {f.idx} (missing source): {f.src_path}")
            continue
        wh = _ensure_journal_png(f.src_path, dst)
        manifest_lines.append(f"OK Figure {f.idx} {f.title} -> {dst.name} ({wh[0]}x{wh[1]})")

    (out_dir / f"manifest_{vid}.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    # Cleanup temp Fig4 file
    try:
        tmp_fig4.unlink(missing_ok=True)
    except TypeError:
        if tmp_fig4.exists():
            tmp_fig4.unlink()

    print("Paper Figure 2–7 exported to:", out_dir)
    print("\n".join(manifest_lines))


if __name__ == "__main__":
    main()

