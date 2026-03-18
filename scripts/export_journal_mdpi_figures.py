#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 !Draft-B-Figures,Graphics,Images 要求.md：
  - 宽度 ≥ 1100 px，300 dpi，PNG（可再打 ZIP 上传）
  - 输出到 figures/journal_MDPI_v001/、v002/…，不覆盖 figures/ 下原图

步骤：
  1) 运行 generate_paper_figures（若缺数据会跳过部分图）
  2) 下游 RUL 图单独宽版写入本批次目录
  3) 可选 assets 框架图
  4) 每张图：若宽 <1100 则左右白边垫至 1100；写入 PNG dpi=(300,300)
  5) 图文摘要 1100×560 @300dpi（单独入口上传，副本放入批次供归档）
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
ASSETS = ROOT / "assets"

JOURNAL_DPI = 300
MIN_WIDTH_PX = 1100


def next_batch_dir() -> Path:
    k = 1
    while (FIG / f"journal_MDPI_v{k:03d}").exists():
        k += 1
    d = FIG / f"journal_MDPI_v{k:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_mdpi_png(src: Path, dst: Path) -> tuple[int, int]:
    """RGB, min width MIN_WIDTH_PX, dpi 300. Returns (w,h)."""
    from PIL import Image

    im = Image.open(src)
    if im.mode in ("RGBA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "RGBA":
            bg.paste(im, mask=im.split()[3])
        else:
            bg.paste(im)
        im = bg
    else:
        im = im.convert("RGB")
    w, h = im.size
    if w < MIN_WIDTH_PX:
        out = Image.new("RGB", (MIN_WIDTH_PX, h), (255, 255, 255))
        out.paste(im, ((MIN_WIDTH_PX - w) // 2, 0))
        im = out
        w = MIN_WIDTH_PX
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "PNG", dpi=(JOURNAL_DPI, JOURNAL_DPI), compress_level=6)
    return im.size


def main() -> None:
    batch = next_batch_dir()
    vid = batch.name.replace("journal_MDPI_", "")
    readme = batch / "README_MDPI_export.txt"
    readme.write_text(
        f"""MDPI / Electronics — Figures, Graphics, Images 批次 {vid}

要求：单图宽 ≥ {MIN_WIDTH_PX} px，{JOURNAL_DPI} dpi，PNG/TIFF。
图文摘要请仍在投稿系统「Graphical Abstract」单独上传；本目录中的 GA 仅为归档副本。

打包：将此文件夹内 PNG 压缩为 ZIP（<200MB）上传。

Figure 1 定稿（FIG1-04）：manifest 中含「FIG1-04」「定稿框架」的条目，对应源文件
fig1_ddie_framework_redesign.png；批次内文件名形如 NN_fig1_framework_redesign_MDPI_{vid}.png
（NN 依导出顺序，v001 示例曾为 10_...）。详见项目 docs/Figure1_工作记录与代码对照.md。
""",
        encoding="utf-8",
    )

    # 1) 主图脚本
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_paper_figures.py")], cwd=str(ROOT))
    if r.returncode != 0:
        print("Warning: generate_paper_figures 退出码", r.returncode)

    # 2) 下游 RUL 期刊宽版（不覆盖 figures/fig4_downstream_tasks.png）
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_fig4_downstream_rul import save_journal_copy

        save_journal_copy(batch / f"Fig_downstream_RUL_MDPI_{vid}.png")
    except Exception as e:
        print("Fig downstream journal copy skipped:", e)

    # 3) 收集源文件（原 figures 内，不删不改）
    # LOOCV：正文 Figure 3 为 fig3_nasa_loocv_original_scale；若无则用旧名 fig8
    _loocv = FIG / "fig3_nasa_loocv_original_scale.png"
    if not _loocv.is_file():
        _loocv = FIG / "fig8_nasa_loocv_original_scale.png"
    # 01 主框架 FIG1-01
    patterns = [
        "fig1_ddie_framework.png",
        "fig2_nasa_imputation_mae_rmse.png",
        "fig3_nasa_igbt_imputation_mae_rmse.png",
        "fig4_method_selection_framework.png",
        "fig5_applicability_boundary.png",
        "fig6_combined_nasa_battery_igbt_mae.png",
        "fig7_downstream_tasks.png",
        "__LOOCV__",
        "fig9_imputation_example.png",
    ]
    seq = 0
    manifest = []

    for name in patterns:
        if name == "__LOOCV__":
            src = _loocv
            name = src.name
            if not src.is_file():
                manifest.append("SKIP (missing): fig3/fig8 LOOCV")
                continue
        else:
            src = FIG / name
            if not src.is_file():
                manifest.append(f"SKIP (missing): {name}")
                continue
        seq += 1
        dst = batch / f"{seq:02d}_{name.replace('.png', '')}_MDPI_{vid}.png"
        wh = ensure_mdpi_png(src, dst)
        manifest.append(f"OK {seq:02d} {name} -> {dst.name} ({wh[0]}x{wh[1]})")

    # ---------- FIG1-04：定稿框架图（手工/外部矢量）→ 期刊批次 ----------
    # 源固定名 fig1_ddie_framework_redesign.png；批次内示例见 journal_MDPI_v001/10_fig1_framework_redesign_MDPI_v001.png
    # （序号 seq 随本批次前面图片数量变化，以 manifest 为准；工作记录 docs/Figure1_工作记录与代码对照.md）
    for extra in [
        ASSETS / "fig1_ddie_framework_redesign.png",
        FIG / "fig1_ddie_framework_redesign.png",
    ]:
        if extra.is_file():
            seq += 1
            dst = batch / f"{seq:02d}_fig1_framework_redesign_MDPI_{vid}.png"
            wh = ensure_mdpi_png(extra, dst)
            manifest.append(
                f"OK {seq:02d} FIG1-04 {extra.name} -> {dst.name} ({wh[0]}x{wh[1]}) [定稿框架 MDPI]"
            )
            break

    # 4) 图文摘要 300 dpi
    ga_out = batch / f"GraphicalAbstract_1100x560_MDPI_{vid}.png"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_graphical_abstract.py"), "--mdpi-out", str(ga_out)],
        cwd=str(ROOT),
    )
    if r.returncode == 0 and ga_out.is_file():
        manifest.append(f"OK GA -> {ga_out.name}")
    else:
        manifest.append("SKIP GA (script failed)")

    (batch / f"manifest_{vid}.txt").write_text("\n".join(manifest), encoding="utf-8")
    print("Batch folder:", batch)
    print("\n".join(manifest))


if __name__ == "__main__":
    main()
