#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FIG1-04 定稿 → 期刊用副本（不覆盖定稿源图 fig1_ddie_framework_redesign.png）

源（优先 figures/，其次 assets/）：fig1_ddie_framework_redesign.png
输出：figures/fig1_FIG1-04_journal_001.png、_002、…（序号递增，自行择优保留）

处理：RGB、宽<2400 则 2×、宽<1100 白边垫齐、300 dpi。
详见 docs/Figure1_工作记录与代码对照.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
ASSETS = ROOT / "assets"

# 与 save_figure_journal_ready 一致
JOURNAL_DPI = 300
MIN_WIDTH_PX = 1100
UPSCALE_THRESHOLD_W = 2400


def _to_rgb(im):
    from PIL import Image

    if im.mode in ("RGBA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "RGBA":
            bg.paste(im, mask=im.split()[3])
        else:
            bg.paste(im)
        return bg
    return im.convert("RGB")


def export_fig1_fig04_journal() -> Path:
    from PIL import Image

    src = FIG / "fig1_ddie_framework_redesign.png"
    if not src.is_file():
        src = ASSETS / "fig1_ddie_framework_redesign.png"
    if not src.is_file():
        raise FileNotFoundError(
            "FIG1-04 源缺失：请将定稿保存为 figures/fig1_ddie_framework_redesign.png "
            "或 assets/fig1_ddie_framework_redesign.png"
        )

    im = _to_rgb(Image.open(src))
    w, h = im.size
    if w < UPSCALE_THRESHOLD_W:
        im = im.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        w, h = im.size
    if w < MIN_WIDTH_PX:
        out_im = Image.new("RGB", (MIN_WIDTH_PX, h), (255, 255, 255))
        out_im.paste(im, ((MIN_WIDTH_PX - w) // 2, 0))
        im = out_im

    n = 1
    while True:
        dst = FIG / f"fig1_FIG1-04_journal_{n:03d}.png"
        if not dst.exists():
            break
        n += 1

    FIG.mkdir(parents=True, exist_ok=True)
    im.save(dst, "PNG", dpi=(JOURNAL_DPI, JOURNAL_DPI), compress_level=6)
    return dst


def main() -> None:
    try:
        dst = export_fig1_fig04_journal()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    from PIL import Image

    i = Image.open(dst)
    print("FIG1-04 源未改动。")
    print("期刊副本:", dst.resolve())
    print("尺寸:", i.size, "dpi:", i.info.get("dpi"))


if __name__ == "__main__":
    main()
