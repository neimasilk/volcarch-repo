"""E229 amendment A12 (2026-10-08): a white band at the top of the five figure TIFFs that are uploaded to the journal.

Why: the journal's submission system builds a "Merged PDF" for editors and reviewers, puts every figure on a page of
its own (landscape figures rotated) and prints a large label "Figure n" over the top centre of the image itself. On the
files uploaded on 2026-10-07 that label covered the first row label of Figure 1, the first box of Figure 2 and parts
of the plots of Figures 3-5 (seen in the merged PDF downloaded on 2026-10-08). Measured on that PDF, the label occupies
roughly the top 400 pixels of each 1,000-dpi image. A white band of 480 pixels (0.48 inch) at the top keeps the
drawing clear of it. Nothing inside the drawing changes; the width (at most 312 pt) is unchanged; the band can be
cropped in production. Only the .tif files carry the band; the .png and .pdf previews do not.

The script is idempotent: a TIFF that already carries the band (recorded in its ImageDescription tag) is skipped.
Run it after 07_figures_revision.py or 08_figure1_legend.py have rewritten a TIFF.

usage: python 10_tiff_top_band.py
"""
from pathlib import Path
from PIL import Image

RES = Path(__file__).resolve().parent / "results"
BAND = 480
MARK = f"top band {BAND} px (white) for the submission system's figure label; E229 A12"
for name in ("F1_input_importance", "F2_workflow", "F3_pmp_classes", "F4_heldout_auc", "F5_glottal_by_list"):
    p = RES / f"{name}.tif"
    with Image.open(p) as im:
        im.load()
        desc = str(im.tag_v2.get(270, "")) if hasattr(im, "tag_v2") else ""
        if "top band" in desc:
            print(f"{name}: band already present, skipped ({im.size[0]} x {im.size[1]})")
            continue
        assert im.mode == "L", im.mode
        w, h = im.size
        out = Image.new("L", (w, h + BAND), 255)
        out.paste(im, (0, BAND))
    out.save(p, compression="tiff_lzw", dpi=(1000, 1000), description=MARK)
    with Image.open(p) as chk:
        assert chk.size == (w, h + BAND) and chk.mode == "L" and abs(chk.info["dpi"][0] - 1000) < 1
        top = chk.crop((0, 0, w, BAND))
        assert top.getextrema() == (255, 255), "band is not white"
        body_same = list(chk.crop((0, BAND, w, h + BAND)).getdata()) == list(Image.open(p).crop((0, BAND, w, h + BAND)).getdata())
    print(f"{name}: {w} x {h} -> {w} x {h + BAND} px, {w / 1000 * 72:.0f} pt wide, band white, {p.stat().st_size:,} bytes")
