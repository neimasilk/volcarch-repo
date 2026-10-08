"""E229 amendment A11 (2026-10-08): Figure 1 redrawn from the stored result file with a corrected legend.

Why: the legend of the uploaded Figure 1 said "more likely a candidate" — the internal name of the class the article
calls "uncoded" (the article has one label with two values, coded and uncoded) — and "(|r| < 0.3)", a symbol the
article does not explain (the rule is now stated in Section 2.4 of the article instead). Nothing is recomputed: the
bars, their order and the direction marks are read from results/F1_input_importance.csv as 01_tables.py wrote it, and
the drawing code below is the figure section of 01_tables.py, unchanged. The legend texts come from labels.csv, which
01_tables.py reads too, so a full re-run draws the same figure.

usage: python 08_figure1_legend.py [--check-old]
  --check-old  draw with the legend texts of the uploaded figure into results/_F1_check.png and compare it pixel by
               pixel with results/as_uploaded_20261007/F1_input_importance.png (test of the drawing code; writes
               nothing else)
"""
import io, sys, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CHECK = "--check-old" in sys.argv
LABELS = pd.read_csv(HERE / "labels.csv", dtype=str, keep_default_na=False, encoding="utf-8")
LAB = dict(zip(LABELS.key, LABELS.label))
if CHECK:   # the three legend texts of the figure uploaded on 2026-10-07
    LAB.update({"fig1_legend_up": "↑  larger value: more likely a candidate", "fig1_legend_down": "↓  larger value: more likely coded",
                "fig1_legend_none": "·  no clear direction (|r| < 0.3)"})
F1 = pd.read_csv(OUT / "F1_input_importance.csv", encoding="utf-8")
assert len(F1) == 25 and list(F1["rank"]) == list(range(1, 26)), "unexpected input table"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
FIG_W = 312 / 72
fig, ax = plt.subplots(figsize=(FIG_W, 7.2))
d = F1.iloc[::-1].reset_index(drop=True)         # largest bar at the top
fill = {"written form": "0.30", "meaning": "0.82"}
ax.barh(range(len(d)), d.mean_abs_shap, color=[fill[g] for g in d.group], edgecolor="black", linewidth=0.5, height=0.72)
ax.set_yticks(range(len(d)))
ax.set_yticklabels([textwrap.fill(t, 34) for t in d.label], fontsize=8, linespacing=0.9)
xmax = d.mean_abs_shap.max()
for i, r in d.iterrows():
    ax.text(r.mean_abs_shap + xmax * 0.02, i, r.direction, va="center", ha="left", fontsize=9, fontweight="bold")
ax.set_xlim(0, xmax * 1.14)
ax.set_xlabel(textwrap.fill(LAB["fig1_axis_x"], 34), fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylim(-0.7, len(d) - 0.3)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fill["written form"], edgecolor="black", linewidth=0.5),
           plt.Rectangle((0, 0), 1, 1, facecolor=fill["meaning"], edgecolor="black", linewidth=0.5),
           plt.Line2D([], [], linestyle="none"), plt.Line2D([], [], linestyle="none"), plt.Line2D([], [], linestyle="none")]
fig.tight_layout(pad=0.4, rect=[0, 0.125, 1, 1])
fig.legend(handles, [LAB["fig1_legend_form"], LAB["fig1_legend_meaning"], LAB["fig1_legend_up"], LAB["fig1_legend_down"],
                     LAB["fig1_legend_none"]],
           loc="lower center", fontsize=8, frameon=False, handlelength=1.3, borderaxespad=0.1, labelspacing=0.3)

if CHECK:
    fig.savefig(OUT / "_F1_check.png", dpi=600)
    a = np.asarray(Image.open(OUT / "_F1_check.png").convert("L"), dtype=int)
    b = np.asarray(Image.open(OUT / "as_uploaded_20261007" / "F1_input_importance.png").convert("L"), dtype=int)
    same = a.shape == b.shape and int(np.abs(a - b).max()) == 0
    print("drawing code reproduces the uploaded figure pixel for pixel:", same, a.shape, b.shape,
          "" if same or a.shape != b.shape else f"pixels differing: {int((a != b).sum())}")
    (OUT / "_F1_check.png").unlink()
    sys.exit(0 if same else 1)

fig.savefig(OUT / "F1_input_importance.png", dpi=600)
fig.savefig(OUT / "F1_input_importance.pdf")
_buf = io.BytesIO()
fig.savefig(_buf, format="png", dpi=1000)
_buf.seek(0)
_im = Image.open(_buf).convert("RGBA")
_bg = Image.new("RGBA", _im.size, (255, 255, 255, 255))
Image.alpha_composite(_bg, _im).convert("L").save(OUT / "F1_input_importance.tif", compression="tiff_lzw", dpi=(1000, 1000))
plt.close(fig)
im = Image.open(OUT / "F1_input_importance.tif")
print("F1 written:", im.size, im.mode, im.info.get("dpi"), f"{im.size[0] / 1000 * 72:.0f} pt wide")
assert im.size[0] / 1000 * 72 <= 312.5
