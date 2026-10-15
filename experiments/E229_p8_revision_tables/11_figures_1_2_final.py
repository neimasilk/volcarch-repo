"""E229 amendment A13 (2026-10-12): Figures 1 and 2 redrawn after the final review of the P8 revision. No number changes.

Figure 1 (input contributions) — same bars, order and direction marks, read from results/F1_input_importance.csv as
01_tables.py wrote it. Changed: (a) the two-line row labels touched each other (line spacing 0.9 -> 1.05, wrap 34 -> 36
characters); (b) three labels now come from labels.csv by the input's key instead of the label column stored in the
result file: "Meaning: number" (the article's name of the domain; was "numeral"), and the serial comma of the journal's
style in "(hyphens, spaces, and glottal marks included)" and "... na, po, or aŋ". Table 5 of the article takes the same
labels from labels.csv (build_docx.py), so table and figure cannot differ.

Figure 2 (workflow) — redrawn, because the diagram of 2026-10-07 showed the profile (table 5, figure 5) as an output of
the classifier. It is not: table 5 and figure 5 are counted directly from the label and the inputs, and only the
ranking (tables 3-4, figure 4), the contribution of each input (figure 1) and the label-by-score cells (tables 6-7)
come from the classifier (figure 1 from a model fitted to all forms, as its box says). Also: "(table 1)" in the label box; "(table 8)" attached to the glottal conventions only,
which is all that table holds; "10 repetitions" and en-dashes as in the article.

The files uploaded on 2026-10-08 are copied to results/as_uploaded_20261008/ before anything is overwritten.
Run 10_tiff_top_band.py afterwards (white band for the submission system's figure label).

usage: python 11_figures_1_2_final.py
"""
import io, shutil, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
KEEP = OUT / "as_uploaded_20261008"
KEEP.mkdir(exist_ok=True)
for stem in ("F1_input_importance", "F2_workflow"):
    for ext in ("tif", "png", "pdf"):
        src = OUT / f"{stem}.{ext}"
        if src.exists() and not (KEEP / src.name).exists():
            shutil.copy2(src, KEEP / src.name)

LABELS = pd.read_csv(HERE / "labels.csv", dtype=str, keep_default_na=False, encoding="utf-8")
LAB = dict(zip(LABELS.key, LABELS.label))
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
MAX_PT = 312


def to_tif(fig, path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=1000)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    Image.alpha_composite(bg, im).convert("L").save(path, compression="tiff_lzw", dpi=(1000, 1000))
    chk = Image.open(path)
    w_pt = chk.size[0] / 1000 * 72
    assert w_pt <= MAX_PT + 0.5, w_pt
    return chk.size, w_pt


# ------------------------------------------------------------------------------------------------ Figure 1
F1 = pd.read_csv(OUT / "F1_input_importance.csv", encoding="utf-8")
assert len(F1) == 25 and list(F1["rank"]) == list(range(1, 26)), "unexpected input table"
missing = [k for k in F1.feature if k not in LAB]
assert not missing, missing
labels = [LAB[k] for k in F1.feature]
changed = [(old, new) for old, new in zip(F1.label, labels) if old != new]
print("Figure 1 labels that differ from the stored result file:", changed)
assert len(changed) == 3, changed

fig, ax = plt.subplots(figsize=(MAX_PT / 72, 7.2))
d = F1.iloc[::-1].reset_index(drop=True)          # largest bar at the top
dl = labels[::-1]
fill = {"written form": "0.30", "meaning": "0.82"}
ax.barh(range(len(d)), d.mean_abs_shap, color=[fill[g] for g in d.group], edgecolor="black", linewidth=0.5, height=0.72)
ax.set_yticks(range(len(d)))
ax.set_yticklabels([textwrap.fill(t, 36) for t in dl], fontsize=8, linespacing=1.05)
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
fig.savefig(OUT / "F1_input_importance.png", dpi=600)
fig.savefig(OUT / "F1_input_importance.pdf")
size, w_pt = to_tif(fig, OUT / "F1_input_importance.tif")
plt.close(fig)
print("F1 written:", size, f"{w_pt:.0f} pt wide")


# ------------------------------------------------------------------------------------------------ Figure 2
def make_f2():
    fig = plt.figure(figsize=(MAX_PT / 72, 4.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    FS = 6.3

    def box(x, y, text, fill="white"):
        return ax.text(x, y, text, ha="center", va="center", fontsize=FS, linespacing=1.2,
                       bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.3", linewidth=0.6, edgecolor="black", facecolor=fill))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=7, linewidth=0.6, color="black", shrinkA=0, shrinkB=0))

    # data
    box(0.5, 0.945, "ABVD: six Sulawesi word lists (Muna, Bugis, Makasar, Wolio, Sa'dan Toraja, Tolaki)\n"
                    "1,357 forms for 210 meanings; forms taken as written", "0.92")
    arrow(0.26, 0.905, 0.26, 0.868); arrow(0.74, 0.905, 0.74, 0.868)
    # label and inputs
    box(0.26, 0.815, "Label from ABVD's cognate coding\ncoded = has a cognate-set number (919)\nuncoded = has none (438) (table 1)")
    box(0.74, 0.815, "25 inputs per form\n17 from the written form (length, vowels,\nglottal mark, letter strings), 8 from the meaning")
    # counted directly from label and inputs: the profile
    arrow(0.38, 0.762, 0.43, 0.728); arrow(0.62, 0.762, 0.57, 0.728)
    box(0.5, 0.655, "Counted directly, without the classifier:\nprofile of uncoded against coded forms,\nproperty by property within lists\n(table 5, fig. 5)")
    # the classifier takes label and inputs too
    arrow(0.235, 0.762, 0.235, 0.522); arrow(0.765, 0.762, 0.765, 0.522)
    box(0.5, 0.465, "Classifier (gradient-boosted trees), scored on forms it has not seen:\n"
                    "five-fold cross-validation, 10 repetitions; one whole list held out", "0.92")
    arrow(0.30, 0.418, 0.20, 0.358); arrow(0.5, 0.418, 0.5, 0.358); arrow(0.70, 0.418, 0.80, 0.358)
    box(0.17, 0.305, "Ranking of uncoded\nabove coded forms\n(AUC; tables 3–4, fig. 4)")
    box(0.5, 0.305, "Contribution of each input\nto the score (model fitted\nto all forms; fig. 1)")
    box(0.83, 0.305, "Label crossed with\nscore, with examples\n(tables 6–7)")
    # checks
    box(0.5, 0.125, "Three checks on what \"uncoded\" means: Makasar against the PMP entry, by meaning (table 2, fig. 3);\n"
                    "Tolaki against its subgroup (look-alike screen with a chance control); uncoded forms across lists\n"
                    "(permutation test). Spelling tests: glottal conventions (table 8), digraphs, size inputs", "0.92")
    return fig


fig = make_f2()
fig.savefig(OUT / "F2_workflow.png", dpi=600)
fig.savefig(OUT / "F2_workflow.pdf")
size, w_pt = to_tif(fig, OUT / "F2_workflow.tif")
plt.close(fig)
print("F2 written:", size, f"{w_pt:.0f} pt wide")
