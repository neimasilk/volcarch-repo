"""E229 amendment A10 (2026-10-07): four further figures for the P8 revision, drawn from stored result files only.

F2 — the workflow of the study (data, label, inputs, classifier, checks), for readers who do not work with classifiers.
F3 — meanings with a PMP entry, by the coding of the list's form (three classes), six lists; data: E228
     TABLE_R1-2_retention_by_language.csv (the numbers of Table 2 of the article).
F4 — AUC for a held-out list, two input sets, with the chance line; data: E229 T3_lolo.csv (Table 4).
F5 — share of forms with a written glottal mark, uncoded against coded, by list; data: E231 P13_glottal_by_list.csv.
Press guidelines (VENUE.md): greyscale, at most 312 pt (26 picas) wide, 1,000 dpi TIFF with LZW for line art, Arial,
8–9 pt. Outputs: results/F{2..5}_*.{tif,png,pdf}, results/F2_F5_anchors.json. Each TIFF is checked to be <= 312 pt wide.
"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
E228 = HERE.parent / "E228_p8_revision_analyses" / "results"
E231 = HERE.parent / "E231_p8_what_coded_means" / "results"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
MAX_PT = 312


def save(make, stem, width_in, height_in, tight=True):
    """Render; if the tight bounding box exceeds 312 pt, shrink the figure width and render again (up to six times)."""
    bb = "tight" if tight else None
    for attempt in range(6):
        fig = make(width_in, height_in)
        tmp = RES / f"{stem}_tmp.png"
        fig.savefig(tmp, dpi=1000, bbox_inches=bb)
        with Image.open(tmp) as im0:
            im0.load(); size = im0.size; gray = im0.convert("L")
        w_pt = size[0] / 1000 * 72
        if w_pt <= MAX_PT or attempt == 5:
            fig.savefig(RES / f"{stem}.pdf", bbox_inches=bb)
            fig.savefig(RES / f"{stem}.png", dpi=600, bbox_inches=bb)
            if w_pt > MAX_PT:   # last resort: a resample of at most a few percent to meet the 26-pica limit
                target = int(MAX_PT / 72 * 1000); gray = gray.resize((target, round(gray.size[1] * target / gray.size[0])), Image.LANCZOS)
                size = gray.size; w_pt = size[0] / 1000 * 72
            gray.save(RES / f"{stem}.tif", compression="tiff_lzw", dpi=(1000, 1000))
            tmp.unlink(); plt.close(fig)
            return w_pt, size
        tmp.unlink(); plt.close(fig)
        width_in *= (MAX_PT - 4) / w_pt


anchors = {}

# ---------------------------------------------------------------- F2 workflow
def make_f2(w, h):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    FS = 6.3
    def box(x, y, text, fill="white", fs=FS):
        return ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.2,
                       bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.3", linewidth=0.6, edgecolor="black", facecolor=fill))
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", mutation_scale=7, linewidth=0.6, color="black"))
    box(0.5, 0.925, "ABVD: six Sulawesi word lists (Muna, Bugis, Makasar, Wolio, Sa'dan Toraja, Tolaki)\n1,357 forms for 210 meanings; forms taken as written", "0.92")
    arrow(0.5, 0.875, 0.5, 0.80)
    box(0.26, 0.735, "Label from ABVD's cognate coding\ncoded = has a cognate-set number (919)\nuncoded = has none (438)")
    box(0.74, 0.735, "25 inputs per form\n17 from the written form (length, vowels,\nglottal mark, letter strings), 8 from the meaning")
    arrow(0.30, 0.675, 0.44, 0.585); arrow(0.70, 0.675, 0.56, 0.585)
    box(0.5, 0.525, "Classifier (gradient-boosted trees), scored on forms it has not seen:\nfive-fold cross-validation, ten repetitions; one whole list held out", "0.92")
    arrow(0.5, 0.465, 0.5, 0.40)
    box(0.17, 0.325, "Ranking of uncoded\nabove coded forms\n(AUC; tables 3-4, fig. 4)")
    box(0.5, 0.325, "Profile: what distinguishes\nuncoded forms\n(table 5, figs 1 and 5)")
    box(0.83, 0.325, "Label x score cells,\nwith examples\n(tables 6-7)")
    arrow(0.5, 0.40, 0.19, 0.39); arrow(0.5, 0.40, 0.81, 0.39)
    box(0.5, 0.10, "Three checks on what 'uncoded' means: Makasar against the PMP entry, by meaning (table 2, fig. 3);\n"
                   "Tolaki against its subgroup (look-alike screen with a chance control); uncoded forms across lists\n"
                   "(permutation test). Spelling tests: glottal conventions, digraphs, size inputs (table 8)", "0.92")
    return fig
anchors["F2_width_pt"], _ = save(make_f2, "F2_workflow", 312 / 72, 3.3, tight=False)

# ---------------------------------------------------------------- F3 PMP classes
NAME = {"Makasar": "Makasar", "Bugis": "Bugis", "Tae' (Sa'dan Toraja)": "Sa'dan Toraja", "Wolio": "Wolio", "Muna": "Muna", "Tolaki": "Tolaki"}
ORDER = ["Makasar", "Bugis", "Tae' (Sa'dan Toraja)", "Wolio", "Muna", "Tolaki"]
rows = {r["language"]: r for r in csv.DictReader(open(E228 / "TABLE_R1-2_retention_by_language.csv", encoding="utf-8"))}
labels, a, b, c, n = [], [], [], [], []
for k in ORDER:
    r = rows[k]; tot = int(r["meanings_compared_with_PMP"])
    labels.append(NAME[k]); n.append(tot)
    a.append(100 * int(r["retained_from_PMP_n"]) / tot); b.append(100 * int(r["coded_not_PMP_etymon_n"]) / tot); c.append(100 * int(r["no_cognate_set_n"]) / tot)
anchors["F3_makasar_pct"] = [round(a[0], 1), round(b[0], 1), round(c[0], 1)]
def make_f3(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); y = range(len(labels))
    ax.barh(y, a, color="0.25", edgecolor="black", linewidth=0.5, label="in the PMP entry's cognate set")
    ax.barh(y, b, left=a, color="0.65", edgecolor="black", linewidth=0.5, label="coded in another set")
    ax.barh(y, c, left=[i + j for i, j in zip(a, b)], color="white", edgecolor="black", linewidth=0.5, hatch="///", label="no cognate set")
    for yi, (ai, bi, ci) in enumerate(zip(a, b, c)):
        ax.text(ai / 2, yi, f"{ai:.0f}", ha="center", va="center", color="white", fontsize=7)
        ax.text(ai + bi / 2, yi, f"{bi:.0f}", ha="center", va="center", color="black", fontsize=7)
        ax.text(ai + bi + ci / 2, yi, f"{ci:.0f}", ha="center", va="center", color="black", fontsize=7)
    ax.set_yticks(list(y)); ax.set_yticklabels([f"{l}\n(n = {k})" for l, k in zip(labels, n)])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of meanings with a PMP entry (percent)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
    return fig
anchors["F3_width_pt"], _ = save(make_f3, "F3_pmp_classes", 4.0, 2.7)

# ---------------------------------------------------------------- F4 held-out AUC
t3 = list(csv.DictReader(open(RES / "T3_lolo.csv", encoding="utf-8")))
lists = [r["list"] for r in t3 if r["input_set"] == "FM25" and r["list_key"] != "SUMMARY"]
fm = [float(r["auc"]) for r in t3 if r["input_set"] == "FM25" and r["list_key"] != "SUMMARY"]
fo = [float(r["auc"]) for r in t3 if r["input_set"] == "F17" and r["list_key"] != "SUMMARY"]
anchors["F4_tolaki"] = [round(fm[lists.index("Tolaki")], 3), round(fo[lists.index("Tolaki")], 3)]
def make_f4(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); y = range(len(lists))
    ax.axvline(0.5, color="0.5", linewidth=0.6, linestyle="--")
    ax.text(0.503, len(lists) - 0.6, "chance", fontsize=7, color="0.3", va="top")
    for yi, (u, v) in enumerate(zip(fm, fo)):
        ax.plot([v, u], [yi, yi], color="0.6", linewidth=0.8, zorder=1)
    ax.scatter(fm, list(y), marker="o", s=22, color="black", zorder=3, label="form and meaning inputs (25)")
    ax.scatter(fo, list(y), marker="s", s=20, facecolor="white", edgecolor="black", zorder=3, label="form inputs only (17)")
    ax.set_yticks(list(y)); ax.set_yticklabels(lists); ax.invert_yaxis(); ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("AUC for the held-out list (model trained on the other five lists)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4)
    return fig
anchors["F4_width_pt"], _ = save(make_f4, "F4_heldout_auc", 312 / 72, 2.3)

# ---------------------------------------------------------------- F5 glottal mark by list
g = {r["list"]: r for r in csv.DictReader(open(E231 / "P13_glottal_by_list.csv", encoding="utf-8"))}
order5 = ["Makasar", "Sa'dan Toraja", "Bugis", "Tolaki", "Wolio", "Muna"]
unc = [100 * int(g[k]["candidates_with_mark"]) / int(g[k]["candidates"]) for k in order5]
cod = [100 * int(g[k]["coded_with_mark"]) / int(g[k]["coded"]) for k in order5]
anchors["F5_makasar_unc_cod_pct"] = [round(unc[0], 1), round(cod[0], 1)]
anchors["F5_tolaki_unc_cod_pct"] = [round(unc[3], 1), round(cod[3], 1)]
def make_f5(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); x = range(len(order5)); bw = 0.38
    ax.bar([i - bw / 2 for i in x], unc, width=bw, color="0.25", edgecolor="black", linewidth=0.5, label="uncoded forms")
    ax.bar([i + bw / 2 for i in x], cod, width=bw, color="white", edgecolor="black", linewidth=0.5, hatch="///", label="coded forms")
    for i, (u, c_) in enumerate(zip(unc, cod)):
        ax.text(i - bw / 2, u + 1, f"{u:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + bw / 2, c_ + 1, f"{c_:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{k}\n({g[k]['candidates']} / {g[k]['coded']})" for k in order5], fontsize=7)
    ax.set_ylabel("Forms with a written glottal mark (percent)"); ax.set_ylim(0, 55)
    ax.legend(frameon=False, loc="upper right")
    return fig
anchors["F5_width_pt"], _ = save(make_f5, "F5_glottal_by_list", 312 / 72, 2.4)

(RES / "F2_F5_anchors.json").write_text(json.dumps(anchors, indent=1), encoding="utf-8")
print(json.dumps(anchors))
assert anchors["F3_makasar_pct"] == [39.3, 25.9, 34.8], anchors
assert anchors["F5_tolaki_unc_cod_pct"] == [21.6, 0.0], anchors
assert anchors["F4_tolaki"] == [0.809, 0.76], anchors
for stem in ("F2_workflow", "F3_pmp_classes", "F4_heldout_auc", "F5_glottal_by_list"):
    im = Image.open(RES / f"{stem}.tif"); print(stem, im.size, im.mode, im.info.get("dpi"), f"{im.size[0] / 1000 * 72:.0f} pt wide")
    assert im.size[0] / 1000 * 72 <= MAX_PT + 0.5
print("anchors and widths OK")
