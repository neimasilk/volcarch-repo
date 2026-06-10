"""Regenerate fig3 with the tradition-controlled z_cross values (R1, 2026-06-08)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 300, "savefig.dpi": 300})
v6 = json.load(open(r"D:/documents/volcarch-repo/experiments/E090_transformer_textual_nlp/results/e090_v6_tradition_controlled.json"))
labels = {"SPICE_TRADE":"spice_trade","CAMPHOR_BARUS":"camphor_barus","SUMATRA_GOLD":"sumatra_gold",
          "MARITIME_VOYAGE":"maritime_voyage","JAVA":"java","VOLCANO":"volcano",
          "BUDDHIST_WORLD":"buddhist_world","METAL_TRADE":"metal_trade"}
rows = [(labels[k], v6[k]["z_cross_tradition"]) for k in labels if "z_cross_tradition" in v6[k]]
rows.sort(key=lambda x: x[1], reverse=True)
names = [r[0] for r in rows]; z = [r[1] for r in rows]
colors = ["#c0392b" if n == "volcano" else "#2c6fbb" for n in names]

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(range(len(names)), z, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(1.96, ls="--", color="gray", lw=1)
ax.text(len(names)-0.4, 2.6, "z = 1.96 (p = 0.05)", color="gray", fontsize=8, ha="right")
for b, zz in zip(bars, z):
    ax.text(b.get_x()+b.get_width()/2, zz+0.4, f"{zz:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_ylabel(r"Cross-tradition convergence  $z_{\mathrm{cross}}$")
ax.set_title("Cross-tradition semantic convergence (tradition-controlled test)")
ax.set_ylim(0, max(z)*1.15)
plt.tight_layout()
out = r"D:/documents/volcarch-repo/papers/P16_computational_textual_archaeology/figures/fig3_convergence_zscores.png"
plt.savefig(out, bbox_inches="tight"); print("Saved", out, "| z order:", list(zip(names, z)))
