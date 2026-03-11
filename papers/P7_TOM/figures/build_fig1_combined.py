"""Combine 1860 and present Dwarapala photos side-by-side for Figure 1."""
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except ImportError:
    print("matplotlib required")
    sys.exit(1)

P1_FIGS = Path("D:/documents/volcarch-repo/papers/P1_taphonomic_framework/figures")
OUT_DIR = Path("D:/documents/volcarch-repo/papers/P7_TOM/figures")

img_1860 = mpimg.imread(P1_FIGS / "fig0a_dwarapala_1860.jpg")
img_now = mpimg.imread(P1_FIGS / "fig0b_dwarapala_present.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left: 1860 photo
ax1.imshow(img_1860)
ax1.set_title("~1860 CE — half-buried (~185 cm)", fontsize=10, fontweight="bold", pad=8)
ax1.text(0.02, 0.02, "Source: Leiden University Library",
         transform=ax1.transAxes, fontsize=6, color="white",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
ax1.axis("off")

# Right: present photo
ax2.imshow(img_now)
ax2.set_title("Present (2026) — excavated, person for scale", fontsize=10, fontweight="bold", pad=8)
ax2.text(0.02, 0.02, "Total height: ~370 cm",
         transform=ax2.transAxes, fontsize=7, color="white",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
ax2.axis("off")

fig.suptitle("Dwarapala Guardian Statue, Singosari, East Java",
             fontsize=13, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save PNG + TIF
png_out = OUT_DIR / "fig1_dwarapala_1860.png"
plt.savefig(png_out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {png_out}")

tif_out = OUT_DIR / "fig1_dwarapala_1860.tif"
plt.savefig(tif_out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {tif_out}")

plt.close()
