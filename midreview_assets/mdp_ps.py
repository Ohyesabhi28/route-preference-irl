import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib import font_manager as fm

INK, MUTED, ACC, SLATE = "#262626", "#6B6B6B", "#8A2432", "#3E5C76"
PAPER, BAND, LINE = "#FAF8F4", "#F1EDE5", "#CFC8BB"
names = {f.name for f in fm.fontManager.ttflist}
SERIF = "Georgia" if "Georgia" in names else "DejaVu Serif"
SANS = "Calibri" if "Calibri" in names else "DejaVu Sans"

fig = plt.figure(figsize=(10.0, 4.4), dpi=220)
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

# states: junction + incoming road (real Porto segments)
S = {"S1": (.9, 2.2, "Origin"), "S2": (4.9, 3.5, "Junction"), "S3": (4.9, .9, "Carvalhosa"),
     "S4": (9.0, 3.5, "Destination"), "S5": (9.0, 1.0, "Torrinha")}
A = {"A1": (2.9, 3.5, "Aníbal Cunha"), "A2": (2.9, .9, "Trav. Carvalhosa"),
     "B1": (7.0, 3.5, "Rua da Boa Hora"), "B2": (7.0, 1.7, "Rua da Torrinha")}
P = {k: v[:2] for k, v in {**S, **A}.items()}
RS, RA = .38, .52
ux = 72
for k, (x, y, l) in S.items():
    term = k == "S4"; off = k in ("S3", "S5")
    col = SLATE if term else (MUTED if off else INK)
    ax.add_patch(Circle((x, y), RS + (.07 if term else 0), fc="white" if term else BAND, ec=col,
                        lw=1.6 if term else 1.3, zorder=3))
    if term:
        ax.add_patch(Circle((x, y), RS - .01, fc="white", ec=col, lw=1.1, zorder=3))
    ax.text(x, y + .07, k, ha="center", va="center", fontsize=11, fontfamily=SERIF, fontweight="bold", color=col, zorder=4)
    ax.text(x, y - .16, l, ha="center", va="center", fontsize=6.3 if (term or off) else 6.8,
            fontfamily=SANS, color=MUTED, zorder=4)
for k, (x, y, l) in A.items():
    ax.add_patch(Polygon([(x - RA, y), (x, y + .36), (x + RA, y), (x, y - .36)], fc="white", ec=SLATE, lw=1.3,
                         joinstyle="round", zorder=3))
    ax.text(x, y + .08, k, ha="center", va="center", fontsize=10.5, fontfamily=SERIF, fontweight="bold", color=SLATE, zorder=4)
    ax.text(x, y - .13, l, ha="center", va="center", fontsize=5.8, fontfamily=SANS, color=MUTED, zorder=4)


def rad(k):
    if k in A:
        return .46 * ux
    return (RS + (.07 if k == "S4" else 0) + .02) * ux


def arrow(a, b, label=None, color=MUTED, lw=1.2, off=(0, 0), t=.5):
    ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=9, shrinkA=rad(a), shrinkB=rad(b) + 2,
                                 color=color, lw=lw, zorder=2))
    if label:
        p, q = np.array(P[a]), np.array(P[b]); x, y = p + (q - p) * t
        ax.text(x + off[0], y + off[1], label, ha="center", va="center", fontsize=7, fontfamily=SANS, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none"))


for s_, a_ in [("S1", "A1"), ("S1", "A2"), ("S2", "B1"), ("S2", "B2")]:
    arrow(s_, a_, color=LINE, lw=1.4)
arrow("A1", "S2", "P=1.0  R=−1.03")
arrow("A2", "S3", "P=1.0  R=−2.08")
arrow("B1", "S4", "P=1.0  R=−0.58")
arrow("B2", "S5", "P=1.0  R=−1.74", off=(0, -.1))
fig.savefig("midreview_assets/mdp_ps.png", facecolor=PAPER)
