import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager as fm

INK, MUTED, ACC, SLATE = "#262626", "#6B6B6B", "#8A2432", "#3E5C76"
PAPER, BAND, LINE = "#FAF8F4", "#F1EDE5", "#CFC8BB"
names = {f.name for f in fm.fontManager.ttflist}
SERIF = "Georgia" if "Georgia" in names else "DejaVu Serif"
SANS = "Calibri" if "Calibri" in names else "DejaVu Sans"

fig = plt.figure(figsize=(10.0, 4.6), dpi=220)
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
ux = 72

# Real numbers: mean learned reward R = w.phi over all 14,788 transitions of each turn type (Porto graph);
# share = fraction of the 77,531 expert (real taxi) steps that take that turn type.
S = {"S0": (.85, 2.75, "S$_0$", "Origin"),
     "S1": (5.3, 2.75, "S$_t$", "junction + road"),
     "ST": (9.1, 2.75, "S$_T$", "Destination")}
A = {"A1": (3.0, 4.0, "Straight on", "73% of driver steps"),
     "A2": (3.0, 2.75, "Slight turn", "20% of driver steps"),
     "A3": (3.0, 1.5, "Sharp turn / U-turn", "7% of driver steps")}
P = {k: v[:2] for k, v in {**S, **A}.items()}
RS, RA = .38, .55

for k, (x, y, lab, sub) in S.items():
    term = k == "ST"
    col = SLATE if term else INK
    ax.add_patch(Circle((x, y), RS + (.07 if term else 0), fc="white" if term else BAND, ec=col,
                        lw=1.6 if term else 1.3, zorder=3))
    if term:
        ax.add_patch(Circle((x, y), RS - .01, fc="white", ec=col, lw=1.1, zorder=3))
    ax.text(x, y + .07, lab, ha="center", va="center", fontsize=12, fontfamily=SERIF, fontweight="bold", color=col, zorder=4)
    ax.text(x, y - .17, sub, ha="center", va="center", fontsize=5.8 if "junction" in sub else 6.6,
            fontfamily=SANS, color=MUTED, zorder=4)
for k, (x, y, name, share) in A.items():
    ax.add_patch(Polygon([(x - RA, y), (x, y + .38), (x + RA, y), (x, y - .38)], fc="white", ec=SLATE, lw=1.3,
                         joinstyle="round", zorder=3))
    ax.text(x, y + .09, k, ha="center", va="center", fontsize=10.5, fontfamily=SERIF, fontweight="bold", color=SLATE, zorder=4)
    ax.text(x, y - .12, name, ha="center", va="center", fontsize=5.6, fontfamily=SANS, color=INK, zorder=4)
    ax.text(x, y - .53, share, ha="center", va="center", fontsize=6.2, fontfamily=SANS, color=MUTED, zorder=4)


def rad(k):
    if k in A:
        return .5 * ux
    return (RS + (.07 if k == "ST" else 0) + .02) * ux


def arrow(a, b, label=None, color=MUTED, lw=1.2, off=(0, 0), t=.5, lcol=INK):
    ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=9, shrinkA=rad(a), shrinkB=rad(b) + 2,
                                 color=color, lw=lw, zorder=2))
    if label:
        p, q = np.array(P[a]), np.array(P[b]); x, y = p + (q - p) * t
        ax.text(x + off[0], y + off[1], label, ha="center", va="center", fontsize=7, fontfamily=SANS, color=lcol, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none"))


for a in A:
    arrow("S0", a, color=LINE, lw=1.4)
arrow("A1", "S1", "P=1.0   R=−0.58", off=(.05, .06))
arrow("A2", "S1", "P=1.0   R=−1.14", t=.55)
arrow("A3", "S1", "P=1.0   R=−2.24", off=(.05, -.06))
arrow("S1", "ST", "repeat at every junction", color=MUTED, lcol=MUTED)

fx, fy, fw, fh = .45, .12, 9.1, .62
ax.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0,rounding_size=.08", fc=BAND, ec=LINE, lw=1, zorder=1))
items = [("6,644", "states  (junction + incoming road)"),
         ("14,788", "actions  (road → next road)"),
         ("P = 1.0", "deterministic transitions"),
         ("R = w·φ(s,a)", "mean per action type, learned weights")]
for i, (big, small) in enumerate(items):
    x = fx + .35 + i * 2.28
    ax.text(x, fy + fh * .64, big, ha="left", va="center", fontsize=10.5, fontfamily=SERIF, fontweight="bold",
            color=ACC, zorder=4)
    ax.text(x, fy + fh * .27, small, ha="left", va="center", fontsize=6.6, fontfamily=SANS, color=MUTED, zorder=4)

fig.savefig("midreview_assets/mdp_proj.png", facecolor=PAPER)
