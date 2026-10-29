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
ux = 72
RS, RA = .4, .5

S = {"s0": (.9, 2.2, "s$_0$", "origin"),
     "s1": (5.0, 3.6, "s$_1$", "junction + road"),
     "s2": (5.0, 2.2, "s$_2$", "junction + road"),
     "s3": (5.0, .8, "s$_m$", "junction + road"),
     "sT": (9.1, 2.2, "s$_T$", "destination")}
A = {"a1": (2.95, 3.6, "a$_1$", "road 1"),
     "a2": (2.95, 2.2, "a$_2$", "road 2"),
     "a3": (2.95, .8, "a$_k$", "road k"),
     "b1": (7.05, 3.0, "a$_t$", "next road"),
     "b2": (7.05, 1.4, "a$_t$", "next road")}
P = {k: v[:2] for k, v in {**S, **A}.items()}

for k, (x, y, lab, sub) in S.items():
    term = k == "sT"
    col = SLATE if term else INK
    ax.add_patch(Circle((x, y), RS + (.07 if term else 0), fc="white" if term else BAND, ec=col,
                        lw=1.6 if term else 1.3, zorder=3))
    if term:
        ax.add_patch(Circle((x, y), RS - .01, fc="white", ec=col, lw=1.1, zorder=3))
    ax.text(x, y + .07, lab, ha="center", va="center", fontsize=13, fontfamily=SERIF, fontweight="bold", color=col, zorder=4)
    ax.text(x, y - .18, sub, ha="center", va="center", fontsize=5.8, fontfamily=SANS, color=MUTED, zorder=4)
for k, (x, y, lab, sub) in A.items():
    ax.add_patch(Polygon([(x - RA, y), (x, y + .37), (x + RA, y), (x, y - .37)], fc="white", ec=SLATE, lw=1.3,
                         joinstyle="round", zorder=3))
    ax.text(x, y + .06, lab, ha="center", va="center", fontsize=12, fontfamily=SERIF, fontweight="bold", color=SLATE, zorder=4)
    ax.text(x, y - .17, sub, ha="center", va="center", fontsize=6, fontfamily=SANS, color=MUTED, zorder=4)


def rad(k):
    if k in A:
        return .47 * ux
    return (RS + (.07 if k == "sT" else 0) + .02) * ux


def arrow(a, b, label=None, color=MUTED, lw=1.2, off=(0, 0), t=.5, rad_=0.0):
    ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=9, shrinkA=rad(a), shrinkB=rad(b) + 2,
                                 color=color, lw=lw, zorder=2, connectionstyle=f"arc3,rad={rad_}"))
    if label:
        p, q = np.array(P[a]), np.array(P[b]); x, y = p + (q - p) * t
        ax.text(x + off[0], y + off[1], label, ha="center", va="center", fontsize=7, fontfamily=SANS, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none"))


for a in ("a1", "a2", "a3"):
    arrow("s0", a, color=LINE, lw=1.4)
for a, s in [("a1", "s1"), ("a2", "s2"), ("a3", "s3")]:
    arrow(a, s, "P=1   R=w·φ(s,a)")
for s, b in [("s1", "b1"), ("s2", "b1"), ("s2", "b2"), ("s3", "b2")]:
    arrow(s, b, color=LINE, lw=1.4)
arrow("b1", "sT", "P=1   R=w·φ(s,a)", off=(0, .1))
arrow("b2", "sT", "P=1   R=w·φ(s,a)", off=(0, -.1))

# a few dots between columns: the pattern repeats until the destination
ax.text(5.95, 2.2, "· · ·", ha="center", va="center", fontsize=12, color=MUTED, fontfamily=SANS)

fig.savefig("midreview_assets/mdp_general.png", facecolor=PAPER)
