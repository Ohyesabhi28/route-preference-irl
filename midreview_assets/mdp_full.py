import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager as fm

INK, MUTED, ACC, SLATE = "#262626", "#6B6B6B", "#8A2432", "#3E5C76"
PAPER, BAND, LINE, WHITE = "#FAF8F4", "#F1EDE5", "#CFC8BB", "#FFFFFF"
names = {f.name for f in fm.fontManager.ttflist}
SERIF = "Georgia" if "Georgia" in names else "DejaVu Serif"
SANS = "Calibri" if "Calibri" in names else "DejaVu Sans"

fig = plt.figure(figsize=(10.0, 5.6), dpi=220)
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
ux = 72
RS, RA = .36, .5


def rbox(x, y, w, h, fc=WHITE, ec=LINE, lw=1.1, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.08", fc=fc, ec=ec, lw=lw, zorder=z))


def label(x, y, s, size=7, color=INK, weight="normal", fam=None, ha="center", va="center", z=5, bg=False):
    kw = dict(bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none")) if bg else {}
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, fontweight=weight,
            fontfamily=fam or SANS, zorder=z, **kw)


# ---------------- band 1: the MDP ----------------
rbox(.3, 2.75, 9.4, 2.65, fc=BAND, ec=LINE)
label(.5, 5.2, "Environment: Porto road graph", 9, ACC, "bold", SERIF, ha="left")
S = {"s0": (1.0, 3.95, "s$_t$", "junction + road"),
     "s1": (5.0, 3.95, "s$_{t+1}$", "junction + road"),
     "sT": (9.0, 3.95, "s$_T$", "destination")}
A = {"a1": (3.0, 4.6, "a$_1$", "road 1"), "a2": (3.0, 3.95, "a$_2$", "road 2"), "a3": (3.0, 3.3, "a$_k$", "road k")}
P = {k: v[:2] for k, v in {**S, **A}.items()}
for k, (x, y, lab, sub) in S.items():
    term = k == "sT"
    col = SLATE if term else INK
    ax.add_patch(Circle((x, y), RS + (.07 if term else 0), fc=WHITE, ec=col, lw=1.6 if term else 1.3, zorder=3))
    if term:
        ax.add_patch(Circle((x, y), RS - .01, fc=WHITE, ec=col, lw=1.1, zorder=3))
    label(x, y + .06, lab, 12, col, "bold", SERIF, z=4)
    label(x, y - .17, sub, 5.6, MUTED, z=4)
for k, (x, y, lab, sub) in A.items():
    ax.add_patch(Polygon([(x - RA, y), (x, y + .27), (x + RA, y), (x, y - .27)], fc=WHITE, ec=SLATE, lw=1.3,
                         joinstyle="round", zorder=3))
    label(x, y + .06, lab, 11, SLATE, "bold", SERIF, z=4)
    label(x, y - .11, sub, 5.4, MUTED, z=4)


def rad(k):
    if k in A:
        return .45 * ux
    return (RS + (.07 if k == "sT" else 0) + .02) * ux


def arrow(a, b, lab=None, color=MUTED, lw=1.2, off=(0, 0), t=.5):
    ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=9, shrinkA=rad(a), shrinkB=rad(b) + 2,
                                 color=color, lw=lw, zorder=2))
    if lab:
        p, q = np.array(P[a]), np.array(P[b]); x, y = p + (q - p) * t
        label(x + off[0], y + off[1], lab, 7, INK, bg=True)


for a in A:
    arrow("s0", a, color=LINE, lw=1.4)
arrow("a1", "s1", "P=1  R=r$_t$", off=(.2, .1), t=.62)
arrow("a2", "s1", "P=1  R=r$_t$", t=.64)
arrow("a3", "s1", "P=1  R=r$_t$", off=(.2, -.1), t=.62)
arrow("s1", "sT", "repeat every junction", color=MUTED)

label(5.0, 2.95, "episode = one trip  ·  destination absorbing  ·  horizon H = 80, no discounting  ·  "
      "policy π(a|s) = softmax Q(s,a)", 7, MUTED)

# ---------------- band 2: the reward ----------------
rbox(.3, .25, 9.4, 2.3, fc=BAND, ec=LINE)
label(.5, 2.35, "Reward function", 9, ACC, "bold", SERIF, ha="left")

# phi group
rbox(.5, .5, 4.1, 1.55, fc=WHITE, ec=SLATE, lw=1.2)
label(.65, 1.88, "φ(s,a)  ·  10 road features", 8, SLATE, "bold", ha="left")
cols = [("Road", ["length", "speed", "travel time", "road class (3)", "lanes"]),
        ("Turn", ["straight  < 30°", "slight  30–90°", "sharp / U-turn  > 90°"])]
x0 = .7
for title, items in cols:
    label(x0, 1.62, title, 7.5, ACC, "bold", SERIF, ha="left")
    for i, it in enumerate(items):
        label(x0, 1.38 - i * .2, it, 7, INK, ha="left")
    x0 += 1.75

# w
rbox(5.2, .8, 1.9, 0.95, fc=WHITE, ec=ACC, lw=1.4)
label(6.15, 1.5, "w", 16, ACC, "bold", SERIF)
label(6.15, 1.14, "learned by MaxEnt IRL", 6.5, MUTED)
label(6.15, .95, "from real taxi trips", 6.5, MUTED)
# R
rbox(7.7, .8, 1.8, 0.95, fc=WHITE, ec=INK, lw=1.4)
label(8.6, 1.42, "R(s,a)", 13, INK, "bold", SERIF)
label(8.6, 1.08, "= wᵀ φ(s,a)", 8.5, INK)
label(8.6, .9, "< 0 on every road", 6.5, MUTED)

for (x1, x2, sym) in [(4.6, 5.2, "·"), (7.1, 7.7, "=")]:
    ax.add_patch(FancyArrowPatch((x1 + .02, 1.27), (x2 - .02, 1.27), arrowstyle="-|>", mutation_scale=9,
                                 color=MUTED, lw=1.2, zorder=2))
    label((x1 + x2) / 2, 1.42, sym, 14 if sym == "·" else 10, MUTED)

# reward -> edges
ax.add_patch(FancyArrowPatch((8.6, 1.78), (8.6, 2.72), arrowstyle="-|>", mutation_scale=9, color=ACC, lw=1.3, zorder=2))
label(9.12, 2.25, "R on every\nedge", 6.5, ACC)

fig.savefig("midreview_assets/mdp_full.png", facecolor=PAPER)
