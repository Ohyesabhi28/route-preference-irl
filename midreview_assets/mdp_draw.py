import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, Polygon, FancyArrowPatch

INK, MUTED, ACC, SLATE = "#262626", "#6B6B6B", "#8A2432", "#3E5C76"
PAPER, BAND, LINE = "#FAF8F4", "#F1EDE5", "#CFC8BB"
names = {f.name for f in fm.fontManager.ttflist}
SERIF = "Georgia" if "Georgia" in names else "DejaVu Serif"
SANS = "Calibri" if "Calibri" in names else "DejaVu Sans"

fig = plt.figure(figsize=(9.0, 4.2), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(-0.55, 4.2)
ax.axis("off")
fig.patch.set_alpha(0)

S = {"S1": (.75, 2.05), "S2": (4.6, 2.85), "S3": (4.6, .55), "S4": (9.0, 3.25), "S5": (9.0, 1.2)}
A = {"A1": (2.6, 3.55, "Wait"), "A2": (2.6, 2.05, "Drive ahead"), "A3": (2.6, .55, "Refuel"),
     "B1": (6.8, 3.7, "Arterial"), "B2": (6.8, 2.45, "Shortcut"), "B3": (6.8, 1.35, "Bypass"),
     "C1": (6.8, .4, "Resume")}
P = {k: (v[0], v[1]) for k, v in A.items()}
P.update(S)
R_S, R_A_W, R_A_H = .36, .44, .26
terminal = {"S4": SLATE, "S5": ACC}
NP = {}

for k, (x, y) in S.items():
    if k in terminal:
        NP[k] = Circle((x, y), R_S + .07, fc="white", ec=terminal[k], lw=1.6, zorder=3)
        ax.add_patch(NP[k])
        ax.add_patch(Circle((x, y), R_S - .01, fc="white", ec=terminal[k], lw=1.1, zorder=3))
        col = terminal[k]
    else:
        NP[k] = Circle((x, y), R_S, fc=BAND, ec=INK, lw=1.3, zorder=3)
        ax.add_patch(NP[k])
        col = INK
    ax.text(x, y + .06, k, ha="center", va="center", fontsize=11, fontfamily=SERIF, fontweight="bold", color=col, zorder=4)
lab = {"S1": "Origin", "S2": "Junction", "S3": "Rest stop", "S4": "Goal", "S5": "Blocked"}
for k, (x, y) in S.items():
    ax.text(x, y - .17, lab[k], ha="center", va="center", fontsize=6.8, fontfamily=SANS, color=MUTED, zorder=4)
for k, (x, y, name) in A.items():
    NP[k] = Polygon([(x - R_A_W, y), (x, y + R_A_H + .08), (x + R_A_W, y), (x, y - R_A_H - .08)],
                    fc="white", ec=SLATE, lw=1.3, joinstyle="round", zorder=3)
    ax.add_patch(NP[k])
    ax.text(x, y + .07, k, ha="center", va="center", fontsize=10.5, fontfamily=SERIF, fontweight="bold", color=SLATE, zorder=4)
    ax.text(x, y - .13, name, ha="center", va="center", fontsize=6.8, fontfamily=SANS, color=MUTED, zorder=4)


def edge_pts(a, b, rad=0):
    return P[a], P[b]


def trim(p, q, ka, kb):
    """shrink the segment so arrows start/end on node borders"""
    import numpy as np
    p, q = np.array(p), np.array(q)
    d = q - p
    L = np.hypot(*d)
    u = d / L
    ra = (R_S + (.07 if ka in terminal else 0)) if ka in S else .5
    rb = (R_S + (.07 if kb in terminal else 0)) if kb in S else .5
    if ka in A:
        ra = .42
    if kb in A:
        rb = .42
    return tuple(p + u * (ra + .02)), tuple(q - u * (rb + .04))


def rad_pt(k):
    u = 64.8
    if k in A:
        return 0.40 * u
    return (R_S + (.07 if k in terminal else 0) + .02) * u


def arrow(a, b, rad=0.0, label=None, lpos=.5, color=MUTED, lw=1.2, loff=(0, 0), dash=False):
    p, q = P[a], P[b]
    ar = FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=9,
                         shrinkA=rad_pt(a), shrinkB=rad_pt(b) + 2,
                         color=color, lw=lw, zorder=2, linestyle=(0, (4, 3)) if dash else "-")
    ax.add_patch(ar)
    if label:
        import numpy as np
        p0, p2 = np.array(P[a]), np.array(P[b])
        d = p2 - p0
        c = (p0 + p2) / 2 + rad * np.array([d[1], -d[0]])
        t = lpos
        pt = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2
        x, y = pt
        ax.text(x + loff[0], y + loff[1], label, ha="center", va="center", fontsize=7, fontfamily=SANS, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none"))


# choose action
for a in ("A1", "A2", "A3"):
    arrow("S1", a, .0, color=LINE, lw=1.3)
for b in ("B1", "B2", "B3"):
    arrow("S2", b, .0, color=LINE, lw=1.3)
arrow("S3", "C1", 0, color=LINE, lw=1.3)
# outcomes
arrow("A1", "S2", -.15, "0.65 · −1")
arrow("A1", "S1", -.25, "0.35 · −1", .45)
arrow("A2", "S2", -.1, "0.90 · −2")
arrow("A2", "S1", -.45, "0.10 · −2", .5, loff=(0, -.12))
arrow("A3", "S3", 0, "1.00 · −3")
arrow("C1", "S1", -.4, "0.95 · −2", .5)
arrow("C1", "S3", .45, "0.05 · −2", .5)
arrow("B1", "S4", -.1, "0.65 · +15")
arrow("B1", "S2", -.35, "0.35 · −1", .5, loff=(0, .14))
arrow("B2", "S4", .0, "0.55 · +12", .5)
arrow("B2", "S2", -.3, "0.30 · −2", .5, loff=(0, -.08))
arrow("B2", "S5", -.0, "0.15 · −5", .55, color=ACC)
arrow("B3", "S4", .0, "0.60 · +10", .45, loff=(.0, 0))
arrow("B3", "S5", .15, "0.40 · −10", .5, color=ACC)

fig.savefig("midreview_assets/mdp_redraw.png", transparent=True)
print(SERIF, SANS)
