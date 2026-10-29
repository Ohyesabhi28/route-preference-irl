import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon, Arc, FancyArrowPatch

ACC, SLATE, MUTED = "#8A2432", "#3E5C76", "#6B6B6B"
LW = 3.2


def canvas():
    fig = plt.figure(figsize=(1, 1), dpi=200)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


def save(fig, name):
    fig.savefig(f"midreview_assets/icons/{name}.png", transparent=True)
    plt.close(fig)


def line(ax, xs, ys, c, lw=LW, **k):
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="round", solid_joinstyle="round", **k)


def graph(c):
    fig, ax = canvas()
    pts = [(.15, .2), (.5, .45), (.85, .2), (.3, .85), (.8, .8)]
    for a, b in [(0, 1), (1, 2), (1, 3), (1, 4), (3, 4)]:
        line(ax, [pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], c, 2.4)
    for x, y in pts:
        ax.add_patch(Circle((x, y), .09, fc="white", ec=c, lw=LW))
    save(fig, "graph")


def pin(c):
    fig, ax = canvas()
    t = np.linspace(0, 2 * np.pi, 100)
    ax.add_patch(Polygon([(.5, .08), (.2, .5), (.8, .5)], fc="white", ec=c, lw=LW, joinstyle="round"))
    ax.add_patch(Circle((.5, .6), .27, fc="white", ec=c, lw=LW))
    ax.add_patch(Circle((.5, .6), .09, fc=c, ec=c))
    ax.add_patch(Polygon([(.3, .5), (.7, .5), (.5, .1)], fc="white", ec="none"))
    line(ax, [.26, .5, .74], [.46, .08, .46], c)
    save(fig, "pin")


def snap(c):
    fig, ax = canvas()
    line(ax, [.08, .35, .6, .92], [.3, .3, .72, .72], c, 3.4)
    for x, y in [(.2, .42), (.45, .22), (.62, .84), (.82, .62), (.35, .38)]:
        ax.add_patch(Circle((x, y), .05, fc=MUTED, ec="none"))
    save(fig, "snap")


def route(c):
    fig, ax = canvas()
    ax.add_patch(Circle((.18, .2), .09, fc="white", ec=c, lw=LW))
    line(ax, [.18, .18, .6, .6, .8], [.3, .55, .55, .8, .8], c, 3.4)
    line(ax, [.8, .8], [.62, .95], c, 2.6)
    ax.add_patch(Polygon([(.8, .95), (.8, .78), (.96, .865)], fc=c, ec=c, lw=1))
    save(fig, "route")


def dice(c):
    fig, ax = canvas()
    ax.add_patch(FancyBboxPatch((.15, .15), .7, .7, boxstyle="round,pad=0,rounding_size=.12", fc="white", ec=c, lw=LW))
    for x, y in [(.33, .67), (.5, .5), (.67, .33), (.33, .33), (.67, .67)]:
        ax.add_patch(Circle((x, y), .06, fc=c, ec="none"))
    save(fig, "dice")


def target(c):
    fig, ax = canvas()
    for r in (.4, .24):
        ax.add_patch(Circle((.5, .5), r, fc="white", ec=c, lw=LW))
    ax.add_patch(Circle((.5, .5), .08, fc=c, ec=c))
    save(fig, "target")


def layers(c):
    fig, ax = canvas()
    for i, y in enumerate([.2, .42, .64]):
        ax.add_patch(Polygon([(.5, y + .22), (.9, y + .08), (.5, y - .06), (.1, y + .08)], fc="white", ec=c, lw=LW, joinstyle="round"))
    save(fig, "layers")


def bars(c):
    fig, ax = canvas()
    for x, h in [(.15, .35), (.4, .65), (.65, .5)]:
        ax.add_patch(Rectangle((x, .12), .2, h, fc="white", ec=c, lw=LW, joinstyle="round"))
    line(ax, [.08, .92], [.12, .12], c)
    save(fig, "bars")


def gradient(c):
    fig, ax = canvas()
    line(ax, [.12, .88], [.2, .85], c, 2.2, ls=(0, (2, 2)))
    line(ax, [.12, .12, .88], [.9, .2, .2], MUTED, 2.4)
    ax.add_patch(FancyArrowPatch((.35, .73), (.7, .42), arrowstyle="-|>", mutation_scale=22, color=c, lw=LW))
    save(fig, "gradient")


def cycle(c):
    fig, ax = canvas()
    for a0, a1 in [(30, 150), (210, 330)]:
        ax.add_patch(Arc((.5, .5), .66, .66, theta1=a0, theta2=a1, color=c, lw=LW))
    for (x, y, d) in [(.22, .66, -1), (.78, .34, 1)]:
        ax.add_patch(Polygon([(x - .09, y), (x + .09, y), (x, y + .13 * (-d))], fc=c, ec=c))
    save(fig, "cycle")


def monitor(c):
    fig, ax = canvas()
    ax.add_patch(FancyBboxPatch((.08, .3), .84, .56, boxstyle="round,pad=0,rounding_size=.05", fc="white", ec=c, lw=LW))
    line(ax, [.5, .5], [.3, .16], c); line(ax, [.3, .7], [.14, .14], c)
    ax.add_patch(Circle((.5, .6), .09, fc="white", ec=c, lw=2.4))
    line(ax, [.2, .38, .5, .65, .8], [.42, .5, .45, .58, .5], MUTED, 2)
    save(fig, "monitor")


def code(c):
    fig, ax = canvas()
    line(ax, [.34, .1, .34], [.78, .5, .22], c, 3.6)
    line(ax, [.66, .9, .66], [.78, .5, .22], c, 3.6)
    line(ax, [.56, .44], [.8, .2], MUTED, 3)
    save(fig, "code")


def fork(c):
    fig, ax = canvas()
    line(ax, [.5, .5], [.1, .45], c, 3.4)
    line(ax, [.5, .22, .22], [.45, .6, .85], c, 3.4)
    line(ax, [.5, .78, .78], [.45, .6, .85], MUTED, 3.4, ls=(0, (2, 1.5)))
    for x, y in [(.22, .88), (.78, .88)]:
        ax.add_patch(Circle((x, y), .07, fc="white", ec=c, lw=2.6))
    save(fig, "fork")


def check(c):
    fig, ax = canvas()
    ax.add_patch(FancyBboxPatch((.12, .1), .76, .8, boxstyle="round,pad=0,rounding_size=.08", fc="white", ec=c, lw=LW))
    line(ax, [.28, .45, .74], [.5, .34, .7], c, 4)
    line(ax, [.26, .74], [.2, .2], MUTED, 2.4)
    save(fig, "check")


import os
os.makedirs("midreview_assets/icons", exist_ok=True)
for f, col in [(graph, ACC), (pin, ACC), (snap, ACC), (route, ACC), (dice, MUTED)]:
    f(col)
for f in (target, layers, bars, gradient, cycle):
    f(ACC)
for f in (monitor, code, fork, check):
    f(SLATE)
# muted variants for band 1 (grey-ink look)
INK = "#262626"
for f, n in [(graph, "graph"), (pin, "pin"), (snap, "snap"), (route, "route")]:
    f(INK)
    os.replace("midreview_assets/icons/%s.png" % n, "midreview_assets/icons/%s_ink.png" % n)
    f(ACC)
for f in (graph, pin, snap, route):
    pass


def db(c):
    fig, ax = canvas()
    from matplotlib.patches import Ellipse
    ax.add_patch(Rectangle((.18, .2), .64, .6, fc="white", ec="none"))
    line(ax, [.18, .18], [.2, .8], c); line(ax, [.82, .82], [.2, .8], c)
    ax.add_patch(Ellipse((.5, .8), .64, .22, fc="white", ec=c, lw=LW))
    for y in (.5, .2):
        ax.add_patch(Arc((.5, y), .64, .22, theta1=180, theta2=360, color=c, lw=LW))
    save(fig, "db_%s" % ("ink" if c == INK else "acc"))


db(INK)
db(ACC)


import shutil
for f_,n in [(layers,"layers_ink"),(bars,"bars_ink")]:
    f_(INK); shutil.move("midreview_assets/icons/%s.png" % n.split("_")[0], "midreview_assets/icons/%s.png" % n)
layers(ACC); bars(ACC)
graph(SLATE); shutil.move("midreview_assets/icons/graph.png","midreview_assets/icons/graph_s.png")
graph(ACC)
