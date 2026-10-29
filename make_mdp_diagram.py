"""Regenerates mdp_diagram.svg (worked example MDP, MDP_FORMULATION.md section 7). Run from the project root."""
import math, re, json

# ---------------- learned v3 weights (single split) ----------------
W = dict(length=-0.2355, travel_time=0.1570, step=-0.6740, road_motorway=0.2298,
         road_primary=-0.1110, turn_slight=-0.5192, turn_sharp=-1.4644,
         num_lanes=-0.0084, same_road_name=-0.0419, away_from_goal=-0.2950,
         edge_familiarity=0.8723)


def reward(L, T, primary=0, slight=0, sharp=0, away=0, fam=0.0, same=0, lanes=1.0):
    return (W['step'] + W['length'] * L + W['travel_time'] * T + W['road_primary'] * primary
            + W['turn_slight'] * slight + W['turn_sharp'] * sharp + W['away_from_goal'] * away
            + W['edge_familiarity'] * fam + W['same_road_name'] * same + W['num_lanes'] * lanes)


# transitions: (state, action_id, label, next, reward)
T = [
    ('S1', 'A1', 'straight onto arterial', 'S2', reward(1.2, .15, primary=1, fam=.8, same=1)),
    ('S1', 'A2', 'slight turn, side street', 'S3', reward(.8, .20, slight=1, fam=.3)),
    ('S1', 'A3', 'U-turn', 'S4', reward(.8, .20, sharp=1, away=1, fam=.3)),
    ('S2', 'B1', 'straight to destination', 'GOAL', reward(1.5, .18, primary=1, fam=.9, same=1)),
    ('S2', 'B2', 'enter dead-end stub', 'DEAD', reward(.6, .10, slight=1, away=1, fam=.1)),
    ('S3', 'C1', 'straight (long, quiet)', 'GOAL', reward(2.0, .40, fam=.2, same=1)),
    ('S3', 'C2', 'slight turn, rejoin arterial', 'S2', reward(.7, .15, primary=1, slight=1, fam=.6)),
    ('S4', 'D1', 'U-turn again', 'S1', reward(.8, .20, sharp=1, fam=.3)),
]
R = {(a, n): r for (_, a, _, n, r) in T}
acts = {}
for s, a, lab, n, r in T:
    acts.setdefault(s, []).append((a, n, r))

V = {'GOAL': 0.0, 'S1': -1, 'S2': -1, 'S3': -1, 'S4': -1}
for _ in range(500):
    for s in ('S1', 'S2', 'S3', 'S4'):
        terms = []
        for a, n, r in acts[s]:
            if n == 'DEAD':
                continue            # V = -inf, exp(-inf) = 0
            terms.append(math.exp(r + V[n]))
        V[s] = math.log(sum(terms))
Q = {a: r + (V[n] if n != 'DEAD' else -math.inf) for s in acts for a, n, r in acts[s]}
PI = {}
for s in acts:
    for a, n, r in acts[s]:
        PI[a] = 0.0 if n == 'DEAD' else math.exp(Q[a] - V[s])
res = dict(V=V, rewards={a: round(R[(a, n)], 2) for (_, a, _, n, _) in T for _ in [0]}, PI=PI)
rw = {a: r for (_, a, _, n, r) in T}
print({k: round(v, 3) for k, v in V.items()})
print({k: round(v, 3) for k, v in rw.items()})
print({k: round(v, 3) for k, v in PI.items()})

# ---------------- SVG ----------------
POS = {
    'S1': (80, 250), 'S2': (420, 110), 'S3': (420, 250), 'S4': (420, 410),
    'A1': (250, 110), 'A2': (250, 250), 'A3': (250, 410),
    'B2': (600, 45), 'B1': (600, 135), 'C2': (600, 215), 'C1': (600, 305), 'D1': (600, 440),
    'GOAL': (800, 210), 'DEAD': (800, 55),
}
COL = dict(state='#2563eb', act='#4f6fd8', goal='#15803d', dead='#b91c1c', line='#94a3b8', txt='#1e293b')


def f2(x):
    return f'{x:+.2f}'.replace('+', '+').replace('-', '−')


def path(p, q, bend=0, shrink_p=0, shrink_q=0):
    (x1, y1), (x2, y2) = p, q
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    x1, y1 = x1 + ux * shrink_p, y1 + uy * shrink_p
    x2, y2 = x2 - ux * shrink_q, y2 - uy * shrink_q
    mx, my = (x1 + x2) / 2 - uy * bend, (y1 + y2) / 2 + ux * bend
    return f'M {x1:.1f} {y1:.1f} Q {mx:.1f} {my:.1f} {x2:.1f} {y2:.1f}', (0.25 * x1 + 0.5 * mx + 0.25 * x2, 0.25 * y1 + 0.5 * my + 0.25 * y2)


out = []


def label(x, y, lines, col=COL['txt'], bg=True):
    w = max(len(l) for l in lines) * 5.6 + 8
    h = 12 * len(lines) + 4
    if bg:
        out.append(f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" height="{h}" rx="3" fill="#ffffff" opacity="0.92"/>')
    for i, l in enumerate(lines):
        out.append(f'<text x="{x:.1f}" y="{y-h/2+11+12*i:.1f}" text-anchor="middle" fill="{col}" font-size="10">{l}</text>')


def edge(p, q, labs, col, bend=0, sp=0, sq=0, dash=False, lx=0, ly=0, w=1.6):
    d, mid = path(p, q, bend, sp, sq)
    da = ' stroke-dasharray="5 4"' if dash else ''
    out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}"{da} marker-end="url(#ar)"/>')
    if labs:
        label(mid[0] + lx, mid[1] + ly, labs)


# defs
defs = '<defs><marker id="ar" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0,9 3.5,0 7" fill="#334155"/></marker></defs>'

# state -> action arrows (policy) and action -> next-state arrows (P=1, R)
names = {'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S4': 'S4'}
state_of = {a: s for s, a, _, _, _ in T}
for s, a, lab, n, r in T:
    pa = PI[a]
    plab = [f'π = {pa:.2f}'] if pa > 0 else ['π = 0  (never)']
    bend = {'A1': -30, 'A2': 0, 'A3': 30, 'B2': -10, 'B1': 10, 'C2': -10, 'C1': 10, 'D1': 0}.get(a, 0)
    edge(POS[s], POS[a], plab, '#475569', bend=bend, sp=30, sq=22, lx=0, ly=-2 if a != 'A2' else -10)
    ncol = COL['goal'] if n == 'GOAL' else COL['dead'] if n == 'DEAD' else '#64748b'
    lines = ['P = 1.00', f'R = {f2(r)}']
    if a == 'D1':
        out.append('<path d="M 575 452 C 470 535, 190 535, 92 282" fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#ar)"/>')
        label(330, 508, lines)
    elif a == 'C2':
        edge(POS[a], POS['S2'], lines, '#94a3b8', bend=-25, sp=22, sq=30, lx=0, ly=-4)
    else:
        sq = 48 if n in ('GOAL', 'DEAD') else 30
        edge(POS[a], POS[n], lines, ncol, bend=0, sp=22, sq=sq, dash=(n == 'DEAD'), lx=0, ly=-2, w=2.4 if n == 'GOAL' else 1.6)

# nodes
for s in ('S1', 'S2', 'S3', 'S4'):
    x, y = POS[s]
    out.append(f'<circle cx="{x}" cy="{y}" r="30" fill="#ffffff" stroke="{COL["state"]}" stroke-width="2.4"/>')
    out.append(f'<text x="{x}" y="{y-2}" text-anchor="middle" fill="#0f172a" font-size="13" font-weight="800">{s}</text>')
    out.append(f'<text x="{x}" y="{y+12}" text-anchor="middle" fill="#475569" font-size="9">V={V[s]:.2f}</text>')
for a, (x, y) in POS.items():
    if a[0] in 'ABCD' and a not in ('GOAL', 'DEAD'):
        s_, _, lab_, n_, _ = [t for t in T if t[1] == a][0]
        out.append(f'<polygon points="{x},{y-22} {x+30},{y} {x},{y+22} {x-30},{y}" fill="#e0e7ff" stroke="#4f6fd8" stroke-width="2"/>')
        out.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" fill="#1e1b4b" font-size="12" font-weight="800">{a}</text>')
for t, (x, y), c, ttl, sub in (('GOAL', POS['GOAL'], COL['goal'], 'GOAL', 'V = 0'),
                               ('DEAD', POS['DEAD'], COL['dead'], 'DEAD END', 'V = −∞')):
    for r_, w_ in ((44, 3), (37, 1.5)):
        out.append(f'<circle cx="{x}" cy="{y}" r="{r_}" fill="{("#dcfce7" if t=="GOAL" else "#fee2e2") if r_==37 else "none"}" stroke="{c}" stroke-width="{w_}"/>')
    out.append(f'<text x="{x}" y="{y-2}" text-anchor="middle" fill="{c}" font-size="12" font-weight="800">{ttl}</text>')
    out.append(f'<text x="{x}" y="{y+11}" text-anchor="middle" fill="{c}" font-size="8">{sub}</text>')

# action captions under diamonds
cap = {'A1': 'straight → arterial', 'A2': 'slight turn → side street', 'A3': 'U-turn',
       'B1': 'straight to dest.', 'B2': 'enter dead-end stub', 'C1': 'straight, long & quiet',
       'C2': 'slight turn, rejoin', 'D1': 'U-turn again'}
for a, c_ in cap.items():
    x, y = POS[a]
    out.append(f'<text x="{x}" y="{y+37}" text-anchor="middle" fill="#4f46e5" font-size="9" font-style="italic">{c_}</text>')

# legend
lg = ['<rect x="14" y="14" width="205" height="106" rx="8" fill="#ffffff" stroke="#cbd5e1"/>']
lg.append('<circle cx="32" cy="34" r="9" fill="#ffffff" stroke="#2563eb" stroke-width="2"/><text x="48" y="38" fill="#1e293b" font-size="10">state = incoming road edge</text>')
lg.append('<polygon points="32,50 42,60 32,70 22,60" fill="#e0e7ff" stroke="#4f6fd8" stroke-width="1.8"/><text x="48" y="64" fill="#1e293b" font-size="10">action = choose outgoing road</text>')
lg.append('<circle cx="32" cy="86" r="9" fill="#dcfce7" stroke="#15803d" stroke-width="2.4"/><text x="48" y="90" fill="#1e293b" font-size="10">positive terminal (destination)</text>')
lg.append('<circle cx="32" cy="106" r="9" fill="#fee2e2" stroke="#b91c1c" stroke-width="2.4"/><text x="48" y="110" fill="#1e293b" font-size="10">negative terminal (dead end)</text>')
out.extend(lg)
out.append('<text x="440" y="548" text-anchor="middle" fill="#475569" font-size="10">Rewards R = wᵀφ from the learned v3 weights · V, π from soft value iteration (γ = 1) · all transitions deterministic (P = 1)</text>')

svg = (f'<svg viewBox="0 0 880 560" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#f8fafc;'
       f'border-radius:14px;border:1px solid #cbd5e1;font-family:Inter,Arial,sans-serif">{defs}' + ''.join(out) + '</svg>')
open(r'C:\Desktop\RL_1\mdp_diagram.svg', 'w', encoding='utf8').write(svg.replace('<svg ', '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ', 1).replace('style="width:100%;', 'width="1320" height="840" style="').replace('font-family:Inter,Arial,sans-serif', 'font-family:Arial,sans-serif'))
print('ok')
