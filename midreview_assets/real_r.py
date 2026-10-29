import sys, pickle, numpy as np
sys.path.insert(0, '.')
from porto_data import load_porto_graph
from mdp import RouteMDP
from config import FEATURES
G = load_porto_graph(); m = RouteMDP(G)
w = np.load('../models/irl_weights_porto.npy')
R = m.T_feat @ w
d = pickle.load(open('../data/porto_trajectories.pkl','rb'))
print(type(d), list(d.keys())[:5] if isinstance(d, dict) else len(d))
tr = d['train'] if 'train' in d else d
print(type(tr), len(tr)); t = tr[0]; print(t.keys() if isinstance(t, dict) else t)
print("R range", R.min(), R.max(), np.percentile(R,[5,50,95]))
import networkx as nx
def desc(e):
    u,v,k = m.edges[e]; dd = G[u][v][k]
    return dd.get('highway'), round(float(dd.get('length',0))), dd.get('name')
best=None
for t in tr[:400]:
    es = t['edges']
    if len(es) < 6: continue
    ids = m.traj_edge_ids(t)
    alts=[]
    for a,b in zip(ids[:-1], ids[1:]):
        o=[x for x in m.t_dst[m.t_src==a] if x!=b]
        alts.append(len(o))
    score=sum(1 for x in alts if x>=1)
    if best is None or score>best[0]: best=(score,t,ids)
print(best[0]); t=best[1]; ids=best[2]
for a,b in zip(ids[:-1], ids[1:]):
    print("in",desc(a),"-> chosen",desc(b),"R=%.2f"%R[m.tid[(a,b)]])
    for x in m.t_dst[m.t_src==a]:
        if x!=b: print("      alt",desc(x),"R=%.2f"%R[m.tid[(a,int(x))]])
