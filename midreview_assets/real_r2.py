import sys, pickle, numpy as np
sys.path.insert(0,'.')
from porto_data import load_porto_graph
from mdp import RouteMDP
from config import FEATURES
G=load_porto_graph(); m=RouteMDP(G); w=np.load('../models/irl_weights_porto.npy')
d=pickle.load(open('../data/porto_trajectories.pkl','rb'))
def name(e):
    u,v,k=m.edges[e]; return G[u][v][k].get('name')
want={'A':("Rua de An","tertiary",82),}
# locate S1: in-edge tertiary 82m on Rua de Anibal Cunha followed by 250m chosen
found=None
for t in d['train']:
    ids=m.traj_edge_ids(t)
    for a,b in zip(ids[:-1],ids[1:]):
        ua,va,ka=m.edges[a]; ub,vb,kb=m.edges[b]
        if round(G[ua][va][ka]['length'])==82 and round(G[ub][vb][kb]['length'])==250 and G[ua][va][ka].get('highway')=='tertiary':
            found=(t,ids,a,b); break
    if found: break
t,ids,a,b=found
i=ids.index(b); c=ids[i+1]
def show(s,e):
    j=m.tid[(s,e)]; f=m.T_feat[j]; 
    u,v,k=m.edges[e]; dd=G[u][v][k]
    print(dd.get('highway'),round(dd['length']),dd.get('name'),"R=%.2f"%(f@w))
    for n,fv,wv in zip(FEATURES,f,w):
        if abs(fv)>1e-9: print("   %-17s phi=%.3f w=%+.3f -> %+.3f"%(n,fv,wv,fv*wv))
print("S1 in:",name(a))
print("-- at S1");
for e in m.t_dst[m.t_src==a]: show(a,int(e)); print("  chosen" if e==b else "")
print("-- at S2 (after chosen)")
for e in m.t_dst[m.t_src==b]: show(b,int(e)); print("  chosen" if e==c else "")
