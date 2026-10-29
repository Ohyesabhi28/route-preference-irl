import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
NAVY="#262626"; TEAL="#3E5C76"; ORG="#8A2432"; GREY="#8A94A6"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
out="midreview_assets/"
h=pickle.load(open("results/porto/irl_history.pkl","rb"))
it=[a for a,_,_ in h]; nll=[b for _,b,_ in h]
fig,ax=plt.subplots(figsize=(4.6,3.0),dpi=200)
ax.plot(it,nll,color=ORG,lw=2.2); ax.fill_between(it,nll,min(nll),color=ORG,alpha=.12)
ax.set_xlabel("L-BFGS iteration"); ax.set_ylabel("Negative log-likelihood")
ax.set_title("Training curve (Porto)",color=NAVY,fontsize=11,loc="left",fontweight="bold")
ax.annotate(f"{nll[0]:.1f}",(0,nll[0]),color=NAVY,xytext=(3,-2),textcoords="offset points",fontsize=9)
ax.annotate(f"{nll[-1]:.2f}",(it[-1],nll[-1]),color=NAVY,xytext=(-30,8),textcoords="offset points",fontsize=9)
fig.tight_layout(); fig.savefig(out+"curve.png"); plt.close()
w=np.load("models/irl_weights_porto.npy")
names=["length","speed","travel time","motorway/trunk","primary/secondary","residential","straight","slight turn","sharp turn / U","lanes"]
o=np.argsort(w); fig,ax=plt.subplots(figsize=(4.6,3.0),dpi=200)
ax.barh([names[i] for i in o],w[o],color=[ORG if w[i]<0 else TEAL for i in o])
ax.axvline(0,color=NAVY,lw=.8); ax.tick_params(labelsize=8)
ax.set_title("Learned reward weights",color=NAVY,fontsize=11,loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig(out+"weights.png"); plt.close()
