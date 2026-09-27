import json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import *
g=json.load(open(MAP_JSON,encoding='utf-8'))
ST=pd.DataFrame(g['stops_empirical']); print(ST.type.value_counts().to_dict()); print(ST.sort_values('p_stop',ascending=False).head(30).to_string())
S=pd.read_parquet(ROOT/'map'/'pathgraph_samples.parquet'); tr=cKDTree(S[['x','y']].values)
R=6378137
def xy(lat,lon): return np.radians(lon-37.419565)*R*np.cos(np.radians(55.804767)), np.radians(lat-55.804767)*R
sp=splits(); rows=[]
for split in ('train','val'):
  for b in sp[split]:
    run=load_run(b); fx=run['master_fix']; fx=fx[fx.status==2].sort_values('t_hdr')
    if len(fx)<500: continue
    w=pd.concat([run['front'],run['rear']]).sort_values('t_hdr'); tw=w.t_hdr.values/1e9; vw=w.velocity.values/3.6
    st=vw<0.05; # stop segments
    edges=np.flatnonzero(np.diff(np.r_[0,st.astype(int),0]))
    for a0,a1 in zip(edges[::2],edges[1::2]):
        if tw[a1-1]-tw[a0]<4: continue
        tm=0.5*(tw[a0]+tw[a1-1]); i=np.searchsorted(fx.t_hdr.values/1e9,tm)
        if i<=0 or i>=len(fx): continue
        x,y=xy(fx.lat.values[i],fx.lon.values[i]); d,jj=tr.query([x,y],k=6)
        # heading: pre-stop motion
        j=np.searchsorted(fx.t_hdr.values/1e9,tw[a0]-3)
        x0,y0=xy(fx.lat.values[j],fx.lon.values[j]); hd=np.arctan2(y-y0,x-x0)
        best=None
        for dd,k in zip(d,jj):
            if dd<4 and abs((S.hdg.values[k]-hd+np.pi)%(2*np.pi)-np.pi)<0.8: best=k;break
        if best is None: continue
        e=S.edge.values[best]; s=S.s.values[best]
        cand=ST[ST.edge==e]
        if not len(cand): continue
        k=np.argmin(np.abs(cand.s.values-s)); off=s-cand.s.values[k]
        rows.append(dict(split=split,bag=b,edge=e,s=s,stop=cand.index[k],off=off,type=cand.type.values[k],p=cand.p_stop.values[k],dur=tw[a1-1]-tw[a0]))
D=pd.DataFrame(rows); D.to_csv(ROOT/'reports'/'stop_positions.csv',index=False)
M=D[D.off.abs()<30]
print(M.groupby(['stop','type']).agg(n=('off','size'),p=('p','first'),med=('off','median'),std=('off','std'),iqr=('off',lambda x:x.quantile(.75)-x.quantile(.25)),dur=('dur','median')).round(2).sort_values('n',ascending=False).head(40).to_string())
print('all station offs std', M[M.type=='station'].off.std(), 'MAD', (M[M.type=='station'].off-M[M.type=='station'].groupby('stop').off.transform('median')).abs().median())
