"""Проверка осуществимости физмодели (запускалось из корня проекта). Восстановлено из сессии."""
import pandas as pd, numpy as np, hashlib, json
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
EX=Path('extracted'); s=pd.read_csv(EX/'summary.csv'); R=6378137; g0=9.81
S=pd.read_parquet('map/pathgraph_samples.parquet'); LAT0,LON0=55.804767,37.419565
tree=cKDTree(S[['x','y']].to_numpy()); SH=S.hdg.to_numpy(); SG=S.grade.to_numpy()
seen=set(); rows=[]
for b in s[s.n_master_fix>0].bag:
    fr=pd.read_parquet(EX/b/'front.parquet').sort_values('t_hdr')
    h=hashlib.md5(fr.velocity.round(4).to_numpy().tobytes()).hexdigest()
    if h in seen or len(fr)<600: continue
    seen.add(h)
    cmd=pd.read_parquet(EX/b/'cmd.parquet').sort_values('t_hdr')
    fx=pd.read_parquet(EX/b/'master_fix.parquet').sort_values('t_hdr'); fx=fx[fx.status==2]
    vel=pd.read_parquet(EX/b/'master_vel.parquet').sort_values('t_hdr')
    t=fr.t_hdr.to_numpy()/1e9; tg=np.arange(t[0],t[-1],0.1)
    vg=np.interp(tg,t,fr.velocity.to_numpy()/3.6); a=np.gradient(gaussian_filter1d(vg,5),0.1)
    ug=np.round(np.interp(tg,cmd.t_hdr.to_numpy()/1e9,cmd.position.to_numpy()))
    if len(fx)<100: continue
    la=np.interp(tg,fx.t_hdr/1e9,fx.lat); lo=np.interp(tg,fx.t_hdr/1e9,fx.lon)
    x=np.radians(lo-LON0)*R*np.cos(np.radians(LAT0)); y=np.radians(la-LAT0)*R
    hv=np.arctan2(np.interp(tg,vel.t_hdr/1e9,vel.vy),np.interp(tg,vel.t_hdr/1e9,vel.vx))
    d,j=tree.query(np.c_[x,y])
    sgn=np.where(np.cos(hv-SH[j])>0,1,-1)       # уклон по ходу движения
    gr=SG[j]*sgn; gr[d>4]=np.nan
    inr=(tg>fx.t_hdr.iloc[0]/1e9)&(tg<fx.t_hdr.iloc[-1]/1e9)
    rows.append(pd.DataFrame({'bag':b,'v':vg,'a':a,'u':ug,'gr':np.where(inr,gr,np.nan)}))
D=pd.concat(rows).dropna(); D=D[D.v>1]
for name,q in [('выбег u=0',D[D.u==0]),('тяга u>0',D[D.u>0]),('торможение u<0',D[D.u<0])]:
    print(f'{name}: corr(a, уклон) = {np.corrcoef(q.a,q.gr)[0,1]:+.2f}; наклон a по уклону = {np.polyfit(q.gr,q.a,1)[0]:+.2f} (физика: −9.81)')
q=D[D.u==0]; q=q.assign(gb=pd.cut(q.gr*100,[-5,-3,-1.5,-0.5,0.5,1.5,3,5]))
print('\nвыбег: медиана a по уклону (%):'); print(q.groupby('gb',observed=True).a.agg(['median','size']).round(3).to_string())
X=np.c_[np.ones(len(q)),q.v,q.v**2]; y=-(q.a+g0*q.gr); c,*_=np.linalg.lstsq(X,y,rcond=None)
r2=1-((y-X@c)**2).sum()/((q.a-q.a.mean())**2).sum()
print(f'\nДэвис по выбегу с уклоном из pathgraph: w0 = {c[0]/g0*1000:.2f} + {c[1]/g0*1000:.3f}·v + {c[2]/g0*1000:.4f}·v² Н/кН; R²(a)={r2:.2f}')
# позиция −8
m8=D[D.u==-8]; print(f'\nu=−8: n={len(m8)}, скорость med {m8.v.median():.1f} м/с, a med {m8.a.median():.2f}; доля с v<3: {(m8.v<3).mean():.2f}')
