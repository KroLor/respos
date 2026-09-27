"""Проверка осуществимости физмодели (запускалось из корня проекта). Восстановлено из сессии."""
import pandas as pd, numpy as np, hashlib
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
EX=Path('extracted'); s=pd.read_csv(EX/'summary.csv'); R=6378137; g0=9.81
seen=set(); rows=[]
for b in s.bag:
    fr=pd.read_parquet(EX/b/'front.parquet').sort_values('t_hdr')
    h=hashlib.md5(fr.velocity.round(4).to_numpy().tobytes()).hexdigest()
    if h in seen or len(fr)<600: continue
    seen.add(h)
    re=pd.read_parquet(EX/b/'rear.parquet').sort_values('t_hdr'); cmd=pd.read_parquet(EX/b/'cmd.parquet').sort_values('t_hdr')
    d=pd.merge_asof(fr.rename(columns={'velocity':'vf'}),re[['t_hdr','velocity']].rename(columns={'velocity':'vr'}),on='t_hdr',direction='nearest',tolerance=60_000_000)
    d=pd.merge_asof(d,cmd[['t_hdr','position']],on='t_hdr',direction='backward').dropna()
    t=d.t_hdr.to_numpy()/1e9; v=d[['vf','vr']].mean(axis=1).to_numpy()/3.6
    # равномерная сетка 10 Гц
    tg=np.arange(t[0],t[-1],0.1); vg=np.interp(tg,t,v); ug=np.interp(tg,t,d.position.to_numpy()); ug=np.round(ug)
    vs=gaussian_filter1d(vg,5); a=np.gradient(vs,0.1)   # σ=0.5 с
    grade=np.full(len(tg),np.nan)
    fx=pd.read_parquet(EX/b/'master_fix.parquet')
    if len(fx):
        fx=fx[fx.status==2].sort_values('t_hdr')
        if len(fx)>100:
            dist=np.interp(tg, tg, np.cumsum(vg*0.1))
            z=np.interp(tg, fx.t_hdr.to_numpy()/1e9, fx.alt.to_numpy())
            # уклон как dz/ds по сетке пути 1 м
            sg=np.arange(0,dist[-1],1.0); zs=np.interp(sg,dist,z); zs=gaussian_filter1d(zs,20)
            gr=np.gradient(zs,1.0); grade=np.interp(dist,sg,gr)
            grade[(tg<fx.t_hdr.iloc[0]/1e9)|(tg>fx.t_hdr.iloc[-1]/1e9)]=np.nan
    rows.append(pd.DataFrame({'bag':b,'t':tg,'v':vg,'a':a,'u':ug,'grade':grade}))
D=pd.concat(rows); D=D[D.v>0.5]
Dg=D.dropna(subset=['grade']).copy(); Dg['ac']=Dg.a+g0*Dg.grade   # ускорение без гравитационной составляющей
print('точек (движение):',len(D),' с уклоном:',len(Dg))

# 1. задержка команда → ускорение
lags=np.arange(0,31)
cc=[]
for L in lags:
    x=Dg.groupby('bag').apply(lambda q: np.corrcoef(q.u.to_numpy()[:len(q)-L] if L else q.u, q.ac.to_numpy()[L:])[0,1] if len(q)>L+10 else np.nan).median()
    cc.append(x)
Lb=lags[int(np.nanargmax(cc))]; print(f'корреляция u→a: при 0 с {cc[0]:.2f}, максимум {np.nanmax(cc):.2f} при задержке {Lb*0.1:.1f} с')

# 2. выбег: удельное сопротивление по Дэвису
c=Dg[(Dg.u==0)&(Dg.v>1)]
X=np.c_[np.ones(len(c)),c.v,c.v**2]; y=-c.ac
coef,*_=np.linalg.lstsq(X,y,rcond=None); pred=X@coef
r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
print(f'Выбег: w0 = {coef[0]/g0*1000:.2f} + {coef[1]/g0*1000:.3f}·v + {coef[2]/g0*1000:.4f}·v² Н/кН (v в м/с); n={len(c)}, R²={r2:.2f}, СКО ост. {np.std(y-pred):.3f} м/с²')
cn=Dg[(Dg.u==0)&(Dg.v>1)]
print(f'   без учёта уклона R² падает до: {1-((-cn.a-(np.c_[np.ones(len(cn)),cn.v,cn.v**2]@np.linalg.lstsq(np.c_[np.ones(len(cn)),cn.v,cn.v**2],-cn.a,rcond=None)[0]))**2).sum()/((-cn.a+cn.a.mean())**2).sum():.2f}')

# 3. таблица удельного ускорения по позициям и скоростям (со сдвигом на задержку)
Dg['u_l']=Dg.groupby('bag').u.shift(Lb)
Dg['vb']=pd.cut(Dg.v,[0.5,3,6,9,12,16],labels=['0.5-3','3-6','6-9','9-12','12-16'])
T=Dg.dropna(subset=['u_l']).pivot_table(index='u_l',columns='vb',values='ac',aggfunc='median',observed=False).round(2)
N=Dg.dropna(subset=['u_l']).groupby('u_l').size()
T['n']=N; print('\nМедианное ускорение (без уклона), м/с², по позиции контроллера (с задержкой) и скорости:'); print(T.to_string())

# 4. предсказуемость: табличная модель a(u, v) + уклон — доля объяснённой дисперсии
Q=Dg.dropna(subset=['u_l']).copy()
Q['vb2']=(Q.v//1).astype(int)
tab=Q.groupby(['u_l','vb2']).ac.transform('median')
for name,pred in [('a(u,v) без уклона, по a', tab), ('a(u,v) − g·уклон, по a', tab-g0*Q.grade)]:
    r2=1-((Q.a-pred)**2).sum()/((Q.a-Q.a.mean())**2).sum(); print(f'R² {name}: {r2:.2f}, СКО ошибки ускорения {np.std(Q.a-pred):.3f} м/с²')
# что это значит для счисления: ошибка скорости через 10 с только по модели
print('\nЧастоты позиций:', D.u.value_counts().sort_index().to_dict())
