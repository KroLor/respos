import sys
import pandas as pd, numpy as np
from common import *; import metrics
b=sys.argv[1]; thr=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
sets=dict(a.split('=') for a in sys.argv[3:])
sets={k:float(v) for k,v in sets.items()}
run=load_run(b); ref=reference(run)
est=make_estimator(sets); out,_=run_estimator(est, events(run))
vr=ref['vel']; ir,ie=metrics.match(out,vr.t.to_numpy())
err=out.v.to_numpy()[ie]-vr.v.to_numpy()[ir]
t=vr.t.to_numpy()[ir]
big=np.nonzero((np.abs(err)>thr)&~vr.glitch.to_numpy()[ir])[0]; print(len(big), 'of', len(err))
if len(big):
  tb=t[big[0]]
  o=out[(out.t>tb-4)&(out.t<tb+3)]
  f=run['front']; f=f[(f.t_hdr/1e9>tb-5)&(f.t_hdr/1e9<tb+4)]
  r=run['rear']; r=r[(r.t_hdr/1e9>tb-5)&(r.t_hdr/1e9<tb+4)]
  c=run['cmd']
  print(pd.DataFrame({'t':o.t-tb,'v':o.v,'ref':np.interp(o.t,vr.t,vr.v),'front':np.interp(o.t, f.t_hdr/1e9, f.velocity/3.6),'rear':np.interp(o.t, r.t_hdr/1e9, r.velocity/3.6),
    'u':np.interp(o.t, c.t_hdr/1e9, c.position),'a':o.a,'slip':o.slip_kind,'flt':o.sensor_fault if 'sensor_fault' in o else ''}).iloc[::3].round(2).to_string())
