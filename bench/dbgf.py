import sys
import pandas as pd, numpy as np
from common import *; import faults
b, scen = sys.argv[1], sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 0
run=load_run(b); ref=reference(run); fr,w=faults.apply(run,b,scen)
est=make_estimator(); out,_=run_estimator(est, events(fr))
vr=ref['vel']; a0,a1=w[k]
o=out[(out.t>a0-1)&(out.t<a1+3)]
f=fr['front']; r=fr['rear']; c=fr['cmd']
print(pd.DataFrame({'t':o.t-a0,'v':o.v,'ref':np.interp(o.t,vr.t,vr.v),'front':np.interp(o.t, f.t_hdr/1e9, f.velocity/3.6),'rear':np.interp(o.t, r.t_hdr/1e9, r.velocity/3.6),
  'u':np.interp(o.t, c.t_hdr/1e9, c.position),'a':o.a,'slip':o.slip_kind}).iloc[::3].round(2).to_string())
