"""Reference baseline: before/after independence."""
from itertools import product
STATES=tuple((a,b) for a,b in product(range(-2,3),repeat=2) if max(0,a,b)-min(0,a,b)<=2)
CHANNELS=tuple(f'{a}:{n}' for n in (32,64) for a in ('K16N','K16BAL','K17EXP'))
HVALUES=(None,1.,.5,.25)
def estimate(train,x,h):
    return {'channels':{c:{'delta':0.} for c in CHANNELS}}
