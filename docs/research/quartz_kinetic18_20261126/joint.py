"""Unconditional reference implementation (pre-kernel red-test baseline)."""
from itertools import product
STATES=tuple(sorted({(b-a,c-a) for a,b,c in product(range(3),repeat=3)}))
HVALUES=(None,1.,.5,.25)
def predict(train,x,h):
    c=[1/19]*19
    for r in train:c[STATES.index(tuple(r['y']))]+=1
    return [v/(len(train)+1) for v in c],len(train)
