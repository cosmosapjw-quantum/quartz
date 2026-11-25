from itertools import product
from collections import Counter
from fractions import Fraction
n=0
def comp(k,b):
    if k==1:yield (b,);return
    for i in range(b+1):
        for x in comp(k-1,b-i):yield (i,)+x
for k in range(2,6):
    for x in product(range(4),repeat=k):
        if sum(x)==0:continue
        L=max(range(k),key=lambda i:(x[i],i))
        d={j:x[L]-x[j]+(j<L) for j in range(k) if j!=L}
        assert min(d.values())>0;n+=1
        for b in range(5):
            actual={max(range(k),key=lambda j:(x[j]+v[j],j)) for v in comp(k,b)}
            # Full remaining budget can always leave leader intact.
            expected={L}|{j for j,c in d.items() if c<=b}
            assert actual==expected;n+=1
            assert (actual=={L})==(min(d.values())>b);n+=1
for a,b,c in product((Fraction(0),Fraction(1,2),Fraction(1)),repeat=3):
    assert a-c-(b-c)==a-b;n+=1
print('Exact finite assertions',n)
print('Wolfram outputs {1,0,0,16}: strict tie and later-rank tie increments and contrast cancellation.')
