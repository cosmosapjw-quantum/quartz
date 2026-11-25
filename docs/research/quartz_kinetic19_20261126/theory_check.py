from itertools import product
from fractions import Fraction as F
S=[(0,a,b) for a,b in product(range(-2,3),repeat=2) if max(0,a,b)-min(0,a,b)<=2]
n=0
# Four equiprobable scenarios, arbitrary two-bit signal; compare full signal to its first-bit garbling.
for ids in product(range(len(S)),repeat=3):
    for zs in product(range(4),repeat=3):
        us=[S[i] for i in ids];m=[sum(u[i] for u in us) for i in range(3)]
        v=max(m);h=sum(max(u) for u in us)
        def val(z):
            return sum(max(sum(u[i] for u,r in zip(us,z) if r==k) for i in range(3)) for k in set(z))
        fine=val(zs);coarse=val([z//2 for z in zs])
        assert v<=coarse<=fine<=h
        n+=1
# exact XOR complementary signal
u=[-1,1,1,-1]
def xor_value(z):
    return F(sum(max(0,sum(v for v,r in zip(u,z) if r==k)) for k in set(z)),4)
assert xor_value([0,0,1,1])==xor_value([0,1,0,1])==0
assert xor_value([0,1,2,3])==F(1,2)
print('exact shared-prior/garbling cases',n,'XOR one-step counterexample PASS')
