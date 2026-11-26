from fractions import Fraction as F
from random import Random
r=Random(261135)
H=[[(-1)**((i&(~j)).bit_count()) for j in range(16)] for i in range(16)]
# Python negative int bit_count is absolute value, but i&~j is nonnegative.
assert all(sum(a*b for a,b in zip(H[i],H[j]))==16*(i==j) for i in range(16) for j in range(16))
for _ in range(4096):
    u=[F(r.randrange(3),2) for j in range(16)]
    k=[r.randrange(81) for j in range(16)]
    f=[a-F(b,500) for a,b in zip(u,k)]
    c=[sum(a*b for a,b in zip(row,f))/16 for row in H]
    assert all(sum(c[i]*H[i][j] for i in range(16))==f[j] for j in range(16))
    assert f[15]-f[0]==2*sum(c[i] for i in range(16) if i.bit_count()%2)
    for j in (1,2,4,8):
        assert 2*c[j]==sum(f[i|j]-f[i] for i in range(16) if not i&j)/8
        for i in range(16):
            if not i&j:
                assert f[i|j]-f[i]==2*sum(c[t]*H[t^j][i] for t in range(16) if t&j)
    g=F(r.randrange(-100,101),100);z=r.randrange(1,101)
    assert (g/z>F(1,500))==(g-F(z,500)>0)
print('4096 complete rational factorial decompositions, conditional contrasts and price identities')
