from fractions import Fraction as F
from random import Random
rng=Random(2026112625)
H=[[(-1)**((a&m).bit_count()) for m in range(16)] for a in range(16)]
assert all(sum(H[a][m]*H[b][m] for m in range(16))==16*(a==b) for a in range(16) for b in range(16))
for _ in range(4096):
    f=[F(rng.randrange(-100,101),100) for m in range(16)]
    beta=[sum(x*y for x,y in zip(h,f))/16 for h in H]
    assert all(sum(H[m][a]*beta[a] for a in range(16))==f[m] for m in range(16))
    assert f[15]-f[0]==-2*sum(beta[a] for a in range(16) if a.bit_count()%2)
    for j in range(4):
        k=1<<j
        assert sum(f[m|k]-f[m] for m in range(16) if not m&k)/8==-2*beta[k]
        for l in range(j+1,4):
            u=1<<l
            assert sum(f[m|k|u]-f[m|k]-f[m|u]+f[m] for m in range(16) if not m&(k|u))/4==4*beta[k|u]
    direct={i for i in range(10) if rng.randrange(2)};received={i for i in range(10) if rng.randrange(2)}
    other={i for i in range(10) if rng.randrange(2)};cur={rng.randrange(10)}
    novel=other-direct-received-cur
    assert not (other-direct-(received|novel)-cur)
print('4096 exact rational factorial/interaction and receipt replay cases; orthogonality exact')
