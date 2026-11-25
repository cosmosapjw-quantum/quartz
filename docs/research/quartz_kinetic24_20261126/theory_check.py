from fractions import Fraction as F
from random import Random
from math import log
rng=Random(2026112624)
for _ in range(4096):
    v=F(rng.randrange(1,8));w=F(rng.randrange(1,8));c=F(rng.randrange(-5,6),4)
    if c*c>=v*w:c=F(0)
    a,b=F(rng.randrange(-5,6)),F(rng.randrange(-5,6))
    full=(w*a*a-2*c*a*b+v*b*b)/(v*w-c*c)
    inn=a*a/v+(b-c*a/v)**2/(w-c*c/v)
    assert full==inn and full>=(a+b)**2/(v+2*c+w)
    sets=[{j:F(j,10) for j in range(9) if rng.randrange(2)} for _ in range(3)]
    A,B,C=sets
    assert A|A==A and A|B==B|A and (A|B)|C==A|(B|C)
    assert set(A|B)==set(A)|set(B)
    x=[F(rng.randrange(-10,11),10) for i in range(5)]
    y=[F(rng.randrange(-10,11),10) for i in range(5)]
    assert abs(max(-z for z in x)-max(-z for z in y))<=max(abs(a-b) for a,b in zip(x,y))
print('4096 exact rational Schur/aggregation, idempotent union, negamax Lipschitz cases')
