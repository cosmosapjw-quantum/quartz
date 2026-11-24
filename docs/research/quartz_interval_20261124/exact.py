from fractions import Fraction as F
from itertools import product
from random import Random
from math import prod

rng = Random(203)
for n in range(6):
    for _ in range(20):
        def law(k):
            x = [rng.randint(1, 11) for _ in range(k)]
            return [F(i, sum(x)) for i in x]
        q = law(2)
        p = [[law(3) for _ in q] for _ in range(n)]
        b = F(rng.randrange(9), 8)
        mix = lambda k: sum(q[z]*prod(sum(row[z][:k]) for row in p) for z in range(2))
        m = 1-max(F(0), F(1,2)-b)*mix(1)-(1-max(b,F(1,2)))*mix(2)
        direct = sum(max((b, *(F(k,2) for k in path)))*
                     sum(q[z]*prod(row[z][k] for row,k in zip(p,path)) for z in range(2))
                     for path in product(range(3), repeat=n))
        assert m == direct
print('120 exact normalized mixture/world comparisons')
