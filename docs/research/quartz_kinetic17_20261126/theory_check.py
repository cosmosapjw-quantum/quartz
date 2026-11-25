from fractions import Fraction as F
from itertools import product
count=0
for u in product(product(range(3), repeat=3),repeat=2):
    worlds=[tuple(F(x,2) for x in row) for row in u]
    for A in [(0,), (0,1), (1,2)]:
        for B in [(0,1,2), A]:
            m=lambda S:max(sum(w[i] for w in worlds)/2 for i in S)
            em=lambda S:sum(max(w[i] for i in S) for w in worlds)/2
            h=em((0,1,2))
            assert h-m(A)-(h-em(B))==em(B)-m(A)
            assert em(B)-m(A)==m(B)-m(A)+em(B)-m(B)
            assert em(B)-m(A)==em(A)-m(A)+em(B)-em(A)
            assert m(B)>=m(A) and em(B)>=m(B) and em(A)>=m(A)
            for w in worlds:
                for d in A:
                    assert max(w)-w[d]==max(w)-max(w[i] for i in A)+max(w[i] for i in A)-w[d]
                for a in A:
                    assert max(w)-w[a] <= max([F(0)]+[w[i]-w[a] for i in range(3) if i!=a])
            Bs=[A, tuple(range(3))]
            v=sum(max(w[i] for i in b) for w,b in zip(worlds,Bs))/2
            assert v-m(A)==em(A)-m(A)+v-em(A) and v>=em(A)
            count+=1
print('EXACT finite cases',count)
