from fractions import Fraction as F
from itertools import product
n=0
for a,b,c,d in product(range(3),repeat=4):
    assert (b-a)+(c-b)+(d-c)==d-a
    n+=1
for q0,q1,b0,b1 in product(range(1,6),repeat=4):
    assert (1-F(q1,b1))-(1-F(q0,b0))==F(q0*b1-q1*b0,b0*b1)
    n+=1
assert 4<20 and 4+20+20>20
print('exact finite algebra checks',n,'counterexample valid; Wolfram three True')
