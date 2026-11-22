#!/usr/bin/env python3
from fractions import Fraction as F
import json,pathlib
A=[[F(x) for x in row] for row in [[5,-1,-1,0],[-1,5,0,-1],[-1,0,5,-1],[0,-1,-1,5]]]
def dot(x,y): return sum(a*b for a,b in zip(x,y))
def mv(x): return [dot(row,x) for row in A]
def solve(b):
 M=[row[:]+[v] for row,v in zip(A,b)]
 for i in range(4):
  z=M[i][i]; M[i]=[v/z for v in M[i]]
  for j in range(4):
   if j!=i:
    z=M[j][i];M[j]=[u-z*v for u,v in zip(M[j],M[i])]
 return [row[-1] for row in M]
rows=[]
for b in [[F(1),F(0),F(0),F(0)],[F(0),F(1,2),F(1,2),F(0)]]:
 x=[v/5 for v in b];r=[v-y for v,y in zip(b,mv(x))]
 ell=2*dot(b,x)-dot(x,mv(x));q=dot(b,solve(b));d=dot(r,r)
 assert q-ell==dot(r,solve(r))
 lo,hi=ell+d/9,ell+d
 assert lo<=q<=hi
 rows.append({'q':str(q),'ell':str(ell),'norm2':str(d),'lo':str(lo),'hi':str(hi)})
assert F(rows[0]['lo'])>F(rows[1]['hi'])
out={'method':'independent Python Fraction Gaussian elimination; no production imports','rows':rows,'strict_choice':0,'wolfram_output':'{{{23/105,1/5,2/25,47/225,7/25,0},{5/42,1/10,2/25,49/450,9/50,0}},525,{7,5,5,3},True}','wolfram_context_status':'internal tool error on initial context retrieval; evaluator actually succeeded','status':'PASS'}
p=pathlib.Path(__file__).resolve().parents[1]/'evidence/R07_EXACT.json'
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
