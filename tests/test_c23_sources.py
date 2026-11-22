import importlib.util,pathlib,unittest
from fractions import Fraction as F
P=pathlib.Path(__file__).resolve().parents[1]/'docs/research/quartz_general_residual_stop_20261123/coding/c23_sources.py'
sp=importlib.util.spec_from_file_location('sources',P); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
def exact_solve(task,b):
 n=task['side'];d=n*n;rho=F(task['rho']);A=[[F(0) for j in range(d)] for i in range(d)]
 for i in range(d):
  A[i][i]=1+4*rho
  for j in range(d):
   if (i//n==j//n and abs(i-j)==1) or abs(i-j)==n:A[i][j]=-rho
 M=[row+[F(v)] for row,v in zip(A,b)]
 for i in range(d):
  pivot=M[i][i];M[i]=[v/pivot for v in M[i]]
  for j in range(d):
   if j!=i:
    z=M[j][i];M[j]=[a-z*c for a,c in zip(M[j],M[i])]
 return [row[-1] for row in M]
class SourcesTest(unittest.TestCase):
 def test_oracle_matches_inverse_of_stored_rhs(self):
  task,truth=mod.make_source({'id':'unit','side':2,'rho':16,'target_gap':'medium','seed':17,'split':'UNIT','stratum':0})
  self.assertEqual(set(task),{'id','side','rho','scale','rhs'})
  for b,q in zip(task['rhs'],truth['u_exact']):
   x=exact_solve(task,b); value=sum(F(v)*w for v,w in zip(b,x))/F(task['scale'])
   self.assertEqual(value,F(q))
   self.assertGreaterEqual(value,0);self.assertLessEqual(value,1)
 def test_tie_is_exact_and_distinct(self):
  task,truth=mod.make_source({'id':'tie','side':3,'rho':64,'target_gap':'tie','seed':23,'split':'UNIT','stratum':1})
  vals=[F(q) for q in truth['u_exact']];best=max(vals);inds=[i for i,q in enumerate(vals) if q==best]
  self.assertEqual(len(inds),2)
  self.assertNotEqual(task['rhs'][inds[0]],task['rhs'][inds[1]])
 def test_deterministic_and_public_only(self):
  spec={'id':'unit','side':2,'rho':1,'target_gap':'wide','seed':41,'split':'UNIT','stratum':0}
  self.assertEqual(mod.make_source(spec),mod.make_source(spec))
  task,truth=mod.make_source(spec)
  self.assertNotIn('x_num',task);self.assertNotIn('u_exact',task)
if __name__=='__main__':unittest.main()
