import unittest
from planner import plan
class PlanTests(unittest.TestCase):
    def test_information_only_first(self):
        # R05 counterexample; second identical signal decreases U, different increases it.
        def row(c):
            known,h=c
            if known is not None:return (known,0,0,0)
            if sum(h)==0:return (.5,.5,0,.5)
            if sum(h)==1:return (.5,.66,0,.34) if h[0] else (.5,.34,0,.66)
            return ((25/34 if h[0] and h[2] else 25/66),0,0,0)
        s=[(.5,(0,0,0)),(None,(0,0,0))]
        self.assertEqual(plan(s,row,[.01,.01],1),(.5,None))
        v,i=plan(s,row,[.01,.01],2)
        self.assertAlmostEqual(v,.56,places=12)
        self.assertEqual(i,1)
    def test_uninformative(self):
        def row(c):return (.5,1/3,1/3,1/3) if c[0] is None else (c[0],0,0,0)
        self.assertEqual(plan([(.5,(0,0,0)),(None,(0,0,0))],row,[.01,.01],2),(.5,None))
if __name__=='__main__':unittest.main()
