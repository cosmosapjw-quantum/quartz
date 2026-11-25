import unittest,math
from joint import *
class Extended(unittest.TestCase):
    def test_support(self):
        self.assertEqual(len(STATES),19)
        self.assertTrue(all(max(0,*s)-min(0,*s)<=2 for s in STATES))
    def test_anchor_tie_and_joint_risk(self):
        q=[0.]*19;q[INDEX[(1,0)]]=q[INDEX[(0,1)]]=.5
        self.assertEqual(decide(q),1)
        self.assertAlmostEqual(metrics(q,[1,0])['risk'],.25)
        t=[0.]*19;t[INDEX[(0,0)]]=1.
        self.assertEqual(decide(t),0)
        self.assertEqual(metrics(t,[0,0])['risk'],0.)
        self.assertEqual(metrics(t,[0,0])['brier'],0.)
    def test_reject_invalid(self):
        with self.assertRaises(ValueError):predict([], [0.]*10,.5)
        with self.assertRaises(ValueError):predict([dict(x=[0.]*10,y=[2,-2])],[0.]*10,.5)
        with self.assertRaises(ValueError):predict([dict(x=[0.]*10,y=[0,0])],[float('nan')]*10,.5)
    def test_joint_global(self):
        p,ess=predict([dict(x=[0.]*10,y=[2,0]),dict(x=[1.]*10,y=[0,-2])],[0.]*10,None)
        self.assertAlmostEqual(p[INDEX[(2,0)]],(1+1/19)/3)
        self.assertEqual(ess,2)
        self.assertTrue(math.isfinite(metrics(p,[-2,0])['log']))
if __name__=='__main__':unittest.main()
