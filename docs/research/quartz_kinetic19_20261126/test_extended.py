import unittest
from transition import *
class MoreTests(unittest.TestCase):
    def test_marginal_and_bound(self):
        tr=[dict(x=[0.]*6,y=[-2,0],obs={c:dict(z=0,cost=1.) for c in CHANNELS}),dict(x=[1.]*6,y=[2,0],obs={c:dict(z=1,cost=2.) for c in CHANNELS})]
        q=estimate(tr,[.5]*6,.5)
        for c in CHANNELS:
            a=q['channels'][c]
            self.assertTrue(0<=a['delta']<=q['risk']+1e-12)
            self.assertTrue(all(abs(sum(r)-v)<1e-12 for r,v in zip(a['J'],q['p'])))
            self.assertAlmostEqual(sum(a['pi']),1.)
        self.assertEqual(choose(q,100)[0],'STOP')
        self.assertNotEqual(choose(q,0)[0],'STOP')
    def test_invalid(self):
        with self.assertRaises(ValueError):estimate([], [0.]*6,None)
        with self.assertRaises(ValueError):estimate([dict(x=[0.]*6,y=[0,0],obs={})],[0.]*6,None)
if __name__=='__main__':unittest.main()
