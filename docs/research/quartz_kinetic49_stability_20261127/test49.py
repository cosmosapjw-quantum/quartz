import unittest
from allocation49 import fit
def row(z,s,c,g):return dict(bin=z,seed=s,cost=c,gains=dict(zip(('32','64','128'),g)))
class Test(unittest.TestCase):
    def test_robust_nonzero(self):
        rs=[row(1,s,1,[.1,.2,.3]) for s in (1,2)]+[row(3,s,4,[-.1,.2,-.3]) for s in (1,2)]
        q=fit(rs,(1,2));self.assertEqual(q['mask'],2);self.assertGreater(q['worst_mean'],0)
    def test_robust_zero(self):
        rs=[row(1,1,1,[.9,.2,.3]),row(1,2,1,[-.1,.2,.3]),row(2,1,8,[0,0,0]),row(2,2,8,[0,0,0])]
        q=fit(rs,(1,2));self.assertEqual(q['mask'],0)
if __name__=='__main__':unittest.main()
