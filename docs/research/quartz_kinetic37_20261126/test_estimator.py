import unittest
from fractions import Fraction as F
from itertools import combinations
from estimator import sample_stable,estimate_absolute,delta,fit_table
class Tests(unittest.TestCase):
    def test_srs(self):
        a=list(range(8));self.assertEqual(len(sample_stable(a,19)),2)
        self.assertEqual(sample_stable([1],19),[1])
        self.assertEqual(sample_stable([],19),[])
        y=[F(0),F(1),F(0),F(1)]
        vals=[estimate_absolute(4,[(4,[y[i] for i in p])])['mean'] for p in combinations(range(4),2)]
        self.assertEqual(sum(vals)/len(vals),F(1,2))
    def test_zero(self):
        self.assertEqual(delta(7,7,None),0)
        self.assertEqual(delta(7,8,{'7':F(1,4),'8':F(3,4)}),F(-1,2))
        with self.assertRaises(AssertionError):delta(7,8,None)
    def test_fit(self):
        t=fit_table([dict(bin=0,cost={80:40,128:80},gain={80:F(-1,2),128:F(0)}),dict(bin=1,cost={80:40,128:80},gain={80:F(0),128:F(0)})])
        self.assertEqual(t['table'],[128,80,80,80,80,80,80,80])
if __name__=='__main__':unittest.main()
