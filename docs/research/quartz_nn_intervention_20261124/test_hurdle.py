import unittest
from hurdle import shrink
class Test(unittest.TestCase):
    def test_signed_and_empty_fallback(self):
        # Global 1 loss, 2 half gains, 1 full gain -> g=1/4; mean cost 10.
        m=shrink([(0,-2,10),(0,1,10),(1,1,10),(1,2,10)])
        self.assertEqual(m['global'],[.25,0,0,.5,.25])
        self.assertAlmostEqual(m['gain_cost'][15][0],.25)
        self.assertAlmostEqual(m['gain_cost'][15][1],10.)
        self.assertAlmostEqual(m['gain_cost'][0][0],.15)
        self.assertAlmostEqual(m['gain_cost'][1][0],.35)
    def test_zero_and_negative(self):
        m=shrink([(3,-2,20),(3,0,40)])
        self.assertEqual(m['global'],[.5,0,.5,0,0])
        self.assertEqual(m['gain_cost'][3],[-.5,30.])
if __name__=='__main__':unittest.main()
