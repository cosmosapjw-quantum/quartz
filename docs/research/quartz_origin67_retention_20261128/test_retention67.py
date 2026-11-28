import unittest
from retention67 import choose, features
class TestRetention(unittest.TestCase):
    def test_signed_switch(self):
        self.assertEqual(choose(12,13,[.02,.01,.94,.01,.02]),12)
        self.assertEqual(choose(12,13,[.02,.01,.92,.02,.03]),13)
        self.assertEqual(choose(12,13,[.03,.02,.92,.01,.02]),12)
        self.assertEqual(choose(12,12,[.02,.01,.92,.02,.03]),12)
    def test_missing_and_orientation(self):
        r={'history':[0,1,2,3],'partial':{'bounds':[[4,-1,1],[5,-1,1],[6,-1,-1]]},'move':5,
          'allocation65':{'edges':[[4,3,-.3,.2],[5,2,.6,.3],[6,0,0,.5]],'safe':[4,5],'fallback':4,'chosen':5},
          'operation66':{'prefix_edges':[[4,3,-.3,.2],[5,0,0,.3],[6,0,0,.5]],'prefix_move':4}}
        a,b,x=features(r,'EXPAND')
        self.assertEqual((a,b),(4,5));self.assertAlmostEqual(x[3],.2);self.assertEqual(x[9],1);self.assertEqual(x[10],0);self.assertEqual(x[13:], [0,0])
if __name__=='__main__':unittest.main()
