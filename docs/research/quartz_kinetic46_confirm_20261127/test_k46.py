import unittest
from pair46 import query,gain
class Test(unittest.TestCase):
    def test_zero_budget_unknown(self):
        j=dict(a=1,b=1<<48,actions=[1,2],input=[[-1,1],[-1,1]],optimal=[False,False])
        z=query(j,0);self.assertEqual(z['bounds'],j['input']);self.assertEqual(z['ask'],[-1]*4);self.assertEqual(sum(z['nodes']),0)
    def test_win_and_legal_bounds(self):
        j=dict(a=7,b=(1<<42)|(1<<43)|(1<<48),actions=[3,10],input=[[-1,1],[-1,1]],optimal=[False,False])
        z=query(j,256);self.assertEqual(z['bounds'][0],[1,1]);self.assertLessEqual(sum(z['nodes']),1024)
        self.assertEqual(gain([[-1,1],[-1,1]],[True,False]),[0,2])
if __name__=='__main__':unittest.main()
