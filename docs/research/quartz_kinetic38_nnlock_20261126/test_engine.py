import unittest
from engine import lock_distance, Puct, load_eval, proof, won
class Test(unittest.TestCase):
    def test_ties(self):
        self.assertEqual(lock_distance([0,1],[2,1]),2)
        self.assertEqual(lock_distance([0,1],[1,2]),1)
        self.assertEqual(lock_distance([0,1],[2,2]),1)
    def test_common_gate(self):
        nn=load_eval(101)
        x=Puct(nn).search([23,0,25,48],32,True)
        self.assertEqual(x['kind'],'CERTIFICATE')
        self.assertEqual(x['nn'],0)
    def test_hard_cap_same_decision(self):
        nn=load_eval(101);h=[0,48,7,47]
        a=Puct(nn).search(h,32,False);b=Puct(nn).search(h,32,True)
        self.assertEqual(a['move'],b['move'])
        self.assertLessEqual(b['nn'],a['nn'])
        for x in (a,b):
            self.assertEqual(x['nn'],1+x['visits']-x['terminal_leaves'])
            self.assertLessEqual(x['visits'],32)
if __name__=='__main__':unittest.main()
