import unittest
from k40_engine import Puct,load_eval
from budget import solve,F
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(c):c.ev=load_eval(101)
    def test_three_horizon_prefix(self):
        e=Puct(self.ev);self.assertTrue(hasattr(e,'atlas40'),'missing three-horizon atlas')
        h=[0,48,7,47];a=e.atlas40(h)
        for H in (20,26,32):
            b=e.search(h,H,False);l=a[f'L{H}']
            self.assertEqual(b['move'],l['move']);self.assertLessEqual(l['nn'],b['nn'])
            x=e.adaptive40(h,{str(a['bin']):{str(H):'1'}})
            for k in ('move','nn','visits','votes'):self.assertEqual(x[k],l[k])
        self.assertLessEqual(a['L20']['nn'],a['L26']['nn']);self.assertLessEqual(a['L26']['nn'],a['L32']['nn'])
        self.assertTrue(0<=a['bin']<8)
    def test_gate(self):
        e=Puct(self.ev);self.assertTrue(hasattr(e,'adaptive40'),'missing budgeted live path')
        a=e.adaptive40([23,0,25,48],{})
        self.assertEqual(a['nn'],0);self.assertEqual(a['kind'],'CERTIFICATE')
    def test_budget(self):
        a=solve({0:{20:(0,0),26:(3,4),32:(8,8)},1:{20:(0,0),26:(2,4),32:(4,4)}},4)
        self.assertEqual(a['gain'],F(20,3));self.assertEqual(a['dual'],F(4,3))
if __name__=='__main__':unittest.main()
