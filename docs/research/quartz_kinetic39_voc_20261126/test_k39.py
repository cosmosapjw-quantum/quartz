import unittest
from k39_engine import Puct,load_eval
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(c):c.ev=load_eval(101)
    def test_resumable_same_action_cost(self):
        engine=Puct(self.ev)
        self.assertTrue(hasattr(engine,'atlas'),'Missing single-prefix atlas; must not restart low prefix')
        h=[0,48,7,47]
        a=engine.atlas(h)
        self.assertEqual(a['B20']['move'],a['L20']['move'])
        self.assertEqual(a['B32']['move'],a['L32']['move'])
        self.assertLessEqual(a['L20']['nn'],a['L32']['nn'])
        for m,k in ((0,'L20'),(15,'L32')):
            r=engine.adaptive(h,m)
            for field in ('move','nn','visits','votes','terminal_leaves'):
                self.assertEqual(r[field],a[k][field])
    def test_common_gate(self):
        engine=Puct(self.ev)
        self.assertTrue(hasattr(engine,'adaptive'),'Missing adaptive path')
        z=engine.adaptive([23,0,25,48],15)
        self.assertEqual(z['nn'],0);self.assertEqual(z['kind'],'CERTIFICATE')
    def test_trained_gen5(self):
        ev=load_eval(105);z=Puct(ev).search([0,48,7,47],20,False)
        self.assertGreater(z['nn'],0)
if __name__=='__main__':unittest.main()
