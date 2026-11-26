import unittest
from partial43 import probe
from k43_engine import Puct,load_eval
class Test(unittest.TestCase):
    def test_unknown_and_gate(self):
        x=probe([0,48,7,47],7,7,0)
        self.assertEqual(x['queries'],0);self.assertIsNone(x['cert'])
        self.assertTrue(any(l==-1 and u==1 for _,l,u in x['bounds']))
        x=probe([23,0,25,48],7,7,4096)
        self.assertIsNotNone(x['gate']);self.assertEqual(x['gate'],x['cert'])
        self.assertEqual(x['queries'],0)
    def test_rapfi_gate(self):
        h=[112,0,113,1,114,2,115,3]
        x=probe(h,15,7,4096);self.assertEqual(x['gate'],111)
    def test_live(self):
        e=Puct(load_eval(101));self.assertTrue(hasattr(e,'quartz43'),'missing partial proof policy')
        q=e.quartz43([0,48,7,47]);self.assertIn('partial',q);self.assertLessEqual(q['partial']['queries'],4096)
if __name__=='__main__':unittest.main()
