import unittest
from inc44 import probe,semantic
class Test(unittest.TestCase):
    def test_parity(self):
        for n,h in [(7,[0,48,7,47]),(7,[23,0,25,48]),(15,[112,0,113,1,114,2,115,3]),(15,[112,111,97,96,128,127])]:
            for d,b in ((3,0),(3,1),(3,256),(5,1024),(7,4096)):
                self.assertEqual(semantic(probe(h,n,d,b,'I')),semantic(probe(h,n,d,b,'OLD')))
    def test_filter(self):
        h=[0,224,14,210,15,209]
        a=probe(h,15,7,4096,'I');b=probe(h,15,7,4096,'F')
        self.assertEqual(a['bounds'],b['bounds']);self.assertEqual(a['cert'],b['cert'])
        self.assertGreater(b['skipped'],0);self.assertLess(b['queries'],a['queries'])
if __name__=='__main__':unittest.main()
