import unittest
from seq45 import Session,staged,gain2
class Test(unittest.TestCase):
    def test_stages(self):
        for n,h in [(7,[0,48,7,47]),(15,[112,111,97,96,128,127]),(7,[23,0,25,48])]:
            with Session(h,n) as s:
                z=s.advance(3,0)
                self.assertEqual(z['queries'],0)
                q=s.advance(5,1024);r=s.advance(7,4096)
                for a,b in zip(z['bounds'],q['bounds']):self.assertTrue(a[0]==b[0] and a[1]<=b[1]<=b[2]<=a[2])
                for a,b in zip(q['bounds'],r['bounds']):self.assertTrue(a[0]==b[0] and a[1]<=b[1]<=b[2]<=a[2])
                self.assertLessEqual(z['queries'],q['queries']);self.assertLessEqual(q['queries'],r['queries']);self.assertLessEqual(r['queries'],4096)
    def test_gain(self):
        q={0:(-1,1),1:(-1,-1),2:(1,1)}
        self.assertEqual(gain2(0,0,q),(0,0));self.assertEqual(gain2(0,1,q),(0,2))
        self.assertEqual(gain2(2,1,q,True),(2,2));self.assertEqual(gain2(0,2,q),(-2,0))
if __name__=='__main__':unittest.main()
