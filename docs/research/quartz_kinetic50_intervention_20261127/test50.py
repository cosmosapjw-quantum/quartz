import unittest
from intervention50 import request_mask
class Test(unittest.TestCase):
    def test_ordinal(self):
        self.assertEqual([request_mask(i,2,'EXTEND') for i in range(3)],[0,255,None])
        self.assertEqual([request_mask(i,2,'CONTROL') for i in range(3)],[0,0,None])
    def test_first(self):
        self.assertEqual(request_mask(0,1,'EXTEND'),255)
        self.assertIsNone(request_mask(1,1,'EXTEND'))
if __name__=='__main__':unittest.main()
