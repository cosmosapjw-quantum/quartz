import unittest
from scheduler import signed_order
class Test(unittest.TestCase):
    def test_signed(self):
        self.assertEqual(signed_order([-.1,.125,0],['a','b','c']),[1,2,0])
    def test_ties(self):
        self.assertEqual(signed_order([0,0,0],['b','c','a']),[2,0,1])
if __name__=='__main__':unittest.main()
