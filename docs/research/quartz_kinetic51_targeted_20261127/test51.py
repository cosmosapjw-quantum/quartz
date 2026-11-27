import unittest
from target51 import candidate,choose
class Tests(unittest.TestCase):
    def test_rival(self):
        low={'move':4,'votes':[[1,5],[2,5],[3,7],[4,8]]}
        self.assertEqual(candidate(low,[[1,-1,1],[2,-1,1],[3,-1,-1],[4,-1,1]]),1)
    def test_dominance(self):
        low={'move':4,'votes':[[1,5],[2,3],[4,8]]}
        self.assertIsNone(candidate(low,[[1,-1,0],[2,-1,-1],[4,0,1]]))
    def test_strict(self):
        self.assertEqual(choose(4,1,-.2,.1),1)
        self.assertEqual(choose(4,1,.1,.1),4)
if __name__=='__main__':unittest.main()
