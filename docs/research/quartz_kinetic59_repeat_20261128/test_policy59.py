import unittest
from policy59 import operation, advance
class PolicyTests(unittest.TestCase):
    def test_baseline(self):
        for arm in ('CONTROL','FIRST','REPEAT'):
            for used in (False,True):
                self.assertEqual(operation(arm,False,used),('normal',False))
    def test_q(self):
        for arm,op in [('CONTROL',0),('FIRST',1),('REPEAT',1)]:
            self.assertEqual(operation(arm,True,False),('early',op))
        self.assertEqual(operation('CONTROL',True,True),('normal',True))
        self.assertEqual(operation('FIRST',True,True),('normal',True))
        self.assertEqual(operation('REPEAT',True,True),('early',1))
    def test_attempt(self):
        self.assertFalse(advance(False,{'kind':'CERTIFICATE'}))
        self.assertFalse(advance(False,{'kind':'PARTIAL47'}))
        self.assertTrue(advance(False,{'early':{'reached':False}}))
        self.assertTrue(advance(True,{}))
if __name__=='__main__':unittest.main()
