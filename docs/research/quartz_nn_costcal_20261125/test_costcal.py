import unittest
from costcal import point_cost,envelope,tables
class Test(unittest.TestCase):
    def test_median_and_rank(self):
        self.assertEqual(point_cost([[100,200,400],[110,220,440],[100000,200000,300000]]),[110,220,440])
        self.assertEqual(envelope(list(range(1,20))),(18,18))
        self.assertEqual(envelope(list(range(1,9))),(9,None))
        self.assertEqual(envelope([.5]*19),(18,1.))
    def test_gain_is_immutable(self):
        gain=[[.1,-.1,0.] for _ in range(16)]
        tab=tables(gain,[100,200,400],1.5)
        self.assertEqual(tab[0],[[.1,150.],[-.1,300.],[0.,600.]])
        self.assertEqual(gain[0],[.1,-.1,0.])
if __name__=='__main__':unittest.main()
