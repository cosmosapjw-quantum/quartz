import unittest
from transition import estimate,CHANNELS
class TestTransition(unittest.TestCase):
    def test_informative_vs_independent(self):
        tr=[]
        for sign,z in [(-2,0),(2,1)]:
            tr.append(dict(x=[0.]*6,y=[sign,0],obs={c:dict(z=z if c==CHANNELS[0] else 0,cost=1.) for c in CHANNELS}))
        q=estimate(tr,[0.]*6,None)
        self.assertGreater(q['channels'][CHANNELS[0]]['delta'],.3)
        self.assertAlmostEqual(q['channels'][CHANNELS[1]]['delta'],0.)
if __name__=='__main__':unittest.main()
