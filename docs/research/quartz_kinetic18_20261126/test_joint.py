import unittest, math
from joint import STATES,predict
class TestJoint(unittest.TestCase):
    def test_kernel_conditions_on_observation(self):
        train=[dict(x=[0.]*10,y=[2,0]),dict(x=[1.]*10,y=[-2,0])]
        p,ess=predict(train,[0.]*10,.25)
        q,_=predict(train,[1.]*10,.25)
        assert len(p)==len(q)==19
        assert abs(sum(p)-1)<1e-12 and min(p)>0
        self.assertGreater(p[STATES.index((2,0))],q[STATES.index((2,0))])
        self.assertLess(ess,1.01)
    def test_unconditional_is_feature_invariant(self):
        train=[dict(x=[0.]*10,y=[2,0]),dict(x=[1.]*10,y=[-2,0])]
        self.assertEqual(predict(train,[0.]*10,None),predict(train,[1.]*10,None))
if __name__=='__main__':unittest.main()
