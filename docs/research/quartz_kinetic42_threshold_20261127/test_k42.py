import unittest
from k42_plan import rule,load,BUDGET
from threshold42 import solve
from k42_engine import Puct,load_eval
class Test(unittest.TestCase):
    def test_action_and_full(self):
        h=next(j['opening'] for j in load('PLAN.json')['jobs'] if j['split']=='ATLAS')
        a,b=rule.from_history(h)
        if len(h)%2:a,b=b,a
        q=solve(a,b);f=solve(a,b,census=True)
        self.assertTrue(q['complete']);self.assertTrue(f['complete'])
        self.assertEqual(q['best'],f['best']);self.assertEqual(q['best_value'],max(f['values'].values()))
        for m,v in f['values'].items():self.assertLessEqual(q['bounds'][m][0],v);self.assertLessEqual(v,q['bounds'][m][1])
        self.assertGreaterEqual(q['bounds'][str(q['best'])][0],max(v[1] for v in q['bounds'].values()))
        z=solve(a,b,0);self.assertFalse(z['complete']);self.assertEqual(z['nodes'],0);self.assertNotIn('best',z)
    def test_live(self):
        e=Puct(load_eval(101));self.assertTrue(hasattr(e,'quartz42'),'missing native action-proof path')
        h=[0,48,7,47];q=e.quartz42(h);l=e.search(h,32,True)
        for k in ('move','nn','visits','votes'):self.assertEqual(q[k],l[k])
        h=next(j['opening'] for j in load('PLAN.json')['jobs'] if j['split']=='LATE')
        z=e.quartz42(h);self.assertTrue(z['exact_attempt']);self.assertLessEqual(z['exact_nodes'],BUDGET)
        if z['exact_complete']:self.assertEqual(z['nn'],0);self.assertEqual(z['kind'],'EXACT42')
if __name__=='__main__':unittest.main()
