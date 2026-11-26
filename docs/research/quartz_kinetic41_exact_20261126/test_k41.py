import unittest
from k41_plan import rule,load,BUDGET
from exact41 import solve
from k41_engine import Puct,load_eval
class Test(unittest.TestCase):
    def test_census_and_budget(self):
        h=next(j['opening'] for j in load('PLAN.json')['jobs'] if j['split']=='TRAIN')
        a,b=rule.from_history(h)
        if len(h)%2:a,b=b,a
        r=solve(a,b,BUDGET);self.assertIsNotNone(r,'missing exact action census')
        self.assertTrue(r['complete']);self.assertEqual(set(map(int,r['values'])),set(rule.cells(rule.FULL^(a|b))))
        self.assertEqual(r['best_value'],max(r['values'].values()))
        self.assertEqual(r['best'],min(int(m) for m,v in r['values'].items() if v==r['best_value']))
        z=solve(a,b,0);self.assertFalse(z['complete']);self.assertEqual(z['nodes'],0);self.assertNotIn('values',z)
    def test_same_evaluator_and_gate(self):
        e=Puct(load_eval(101));self.assertTrue(hasattr(e,'quartz41'),'missing bounded proof path')
        q=e.quartz41([23,0,25,48]);self.assertEqual(q['nn'],0);self.assertEqual(q['kind'],'CERTIFICATE')
        h=[0,48,7,47];q=e.quartz41(h);l=e.search(h,32,True)
        for k in ('move','nn','visits','votes'):self.assertEqual(q[k],l[k])
        h=next(j['opening'] for j in load('PLAN.json')['jobs'] if j['split']=='LATE')
        q=e.quartz41(h);self.assertEqual(q['kind'],'EXACT41');self.assertEqual(q['nn'],0);self.assertTrue(q['exact_complete'])
        self.assertLessEqual(q['exact_nodes'],BUDGET)
if __name__=='__main__':unittest.main()
