#!/usr/bin/env python3
"""C26 finite exact-query model checks; no GameState execution or repo mutation."""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product, combinations_with_replacement
from decimal import Decimal, localcontext
import json, platform, sys
from pathlib import Path

VALUES = (F(0), F(1, 2), F(1))

def gain(p, b):
    return sum((prob * max(F(0), u-b) for prob, u in zip(p, VALUES)), F(0))

def reservation(p, d):
    assert d > 0 and sum(p) == 1 and all(x >= 0 for x in p)
    p0, ph, p1 = p
    mu = ph/2 + p1
    if p1 and d <= p1/2:
        return 1-d/p1
    if ph+p1 and d <= mu:
        return (mu-d)/(ph+p1)
    return mu-d

def exact_values(boxes, incumbent):
    z = tuple(reservation(p,d) for p,d in boxes)
    @lru_cache(None)
    def dp(b, remaining, mode):
        if not remaining:
            return b
        q = {a: gain(boxes[a][0], b)-boxes[a][1] for a in remaining}
        if mode == 'optimal':
            candidates = remaining
        elif mode == 'greedy':
            a = max(remaining, key=lambda a: (q[a], -a))
            if q[a] <= 0:
                return b
            candidates = (a,)
        elif mode == 'pandora':
            a = max(remaining, key=lambda a: (z[a], -a))
            if z[a] <= b:
                return b
            candidates = (a,)
        elif mode == 'all':
            candidates = (remaining[0],)
        else:
            raise ValueError(mode)
        next_values = []
        for a in candidates:
            p, d = boxes[a]
            rest = tuple(k for k in remaining if k != a)
            next_values.append(-d+sum((prob*dp(max(b,u),rest,mode)
                                      for prob,u in zip(p, VALUES)), F(0)))
        return max([b] + next_values) if mode == 'optimal' else max(next_values)
    remaining = tuple(range(len(boxes)))
    result = {mode: dp(incumbent, remaining, mode)
              for mode in ('optimal','greedy','pandora','all')}
    stop_condition = all(gain(p, incumbent) <= d for p,d in boxes)
    assert stop_condition == (result['optimal'] == incumbent)
    assert result['pandora'] == result['optimal']
    for (p,d), zi in zip(boxes,z):
        assert gain(p,zi) == d
    return result, z

def dec(x):
    return Decimal(x.numerator)/Decimal(x.denominator)

def gibbs_value(boxes, incumbent, tau=F(1,10), mask_at_one=True):
    """Numerical expectation, explicit full branch recursion; no RNG sampling."""
    @lru_cache(None)
    def dp(b, remaining):
        if not remaining or (mask_at_one and b == 1):
            return dec(b)
        raw_q = [F(0)] + [gain(boxes[a][0],b)-boxes[a][1] for a in remaining]
        shift = max(raw_q)
        w = [(dec(x-shift)/dec(tau)).exp() for x in raw_q]
        weights = [x/sum(w) for x in w]
        answer = weights[0]*dec(b)
        for a, qa in zip(remaining, weights[1:]):
            p,d = boxes[a]
            rest = tuple(x for x in remaining if x != a)
            continuation = -dec(d)+sum(dec(prob)*dp(max(b,u),rest)
                                        for prob,u in zip(p,VALUES))
            answer += qa*continuation
        return answer
    return dp(incumbent, tuple(range(len(boxes))))

def correlated_counterexample():
    support = ((F(1,2),(F(0),F(1),F(0))),
               (F(1,2),(F(1,2),F(0),F(1))))
    costs = (F(1,100),F(3,10),F(3,10))
    @lru_cache(None)
    def dp(indices, remaining, b):
        mass = sum(support[i][0] for i in indices)
        opts = [('STOP',b)]
        for a in remaining:
            total = -costs[a]
            for u in VALUES:
                sub = tuple(i for i in indices if support[i][1][a] == u)
                if sub:
                    submass = sum(support[i][0] for i in sub)
                    total += submass/mass*dp(sub,tuple(k for k in remaining if k!=a),max(b,u))[1]
            opts.append((str(a),total))
        return max(opts,key=lambda t:t[1])
    b = F(1,2)
    q = tuple(sum(prob*max(F(0), us[a]-b) for prob,us in support)-costs[a] for a in range(3))
    optimum = dp((0,1),(0,1,2),b)
    assert all(x < 0 for x in q)
    assert optimum == ('0', F(69,100))
    return {'incumbent':str(b),'initial_net_one_step':[str(x) for x in q],
            'optimal_first_query':optimum[0],'optimal_value':str(optimum[1]),
            'optimal_gain':str(optimum[1]-b),
            'joint_support':[[str(p),[str(x) for x in us]] for p,us in support],
            'costs':[str(x) for x in costs]}

def main(output):
    costs = (F(1,20),F(1,10),F(1,5))
    pgrid = tuple((F(a,4),F(b,4),F(4-a-b,4))
                  for a in range(5) for b in range(5-a))
    boxgrid = tuple(product(pgrid,costs))
    counts = {'two_boxes_ordered':0,'three_boxes_up_to_relabeling':0}
    for boxes in product(boxgrid, repeat=2):
        for b in VALUES:
            exact_values(boxes,b)
            counts['two_boxes_ordered'] += 1
    phalf = tuple((F(a,2),F(b,2),F(2-a-b,2))
                  for a in range(3) for b in range(3-a))
    triplegrid = tuple(product(phalf,costs))
    for ids in combinations_with_replacement(range(len(triplegrid)),3):
        boxes = tuple(triplegrid[i] for i in ids)
        for b in VALUES:
            exact_values(boxes,b)
            counts['three_boxes_up_to_relabeling'] += 1
    assert counts == {'two_boxes_ordered':6075,'three_boxes_up_to_relabeling':3420}
    A = ((F(0),F(1),F(0)),F(1,10))
    B = ((F(3,5),F(0),F(2,5)),F(1,20))
    C = ((F(4,5),F(0),F(1,5)),F(1,50))
    examples = {}
    for label, boxes in [('two_box_order_counterexample',(A,B)),('three_box_order_comparison',(A,B,C))]:
        vals,z = exact_values(boxes,F(0))
        with localcontext() as ctx:
            ctx.prec = 60
            masked60 = gibbs_value(boxes,F(0),mask_at_one=True)
            unmasked60 = gibbs_value(boxes,F(0),mask_at_one=False)
        with localcontext() as ctx:
            ctx.prec = 90
            masked90 = gibbs_value(boxes,F(0),mask_at_one=True)
            unmasked90 = gibbs_value(boxes,F(0),mask_at_one=False)
            residual = max(abs(masked60-masked90),abs(unmasked60-unmasked90))
            assert residual < Decimal('1e-55')
        examples[label] = {'boxes':[{'p':[str(x) for x in p],'priced_cost':str(d)} for p,d in boxes],
                           'initial_net_one_step':[str(gain(p,F(0))-d) for p,d in boxes],
                           'reservation_values':[str(x) for x in z],
                           'exact_expected_values':{k:str(v) for k,v in vals.items()},
                           'gibbs_tau':'1/10','gibbs_prior':'uniform over STOP + remaining queries',
                           'gibbs_masked_at_incumbent_one':str(masked90),
                           'gibbs_unmasked':str(unmasked90),
                           'precision_60_90_max_residual':str(residual)}
    assert examples['two_box_order_counterexample']['exact_expected_values']['greedy']=='11/20'
    assert examples['two_box_order_counterexample']['exact_expected_values']['optimal']=='59/100'
    result = {'status':'completed','assertion_tested':'Exact Bellman optimal value equals Pandora value; all nonpositive one-step gains iff stopping is optimal, over stated finite grids; three-point reservation formula solves expected-excess equation exactly.',
              'counts':counts,'total_cases':sum(counts.values()),'coefficient_domain':'Fraction rational; Decimal only for Gibbs expectations',
              'software':{'python':platform.python_version()},'randomness':{'used':False,'reason':'all states and outcomes enumerated; no empirical Monte Carlo or GameState run'},
              'two_box_domain':{'probability_grid':'all 15 triples with nonnegative numerator summing to 4, denominator 4','priced_costs':[str(x) for x in costs],'incumbents':[str(x) for x in VALUES],'ordered_box_types':45},
              'three_box_domain':{'probability_grid':'all 6 triples with nonnegative numerator summing to 2, denominator 2','priced_costs':[str(x) for x in costs],'incumbents':[str(x) for x in VALUES],'box_types':18,'enumeration':'all 1140 three-multisets of 18 types, representing labels up to permutation'},
              'examples':examples,'correlation_counterexample':correlated_counterexample(),
              'non_claims':['No production/GameState implementation verification','No proof of empirical conditional independence','No novelty claim','Gibbs probabilities/expectations are numerical, not rational identities','Finite grid is a check of the proof, not a substitute for the uniform proof']}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'counts':counts,'examples':examples,'correlation_counterexample':result['correlation_counterexample']},indent=2))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/tmp/quartz_c26_exactcheck.json')
