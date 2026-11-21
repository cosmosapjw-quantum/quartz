#!/usr/bin/env python3
"""Exact finite PUCT reactivation rank reversal, no game-theoretic extrapolation."""
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
import hashlib
import json

def test_and_save():
    getcontext().prec=70
    priors=[Fraction(1,2),Fraction(1,3),Fraction(1,6)]
    n=[8,0,0]
    q=[Fraction(2,5),Fraction(0),Fraction(0)]
    c=Fraction(2)
    parent=sum(n)
    assert sum(priors)==1 and priors[0]+priors[1]==Fraction(5,6)
    root=(Decimal(parent)).sqrt()
    def dec(v):return Decimal(v.numerator)/Decimal(v.denominator)
    scores=[dec(q[i])+dec(c)*dec(priors[i])*root/Decimal(1+n[i]) for i in range(3)]
    assert scores[1]>scores[0]>0
    assert sum(n)==8
    before={'root_visits':8,'n':n,'w':['16/5','0','0']}
    after_mask_activation={'root_visits':8,'n':n.copy(),'w':before['w'].copy()}
    assert before==after_mask_activation
    hypothetical_after_16={'root_visits':24,'n':[8,12,4]}
    assert sum(hypothetical_after_16['n'])==24
    p=Path(__file__).with_name('FINITE_COUNTEREXAMPLE13.json')
    if p.exists():raise FileExistsError('finite result already locked')
    result={
      'schema':'quartz_cog03_resident13_exact_finite_puct_counterexample_v1',
      'status':'DERIVED_FINITE_CONSTRUCTION_NOT_ACTUAL_RUST_MCTS',
      'prior_fractions':[str(v) for v in priors],
      'original_full_prior_sum':str(sum(priors)),
      'restricted_prior_mass_first_two':str(priors[0]+priors[1]),
      'completed_visits_before_activation':8,
      'edge_visits_before':n,
      'edge_q_values':[str(v) for v in q],
      'c_PUCT':str(c),
      'scores_decimal_70digits':[str(s) for s in scores],
      'newly_activated_action_1_beats_old_action_0':scores[1]>scores[0],
      'score_0_exact':'2/5 + 2 Sqrt[2]/9',
      'score_1_exact':'4 Sqrt[2]/3',
      'score_2_exact':'2 Sqrt[2]/3',
      'score_1_minus_score_0_exact':'10 Sqrt[2]/9 - 2/5 > 0',
      'wolfram_crosscheck_tool_output':'{{2/5+2sqrt(2)/9,4sqrt(2)/3,2sqrt(2)/3},{5/6,1},True,{8,8}}',
      'untested_claims':['actual_rust_root_masks','Connect4 move utility monotonicity','finite sample win rate']
    }
    p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('COG03_R07_EXACT_FINITE_ARITHMETIC_PASS',hashlib.sha256(p.read_bytes()).hexdigest(),result['scores_decimal_70digits'])

if __name__=='__main__':test_and_save()
