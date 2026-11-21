#!/usr/bin/env python3
"""A four-world exact XOR example of genuinely complementary computations."""
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

def evaluate():
    worlds=[(x,y,x^y) for x in (0,1) for y in (0,1)]
    assert len(worlds)==4
    def optimal_expected_accuracy(observed):
        partitions=defaultdict(list)
        for i,(b1,b2,z) in enumerate(worlds):
            key={'none':None,'B1':b1,'B2':b2,'both':(b1,b2)}[observed]
            partitions[key].append(z)
        return sum(Fraction(max(count.count(0),count.count(1)),4)
                   for count in partitions.values())
    without=optimal_expected_accuracy('none')
    after1=optimal_expected_accuracy('B1')
    after2=optimal_expected_accuracy('B2')
    both=optimal_expected_accuracy('both')
    cost=Fraction(1,10)
    assert (without,after1,after2,both)==(Fraction(1,2),Fraction(1,2),Fraction(1,2),Fraction(1))
    assert both-without-2*cost==Fraction(3,10)>0
    assert after1-without-cost==Fraction(-1,10)<0
    result={'schema':'quartz_cog02_dynamic10_exact_xor_synergy_v1',
      'worlds':[{'B1':x,'B2':y,'target_Z_XOR':z,'probability':'1/4'} for x,y,z in worlds],
      'prior_optimal_action_utility':str(without),
      'after_B1_optimal_utility':str(after1),
      'after_B2_optimal_utility':str(after2),
      'after_B1_and_B2_optimal_utility':str(both),
      'myopic_gross_information_gain':str(after1-without),
      'joint_two_observation_gross_information_gain':str(both-without),
      'per_observation_cost':str(cost),
      'one_step_net_gain':str(after1-without-cost),
      'two_step_net_gain':str(both-without-2*cost),
      'interpretation':'exact Bayes information synergy in abstract XOR four-world model, not a theorem of QUARTZ MCTS'}
    return result

if __name__=='__main__':
    p=Path(__file__).with_name('EXACT_XOR_SYNERGY.json')
    if p.exists():raise FileExistsError('refuse overwriting an exact derived result')
    r=evaluate()
    p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('XOR_FOUR_WORLD_DERIVATION',json.dumps(r,sort_keys=True))
