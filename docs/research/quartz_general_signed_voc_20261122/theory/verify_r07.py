#!/usr/bin/env python3
"""Exact Fraction checks for C19 signed utility/cost and interval selection semantics."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import hashlib,json
q=F(1,4);pos=F(1,2);neg=-pos;cost=F(1,10)
assert q*pos-cost==F(1,40)
assert q*neg-cost==-F(9,40)
q_bounds=(F(1,4),F(3,4))
m_bounds=(-F(1,2),F(1,2))
extrema=[p*m-cost for p,m in product(q_bounds,m_bounds)]
assert min(extrema)==-F(19,40) and max(extrema)==F(11,40)
# Finite two-state terminal action exact EVSI; total posterior decision gain nonnegative.
prior=(F(1,2),F(1,2))
U=((F(1),F(0)),(F(0),F(1)))
M0=max(sum(prior[i]*U[i][a] for i in (0,1)) for a in (0,1))
M1=sum(prior[i]*max(U[i]) for i in (0,1))
assert (M0,M1,M1-M0)==(F(1,2),F(1),F(1,2))
# Triangle error bound at fixed numbers (illustration not universal numeric proof).
qp=F(1,4); qh=F(1,3);mp=F(1,2);mh=F(-1,5)
lhs=abs(qp*mp-qh*mh)
rhs=abs(qp-qh)+abs(mp-mh)
assert lhs<=rhs
p=Path(__file__)
record={"schema":"quartz_general_c19_R07_exact_fraction_v1","status":"PASS",
"same_event_probability":str(q),"positive_signed_conditional_gain":str(pos),
"negative_signed_conditional_gain":str(neg),
"positive_net":str(q*pos-cost),"negative_net":str(q*neg-cost),
"joint_uncertain_q_interval":list(map(str,q_bounds)),
"joint_uncertain_m_interval":list(map(str,m_bounds)),
"interval_net_lower":str(min(extrema)),"interval_net_upper":str(max(extrema)),
"finite_perfect_information_EVSI":str(M1-M0),"error_triangle_bound_sample":"PASS",
"source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
"claim_status":"exact mathematical illustration; not MCTS model calibration"}
out=p.with_name("R07_EXACT_CHECKS.json")
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print(json.dumps(record,sort_keys=True))
