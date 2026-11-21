#!/usr/bin/env python3
"""Independent exact Fraction enumeration for finite QUARTZ computation-value semantics."""
from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
import hashlib,json

belief=(F(1,2),F(3,10),F(1,5))
utilities=((F(1),F(0),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(1)))
# likelihoods: state rows, outcome columns
partial=((F(1),F(0)),(F(1),F(0)),(F(0),F(1)))
perfect=((F(1),F(0),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(1)))
null=((F(1,2),F(1,2)),)*3
costs=(F(1,20),F(1,5))
def terminal_value(b,u):
    return max(sum(bi*ui[a] for bi,ui in zip(b,u)) for a in range(len(u[0])))
def risk(b,u):
    oracle=sum(bi*max(ui) for bi,ui in zip(b,u))
    return oracle-terminal_value(b,u)
def posteriors(b,likelihood):
    for row in likelihood:
        assert sum(row)==1 and all(q>=0 for q in row)
    assert len(b)==len(likelihood)
    for y in range(len(likelihood[0])):
        mass=sum(bi*row[y] for bi,row in zip(b,likelihood))
        if mass>0:
            next_b=tuple(bi*row[y]/mass for bi,row in zip(b,likelihood))
            assert sum(next_b)==1
            yield mass,next_b
def evsi(b,u,likelihood):
    return sum(mass*terminal_value(next_b,u) for mass,next_b in posteriors(b,likelihood))-terminal_value(b,u)
def optimal(b,u,channel_and_cost,h,lam=F(1)):
    if h==0:return risk(b,u),"STOP"
    options=[(risk(b,u),"STOP")]
    for idx,(channel,cost) in enumerate(channel_and_cost):
        cont=lam*cost+sum(prob*optimal(next_b,u,channel_and_cost,h-1,lam)[0]
                            for prob,next_b in posteriors(b,channel))
        options.append((cont,idx))
    return min(options,key=lambda row:row[0])
def main():
    G=(evsi(belief,utilities,partial),evsi(belief,utilities,perfect))
    ratios=(G[0]/costs[0],G[1]/costs[1])
    net=(G[0]-costs[0],G[1]-costs[1])
    assert risk(belief,utilities)==F(1,2)
    assert G==(F(1,5),F(1,2))
    assert ratios==(F(4),F(5,2))
    assert net==(F(3,20),F(3,10))
    assert ratios[0]>ratios[1] and net[1]>net[0]
    assert all((g/k>1)==(g-k>0) for g,k in zip(G,costs))
    channel_cost=((partial,costs[0]),(perfect,costs[1]))
    solution=[optimal(belief,utilities,channel_cost,h) for h in range(4)]
    assert solution[0]==(F(1,2),"STOP")
    assert solution[1]==(F(1,5),1)
    assert solution[2]==solution[1]
    assert solution[3]==solution[1]
    assert evsi(belief,utilities,null)==0
    # Permuting terminal action columns preserves Bayes values (even if selected ID changes).
    for perm in permutations(range(3)):
        mutated=tuple(tuple(ui[j] for j in perm) for ui in utilities)
        assert risk(belief,mutated)==risk(belief,utilities)
        assert evsi(belief,mutated,partial)==G[0]
        assert evsi(belief,mutated,perfect)==G[1]
    # Relabelling hidden states and all probability/channel rows preserves outcomes.
    for perm in permutations(range(3)):
        b=tuple(belief[i] for i in perm)
        u=tuple(utilities[i] for i in perm)
        ch=tuple(partial[i] for i in perm)
        assert risk(b,u)==F(1,2)
        assert evsi(b,u,ch)==G[0]
    # Costs converted to smaller units while inverse price changes, no policy change.
    scaled=((partial,costs[0]*1000),(perfect,costs[1]*1000))
    assert optimal(belief,utilities,scaled,1,lam=F(1,1000))==solution[1]
    # A perfect experiment is at least as informative as its binary garbling.
    assert G[1]>=G[0]>=evsi(belief,utilities,null)==0
    # Dimensionally distinct: no out-of-support observations enter Bayes update.
    degenerate=((F(1),F(0)),(F(1),F(0)),(F(1),F(0)))
    assert evsi(belief,utilities,degenerate)==0
    report={
       "schema":"quartz_general_meta17_R07_exact_finite_Bayes_v1",
       "status":"PASS",
       "n_hidden_states":3,"n_terminal_actions":3,"n_compute_choices":2,
       "source_python_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "posterior":list(map(str,belief)),
       "initial_posterior_simple_regret":"1/2",
       "gross_EVSI":list(map(str,G)),
       "compute_cost":list(map(str,costs)),
       "index_gain_per_cost":list(map(str,ratios)),
       "net_gain":list(map(str,net)),
       "index_choice":"binary_partial",
       "net_optimal_choice":"complete_observation",
       "finite_horizon_value":[str(x[0]) for x in solution],
       "finite_horizon_choice":[str(x[1]) for x in solution],
       "one_step_stop_sign_unchanged":True,
       "action_permutations_tested":6,
       "latent_state_permutations_tested":6,
       "uninformative_observation_EVSI":"0",
       "cost_unit_invariance":"PASS",
       "garbling_information_order":"PASS",
       "source_reuse":"not a game or MCTS implementation"
    }
    out=Path(__file__).with_name("R07_EXACT_RESULTS.json")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
