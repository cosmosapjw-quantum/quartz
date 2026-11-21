# R5. Exact high-branching candidate-coverage theorem and root bitset lemma

THEOREM A: Fix legal move set A of size B, original candidate subset A0 of size k, with m decisive excluded moves D inside the n=B-k omitted positions. Choose r of those n omitted moves uniformly without replacement. Exactly binomial(n,r) equally likely subsets exist. A sampled subset misses all m decisive positions iff all r are selected from n-m nondecisive positions, of which binomial(n-m,r) subsets exist. Therefore
P_hit = 1 - binomial(n-m,r)/binomial(n,r).
If m=1 this is r/n. This is an exact discovery probability only, not final MCTS choice or game strength.

B=7,k=3,r=2,m=1 gives P_hit=1/2. B=225,k=16,r=4,m=1 gives P_hit=4/209≈0.019138756. For m=5 at B225 the exact value is 2393375/25746292≈0.092959988. At high B, successful tactical detection of a rare omitted decisive action could create a large discovery contrast relative to uninformed same-count reopening, conditional on evidence recall.

THEOREM B: For any subset S of indices 0..224, define four words M_j(S) as the sum of 2^(a mod 64) for all a in S with floor(a/64)=j. The map S↦(M_0,M_1,M_2,M_3) is injective by the uniqueness of binary expansion in disjoint words. S⊆T iff for each j, bitand(M_j(S),M_j(T))=M_j(S). Index224 resides in word3 bit32 and fits [u64;4]. A zero-word array is reserved as the unrestricted sentinel, while an explicit zero-action API input must be rejected.

Conditional expected-net-utility comparison for a single decisive action: let detection q, selection conditional on discovery t, random discovery r/(B-k), its conditional selection t0, action utility delta>0, and costs Ce,Cr. Under the strong assumption no other outcome differences, targeted search is better in expectation only when delta*(q*t - r/(B-k)*t0) > lambda*(Ce-Cr). q,t,t0 and realized costs must be measured, not assumed. This is NOT an unconditional Bayesian VOC theorem.
