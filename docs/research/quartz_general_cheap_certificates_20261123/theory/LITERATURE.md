# C24 R02 — primary floating-point evidence and reuse

Scope: R02 only; no production changes, cohort construction, benchmark or publication.
skill_context_used: PhysMath Research Loop 1.3.0 cloud c43/physmath-research-loop/SKILL.md; references/THEORY_LOOP.md R2; references/POLICY_INTEGRATION.md. Applied primary-source formula/assumption mapping, bounded search, typed source identity and no repeated broad audit. Also read Exa Search/SKILL.md and references/searching.md; Parallel parallel-web-search/SKILL.md.

## Evidence actually read

### 1. Rump (2012), Error estimation of floating-point summation and dot product
BIT Numerical Mathematics 52, 201–220. DOI 10.1007/s10543-011-0342-4.
Author accepted-manuscript PDF: https://www.tuhh.de/ti3/paper/rump/Ru11.pdf

Read §2, §3 Algorithm3.2/Theorem3.5/Corollary3.7, and §4 Algorithm4.2/Theorem4.3/Corollary4.4 including the underflow cases. Binary64 conventions: unit roundoff u=2^-53, smallest subnormal eta=2^-1074, realmin=2^-1022. Overflow is excluded; gradual underflow is included. Algorithm4.2 accumulates separately rounded products p_i, their signed sum s_hat, and their absolute sum T_hat in the same sequential order. With 2(n+2)u<=1, Corollary4.4 proves that the actually computed round-to-nearest quantity

R = fl(fl(fl((n+2)*u)*ufp(T_hat)) + realmin)

bounds |s_hat-x^T y|. ufp(T_hat) is the largest power of two no greater than |T_hat|, and ufp(0)=0. Theorem4.3 gives the underlying real-valued bound (n+2)u*ufp(T_hat)+n*eta/2. The additive underflow term is essential; directly computing eta/2 in f64 would lose it. Corollary4.4 deliberately uses realmin.

C24 applicability: ordinary dot products plus one absolute-product accumulator and scalar final error evaluation can replace interval arithmetic inside every dot operation. This does not by itself bound errors already present in Ax or r; those still require composition.

### 2. Jeannerod and Rump (2013), Improved Error Bounds for Inner Products in Floating-Point Arithmetic
SIAM Journal on Matrix Analysis and Applications 34(2),338–344. DOI 10.1137/120894488.
Author PDF: https://www.tuhh.de/ti3/paper/rump/JeaRu13.pdf
Author repository: https://inria.hal.science/hal-00840926
SciSpace discovery: https://scispace.com/papers/improved-error-bounds-for-inner-products-in-floating-point-1ue37y11na

Read Proposition4.1, Theorem4.2, Corollary5.1 and §5 matrix-vector consequences. Theorem4.2 proves |dot_hat-x^T y|<=n*u*|x|^T|y| for any summation order and any n, explicitly assuming no underflow or overflow. §5 obtains componentwise matrix-vector backward error |Delta A|<=n*u*|A|. For an implemented sparse row containing k actual terms, the same theorem applies in dimension k.

C24 applicability: a simple maximum-based bound for each five-term stencil is literature-supported when its range assumptions are actually established. This result is not an underflow-inclusive certificate. The paper itself supplies the nonzero-product-to-zero counterexample. Do not transport its n*u bound into the C23 tiny-RHS fixture without a domain proof or an additive underflow correction. A bound expressed in an exact absolute sum still needs safe numerical evaluation.

### 3. Reference BLAS DNRM2 — actual safe norm implementation
Official source read in full: https://www.netlib.org/blas/dnrm2.f90
Line-numbered version: https://netlib.org/lapack/explore-html/d6/de0/dnrm2_8f90_source.html

Read thresholds/scales at lines104–111, the three accumulation branches131–156, combination158–196 and final norm197. The code separates large, ordinary and small magnitudes; small values are scaled up before squaring, large ones down. It reconstructs a norm as scale*sqrt(sumsq), instead of forming an unsafe raw sum of squares.

The source identifies Anderson, Algorithm978: Safe Scaling in the Level1 BLAS (2017), DOI10.1145/3061665, and Blue (1978), DOI10.1145/355769.355771. Publisher metadata and abstract were read. The full Anderson PDF fetch timed out; no claim of full-paper verification is made.

C24 applicability: replace unsafe diagnostic sqrt(dot(v,v)) with a scaled norm, or explicitly reject unsupported nonzero-norm inputs. This is a stable norm algorithm, not a rigorous outward residual certificate by itself.

### 4. LAPACK DLASSQ — retaining the scale instead of immediately squaring it back
Official API/source:
https://netlib.org/lapack/explore-html/d8/d76/group__lassq_gae8f40b0a34771b4f2d9c863de3af7be5.html
https://www.netlib.org/lapack/explore-html/db/d2e/dlassq_8f90_source.html

Read the scaled-sum contract and implementation branches156–181, incoming scaled quantity183–208, combination210–247. It retains a (scale,sumsq) representation. C24 should not take a safely computed very small norm and immediately square it back into an underflowing rr before evaluating a ratio. Updating only the displayed norm cannot repair CG alpha=rr/(p.Ap); solver arithmetic needs consistent scaling or a declared range restriction. This last implication is a direct application to the C23 algorithm, not a claim stated by LAPACK.

## Constructive synthesis for the parent

There are two legitimate cheap-enclosure choices:
A. Rump's computed a posteriori bound: additional absolute-product accumulation but no per-element interval endpoints; explicit gradual-underflow coverage.
B. Classical/Jeannerod-Rump maximum-based envelope: fewer extra reductions, potentially more conservative; range assumptions or derived additive-underflow terms must be discharged.

For any chosen approximation y_hat to Ax with rigorous component error e_y, a rigorous response enclosure must propagate both:
- dot/sum rounding at the current stored vectors;
- operator error through x^T(y_hat-Ax) and residual error through ||r||².

The residual-energy identity q-ell=r^T A^-1 r stays unchanged. Literature validity does not establish a faster whole policy. The C24 experiment must charge the estimator itself and compare with PLAIN_ALWAYS and an actual plain initial-STOP.

The C23 regression fixture b0=(2^-600,0,0,0), b1=0, rho1,side2,S=2^-1000 has true relative residual1 at x=0. A safe diagnostic can expose this; consistent solver scaling or explicit rejection is still needed to prevent a falsely successful solve.

## Search provenance and limitations

- Exa: three focused searches, numResults5 each (15 returned-result slots); fetched author Rump and Jeannerod–Rump PDFs and official DNRM2 source. These counts are discovery slots, not15 independently verified papers.
- SciSpace: one natural-language research search,6 results; it independently surfaced the Jeannerod–Rump no-underflow theorem and Neumaier's1985 underflow analysis. Only the directly read author PDFs/official source carry formula-level claims here.
- Parallel: two focused safe-scaling searches. IDs search_062e5251d3b0c1876c369b25ffe40bf9 and search_0967ba49fe9e7eaeb5ec62a695f3a345. Found official DNRM2/DLASSQ and publisher record.
- Actual nonblocking fetch failure: Exa fetch of https://dl.acm.org/doi/pdf/10.1145/3061665 returned CRAWL_LIVECRAWL_TIMEOUT. No blind retry, authentication change or external report was performed; official code was fully available.
- No speedup, global novelty, or all-f64-domain convergence claim is admitted by this literature task.
