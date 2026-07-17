# 05 — Physics-Inspired Shadows, Falsification, and Evidence Discipline

Covered axes: **A21, A22, A26**, plus the analysis-only interpretation of A17,
A25, and the free-energy/RG/open-system lineage.

The purpose of this document is to keep the original physical intuitions alive
without letting them bypass empirical or mathematical contracts.  A physical
analogy may generate an observable or operator; it does not inherit a physical
theorem merely because symbols look similar.

## Claim classes

Every analysis artifact assigns one of:

```text
EXACT_IN_RESTRICTED_MODEL
  proven for a clearly stated finite/stochastic surrogate
STRUCTURAL_ANALOGY
  useful correspondence with explicit points of failure
OPERATOR_INSPIRATION
  inspired a measurable algorithmic feature
FALSIFICATION_ONLY
  diagnostic that may reject the analogy
```

No analysis-only class may modify online search.

---

## A21 — Coherence-gated signed path disagreement shadow

### Question

Does retaining a bounded two-dimensional signed memory of path agreement and
disagreement predict future root revisions beyond ordinary scalar features?

### Safe representation

For path `p`, define a real vector

```text
z_p = magnitude_p [cos phase_p, sin phase_p].
```

Magnitude and phase must be built from observable quantities such as confidence,
prediction-error sign, prior-value disagreement, player parity, or revision
history.  They are not quantum amplitudes.

A coherence gate decays the feature as the root stabilizes.  The skeleton uses
H1 and visit count only as a placeholder; the real gate is fitted and tested
against null features.

### Skeletons

- Rust: `A21CoherenceSignedPathShadow`
- Python: `A21CoherenceSignedPathShadow`

### Required test

Nested, position-grouped prediction models:

```text
base = margin, entropy, H1, P_flip, uncertainty, revisions
base + signed-path feature
```

Promote only if the added feature improves held-out late-flip/regret prediction
and remains bounded, symmetry-compatible, and cheap.  Even then it remains an
operator-inspired feature unless a stronger theorem is proved.

---

## A22 — Physics-analogy falsification dashboard

### Question

Which thermodynamic, RG-like, redundancy, and open-system surrogates actually
fit root-search traces, and where do they fail?

### Dashboard panels

#### Effective inverse temperature

Fit

```text
beta_eff = argmin_beta KL(pi_visits || softmax(beta * score))
```

and always report the residual KL.  A monotone beta without a small residual
does not establish a thermal model.

#### Redundancy / Darwinism surrogate

Create independent or Dirichlet-bootstrap simulation fragments and measure
fragment-size versus full-decision agreement, information, and H1 stability.
Call this classical decision-information redundancy, not quantum mutual
information unless an actual density-operator model is defined.

#### RG-like scale flow

Track entropy, effective branching, Q gaps, curvature/readout correction, and
candidate mass over geometric budgets.  Test explicit rescaling/collapse
hypotheses against null/smooth alternatives.  A visually straight log-log line
is not enough.

#### Susceptibility

On near-tie roots, perturb a root score/prior by a preregistered small field and
measure the change of an order parameter such as top-two policy difference.
Control for finite-budget and saturation effects.

#### Nonequilibrium tests

Do not test FDT, Crooks, or Jarzynski unless all of the following are defined:

- state/energy function;
- forward transition kernel and protocol;
- reverse protocol/kernel;
- work/heat functional;
- equilibrium or stationary reference;
- support/absolute-continuity conditions.

Otherwise record `NOT_APPLICABLE_MODEL_UNDEFINED`, not a failed or successful
physical law.

### Skeletons

- Rust: `A22PhysicsFalsificationDashboard` emits no online action.
- Python: `A22PhysicsFalsificationDashboard` defines trace inputs.

### Artifacts

Machine-readable panel results, null-model definitions, fit residuals,
bootstrap intervals, and an explicit conclusion per hypothesis.

---

## A26 — Exact nested-contour lab

A26 is described in the candidate document.  Its role here is to ensure that
“nested sampling” language is only used where prior mass, likelihood/score,
constrained replacement, and ground truth are explicit.  Approximate live-set
search belongs to A11 and receives no evidence-estimation guarantee.

---

## Free energy and path measures

The defensible shared foundation is a classical KL-regularized control/search
objective:

```text
F[q] = - E_q[return]
       + lambda E_q[compute cost]
       + tau KL(q || proposal/anchor path measure).
```

Its finite-action variational solution supports A02 and parts of A25.  It does
not prove a Keldysh, Lindblad, quantum-Darwinism, or Jarzynski model of the live
Rust search.

## Open-system and non-equilibrium lineage

The original forward selection / backward backup, pending workers,
interference, decoherence, and pointer-state ideas may be recast as questions:

- Does a closed-time-contour bookkeeping view expose useful forward/backward
  trace features?
- Does path diversity act like a repulsive scheduler and reduce duplicate
  computation?
- Does decision stability have a calibrated decay/relaxation time?
- Can a classical stochastic-approximation or MSRJD surrogate predict root
  covariance or scheduler settings?

These questions live in A21/A22 until they predict something beyond existing
statistics.

## Geometry / consistency

Graph geometry is initially restricted to measurable engineering quantities:

- transposition path inconsistency;
- state-sharing gain;
- path semantic overlap;
- cheap Jaccard/Forman-like graph summaries as optional teacher features.

Runtime Ollivier-Ricci calculations or artificially inserted loops are not core
requirements.  Any geometry head must demonstrate incremental predictive value
against depth, branching, entropy, gap, and uncertainty baselines.

## Evidence discipline

A positive fit is not an online improvement.  The ladder is:

```text
observable is numerically well-defined
→ survives invariance/null tests
→ predicts held-out search behavior
→ provides a new proposal or calibration signal
→ improves paired online decision quality per cost
```

Failure at one role is recorded at `(axis, role, game, budget, evaluator,
hardware)` scope; it does not erase the broader source idea.
