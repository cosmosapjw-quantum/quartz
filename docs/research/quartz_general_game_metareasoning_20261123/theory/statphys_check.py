#!/usr/bin/env python3
"""C26 R07: exact algebra and bounded numerical checks, no project imports."""

from __future__ import annotations
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy.special import logsumexp
import sympy as s

BASE = Path("/tmp/quartz_c26_statphys")
CHECKS = []


def require(name, condition, detail=None):
    if not bool(condition):
        raise AssertionError(f"{name}: {detail}")
    CHECKS.append({"name": name, "status": "PASS", "detail": detail})


def zero(M):
    return all(s.simplify(x) == 0 for x in M)


def serial(M):
    if isinstance(M, s.MatrixBase):
        return [[str(M[i, j]) for j in range(M.cols)] for i in range(M.rows)]
    return str(M)


def f64(M):
    return np.array(M, dtype=float)


def kl(p, q):
    return math.fsum(
        float(a) * math.log(float(a) / float(b)) for a, b in zip(p, q) if a > 0
    )


def exact_kl(p, q):
    return s.Add(*(a * s.log(a / b) for a, b in zip(p, q) if a != 0))


def generator(q, kappa=s.Rational(1)):
    m = len(q)
    W = s.zeros(m)
    for a in range(m):
        for b in range(m):
            if a != b:
                W[a, b] = kappa * s.Min(1, q[a] / q[b]) / (m - 1)
    for b in range(m):
        W[b, b] = -sum(W[a, b] for a in range(m) if a != b)
    return W


q = s.Matrix([s.Rational(1, 6), s.Rational(1, 3), s.Rational(1, 2)])
pi = s.ones(3, 1) / 3
tau = s.Rational(1, 5)
kappa = s.Rational(1)
delta = s.Rational(1, 2)
W = generator(q, kappa)
P = s.eye(3) + delta * W
p = s.Matrix([s.Rational(3, 5), s.Rational(3, 10), s.Rational(1, 10)])
require("exact_generator_column_sums", zero(s.ones(1, 3) * W))
require("exact_stationary_gibbs", zero(W * q))
require("exact_detailed_balance", zero(W * s.diag(*q) - s.diag(*q) * W.T))
require("exact_transition_nonnegative", all(x >= 0 for x in P))
require("exact_transition_column_stochastic", zero(s.ones(1, 3) * P - s.ones(1, 3)))
require("exact_transition_stationary", zero(P * q - q))
require("exact_transition_reversible", zero(P * s.diag(*q) - s.diag(*q) * P.T))
require("exact_transition_strictly_positive", all(x > 0 for x in P))
Q = s.Matrix([tau * s.log(q[i] / pi[i]) for i in range(3)])
variational_gap = s.expand_log(
    -p.dot(Q) + tau * exact_kl(p, pi) - tau * exact_kl(p, q), force=True
)
require("exact_gibbs_variational_gap_identity", s.simplify(variational_gap) == 0)
H = (s.diag(*q) - q * q.T) / tau
one = s.ones(3, 1)
Ccommon = s.Rational(1, 16) * one * one.T
require("exact_susceptibility_common_mode_null", zero(H * one))
require("exact_covariance_common_mode_contraction_zero", s.trace(H * Ccommon) == 0)
require("exact_hessian_positive_semidefinite", H.is_positive_semidefinite is True)

# General matrix entries permit checking diagonal closure without assuming a
# diagonal input or inserting a special density matrix.
r = s.Matrix(3, 3, lambda i, j: s.Symbol(f"r{i}{j}", real=True))
D = s.zeros(3)
Ks = []
for a in range(3):
    for b in range(3):
        if a == b:
            continue
        L = s.zeros(3)
        L[a, b] = s.sqrt(W[a, b])
        A = L.T * L
        D += L * r * L.T - (A * r + r * A) / 2
        Ks.append(s.sqrt(delta) * L)
exit_rates = s.Matrix([-W[b, b] for b in range(3)])
require(
    "exact_offdiagonal_jumps_population_generator",
    zero(
        s.Matrix([D[a, a] for a in range(3)])
        - W * s.Matrix([r[a, a] for a in range(3)])
    ),
)
require(
    "exact_offdiagonal_jumps_coherence_damping",
    all(
        s.simplify(D[a, b] + (exit_rates[a] + exit_rates[b]) * r[a, b] / 2) == 0
        for a in range(3)
        for b in range(3)
        if a != b
    ),
)
require("exact_lindblad_trace_preservation", s.simplify(s.trace(D)) == 0)
require(
    "exact_lindblad_gibbs_stationarity",
    zero(
        D.subs({r[a, b]: (q[a] if a == b else 0) for a in range(3) for b in range(3)})
    ),
)
gamma = [s.Rational(1), s.Rational(2), s.Rational(3)]
Ddiag = s.zeros(3)
for a in range(3):
    L = s.zeros(3)
    L[a, a] = s.sqrt(gamma[a])
    A = L.T * L
    Ddiag += L * r * L.T - (A * r + r * A) / 2
require(
    "exact_diagonal_jumps_zero_population_change",
    all(s.simplify(Ddiag[a, a]) == 0 for a in range(3)),
)
require(
    "exact_diagonal_jumps_only_dephase",
    all(
        s.simplify(Ddiag[a, b] + (gamma[a] + gamma[b]) * r[a, b] / 2) == 0
        for a in range(3)
        for b in range(3)
        if a != b
    ),
)
# A genuine CPTP finite map, not an unqualified Euler Lindblad superoperator.
K0 = s.diag(*(s.sqrt(1 - delta * exit_rates[b]) for b in range(3)))
Ks.insert(0, K0)
complete = sum((K.T * K for K in Ks), s.zeros(3))
finite = sum((K * r * K.T for K in Ks), s.zeros(3))
require("exact_kraus_completeness", zero(complete - s.eye(3)))
require(
    "exact_kraus_diagonal_matches_finite_P",
    zero(
        s.Matrix([finite[a, a] for a in range(3)])
        - P * s.Matrix([r[a, a] for a in range(3)])
    ),
)
# Rational probabilities remain exact; only the logarithms are evaluated at
# high precision. The contraction theorem is separately proved in the note.
rat_trace = []
pcur = p
for t in range(13):
    d = exact_kl(pcur, q)
    rat_trace.append(
        {"step": t, "p": [str(x) for x in pcur], "KL_70digits": str(s.N(d, 70))}
    )
    if t:
        diff = s.N(prev - d, 70)
        require(f"rational_step_{t}_KL_strict_contraction", diff > 0, str(diff))
    prev = d
    pcur = P * pcur

# Numeric Gibbs target requested by parent. All tolerances fixed ex ante.
Qn = np.array([0.0, 0.2, -0.1])
pin = np.ones(3) / 3.0
taun = 0.1
logZ = float(logsumexp(np.log(pin) + Qn / taun))
qn = np.exp(np.log(pin) + Qn / taun - logZ)
Wn = np.zeros((3, 3))
for a in range(3):
    for b in range(3):
        if a != b:
            Wn[a, b] = 0.5 * min(1.0, qn[a] / qn[b])
for b in range(3):
    Wn[b, b] = -sum(Wn[a, b] for a in range(3) if a != b)
Pn = np.eye(3) + 0.5 * Wn
require("numeric_stationarity", np.max(abs(Wn @ qn)) < 2e-15)
require(
    "numeric_detailed_balance",
    np.max(abs(Wn @ np.diag(qn) - np.diag(qn) @ Wn.T)) < 2e-15,
)
require("numeric_column_stochastic", np.max(abs(Pn.sum(axis=0) - 1.0)) < 2e-15)
require("numeric_transition_nonnegative", Pn.min() >= 0)
pn = np.array([0.6, 0.3, 0.1])
num_trace = []
for t in range(81):
    d = kl(pn, qn)
    num_trace.append({"step": t, "p": pn.tolist(), "KL": d})
    pn = Pn @ pn
require(
    "numeric_KL_contraction_80_steps",
    all(b["KL"] <= a["KL"] + 3e-15 for a, b in zip(num_trace, num_trace[1:])),
)
require(
    "numeric_final_approaches_stationary",
    max(abs(np.array(num_trace[-1]["p"]) - qn)) < 1e-10,
)
pn = np.array([0.6, 0.3, 0.1])
Fstar = taun * logZ
J = float(pn @ Qn) - taun * kl(pn, pin)
gap_identity_residual = Fstar - J - taun * kl(pn, qn)
require("numeric_gibbs_variational_gap", abs(gap_identity_residual) < 2e-15)

# Exact finite posterior mechanics: theta in {0,1}, utility vector
# U(theta)=(theta,1-theta,1/4). Compute channel has accuracy alpha.
prior = [s.Rational(1, 2)] * 2
U = [s.Matrix([0, 1, s.Rational(1, 4)]), s.Matrix([1, 0, s.Rational(1, 4)])]
mu = sum((prior[i] * U[i] for i in range(2)), s.zeros(3, 1))
Ctotal = sum((prior[i] * (U[i] - mu) * (U[i] - mu).T for i in range(2)), s.zeros(3))


def channel_case(alpha):
    likelihood = [[alpha, 1 - alpha], [1 - alpha, alpha]]
    outcomes = []
    Cmeans = s.zeros(3)
    Econditional = s.zeros(3)
    Emean = s.zeros(3, 1)
    for y in range(2):
        mass = sum(prior[i] * likelihood[i][y] for i in range(2))
        post = [prior[i] * likelihood[i][y] / mass for i in range(2)]
        muy = sum((post[i] * U[i] for i in range(2)), s.zeros(3, 1))
        cov = sum(
            (post[i] * (U[i] - muy) * (U[i] - muy).T for i in range(2)), s.zeros(3)
        )
        Emean += mass * muy
        Cmeans += mass * (muy - mu) * (muy - mu).T
        Econditional += mass * cov
        outcomes.append({"mass": mass, "posterior": post, "mu": muy})
    require(f"posterior_martingale_alpha_{alpha}", zero(Emean - mu))
    require(
        f"total_covariance_decomposition_alpha_{alpha}",
        zero(Ctotal - Cmeans - Econditional),
    )
    evsi = sum(z["mass"] * max(z["mu"]) for z in outcomes) - max(mu)
    smooth = lambda v: taun * float(logsumexp(np.log(pin) + f64(v).ravel() / taun))
    smooth_gain = math.fsum(
        float(z["mass"]) * smooth(z["mu"]) for z in outcomes
    ) - smooth(mu)
    qm = np.exp(
        np.log(pin)
        + f64(mu).ravel() / taun
        - logsumexp(np.log(pin) + f64(mu).ravel() / taun)
    )
    Hm = (np.diag(qm) - np.outer(qm, qm)) / taun
    second = 0.5 * float(np.trace(Hm @ f64(Cmeans)))
    wrong = 0.5 * float(np.trace(Hm @ f64(Ctotal)))
    bound = math.fsum(
        float(z["mass"])
        * float(max(z["mu"] - mu) - min(z["mu"] - mu)) ** 3
        / (6 * taun**2)
        for z in outcomes
    )
    require(f"jensen_nonnegative_alpha_{alpha}", smooth_gain >= -2e-15)
    require(
        f"third_order_remainder_bound_alpha_{alpha}",
        abs(smooth_gain - second) <= bound + 3e-15,
    )
    return {
        "alpha": str(alpha),
        "posterior_means": [serial(z["mu"]) for z in outcomes],
        "mean_covariance": serial(Cmeans),
        "expected_conditional_covariance": serial(Econditional),
        "EVSI_exact": str(evsi),
        "smooth_gain": smooth_gain,
        "second_order_gain": second,
        "wrong_total_latent_variance_gain": wrong,
        "rigorous_abs_remainder_bound": bound,
        "absolute_approximation_error": abs(smooth_gain - second),
    }


cases = [channel_case(s.Rational(1, 2)), channel_case(s.Rational(3, 4))]
require(
    "uninformative_channel_zero_mean_covariance",
    cases[0]["mean_covariance"] == serial(s.zeros(3)),
)
require("uninformative_channel_zero_EVSI", cases[0]["EVSI_exact"] == "0")
require("uninformative_channel_zero_smooth_gain", abs(cases[0]["smooth_gain"]) < 2e-15)
require(
    "latent_variance_would_spuriously_predict_gain",
    cases[0]["wrong_total_latent_variance_gain"] > 0,
)
require("informative_noisy_channel_EVSI_quarter", cases[1]["EVSI_exact"] == "1/4")
small_signal = [
    channel_case(s.Rational(1, 2) + s.Rational(1, n)) for n in (64, 32, 16, 8)
]
# An informative computation whose uncertainty is a common utility shift.
# theta = +/-1/4 revealed perfectly, action 0 strictly best for both theta.
common_base = s.Matrix([2, 1, 0])
xvals = [-s.Rational(1, 4), s.Rational(1, 4)]
common_mu = [common_base + x * one for x in xvals]
common_evsi = sum(max(v) for v in common_mu) / 2 - max(common_base)
common_var = (
    sum(((v - common_base) * (v - common_base).T for v in common_mu), s.zeros(3)) / 2
)
require("informative_common_shift_zero_decision_gain", common_evsi == 0)
require("informative_common_shift_covariance", zero(common_var - Ccommon))
require("informative_common_shift_positive_information", s.N(s.log(2), 70) > 0)
# Common-shift equivariance is exact: F(mu+x1)=F(mu)+x.
common_H = (s.diag(*q) - q * q.T) / tau
require(
    "informative_common_shift_zero_second_order_gain",
    s.trace(common_H * common_var) == 0,
)

result = {
    "schema": "quartz.c26.r07.statphys.v1",
    "status": "ALL_ASSERTIONS_PASSED",
    "scope": "Exact symbolic and small deterministic calculations only; no game data, no repository edits.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "execution": {
        "command": "python3 /tmp/quartz_c26_statphys.py",
        "python": platform.python_version(),
        "sympy": s.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "expected_success_exit_code": 0,
    },
    "conventions": {
        "orientation": "column b is departure state; dp/dt=W p",
        "Q_tau": "utility units",
        "kappa": "inverse algorithmic relaxation time",
        "delta": "algorithmic time",
        "KL": "dimensionless nats",
        "Hamiltonian": "H=0; no hbar=1 convention introduced",
        "finite_map": "explicit Kraus CPTP map, not Euler approximation to full Lindblad superoperator",
    },
    "exact": {
        "q": serial(q),
        "pi": serial(pi),
        "tau": str(tau),
        "Q": serial(Q),
        "kappa": str(kappa),
        "delta": str(delta),
        "W": serial(W),
        "P": serial(P),
        "exit_rates": serial(exit_rates),
        "initial_p": serial(p),
        "hessian": serial(H),
        "rational_KL_trace": rat_trace,
    },
    "numeric": {
        "Q": Qn.tolist(),
        "pi": pin.tolist(),
        "tau": taun,
        "q": qn.tolist(),
        "W": Wn.tolist(),
        "P": Pn.tolist(),
        "Fstar": Fstar,
        "variational_gap_residual": gap_identity_residual,
        "KL_trace": num_trace,
    },
    "posterior_value": {
        "utility_states": [serial(u) for u in U],
        "prior": ["1/2", "1/2"],
        "mean": serial(mu),
        "latent_total_covariance": serial(Ctotal),
        "cases": cases,
        "small_signal": small_signal,
    },
    "positive_information_zero_gain": {
        "theta": ["-1/4", "1/4"],
        "prior": ["1/2", "1/2"],
        "utilities": [serial(v) for v in common_mu],
        "revealed": "theta",
        "mutual_information_exact": "log(2)",
        "mutual_information_nats": math.log(2),
        "decision_EVSI_exact": "0",
        "smooth_value_exact": "0",
        "posterior_mean_covariance": serial(common_var),
        "reason": "All actions share shift; best action unchanged.",
    },
    "checks": CHECKS,
    "claim_limits": [
        "No physical MCTS-Lindblad equivalence established.",
        "No engine performance or policy quality claim.",
        "Finite P is exact for selected discrete map, not exp(delta W).",
        "Cumulant term is second-order with stated remainder, not exact generally.",
        "Unregularized terminal EVSI and smoothed F_tau information value remain separate.",
    ],
}
BASE.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
md = f"""# C26 R07: constructive Gibbs relaxation and computational fluctuation checks

Status: **{len(CHECKS)} assertions passed** in one bounded Python execution. Exact algebra uses SymPy rationals. KL values on rational trajectories use 70-digit log evaluation; the separate contraction proof below supplies the universal statement. Numeric Gibbs checks use fixed tolerances; no project module, game input, production binary or campaign was run.

## 1. Units, definitions and target

There are m=3 states. pi_a>0 is a normalized prior, Q_a is utility, and tau>0 has the same utility units. Define q*_a=pi_a exp(Q_a/tau)/Z. KL is dimensionless and measured in nats. kappa has inverse algorithmic-time units; delta has algorithmic-time units. This is a mathematical stochastic construction. No identification with physical temperature, k_B, hbar, or actual MCTS dynamics is assumed. In the Lindblad construction H=0, so no natural-unit convention is needed.

The variational objective J(p)=p.Q-tau KL(p||pi) obeys exactly

    tau log Z - J(p) = tau KL(p||q*) >= 0.

This follows by substituting log(q*_a/pi_a)=Q_a/tau-log Z. The target is therefore the unique variational maximizer on the positive prior support. The rational target q*=(1/6,1/3,1/2), uniform pi and tau=1/5 is realized by Q_a=tau log(q*_a/pi_a), with Z=1. Both its exact symbolic identity and the requested numeric Q=(0,.2,-.1), tau=.1 identity were checked.

## 2. Constructive population dynamics

For a!=b, set W_ab=kappa/(m-1) min(1,q*_a/q*_b), where column b is the departure state. Set W_bb=-sum_(a!=b) W_ab. Then

    W_ab q*_b = kappa/(m-1) min(q*_a,q*_b) = W_ba q*_a.

Detailed balance plus zero column sums yields W q*=0. All offdiagonal rates are positive; the finite chain is irreducible. Its exit rates are <=kappa. For 0<=delta<=1/kappa, P=I+delta W is column stochastic and nonnegative; Pq*=q*. With 0<delta<1/kappa it has positive diagonal and all entries are positive. P is an exact selected discrete-time transition, not the continuous semigroup exp(delta W).

Exact W at kappa=1:

    {serial(W)}

Exact P at delta=1/2:

    {serial(P)}

For any distribution p, the log-sum inequality applied to each output row gives D(Pp||Pq*)<=D(p||q*); stationarity makes this D(Pp||q*)<=D(p||q*). For the strictly positive P and p!=q*, equality is excluded by the equality condition of log-sum. The supplied nonstationary rational trajectory satisfied strict contraction for all 12 inspected transitions. The numeric trajectory satisfied contraction over 80 transitions.

## 3. Lindblad correction and finite CPTP realization

Set L_ab=sqrt(W_ab)|a><b| for a!=b and use d rho/dt=sum_ab [L_ab rho L_ab^dagger-{{L_ab^dagger L_ab,rho}}/2], with H=0. For arbitrary matrix rho its populations obey

    d rho_aa/dt = sum_(b!=a) W_ab rho_bb - sum_(b!=a) W_ba rho_aa = (W p)_a.

For a!=b its coherences satisfy d rho_ab/dt=-(exit_a+exit_b)rho_ab/2. Thus diag(q*) is stationary. This equality was checked on symbolic entries of a general 3x3 matrix, not just on a special diagonal density matrix.

In contrast, diagonal jumps L_a=sqrt(gamma_a)|a><a| give d rho_aa/dt=0 and d rho_ab/dt=-(gamma_a+gamma_b)rho_ab/2. They only dephase. They cannot supply the population-balance/Gibbs derivation claimed for such jumps in the legacy notes.

A finite completely-positive trace-preserving channel with exactly the selected diagonal transition P is provided by K_ab=sqrt(delta W_ab)|a><b| and K_0=diag(sqrt(1-delta exit_b)). Directly sum K_j^dagger K_j=I, and the diagonal of sum K_j rho K_j^dagger is Pp. The check does NOT claim the naive full-superoperator Euler map I+delta L is CPTP, or that this Kraus channel equals exp(delta L).

## 4. Susceptibility and computation-specific covariance

Let F_tau(mu)=tau log sum_a pi_a exp(mu_a/tau) and q=softmax(log pi+mu/tau). Then grad F=q and H=(diag(q)-q q^T)/tau, a positive-semidefinite susceptibility matrix. H*1=0 exactly. For a calibrated observation Y, mu^Y=E[U|Y] has E mu^Y=mu. Its exact smoothed information value is

    G_tau=E[F_tau(mu^Y)]-F_tau(mu)>=0,

and its second-order term is (1/2) tr[H Cov(mu^Y)]. The expectation is over the *specified computation's outcomes*. It is not generally correct to insert the total latent Cov(U). The exact decomposition is

    Cov(U)=E Cov(U|Y)+Cov(E[U|Y]).

The script verifies this for theta in {{0,1}} with equal prior and U(theta)=(theta,1-theta,1/4). At channel accuracy alpha=3/4, Cov(mu^Y) has leading block [[1/16,-1/16],[-1/16,1/16]], whereas Cov(U)'s block is four times larger. Exact terminal EVSI is 1/4. At alpha=1/2 the channel is uninformative: Cov(mu^Y)=0, EVSI=0 and G_tau=0, despite nonzero Cov(U). Substituting latent variance gives a spurious positive second-order value of {cases[0]["wrong_total_latent_variance_gain"]:.12g}.

A useful finite remainder bound follows from f(t)=F_tau(mu+t v): f'''(t)=tau^(-2) E_qt[(v-E_qt v)^3]. If R(v)=max(v)-min(v), then |f'''(t)|<=R(v)^3/tau^2 and

    |G_tau - (1/2)tr[H Cov(mu^Y)]| <= E[R(mu^Y-mu)^3]/(6 tau^2).

It is a conservative controlled bound, not an asymptotic equality. The actual finite enumerations at alpha=1/2,3/4 and small excess accuracies 1/64,1/32,1/16,1/8 satisfy it. No approximation guarantee is inferred merely from a successful example. beta=1/tau -> infinity is not a uniform validity limit for this local expansion.

## 5. Positive information with zero decision value

Take theta in {{-1/4,+1/4}} equiprobably and U(theta)=(2+theta,1+theta,theta). Perfectly reveal theta. Mutual information is log(2)={math.log(2):.16g} nats, but action 0 is always strictly optimal and terminal EVSI is exactly zero. F_tau is common-shift equivariant, F_tau(mu+x*1)=F_tau(mu)+x, so its exact information value is also zero. Cov(mu^Y)=(1/16)11^T is nonzero; the susceptibility annihilates it. The same example tests why entropy reduction alone is not a decision-value estimator.

## 6. Reproducible execution and outcome

Command: `python3 /tmp/quartz_c26_statphys.py`.
Environment: Python {platform.python_version()}, SymPy {s.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}. Script SHA256: `{result["script_sha256"]}`. JSON stores all matrices, finite outcomes, trajectory rows, fixed tolerances through named assertions, and the complete check list. The invoking host records the actual exit code separately.

Requested numeric Gibbs target: {qn.tolist()}. Initial KL: {num_trace[0]["KL"]:.16g}; step-80 KL: {num_trace[-1]["KL"]:.16g}; variational-gap residual: {gap_identity_residual:.3g}. Floating-point final KL may have a last-bit signed roundoff and is accepted only under the stated absolute tolerance; it is not evidence for negative mathematical KL.

Evidence classification: exact matrix and covariance identities **DERIVED + SYMBOLICALLY CHECKED**; finite high-precision and float trajectories **NUMERICALLY CHECKED**; general contraction and remainder bound **DERIVED** from the displayed inequalities. Physical equivalence to MCTS, real-game benefit, cognitive-brain validity and production runtime efficiency remain **UNRESOLVED / NOT TESTED HERE**. The exact classical/Gibbs mechanism is a valid constructive replacement for the defective diagonal-jump argument without claiming a physical identity.
"""
BASE.with_suffix(".md").write_text(md)
for path in [
    BASE.with_suffix(".py"),
    BASE.with_suffix(".json"),
    BASE.with_suffix(".md"),
]:
    require("artifact_under_1MB_" + path.suffix, path.stat().st_size < 1_000_000)
print(
    json.dumps(
        {
            "status": "PASS",
            "scientific_check_count": len(result["checks"]) - 3,
            "Q_numeric": Qn.tolist(),
            "q_numeric": qn.tolist(),
            "initial_KL": num_trace[0]["KL"],
            "final_KL": num_trace[-1]["KL"],
            "artifacts": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in [
                    BASE.with_suffix(".py"),
                    BASE.with_suffix(".json"),
                    BASE.with_suffix(".md"),
                ]
            ],
        },
        indent=2,
    )
)
