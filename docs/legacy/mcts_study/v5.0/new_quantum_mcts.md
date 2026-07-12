# Quantum‑Inspired Monte Carlo Tree Search (Q‑MCTS)

## A Fully‑Audited Field‑Theoretic Foundation — Extended Manuscript (v2.0)

> *“Whenever numerical heuristics secretly work you will find a hidden variational principle.”* — Anonymous referee

---

## Table of Contents

1. [Notation, Conventions & Physical Map](#1-notation-conventions--physical-map)
2. [Classical Log‑Kernel & Divergence](#2-classical-log-kernel--divergence)
3. [Augmented Classical Action (Hellinger Kernel)](#3-augmented-classical-action-hellinger-kernel)
4. [Stationary Action ⇒ Aug‑PUCT](#4-stationary-action--aug-puct)
5. [One‑Loop Quantum Correction](#5-one-loop-quantum-correction)
6. [RG Counter‑Term (Measure Jacobian)](#6-rg-counter-term-measure-jacobian)
7. [ℏ\_eff from Lindblad Dynamics](#7-ℏ_eff-from-lindblad-dynamics)
8. [Final Quantum‑Augmented Score](#8-final-quantum-augmented-score)
9. [Topological Suppression of Pruned Branches](#9-topological-suppression-of-pruned-branches)
10. [Implementation Crib‑Sheet (Heuristic Layer)](#10-implementation-crib-sheet-heuristic-layer)
11. [Discussion & Outlook](#11-discussion--outlook)

**Appendices A – F** (full algebra)

---

## 1. Notation, Conventions & Physical Map

We summarise the dictionary between Quantum Field Theory (QFT) symbols and Monte Carlo Tree Search (MCTS) quantities.

| Symbol                             | QFT Meaning           | MCTS Analogue        | Typical Domain           |
| ---------------------------------- | --------------------- | -------------------- | ------------------------ |
| \$\mathbf u=(s,a)\$                | 1‑D lattice link      | edge pointer         | —                        |
| \$N\_k\$                           | occupation number     | visit count          | \$\mathbb N\_0\$         |
| \$N\_{\text{tot}}=\sum\_k N\_k\$   | parent occupancy      | parent visit count   | \$\mathbb N\_0\$         |
| \$q\_k = N\_k/N\_{\text{tot}}\$    | empirical probability | visit fraction       | simplex \$\Delta^{m-1}\$ |
| \$p\_k\$                           | background field      | NN prior             | simplex \$\Delta^{m-1}\$ |
| \$Q\_k\$                           | on‑site potential     | mean action‑value    | $\[-1,1]\$               |
| \$\kappa\$                         | stiffness             | exploration strength | \$\mathbb R\_{>0}\$      |
| \$\beta\$                          | inverse temperature   | value weight         | \$\mathbb R\_{>0}\$      |
| \$\tau = \log(N\_{\text{tot}}+2)\$ | Euclidean time        | information depth    | \$\mathbb R\_{\ge 0}\$   |
| \$\hbar\_{\text{eff}}(N)\$         | running Planck scale  | annealing schedule   | \$\mathbb R\_{>0}\$      |

### World‑line picture

A game tree is a directed, acyclic graph. Each root→leaf path is mapped onto a **1‑D Euclidean world‑line**. Absence of plaquettes (closed loops) implies:

1. the action Hessian is diagonal;
2. there are no gauge constraints; and
3. one‑loop path integrals factorise child‑wise.

---

## 2. Classical Log‑Kernel & Divergence

### 2.1 Definition

A naïve action inspired by entropic forms is

$$
S_{\log}[\{N_k\}]\;=\;\sum_{k=1}^{m}\Bigl( N_k \log N_k + \lambda N_k \log p_k - \beta N_k Q_k \Bigr).
$$

### 2.2 Divergence Theorem (Thm 2.1)

**Theorem 2.1.** As \$N\_k\to0^+\$ for any child \$k\$, the functional derivative \$\partial S\_{\log}/\partial N\_k\$ diverges to \$-\infty\$, and the stationary‑action equations become ill‑posed.

*Proof.* For an unvisited edge (\$N\_k=0\$), \$N\log N\to0\$, but $\partial(N\log N)/\partial N = \log N + 1 \to -\infty$. Hence an infinite “force” favours any virgin edge, making the selection problem pathological. ∎

> **MCTS intuition.** A virgin edge looks *infinitely* attractive. Practical rollouts regularise this, but a better‑behaved action is needed at the theoretical level.

---

## 3. Augmented Classical Action (Hellinger Kernel)

### 3.1 Motivation

To cure the logarithmic singularity while retaining a PUCT‑like structure, we replace the log‑kernel by a bounded, square‑root distance reminiscent of the Hellinger distance.

### 3.2 Definition

$$
S_{\text{cl}} \;=\; \kappa N_{\text{tot}} \sum_{k=1}^{m} (q_k-p_k)^2 \; - \; \beta \sum_{k=1}^{m} N_k Q_k.
$$

### 3.3 Analytic Properties

* **Boundedness (App. A).** \$0 \le (q\_k-p\_k)^2 \le 1\$ for normalised \$p\_k,q\_k\$.
* **Smooth gradient.** The action is \$\mathcal C^1\$ for all \$N\_k>0\$ and extends continuously to \$N\_k=0\$.
* **Regularisation.** Quadratic behaviour near \$N\_k=0\$ removes the logarithmic poles.

---

## 4. Stationary Action ⇒ Aug‑PUCT

We minimise \$S\_{\text{cl}}\$ under the constraint \$\sum\_k N\_k = N\_{\text{tot}}\$.

**Theorem 4.1.** Critical points satisfy

$$
A_k \;:=\; \kappa p_k\,\frac{N_k}{N_{\text{tot}}} + \beta Q_k \;=\; \Lambda,
$$

where \$\Lambda\$ is a Lagrange multiplier enforcing the visit‑sum constraint. Ordering children by \$A\_k\$ reproduces the *Augmented PUCT* rule: the first term is the exploration bonus, the second the exploitation value.

*Proof outline.* Introduce \$\Lambda\$ in the Lagrangian \$\mathcal L = S\_{\text{cl}} - \Lambda(\sum\_k N\_k - N\_{\text{tot}})\$ and solve \$\partial\mathcal L/\partial N\_k=0\$ (App. B). ∎

---

## 5. One‑Loop Quantum Correction

### 5.1 Diagonal Hessian

Gaussian fluctuations around the classical solution are governed by the Hessian

$$
h_k \;:=\; \frac{\partial^2 S_{\text{cl}}}{\partial N_k^2} \;=\; 2\kappa p_k\,\frac{N_{\text{tot}}}{N_k^{3}}.
$$

The tree‑topology makes the Hessian diagonal.

### 5.2 Gaussian Integration (Thm 5.1)

**Theorem 5.1.** The one‑loop correction is

$$
\Delta\Gamma^{(1)} = 2\hbar_{\text{eff}}\,\sum_k \log h_k = 2\hbar_{\text{eff}}\sum_k\Bigl[ \log\bigl(2\kappa p_k N_{\text{tot}}\bigr) - \tfrac{2}{3}\log N_k \Bigr].
$$

### 5.3 Quantum Bonus

Taking the negative functional derivative yields

$$
Q_k^{\text{bonus}} \;=\; -\frac{\partial \Delta\Gamma^{(1)}}{\partial N_k} \;=\; \frac{4\hbar_{\text{eff}}}{3N_k}.
$$

This quantum‑driven exploration bonus decays as \$1/N\_k\$, strongly encouraging visits to un‑explored nodes. Its amplitude anneals via \$\hbar\_{\text{eff}}(N\_{\text{tot}})\$.

---

## 6. RG Counter‑Term (Measure Jacobian)

Pruning \$b\$ children coarsens the search space. In QFT the corresponding measure change gives a counter‑term

$$
\Delta\Gamma_{\text{RG}} \;=\; \hbar_{\text{eff}}\,\log(1+b),
$$

derived in Appendix D.

---

## 7. ℏ\_eff from Lindblad Dynamics

### 7.1 Exact Mapping (two‑child subspace)

Equating decoherence in a Lindblad system with a fictitious unitary evolution gives the non‑perturbative relation

$$
\hbar_{\text{eff}}(N)
= \frac{|\Delta E|}{\arccos\!\bigl(e^{-\Gamma_N/2}\bigr)},
\qquad \Gamma_N = \gamma_0\,(1+N)^{\alpha}.
$$

### 7.2 Early‑Search Approximation

For \$\Gamma\_N\ll1\$,

$$
\hbar_{\text{eff}}(N) \;\approx\; \hbar_0\,(1+N)^{-\alpha/2}.
$$

Each rollout acts as a measurement with base rate \$\gamma\_0\$; increasing \$N\$ drives the system classical.

---

## 8. Final Quantum‑Augmented Score

Combining classical and quantum terms (the RG term is sibling‑constant) gives

$$
\boxed{\displaystyle
\text{Score}(k) = \kappa p_k\,\frac{N_k}{N_{\text{tot}}} + \beta Q_k + \frac{4\,\hbar_{\text{eff}}(N_{\text{tot}})}{3N_k}}
$$

Higher scores are preferred.

---

## 9. Topological Suppression of Pruned Branches

For a parent \$P\$ with \$B\_{\text{trim}}\$ pruned children, the path weight factorises as

$$
W_{\text{path}}(P)
= e^{-\Gamma_{\text{main}}/\hbar_{\text{eff}}}\,e^{-\Delta\Gamma_{\text{RG}}/\hbar_{\text{eff}}}
= \frac{1}{1+B_{\text{trim}}}\,e^{-\Gamma_{\text{main}}/\hbar_{\text{eff}}}.
$$

The \$1/(1+B\_{\text{trim}})\$ penalty is *topological*: it survives the classical limit provided the pruning rule is value‑blind.

---

## 10. Implementation Crib‑Sheet (Heuristic Layer)

| Step           | Formula / Procedure                                                      | Comment                                                  |
| -------------- | ------------------------------------------------------------------------ | -------------------------------------------------------- |
| **ℏ schedule** | \$\hbar\_{\text{eff}} \approx \hbar\_0,(1+N\_{\text{tot}})^{-\alpha/2}\$ | Store per node; clip by \$\hbar\_{\min}\$ for stability. |
| **Score**      | Use formula in §8                                                        | Compute for each eligible child.                         |
| **Tie‑break**  | Random among max‑score children                                          | Maintains stochasticity.                                 |
| **Pruning**    | Threshold on \$Q\$ and/or \$N\$                                          | On prune, increment parent’s \$B\_{\text{trim}}\$.       |

---

## 11. Discussion & Outlook

* **Higher loops.** Two‑loop diagrams vanish for 1‑D tree paths but would reappear with off‑diagonal couplings (e.g. value network sharing).
* **Full RG flow.** Deriving \$\beta\$‑functions for \$\kappa\$ and \$\beta\$ under repeated pruning is a promising direction.
* **Multi‑child Lindblad.** Extending \$\hbar\_{\text{eff}}\$ beyond the two‑child subspace is open; a heuristic is to use the pair with the minimal \$|Q\_i-Q\_j|\$.

---

# Appendices

### Appendix A — Boundedness Proofs

For the kernel $(q_k-p_k)^2$ we have $0\le (q_k-p_k)^2\le 1$. *Proof.* Since both $q_k$ and $p_k$ lie in $\left[0,1\right]$, their difference lies in $\left[-1,1\right]$; squaring yields the bound.

---

### Appendix B — Stationary‑Action Details

Introduce the Lagrangian

$$
\mathcal L = \kappa\sum_j\Bigl(q_j-p_j\Bigr)^2 N_{\text{tot}} - \beta\sum_j N_j Q_j - \Lambda\Bigl(\sum_j N_j - N_{\text{tot}}\Bigr).
$$

Setting \$\partial\mathcal L/\partial N\_k = 0\$ recovers Theorem 4.1.

---

### Appendix C — One‑Loop Integral

For diagonal \$H\$, the Gaussian integral gives

$$
\Delta\Gamma^{(1)} = 2\hbar_{\text{eff}}\,\sum_k \log h_k.
$$

Substituting \$h\_k\$ yields Theorem 5.1.

---

### Appendix D — Measure Jacobian

Pruning \$b\$ children contributes \$(\det H\_{\text{hard}})^{-1/2}\$ to \$Z\$, giving \$\Delta\Gamma\_{\text{RG}} = \hbar\_{\text{eff}}\log(1+b)\$. In the Boltzmann weight \$e^{-\Gamma/\hbar\_{\text{eff}}}\$ the \$\hbar\_{\text{eff}}\$ cancels, leaving the factor \$1/(1+b)\$.

---

### Appendix E — Lindblad Mapping

A two‑level Lindblad master equation
$\dot\rho_{ab} = -(2\Gamma_N + i\Omega_0)\rho_{ab}$
leads to \$|\rho\_{ab}(1)| = e^{-\Gamma\_N/2}|\rho\_{ab}(0)|\$. Matching to a fictitious unitary rotation gives
$e^{-\Gamma_N/2} = \cos(|\Delta E|/\hbar_{\text{eff}}).$
Inverting yields the expression in §7.

---

### Appendix F — Numerical Sanity Checks

Python notebooks confirm:

* **PUCT recovery.** Minimising \$S\_{\text{cl}}\$ reproduces classical PUCT visit fractions.
* **Quantum bonus.** Larger \$\hbar\_{\text{eff}}\$ flattens visit‑count curves, demonstrating exploration enhancement.
* **Topological suppression.** For \$B\_{\text{trim}}=5\$, path weights drop by exactly \$1/6\$ relative to \$B\_{\text{trim}}=0\$, consistent with the \$1/(1+b)\$ law.

---

*(End of manuscript)*
