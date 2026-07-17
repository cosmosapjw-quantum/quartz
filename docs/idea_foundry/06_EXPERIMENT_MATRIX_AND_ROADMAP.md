# 06. Experiment Matrix and Implementation Roadmap

이 문서는 24개 축을 실제 저장소에서 어떤 순서와 비용으로 검증할지 정한다.

---

## 1. 실험 plane별 실행 원칙

### Plane P0 — Pure unit/reference

- 수식, 변환, schema
- dependency-light
- 실제 playing-strength claim 금지

### Plane P1 — Synthetic mechanism

- Bernoulli/root candidate world
- priced WIDEN, known true means
- deterministic common-random-number pairing
- neural MCTS claim 금지

### Plane P2 — Frozen Phase-15 trace replay

- 같은 search trace를 여러 readout/operator가 공유
- `search_relevant_signature`와 code salt 고정
- post-hoc calibration/feature screening

### Plane P3 — Resident continuation / counterfactual

- 동일 root checkpoint에서 추가 budget branch
- restart-per-chunk와 root continuation을 pool하지 않음
- 실제 computation-gain label 생성

### Plane P4 — Live engine micro-policy

- 한 control surface만 변경
- exact budget/cost measurement
- paired checkpoint/position/seed

### Plane P5 — Training and arena

- frozen search contract로 evaluator 비교
- frozen evaluator로 controller 비교
- multi-seed paired arena

---

## 2. Baseline matrix

### Search/controller baselines

| ID | Baseline |
|---|---|
| S0 | Pure PUCT fixed budget |
| S1 | current no-refresh Legacy family |
| S2 | current Quartz variants with fixed halt |
| S3 | current progressive widening |
| S4 | Gumbel + Sequential Halving |
| S5 | exact/static-anchor RPO readout/policy |
| S6 | H1 and P_flip virtual stops |
| S7 | KG allocation / KG-stop historical condition |
| S8 | MENTS/BTS/DENTS comparator |

### Parallel/system baselines

| ID | Baseline |
|---|---|
| P0 | serial PUCT |
| P1 | fixed virtual loss |
| P2 | current adaptive split VL |
| P3 | vvisit-only / pending-count correction |
| P4 | explicit fixed threads/batch/inflight |
| P5 | current auto-thread throughput/quality |
| P6 | PMCTS/SMC-inspired challenger when implemented |

### Candidate baselines

| ID | Baseline |
|---|---|
| C0 | no widen |
| C1 | eager prior-order widen |
| C2 | current PW |
| C3 | Gumbel/SH |
| C4 | residual evidence |
| C5 | dynamic live set |
| C6 | JLB root geometry |

### Evaluator baselines

| ID | Baseline |
|---|---|
| E0 | current ResNet |
| E1 | parameter/FLOP-matched residual FCN |
| E2 | plain U-Net/local-global hybrid |
| E3 | denoising-regularized U-Net |
| E4 | RW-ResT Lite |
| E5 | small quantized CNN |
| E6 | incremental pattern student |

---

## 3. Metric matrix

### Decision quality

```text
high-budget oracle top-1 agreement
root simple regret proxy
top-k recall
hidden-best recall
forced-win/block recall
late revision count
commitment time
```

### Calibration

```text
ECE
Brier score
log loss
selective risk
interval empirical coverage
candidate residual-bound coverage
```

### Compute

```text
completed NN evals
requested/cancelled evals
CPU/GPU ms
GPU seconds/game
batch occupancy
queue wait
p50/p95/p99 move time
energy proxy provenance
```

### Parallel

```text
edge duplicate
semantic path overlap
unique leaf/state ratio
max pending
virtual pessimism
TT wait/contention
```

### Training

```text
Elo/Glicko with CI
learning progress per total compute
replay ESS/diversity
external-anchor performance
adversarial/failure-suite performance
```

---

## 4. Campaign sequence

## Campaign 0 — Skeleton and schema gate

목표:

- 24-axis registry와 Python skeleton import/test
- Rust skeleton은 아직 export하지 않음
- 문서와 machine-readable catalog의 ID 일치

실행:

```bash
venv/bin/python -m pytest -q tests/test_idea_foundry_skeleton.py
venv/bin/python -m compileall quartz/idea_foundry
```

Gate:

- all tests green
- no import side effects
- no production configuration change

---

## Campaign 1 — Existing evidence replay

축:

```text
A01 Stop Council feature bank
A11 Entropy-Margin router
A15 B13 readout
A17 temperature/scale diagnostics
A24 symmetry audit
```

필요 artifact:

```text
Stage-7 trace bundles
online rows
posthoc Phase-15 rows
forked-VOC labels
```

작업:

1. trace adapter로 `RootSnapshot` 생성
2. all-shadow feature table 생성
3. position/checkpoint grouped split
4. calibration and incremental-value report

Gate:

- H1/P_flip 외 feature가 held-out prediction에 기여하는지
- H3 continuous feature가 nondegenerate인지
- A15 role은 readout으로 유지할지 결정

---

## Campaign 2 — Candidate coverage synthetic + trace

축:

```text
A02 Static-anchor RPO
A05 Gumbel/SH
A06 Residual mass
A07 JLB
A08 Live set
A09 Tactical sentinel
```

먼저 synthetic candidate bank:

```text
best visible
best hidden low-prior
near tie
multimodal rewards
wrong prior
forced tactic
```

다음 frozen real trace/root bank에서 posthoc policy/candidate replay.

Gate:

- hidden-best recall
- omission regret
- useful WIDEN precision
- computation overhead

---

## Campaign 3 — Counterfactual meta-action teacher

새 infrastructure가 필요한 첫 단계다.

### Snapshot requirements

```text
root state identity
completed edge statistics
candidate visibility
random seed state or deterministic replay key
pending=0 quiescent checkpoint
model/search contract hashes
```

### Branches

```text
STOP
SAMPLE incumbent
SAMPLE challenger
CHALLENGE top1/top2
WIDEN Gumbel
WIDEN residual
PROVE tactical
SET runtime setting
```

Label:

\[
y_c(x)
=
\frac{L_{STOP}-L_c}{C_c+\epsilon}.
\]

Group split는 position/game 단위다. 같은 root의 budget checkpoint가 train/test에 동시에 들어가면 안 된다.

Gate:

- best meta-action prediction
- pairwise ranking
- calibrated positive-gain probability
- feature ablation

---

## Campaign 4 — Online micro-policies

복잡도를 단계적으로 늘린다.

### 4A

```text
SAMPLE vs STOP
```

### 4B

```text
SAMPLE vs WIDEN vs STOP
```

### 4C

```text
SAMPLE/WIDEN/PROVE/SCHEDULE/STOP
```

한 번에 한 surface만 production score와 executor를 변경한다.

Gate:

- fixed NN eval condition
- fixed wall-clock condition
- tactical catastrophe non-inferiority
- policy overhead p99

---

## Campaign 5 — Parallel/system

축:

```text
A12 Service scheduler
A13 Pending flow
A14 Semantic path LSH
A23 Graph sharing
```

실험 grid:

```text
threads: 1,2,4,8,16
budgets: 32,64,128,256,512
batch/inflight: hardware service curve grid
```

H5는 edge duplicate가 아니라 semantic overlap metric으로 판정한다.

Gate:

- decision regret per wall-clock
- no pending leak/stale stop
- path-sketch overhead보다 saved eval cost가 큼

---

## Campaign 6 — Evaluator and training

축:

```text
A18 Diffusion auxiliary
A19 RW-ResT Lite
A20 CPU student
A21 Regret archive
```

순서:

1. frozen supervised proxy
2. throughput Pareto
3. fixed-search arena
4. multi-seed self-play
5. archive/restart under fixed total compute

controller 변경과 같은 campaign에서 evaluator를 바꾸지 않는다.

---

## 5. Rust promotion checklist

각 online 축은 다음을 만족해야 한다.

```text
[ ] Python/reference mechanism test
[ ] Phase-15 shadow artifact
[ ] symmetry/invariance audit
[ ] named config and manifest field
[ ] focused Rust unit tests
[ ] stale-cache STOP prohibition
[ ] hot-path allocation = 0
[ ] hot-path mutex = 0 or measured/approved exception
[ ] exact budget fairness
[ ] telemetry schema and failure rows
[ ] compare against current baseline
```

Promotion sequence:

1. `pub mod foundry_contracts;`
2. focused compile/test
3. `ShadowAxisPolicy` only
4. explicit environment/config registration
5. online executor wiring
6. default remains unchanged

---

## 6. Python/Phase-15 promotion checklist

```text
[ ] operator name registered
[ ] posthoc vs online semantics declared
[ ] search_relevant_signature correct
[ ] trace schema/code salt updated only when search semantics change
[ ] source artifact hashes recorded
[ ] missing joins excluded and counted
[ ] position-grouped split
[ ] paired delta + CI
[ ] prohibited inference text
```

---

## 7. Immediate implementation order

### Milestone M0 — this skeleton set

- docs + 24-axis registry
- Python skeletons/tests
- non-exported Rust contracts

### M1 — Trace adapter and all-shadow table

신규 파일 후보:

```text
quartz/idea_foundry/trace_adapter.py
scripts/idea_foundry_shadow.py
configs/idea_foundry_shadow.v1.json
```

### M2 — A01 stop feature/calibration dataset

```text
quartz/idea_foundry/stop_council.py
scripts/idea_foundry_stop_council.py
```

### M3 — A02/A06 candidate posthoc comparison

```text
quartz/idea_foundry/static_anchor_rpo.py
quartz/idea_foundry/residual_mass.py
scripts/idea_foundry_candidates.py
```

### M4 — Deterministic resident snapshot fork

```text
Rust session snapshot/replay protocol
Python counterfactual runner
cost/decision-loss artifact
```

### M5 — First online executor

먼저 `SAMPLE`과 `STOP`만 지원한다. WIDEN/PROVE/SCHEDULE은 이후 추가한다.

---

## 8. Claim firewall

다음 표현은 금지한다.

```text
synthetic success → playing-strength improvement
readout KL gain → Elo gain
fewer visits → lower GPU cost
bootstrap stability → PAC guarantee
JSD graph → exact Langevin MCMC
live set → exact nested sampling
signed vector → quantum amplitude
entropy decay → physical decoherence
curve shape → universality class
```

대신 각 결과는 정확한 plane, game, budget, evaluator, hardware로 범위를 제한한다.
