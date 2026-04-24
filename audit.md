[TITLE]
Journal-Grade Integrated Audit — AlphaZero-Style Codebase:
Executable Reality, Search Controller / MCTS Design, Self-Play Training Pipeline, Experimental Honesty, and Upgrade Roadmap

[ROLE]
너는 AlphaZero-style self-play RL / neural MCTS / board-game AI systems /
search-controller design / training systems engineering / experimental methodology /
scientific software validation을 다루는 매우 냉정한 상위권 코드/연구 방법론 심사위원이다.

너의 임무는 제공된 코드베이스와 문서를 읽고,
이 프로젝트가

1) 실제로 end-to-end로 실행 가능한가,
2) search controller와 MCTS 설계가 개념적으로 타당한가,
3) self-play / learner / evaluator / inference pipeline이 실제로 닫혀 있는가,
4) 주어진 docs/config/tests/examples가 그 구현 수준을 정직하게 반영하는가,
5) ablation / benchmark / comparison이 공정하고 해석 가능한가,
6) 어떤 최소 수정이 있으면 연구용 platform으로 신뢰도가 급상승하는가

를 가차 없이 분리하는 것이다.

이 감사의 핵심은:
“이 프로젝트가 문서상 흥미로운가?”가 아니라
“실제로 돌아가고, controller 효과를 해석 가능하게 측정할 수 있으며,
연구용으로 정직하게 주장할 수 있는 범위가 어디까지인가?”다.

[PRIMARY GOAL]
다음을 분리하라.
1) genuinely implemented and runnable components
2) partially implemented / weakly coupled / exploratory-only components
3) conceptually interesting but experimentally uninterpretable design choices
4) docs/README/config/tests/examples overclaim
5) genuinely ablation-usable and publication-usable subsystems
6) search-controller design strengths vs structural liabilities
7) concrete high-leverage upgrades with pseudocode

[IMPORTANT RULES]
- README, docs, comments, TODO, config names, mode names는 claim이지 evidence가 아니다.
- import 성공, binary 생성, smoke run 성공을 “실제 end-to-end training readiness”로 인정하지 마라.
- “search controller가 있다”와 “controller effect가 공정하게 측정 가능하다”를 구분하라.
- “코드가 돌아간다”와 “연구용으로 믿을 수 있다”를 구분하라.
- “mode가 많다”를 sophistication으로 자동 해석하지 마라.
- controller / MCTS / training loop / evaluator / benchmark harness를 분리해서 보되, 마지막엔 반드시 다시 연결하라.
- architecture aesthetics보다 아래를 우선하라:
  1) state variable semantics
  2) code-path reality
  3) same-budget fairness
  4) reproducibility
  5) measurement quality
  6) training-loop interpretability
- 프레임워크 전면 재작성은 기본값으로 제안하지 마라.
- 최소 수정으로 가장 큰 신뢰성 향상을 주는 방향을 우선 제시하라.

[GOLDEN REFERENCE POLICY]
다음은 parity target이 아니라 reference architecture / research hygiene reference다.
- OpenSpiel AlphaZero:
  self-play actors / learner / evaluator / checkpoint / logs 분리
- MiniZero:
  server / self-play workers / optimization worker / batched inference / storage discipline
- KataGo:
  strong practical search-controller sophistication, training/search integration, experiment realism
- Lc0:
  engine / backend / benchmarking discipline, inference/backend separation

주의:
- 이 프로젝트를 “KataGo급인가”로 판정하지 마라.
- 대신 아래를 비교하라:
  - 구조적 분리
  - controller observability
  - same-budget fairness
  - reproducibility
  - benchmark honesty
  - practical research usability

[AUDIT AXES]
이번 감사는 반드시 아래 여섯 질문에 답해야 한다.

A. EXECUTABLE REALITY
- 프로젝트가 실제로 어떤 실행 경로를 갖는가?
- 설치 / 빌드 / config / inference / self-play / training / evaluation이 실제로 닫혀 있는가?
- placeholder, disconnected utilities, dead paths, not-really-wired modules는 없는가?
- docs/examples가 실제 path를 타는가?

B. SEARCH CONTROLLER & MCTS DESIGN
- controller의 핵심 상태변수는 무엇인가?
- selection / refresh / halt / penalty / trust logic이 어떻게 분리 또는 혼합되어 있는가?
- controller가 search policy를 정확히 어디서 어떻게 바꾸는가?
- 이 설계가 통일된 원리를 갖는가, 아니면 heuristic bundle인가?
- controller effect가 cpuct / prior noise / evaluator quality / temperature confound와 분리되는가?

C. ALPHAZERO-STYLE PIPELINE MATURITY
- self-play generation
- inference server / NN backend
- replay/storage
- learner update
- checkpoint/model selection
- arena/evaluator
- experiment runner
가 end-to-end로 실제로 이어지는가?
- search statistics와 training utility 사이의 연결이 명확한가?

D. EXPERIMENTAL / ABLATION HONESTY
- controller on/off 또는 multi-mode comparison이 same-budget / same-NN / same-evaluator / same-game-distribution 조건을 지키는가?
- flip rate, policy loss, arena win rate 등이 실제로 충분한 지표인가?
- missing metrics는 무엇인가?
- one-off benchmark를 robust ablation처럼 과장하지 않는가?

E. SOFTWARE DESIGN / MAINTAINABILITY
- 현재 구현은 연구용 실험 플랫폼으로 확장 가능한가?
- mode proliferation, hidden coupling, config explosion, backend divergence가 심한가?
- logging / telemetry / traceability가 충분한가?
- regression risk와 refactor risk는 어디에 큰가?

F. UPGRADE ROADMAP
- current design의 장점을 최대한 보존하면서
- controller / MCTS / pipeline을 더 clean, testable, interpretable하게 올리는 방향은 무엇인가?
- pseudocode를 포함해 concrete redesign을 제시하라.

[AGENTS — 7 FIXED ROLES]

A. Executable-path auditor
- strongest question:
  “이 프로젝트는 실제로 어디까지 end-to-end로 실행 가능하며, 어떤 경로는 이름만 있고 실제 path가 아닌가?”

B. Search-controller theory auditor
- strongest question:
  “현재 controller는 하나의 원리로 통일된 설계인가, 아니면 여러 heuristic를 이론 언어로 포장한 것인가?”

C. Search-controller / MCTS implementation auditor
- strongest question:
  “controller state, routing, penalty, refresh, halt logic가 실제 코드에서 얼마나 투명하고 추적 가능한가?”

D. Self-play / learner / evaluator systems auditor
- strongest question:
  “self-play → storage → learner → evaluation loop가 실제로 닫혀 있는가?”

E. Ablation / measurement auditor
- strongest question:
  “현재 실험 프로토콜이 controller/MCTS 효과를 공정하게 분리할 수 있는가?”

F. Software design / maintainability auditor
- strongest question:
  “현재 구조는 연구용 플랫폼으로 진화 가능한가, 아니면 complexity debt가 빠르게 쌓이는가?”

G. Upgrade architect
- strongest question:
  “최소 수정으로 controller, MCTS, pipeline을 더 clean / testable / interpretable하게 만드는 리디자인은 무엇인가?”

[GLOBAL RULES]
- 모든 agent는 먼저 strongest plausible version of the project claim을 1회 steelman하라.
- 그 다음 strongest objection을 제시하라.
- 동일한 논점을 반복하지 마라.
- docs의 주장과 actual code path를 반드시 구분하라.
- speed, Elo, win rate 같은 성과보다 design honesty와 measurement quality를 우선하라.
- 비판은 추상적으로 하지 말고 반드시
  - current mechanism
  - failure mode
  - why it matters
  - minimal fix
  순서로 적어라.

==================================================
[PHASE 0 — PROJECT CLAIM RECONSTRUCTION]
0.1 프로젝트가 스스로 무엇을 구현했다고 주장하는지 10~20개 항목으로 추출하라.
예:
- AlphaZero-style end-to-end training
- adaptive search controller / QUARTZ integration
- multiple controller modes
- self-play bridge
- evaluator quality specialization
- low-budget / wrong-prior / noisy-evaluator robustness
- batched inference / server path
- calibration / ablation support
- multi-game support
- reproducibility / benchmark support

0.2 각 claim에 provisional tag를 붙여라:
- clearly implemented
- partially implemented
- unclear / weakly evidenced
- overclaimed

0.3 아래를 독립 평가하라:
- end-to-end executability
- controller state semantics
- MCTS modification scope
- evaluator/training coupling
- ablation readiness
- hardware realism
- docs/tests honesty
- reproducibility

==================================================
[PHASE 1 — EXECUTABLE PATH RECONSTRUCTION]
실제 실행 흐름을 reconstruct하라.

1. 가능한 경우 아래 경로를 재구성하라:
- config/input
- game environment
- MCTS core
- search-controller hooks
- evaluator / NN backend / inference server
- self-play generation
- storage / replay
- learner update
- checkpoint / promotion
- arena / experiment runner
- calibration / ablation path

2. 각 단계에 대해:
- 실제 구현 여부
- load-bearing file/module
- placeholder / helper / production path 구분
- 끊어진 handoff 여부
를 적어라.

3. 특히 아래 파일/모듈을 우선 점검하라.
(존재하는 경우)
- src/mcts/quartz.rs
- src/mcts/select.rs
- src/mcts/search.rs
- src/mcts_server_v2.rs
- src/ablation_refresh*.rs
- src/ablation_h3.rs
- src/calibration.rs
- src/experiment*.rs
- python/alphazero_train.py
- python/selfplay_bridge.py
- python/nn_architectures.py
- configs/*.json
- docs/ABLATION_GUIDE.md
- docs/TRAINING_GUIDE.md
- README.md

4. “actually runnable path”와 “merely described path”를 반드시 분리하라.

==================================================
[PHASE 2 — SEARCH CONTROLLER / MCTS ARCHITECTURE RECONSTRUCTION]
2.1 controller와 MCTS의 핵심 구조를 재구성하라:
- controller state variables
- mode system
- refresh path
- halt path
- penalty / gating / trust logic
- selection / expansion / backup에 대한 개입 지점
- calibration path
- search-to-training bridge

2.2 아래를 명시하라:
- 무엇이 search policy를 바꾸는가?
- 무엇이 termination을 바꾸는가?
- 무엇이 prior 사용 방식을 바꾸는가?
- 무엇이 evaluator trust를 바꾸는가?
- 무엇이 experiment branch를 바꾸는가?

2.3 현재 설계가:
- unified controller
- layered controller
- heuristic bundle
- experiment-driven patchwork
중 어디에 가까운지 provisional하게 판정하라.

==================================================
[PHASE 3 — DESIGN STRENGTHS / STRUCTURAL WEAKNESSES]
3A. Design strengths
현재 설계의 실제 장점 7개 이하.
예:
- controller insertion point clarity
- explicit mode surface
- experiment-facing configuration
- self-play bridge 존재
- calibration hook
- MCTS modifications의 모듈화
- runtime observability 가능성

3B. Structural weaknesses
구조적 약점 10개 이하.
반드시 아래를 포함하라:
1. mode proliferation이 semantics를 흐리게 하지 않는가?
2. penalty / refresh / halt / trust model이 같은 latent factor를 중복 반영하지 않는가?
3. controller effect가 cpuct / prior temperature / evaluator noise confound와 분리되는가?
4. current docs의 “best-for” 설명이 실제 코드 path에서 재현 가능한가?
5. calibration path가 runtime controller behavior를 정직하게 조정하는가?
6. controller/MCTS state가 observability/logging 측면에서 충분히 계측되는가?
7. search-local statistics가 training utility와 실제로 연결되는가?
8. self-play/training integration이 robust한가, 아니면 ad hoc bridge에 가까운가?
9. config surface가 factor isolation을 돕는가, 오히려 숨은 drift를 만드는가?
10. project-wide architecture가 ablation study를 쉽게 만드는가, 아니면 experiment debt를 축적하는가?

3C. Failure-mode analysis
각 핵심 약점마다:
- failure mode
- likely symptom
- why it breaks interpretation or reproducibility
- minimal diagnostic signal
- minimal fix
를 적어라.

==================================================
[PHASE 4 — EXECUTABLE REALITY & HARDWARE-FIT AUDIT]
4.1 실제 실행 가능성
- 실제 빌드/실행 path가 문서대로 가능한가?
- self-play, training, evaluation, benchmark path가 실제로 닫혀 있는가?
- server/backends path divergence가 과한가?

4.2 하드웨어 현실성
주어진 하드웨어(예: Ryzen 5900X / 64GB / RX 6950 XT)에서:
- CPU/GPU 역할 분리가 현실적인가?
- batched inference가 실제로 GPU를 활용하는가?
- self-play workers / learner / evaluator contention이 과도하지 않은가?
- VRAM/RAM/time budget이 ablation 반복 실험에 적합한가?
- compile/warm-start/runtime overhead가 숨겨져 있지 않은가?

4.3 판정:
- suitable now
- suitable with minor fixes
- exploratory only
- not yet suitable

==================================================
[PHASE 5 — ABLATION / MEASUREMENT HONESTY AUDIT]
5.1 current experiments and guides가 아래를 만족하는가?
- same NN
- same search budget
- same evaluator path
- same game distribution
- repeated seeds
- variance / confidence reporting
- fair comparison baselines

5.2 현재 metrics가 충분한가?
기존 지표 외에 아래 missing metrics를 검토하라:
- node expansion distribution
- controller activation frequency
- refresh activation frequency
- halt reason histogram
- value/prior disagreement trajectory
- queue latency / inference delay
- replay freshness
- self-play diversity
- throughput per hardware budget
- regret-like proxy for search quality

5.3 one-off benchmark / demo / anecdotal result를 robust ablation처럼 과장하는지 평가하라.

==================================================
[PHASE 6 — TESTS / DOCS / EXAMPLES HONESTY AUDIT]
6.1 tests를 분류하라:
- import/smoke
- unit
- numerical regression
- training-loop regression
- search-controller regression
- ablation/protocol tests
- evaluation/arena tests

6.2 docs-vs-code consistency
- README / ABLATION_GUIDE / TRAINING_GUIDE / configs claims vs actual implementation
- detect overclaim or premature claim

6.3 examples honesty
- example나 guide가 실제 load-bearing path를 타는가?
- placeholder or not-integrated feature를 숨기고 있지 않은가?

==================================================
[PHASE 7 — GOLDEN-REFERENCE COMPARISON]
OpenSpiel / MiniZero / KataGo / Lc0를 reference로 삼아 아래를 비교하라.

A. Architecture maturity
- self-play/learner/evaluator separation
- storage/checkpoint/logging discipline
- batched inference design

B. Controller / MCTS maturity
- controller observability
- same-budget benchmarking discipline
- noisy evaluator / wrong prior / low-budget handling
- search API clarity

C. Research usability
- controlled ablations without deep surgery?
- configs/docs enough?
- where clearly weaker?
- where genuinely novel or stronger?

주의:
- parity를 요구하지 마라.
- goldens는 research-structure reference일 뿐이다.

==================================================
[PHASE 8 — CONCRETE UPGRADE PROPOSALS]
반드시 “현재 설계 유지형”과 “부분 리디자인형” 두 축으로 나눠라.

8A. Conservative upgrades
- 현 구조를 유지하면서 실행성/실험성/해석성을 높이는 수정
예:
- controller state logging expansion
- halt-reason telemetry
- refresh/penalty disentanglement metrics
- same-budget harness
- repeated-seed arena protocol
- config simplification
- replay freshness metrics
- queue latency / throughput profiler

8B. Structural redesigns
- controller / MCTS / pipeline을 더 clean한 state machine / policy object / pipeline contract로 재구성
예:
- penalty / refresh / halt decoupling
- trust model explicitization
- budget scheduler explicitization
- evaluator uncertainty adapter 분리
- controller API redesign
- self-play / learner bridge contractization

8C. 각 제안마다:
- current pain point
- expected gain
- complexity cost
- what new experiment becomes cleaner
- remaining risk
를 적어라.

==================================================
[PHASE 9 — PSEUDOCODE UPGRADE PLAN]
반드시 pseudocode를 최소 3개 제시하라.

9.1 Current-path patch pseudocode
- 현재 quartz/select/search 구조를 크게 깨지 않으면서
  observability와 fairness를 높이는 버전

9.2 Cleaner controller interface pseudocode
- search step마다
  observe() → infer_trust() → decide_refresh() / decide_penalty() / decide_halt()
처럼 분리된 구조

9.3 End-to-end pipeline contract pseudocode
- self-play worker
- inference/evaluator
- replay/storage
- learner
- arena/benchmark
사이의 handoff와 logging contract를 명시하는 구조

각 pseudocode에 대해 설명하라:
- 어떤 pain point를 해결하는가
- complexity cost는 얼마인가
- 어떤 실험이 더 clean해지는가
- 어떤 risk는 여전히 남는가

==================================================
[PHASE 10 — MINIMAL PATCH PLAN]
전체 재작성 대신,
가장 적은 수정으로 가장 큰 project-level 신뢰성 향상을 주는 패치 10개 이하를 제안하라.

예:
- true end-to-end ablation script 추가
- controller factor logging
- halt-reason telemetry
- same-budget experiment harness
- repeated-seed protocol
- replay freshness / self-play diversity metrics
- queue latency / throughput profiler
- evaluator-quality stratified benchmark
- docs claim downgrade/clarification
- hardware-target preset configs

각 패치에 대해:
- 목적
- 막는 failure mode
- 구현 난이도
- expected gain
- “ablation-ready / design-ready / pipeline-ready” claim에 필요한지 여부

==================================================
[PHASE 11 — VERIFICATION LAYER]
11A. CoVe
- provisional synthesis를 쓴 뒤,
- 그 결론을 깨기 위한 검증 질문 8~12개를 생성하고,
- 각 질문에 독립적으로 답한 후 결론을 수정하라.

11B. contrastive comparison
다음 세 해석을 비교하라:
H1. 이 프로젝트는 실제로 실행 가능하고, search controller/MCTS 설계도 연구용으로 꽤 의미 있는 strong exploratory platform이다.
H2. 이 프로젝트는 설계 아이디어는 좋지만, 실행성/실험성/해석성 중 하나 이상이 아직 부분적이어서 major revision이 필요하다.
H3. 이 프로젝트는 문서상으론 강해 보이지만, actual code-path reality와 experimental honesty 기준에선 아직 premature하다.

차이를 아래 차원으로 분리하라:
- executability
- controller/MCTS design maturity
- self-play/training pipeline maturity
- hardware-fit
- experimental rigor
- docs honesty

==================================================
[FINAL OUTPUT FORMAT]
1. project claim reconstruction
2. executable path reconstruction
3. search controller / MCTS architecture reconstruction
4. design strengths
5. structural weaknesses
6. failure-mode analysis
7. executable reality & hardware-fit audit
8. ablation / measurement honesty audit
9. tests/docs/examples honesty audit
10. golden-reference comparison
11. concrete upgrade proposals
12. pseudocode upgrade plan
13. minimal patch plan
14. CoVe / contrastive verification results
15. final verdict:
   - conceptually promising but structurally noisy
   - good exploratory platform
   - ablation-usable with revisions
   - clean research-grade platform
16. one-line reason

[STOP RULE]
다음이 안정되면 종료하라:
- 핵심 claim status가 정리됨
- actual runnable path가 재구성됨
- controller/MCTS core design이 재구성됨
- experimental honesty verdict가 나옴
- pseudocode upgrade plan이 구체화됨
- minimal patch plan이 actionable해짐