# QUARTZ COG-01B RECOGNIZER-GENERALIZATION-07 | CPU 실제 일반화 및 MCTS 결과

2026-11-21 Asia/Seoul. 연구 브랜치 `research/quartz-cog01b-recognizer-generalization-20261121`. Parent `c92e77a1e818c0bb1359836dd4d8be09a529d287`. **Status: scoped implementation-verified + finite experimental numerical check; scientific novelty and cognitive VOC-controller superiority unresolved.**

## 1. 목표, 정확한 평가 기준 및 근거

원래 7×7 비중력 축소판 대신 완전게임 승패가 외부에서 정확히 계산되는 표준 **Connect Four 7×6 중력·4목**을 사용한다. Pascal Pons `connect4` solver의 외부 행동점수 `S(s,a)`에 대해 합법 수의 효용을

```text
U(s,a) = sign(S(s,a)) ∈ {-1,0,+1}
U*(s)  = max_{a legal} U(s,a)
Regret(s,a) = U*(s) - U(s,a)
```

로 정의한다. `-1000`은 불법 열 sentinel이므로 제외한다. `S`의 절댓값은 승리속도/패배지연 순서 정보이며 cardinal payoff로 뺄셈하지 않는다. **교사의 최적 수 집합은 한 수로 축약하지 않고 전체 `argmax U(s,a)`에 정책 타깃을 균등 배분했다.** 이 방법론에는 Kong–Ram–Yu (2026), https://arxiv.org/abs/2607.08984 와 같은 직접적 선행연구가 있다. 유사 방법의 재사용 자체를 출판 가능한 신기술이라고 주장하지 않는다.

두 실험층을 완전히 분리한다:
1. CPU 학습 자료에 대한 model-only greedy policy/value 정확도로 representation 일반화를 평가.
2. 실제 Rust `MctsEngine<Connect4>`에서 같은 root와 8→16→32→64 **누적 방문**을 쓰는 policy/value evaluator를 교체해 얻은 선택 수의 정확 WDL regret 및 물리적 비용을 평가.

Bayesian preposterior expected VOC, nonmyopic compute selection, root attention reopening은 **이번 연구에서 구현되지 않았다.** 고정 예산에서의 realized utility `U(a_2b)-U(a_b)`와 기대 계산가치는 다르다.

## 2. 데이터 authority 및 실험 전제

- Source parent `c92e77a1...`, 첫 contract+input commit `42823519b391bb94d7ee6a1088b4c090fbd4afd2`; `INPUT_LOCK.json` 13개의 소스/모델/외부 solver/book byte SHA256 기록.
- `COHORT_1024_FROZEN.json` SHA256 `cd56b7828a98b8b60f2f8d16e38f841f4a7e6c161446cdf76bfb31a07e1b2b03`: **64개 mirror-canonical 5-ply opening family ×16 roots**, 총1024, train/validation/test 640/192/192 (40/12/12 groups), 흑/백 차례 512/512. 각 group은 9,10,13,14,17,18,21,22 ply 두 번씩을 포함.
- 이전 COG06 192 root와 `standard_external_04`의 기존 12 root는 5-ply prefix/mirror뿐 아니라 실제 보드/mirror-canonical byte state에서도 중복이 없다. 열 이력에 관한 분리이며 더 짧은 prefix 1–4 moves, 분포 대표성, IID성을 보증하는 것은 아니다.
- `evidence/teacher_1024_20261121/labels.json` source `2c0857756840504c7e41e449ccf8b1a343a6afc9`, SHA256 `ac90b136cc6df0d8fc4b69b7ee178806c81b156f1f62bed55b5904b43679a22e`. **1024/1024** solver label OK, per root timeout 5s/total 180s cap, 실측 80.39s. WDL root: train win/draw/loss=452/23/165, val=130/6/56, test=144/5/43. 최적 WDL 동률 행동 개수가 전체 7인 국면 **226개**, 최적 행동 단 하나인 국면 **309개**. 독립 solver 좌우 반전 재질의 16개 전부 일치.
- 강제승이 시험 root 192중 **144**(75%)이고 확정 패배가 **43**(22.4%)이다. 확정패 root는 합법 행동의 WDL이 전부 -1일 수 있으므로 WDL regret 평가를 쉬워지게 할 수 있다. 결론은 정확히 이 집합의 조건부 판정이다.

## 3. Model-only CPU 학습과 실제 일반화

고정 두 architecture, 동일 훈련집합과 80 epochs, 2 CPU Torch threads, Adam lr0.003, wd1e-4, batch64, train-only mirror augmentation (train1280 augmented samples), `L=policy optimal-set CE+0.5 MSE(value, WDL)`, validation-loss 최소 epoch로 각각 선택. No test-time tuning.

- MLP: `126→64(ReLU)→8`, **8,648 params**, model SHA256 `a5bd016c019724fab54110458fcdcb7abf9e61fbb959afe76d6c94b8237b3f3d`.
- ConvNet: `3×6×7→Conv2d(3,16,3,pad1)→ReLU→flatten672→Linear(672,8)`, **5,832 params**, model SHA256 `161f9e6e26840754384cb47335d1082d837fa014735cbfaa2517209422e29f24`.
- 양쪽 모두 **epoch 8** 선택. 기존 COG06 checkpoint은 SHA가 고정된 참조 baseline; 새 모델 선택에 사용하지 않았다.

| Model-only 정확 WDL regret | val (192) | test (192) | test 최적 WDL 행동 선택률 |
|---|---:|---:|---:|
| 예전 192-root 학습 MLP | 0.61458 | 0.83333 | 0.56771 |
| 새 640-root 학습 MLP | 0.53125 | 0.74479 | 0.61979 |
| 새 ConvNet | **0.52604** | **0.63021** | **0.67188** |

가장 중요한 주장은 **새로 분리한 test opening-family 집합에서 ConvNet의 greedy WDL regret가 재학습 MLP보다 0.11458 낮았다**는 유한한 수치 결과다. 12개 test family 단위 3000회 fixed-model bootstrap의 `Conv−newMLP` 조건부 95% 구간 `[-0.21354,-0.02083]`. 반면 val 같은 차이는 `-0.00521`, 구간 `[-0.13542,+0.11458]`. 두 수치를 동시에 유지한다. Bootstrap은 재학습, 모델 선택 불확실성, 새로운 모집단을 포괄하지 않는다.

## 4. 실제 Rust 추론·MCTS 통합

- `src/games/connect4_conv.rs` 신설: float32 Conv3x3 3→16, ReLU, flatten 672, Linear 672→8, current-player scalar tanh, 합법 열 masked softmax. `src/games/connect4_nn.rs`의 기존 MLP evaluator를 재학습 모델에도 재사용. 외부 exact oracle의 행동점수는 **Rust MCTS가 완전히 종료된 뒤에만** 별도로 결합한다.
- Rust endpoint `research_connect4_cpu`의 `eval_mode`에 `conv3x3` 추가. `model_load_wall_us`, `root_setup_wall_us`, `step_wall_us`, `total_request_wall_us_including_fresh16`을 구분한다. 기존 uniform 및 tiny_mlp 모드는 유지.
- `PARITY_REFERENCE.json`에 36 histories(새 test 24, 과거 고정 benchmark 12)의 float32 PyTorch policy/value를 고정. Rust Conv와 재학습 MLP의 합법열 policy·value **절대오차≤3e-5** 테스트 통과. 잘못된 NN weight 형상·비유한 값을 거부한다.
- 실제 Rust MCTS CPU debug build; `mcts_demo --server`, seed17, 1 search thread, `MctsConfig::evaluation(2.0)`, fixed cumulative visits `[8,16,32,64]`, Dirichlet off. 세 model weight 파일과 uniform을 사용한 `(192 new+12 old)×4=816` 실제 root-engine 연구 요청. 모두 성공, 각 root의 fresh16 replay 일치. 기존 12에 대한 과거 uniform action+eval 호출은 정확히 재현.
- `evidence/real_MCTS_816_20261121/{raw_rust_results.json,joined_WDL.json,summary.json,execution_manifest.json,INDEPENDENT_READBACK.json}` source `c29a0b35226f305fff9da648737a8235b5771b53`, binary SHA256 `a6058b00154798271a6f0038b8543be7eabef60991c0ac960b66c7454a6ef41a`. 독립 readback이 **816개 응답·3264개 예산 시점**의 선택 행동과 정확 payoff, 게임판, 호출 수, 모델 identity를 원자료로 다시 계산하고, 새 12개 test family의 대표 국면과 좌우 반전 solver 재질의 **12+12개**에 합격.

## 5. 실제 새 시험계열 MCTS 결과: matched visits, NOT matched CPU

**192개 test roots, 12 family clusters, 흑/백 각각 96개. 낮은 regret가 좋다.**

| cumulative visits | Uniform | 기존 MLP | 새 MLP | ConvNet |
|---|---:|---:|---:|---:|
| 8 | 0.37500 | 0.40625 | 0.30208 | **0.21875** |
| 16 | 0.31771 | 0.24479 | 0.14063 | **0.13542** |
| 32 | 0.17708 | 0.23958 | 0.11458 | **0.08854** |
| 64 | 0.21354 | 0.18750 | **0.05729** | **0.05729** |

- 8 visits: Conv−newMLP mean regret `-0.08333`, family-conditional 95% interval `[-0.15625,-0.00521]`.
- 16 visits: `-0.00521`, interval `[-0.05208,+0.04167]`. 32 visits: `-0.02604` CI `[-0.09896,+0.04167]`. 64 visits: 0, CI `[-0.03125,+0.03125]`. **ConvNet의 robust superiority at all budgets는 입증되지 않는다.**
- Uniform b32→b64 regret **0.17708→0.21354 악화**. 고정된 approximate MCTS의 최종 선택 효용은 추가 탐색에 대해 단조일 필요가 없다. Bayesian posterior expected VOC의 이론적 비음성과 모순되지 않는다.
- 이전 고정 12개 국면에서도 16 방문에서 ConvNet regret=0, uniform=0.25, 구 MLP=0.08333, 새 MLP=0.16667. 그러나 표본은 선택된 흑 차례의 국면 12개이므로 대표적 기력 결론은 불가능하다.

**실제 비용의 대가: 새 test192에서 16 visits의 평균값**

| 지표 | Uniform | 새 MLP | ConvNet |
|---|---:|---:|---:|
| 누적 evaluator calls | 10.74 | 9.91 | 9.36 |
| 누적 실제 MCTS 실행 wall time (ms) | **3.97** | 4.54 | 8.55 |
| 모델 로드+root 초기화+탐색 (ms, 아래 제한) | **5.26** | 11.71 | 13.86 |

- 위 시간은 현재 **Rust debug build, 단일 search thread, 파일에서 model JSON을 root마다 다시 로드**하는 연구용 경로의 실측 평균이다. `step_wall_us`는 MCTS 실행만, `estimated_end_to_end`는 모델로드+root 초기화+MCTS stages를 더한 수치이며 fresh16 검증 재실행 시간은 제외한다. 실제 서비스에서 checkpoint를 한 번만 상주 로드한 경우의 amortized 시간은 별도 측정해야 한다.
- **단일 evaluator call을 '동일한 계산비용'으로 취급하면 안 된다.** Conv forward는 수치적 곱셈·메모리 접근이 더 많고 디버그 구현에 비최적화 루프가 있다. Conv의 WDL 이득을 '동일 CPU 시간당 효율 개선'으로 승격하지 않는다.
- 유한 자료의 평균 비용–regret Pareto에서 uniform 32 visits (7.30ms, regret 0.1771)는 Conv8 visits (9.84ms, regret 0.2188)보다 **빠르고 효용도 좋다**. 따라서 model+search를 합친 실제 시간 예산에서는 최선 선택이 달라질 수 있다. 이 비교도 posthoc aggregate 평균이므로 엄밀한 per-root fixed-time A/B가 아니다.

## 6. 검증, 적대적 결함 분류 및 후속 연구

- Source/theory: Connect Four gravity/game outcomes matched external solver; signed WDL, exact oracle root result. Legacy eval results old12 reproduced; old COG06 and previous base frozen. This is a finite external-solver-supported claim, not a formal independently proved correctness theorem for the solver.
- Implementation: full Rust suite **587 PASS / 0 FAIL / 89 ignored**, related Python research tests **80 PASS / 0 FAIL**, including new G07 tests10, COG06 tests9, prior gates. TDD red observed for missing grouped teacher behavior, missing spatial inference and missing MCTS mode, then green. No global CUDA access.
- Numerics: Torch↔Rust 36 examples policy/value parity ≤3e-5; model weights/teacher labels/solver and compiled Rust binary SHA256 in executed-source manifests; independent solver 12+12 requery, all WDL actions and costs full readback. Stopwatch measurements noisy, debug binary and per-root load constrain extrapolation.
- Scientific limits: teacher labels came from 64 selected (not random population) mirror-disjoint families; 192 descendants in 12 test families correlated. 144/192 new test positions root strong win, 43 forced loss and 5 draw. Forced-loss roots can have optimal regret0 for any legal action. Single training seed, one architecture/optimizer setting, no retraining uncertainty, no statistical population claim.
- Native subagent `NOT_RUN`: current host exposes no native delegation tool. Direct tools, Wolfram arithmetic, exact external oracle, SciSpace/Exa/Parallel primary literature checks and serial independent code readback were used. No simulated agent receipts.
- Neither new learned network nor existing uniform implements **Bayesian expected value of computation**. This work restores a functional, **measurably partially transferable recognition prior**, not the full cognitive control/attention engine.

**Next bounded research node: `QUARTZ-COG-02-COST-MATCHED-VOC-08`.**

1. Use the existing independent 12-family test and separately frozen train/val families, but create any VOC predictor on **training families only**: observed root `F_b` (current policy gap, visit distribution, evaluator Q uncertainty, actual completed calls, pending/TT status) → conditional signed realized `U(a_{2b})-U(a_b)` and measured additional CPU expense. Do not input exact solver actions/values into runtime. Predict **future computation benefit**, not existing action-quality confidence.
2. Implement a reproducible resident-root **wall-time or completed-evaluation budget** scheduler. Amortize model file load once per engine process, and distinguish model initialization overhead from per-move work. Compare fixed-time uniform, flat MLP and spatial ConvNet using true extra completed evaluations and observed time. Same visits are not a cost-match.
3. Compare learned stop/continue decisions to fixed stage policies at matched actual cost; include negative transition events, zero-benefit spending and root candidate omission. Risk: myopic VOC can miss synergistic two-step information, already known from previous R1–R10 theory. Model failure is a legitimate negative result.
4. Only if saved CPU resource and improved exact WDL regret both survive family-heldout tests should reopen `COG-02 Einstellung / candidate attention reopening` to improve low-prior decisive-move recall. Strong Gomoku15/Rapfi transition is a separate scientifically distinct transfer.

## 7. Reproduction and immutable source identity

Frozen evidence and code:

- A `42823519b391bb94d7ee6a1088b4c090fbd4afd2`: 1024-root frozen cohort, input/contract.
- B1 `2c0857756840504c7e41e449ccf8b1a343a6afc9`: bounded exact teacher runner.
- B2 `49408c0d51d33462345f534646bee9f3c6b7aba7`: 1024 exact action labels and mirrored independent audit.
- C1 `588495c705093b8618f9cc41aa4303677df23f8e`: fixed CPU two-architecture training code.
- C2 `f550926e19c557bdd2d94e675cda47e8d88a57f7`: trained MLP+Conv checkpoint, per-split metric, family bootstrap audit.
- D1 `9efe75157cb952e4caf4b5562ec42cda4eab5a9d`: Rust Conv implementation, model loading/timing, 36-position cross-language parity.
- D2 `c29a0b35226f305fff9da648737a8235b5771b53`: actual four-arm Rust MCTS benchmark code. **This clean source generated the MCTS result.**
- D3 `5ae661b5dc132a429cdc6f697d8d43f6db304cc1`: raw 816 engine results + independent readback, SHA manifest.
- E final: this report and next handoff are published separately; read back final commit/tree after publication.

Each completed stage follows Git commit → local Dropbox tar.gz → archive SHA256 → GitHub push → remote `ls-remote` readback. Archives at `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups/`. **Local Dropbox directory backup is not an independently verified cloud sync ACK or archive restore test.** Original developer checkout `/home/cosmosapjw/Dropbox/personal_projects/quartz`, main and unrelated research worktrees untouched. Third-party AGPL source, book and executable are not redistributed through this project branch or archives.
