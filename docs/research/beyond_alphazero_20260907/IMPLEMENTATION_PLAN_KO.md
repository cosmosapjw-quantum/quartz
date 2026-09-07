# Quartz 고예산 반복 탐색 Implementation Plan

> **For agentic workers:** 아래 W1–W7을 작업 단위로 실행하고, 각 단위의 코드·수학·실험 결과를 분리해 기록한다.

**Goal:** 구조적 잠재 효용 belief와 계산 배분이 실제 고예산·반복 검색의 비용 대비 결정 품질을 개선하는지 검증한다.

**Architecture:** coherent observation을 저차 posterior로 갱신하고, 그 posterior의 계산 후 변화에서 KG와 batch 배분을 얻는다. 후보 재개방과 모델 변경 시 재사용을 별도 검증한 뒤 기존 엔진의 좁은 제어 루프로 연결한다.

**Tech Stack:** Python/NumPy/SciPy, stdlib unittest, 기존 Rust MCTS/Foundry 인터페이스, 고정 PyTorch evaluator와 별도 학습 단계.

작성일: 2026-09-07. 소스 기준: `a74adb4e0176b0e7f0ad6cb7c1870a483b787749`, tree `4ea3240170cd4e778a5a9219f9ed49b7fc21758c`.

이 문서는 **앞으로 수행할 작업**을 정의한다. 아래 파일·인터페이스·실험·통과 기준은 별도 표시가 없는 한 제안이며, 구현 완료나 성능 달성을 뜻하지 않는다. 이번 세션의 실제 수학 검증과 작은 Gaussian 계산 실험은 같은 디렉터리의 `RESEARCH_RECORD.md`, `test_research_math.py`, `gaussian_batch_pilot.py`, `evidence/`에 기록한다. 그 결과를 게임 엔진 검증으로 승격하지 않는다.

사용자는 이번 연구 문서 작성·계산 실험·push를 승인했다. 이 계획은 그 작업을 재승인 대기 상태로 바꾸지 않는다. 기존 Foundry 실행·resume·lineage 인프라 변경은 `docs/plans/2026-08-24-idea-foundry-trusted-local-audit-and-codex-pr-plan.md`의 PR 순서와 계약을 따른다. 여기서 제안하는 진단은 별도의 연구 경로다. 과거 v1 설정, 동결된 연구 의미, A01–A26 상태와 역사적 실패 기록을 수정하거나 새 26축 캠페인을 시작하지 않는다.

## 연구 목표와 기여 경계

Quartz의 정체성을 다음처럼 유지한다. **작은 NN은 패턴과 관계를 제안하고, 탐색은 잠재 효용의 차이에 대한 믿음을 갱신하며, 다음 계산은 최종 선택을 개선할 가능성과 비용으로 배분한다.** 대안 영역을 여러 개 유지하고 필요하면 다시 연다. 통계장·자유에너지·정보기하학은 이 믿음의 주변화, 상관, 근사 오차와 다봉성 탐색을 계산하는 방법으로 사용한다. 인간 GM의 선택적 사고는 이 동작의 독립적인 관측 대상이지, 노드 수나 엔트로피를 흉내 내도록 직접 최적화하는 목표가 아니다.

개별 재료의 선행연구 중첩은 강하다. [Sezener–Dayan 2020](https://proceedings.mlr.press/v124/sezener20a.html)은 이미 상관된 Gaussian/GP 효용 믿음과 MCTS의 VOC를 다룬다. [EMCTS 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/bcbd670951a6dede2123961de19e5ea3-Paper-Conference.pdf)는 동일 모델의 반복 백업을 독립 증거로 취급하지 않는 불확실성 전파를 다룬다. 따라서 기여 후보는 “Bayesian+VOC”라는 이름이 아니라 **증거 중복과 모델 변경을 추적하는 지속적 잠재 표현이 실제 비용 안에서 상관 구조를 이용하고, pending 계산과 영역 재개방까지 같은 의사결정 목표로 묶어 고예산 성능을 개선하는가**이다. 아직 입증되지 않았다.

비교 대상은 표준 PUCT뿐 아니라 정확한 상관 KG, Sezener–Dayan식 계산 배분, EMCTS식 상관에 보수적인 불확실성, Gumbel 탐색, pending-count/WU 방식이다. 원 논문의 완전 재현이 아니면 반드시 “해당 원리를 구현한 통제군”으로 표기하고 논문 성능을 재현했다고 쓰지 않는다. 작은·큰 평가기 조합에는 [MPV-MCTS](https://www.ijcai.org/proceedings/2019/0653.pdf)가 직접적인 선행 기준선이다.

## 전체 순서

| 작업 | 산출물 | 의존성 | 다음 단계로 넘길 증거 |
|---|---|---|---|
| W1 | 관측·pending·모델 epoch 계약 | 없음 | 동일 증거가 중복 누적되지 않는 완료 블록 |
| W2 | 지속적 잠재 효용차 믿음 | W1 | 효용차·갱신량의 held-out 보정과 실패 사례 |
| W3 | 올바른 KG와 pending 조건부 배치 배분 | W2 | 계산 개입의 예측 이득과 실현 이득 비교 |
| W4 | 되돌릴 수 있는 영역 휴면·재개방 | W2, W3 | 누락 복구가 소모한 기회비용보다 큰지 |
| W5 | 좁은 live 루프와 고정 NN 파일럿 | W3, W4 | 실제 NN 호출·지연·결정 품질의 예산 곡선 |
| W6 | 고예산 확인 실험과 절약 비용 재투자 | W5 | 독립 문제군에서의 비용·품질 효과 및 CI |
| W7 | 작은·큰 평가기 선택과 독립 학습 | W6의 유효한 배분 모델 | MPV 대비 비용 효과, 독립 seed 학습 효과 |

W4의 기전이 실패하면 W5는 W3만 검증할 수 있다. W2의 학습된 공분산이 실패하면 대각·보수적 상관 통제군으로 축소한다. 구성 요소의 실패를 전체 결합 실험으로 덮지 않는다.

## W1 — 관측 의미와 재사용 수명을 먼저 고정한다

**실제 소스 연결점.** `src/mcts/backup.rs::backprop`, `src/mcts/policy/trait_def.rs::{SearchSnapshot, EdgeView}`, `src/mcts/mod.rs`의 snapshot 생성과 NN evaluation counter, `src/mcts/foundry/types.rs::{FreshnessIdentity, RuntimeSnapshot, UncertaintyChannels}`를 시작점으로 삼는다. 현재 `backup.rs`의 N/W/M2 개별 갱신은 병렬에서 하나의 원자적 Welford 갱신이 아니다. `SearchSnapshot.iteration`은 완료 방문 수와 다를 수 있으며, `FoundrySearchPolicy::identity_and_extras`에는 기본값 runtime 정보가 들어간다. 이 값에서 독립 표본 수나 인증된 표준오차를 바로 만들지 않는다.

**제안 인터페이스.** 별도 연구 Python 모듈 `observation_contract.py`에 다음 구조를 먼저 만든다. 엔진 계측은 검증된 스키마가 확정된 뒤 최소 범위로 연결한다.

```python
ObservationKey(root_hash, state_hash, action_id, leaf_hash,
               model_epoch, representation_epoch, target_epoch,
               independent_block_id, evaluation_id)
CompletedBlock(key, n_completed, sum_value, centered_sum_sq,
               started_ns, completed_ns, depth, provenance)
PendingJob(job_id, root_hash, candidate_generation, model_epoch,
           representation_epoch, expected_observation, enqueued_ns)
BeliefStore.invalidate(reason, old_identity, new_identity)
```

`target_epoch`는 무엇을 추정하는지 고정한다. 작은 정확 게임의 minimax, 특정 고정 continuation 정책의 기대값, 또는 명시된 기준 평가값을 섞지 않는다. 같은 leaf의 동일 모델 deterministic 재평가는 새 독립 증거가 아니다. 확률적 rollout은 별도의 RNG block과 생성 정책을 가진 관측이다. transposition은 경로 수만큼 중복 업데이트하지 않는다. 모델 또는 잠재 표현이 바뀌면 이전 likelihood/posterior를 무효화한다. 원시 관측을 보존해 새 모델로 재해석하는 경로와, 새 NN 출력을 관측했다는 경로를 구별한다.

**검증과 명령.** W1에서 생성할 `tests_research/test_observation_contract.py`를 `python -m unittest discover -s docs/research/beyond_alphazero_20260907/tests_research -p 'test_observation_contract.py'`로 실행한다. 동일 evaluation 재전달, 다른 경로의 동일 leaf, out-of-order completion, root 교체 뒤 늦게 온 결과, 모델 epoch 변경, 독립 noisy 반복이라는 여섯 경우를 확인한다. snapshot은 완료 블록의 merge와 단일 순서 기준값이 일치해야 한다. 실제 Rust 연결을 추가한 경우에만 `cargo test --release --features idea-foundry observation_contract`를 실행하며, 해당 필터의 테스트는 그 연결 작업에서 새로 만든다.

**통과/중단.** 중복 업데이트와 다른 epoch 결과의 혼입은 모두 0이어야 한다. 이 불변식 실패 시 posterior·정지 인증으로 승격하지 않는다. 계측 부하가 크면 모든 노드 이벤트 수집 대신 완료 블록 집계로 축소하되, 관측의 식별 가능성을 없애지 않는다. 현재 환경에 Rust가 없다는 사실은 향후 워크스테이션 테스트 통과를 뜻하지 않는다.

## W2 — 지속적인 잠재 효용차 모델을 보정한다

**실제 소스 연결점.** `EdgeView::sigma_a`, `src/mcts/foundry/types.rs::UncertaintyChannels`, `src/mcts/foundry/control.rs`의 A03, `quartz/phase15_trace.py::build_trace_artifact`를 관측 입력과 비교 대상으로 사용한다. 현재 root action 평균 사이의 분산을 prior variance로 쓰는 경로는 “행동이 다르다”와 “그 차이를 모른다”를 혼동할 수 있다. 기존 의미를 조용히 고치지 말고 새 연구 출력을 나란히 기록한다.

**제안 인터페이스.** `latent_belief.py`에 `UtilityBelief(mean, low_rank_factor, residual_var, bias_floor, identity)`와 `predict_contrast(a,b)`, `predict_update(computation)`, `observe(block)`, `transport(successor_map)`, `reset_epoch(...)`를 만든다. 작은 NN의 특징 `φ(s,a)`를 잠재 계수의 기저로 사용하되 초기 파일럿에서는 특징을 동결한다. 효용차 분산은 `Σaa + Σbb - 2Σab`로 계산한다. 낮은 rank와 대각 residual을 비교하고, bias/drift floor를 표본 수로 자동 소거하지 않는다. 이웃 root로의 전달은 상태·관측 대응이 확인될 때만 허용한다.

**물리·정보기하학 작업.** Gaussian 기준선에서 시작하고 정확한 finite-support 분배함수와 posterior 평균의 차이를 확인한다. `gβ(θ)=β⁻¹log Σa Pa exp(βθa)`의 2차 보정은 `β/4 Σab πaπb Var(θa−θb)`다. 이 식은 posterior soft value의 근사이며 그 자체가 KG나 각 행동의 탐색 bonus는 아니다. simplex tangent measure/Jacobian, 두 basin의 합, local Laplace 실패, 공통 모드 소거를 수치 검증한다. 기존 수학 파일의 범위를 확장할 경우 새 조건과 실행을 기록한다. β 증가와 불확실성 감소의 결합 극한을 확인하지 않고 “고예산이므로 Laplace가 맞다”고 가정하지 않는다.

**검증과 명령.** W2에서 생성할 `test_latent_belief.py`에 완전 공통 모드, 양·음의 상관, 잘못된 부호, 낮은 rank로 표현 불가능한 오차, 모델 교체, 반복 root를 포함한다. `python -m unittest discover -s docs/research/beyond_alphazero_20260907/tests_research -p 'test_latent_belief.py'`로 실행한다. held-out opening family에서 효용차의 50/80/95% 구간 coverage와 폭, 예측된 posterior mean 변화의 분포, 잘못된 확신의 빈도를 함께 평가한다. exact minimax가 없으면 stronger reference에 대한 일치를 사실상 진실의 coverage로 부르지 않는다.

**통과/중단.** 보정된 모델과 실제 공분산을 아는 oracle, 대각 모델, 보수적 상관 bound, 공분산을 섞은 모델을 비교한다. 제안 gate는 held-out 95% 구간의 관측 coverage가 군집 CI와 함께 보고되고, systematic undercoverage가 5%p를 넘는 층에서는 인증 정지를 금지하는 것이다. 이 5%p는 새 실험의 운영 허용치 제안이며 달성 사실이 아니다. 폭만 크게 만들어 coverage를 얻었다면 utility차 예측의 sharpness와 W3 이득이 뒷받침되어야 한다. 추정 공분산이 대각 대비 이득이 없으면 상관 학습을 보류하고 원인·비용을 남긴다.

## W3 — 진짜 한 번의 KG와 pending 조건부 배치를 구현한다

**실제 소스 연결점.** `src/mcts/policy/kg_stop.rs::kg_per_arm`, `src/mcts/foundry/control.rs`의 A04, `src/mcts/foundry/types.rs::{CostVector, CostPrices, ProposalEstimate}`, `src/mcts/foundry/search.rs`의 A13/A15, `RuntimeSnapshot`을 비교한다. 기존 `kg_per_arm`은 현재 gap uncertainty를 계산 후 변화량 대신 사용하고 leader KG를 0으로 둔다. A04의 gain proxy에 0.5를 곱한 값도 검증된 LCB가 아니다. 따라서 기존 기록을 “수정된 KG”로 재해석하지 않는다.

**제안 인터페이스.** `computation_value.py`에 다음을 만든다.

```python
Computation(kind, target, evaluator_id, observation_model, cost_model)
kg(belief, computation, pending) -> GainEstimate
select_batch(belief, pending, candidates, remaining_cost) -> BatchPlan
GainEstimate(mean, uncertainty, approximation, calibration_id)
```

Gaussian 관측 `Yc=hcᵀθ+ε`, `Var ε=rc`에서 posterior mean 변화 방향은 `uc=Σhc/sqrt(hcᵀΣhc+rc)`다. 두 후보의 현재 gap Δ와 변화 표준편차 `s=|uc,a−uc,b|`로 `KG=sφ(Δ/s)−ΔΦ(−Δ/s)`를 계산하며 leader 관측도 포함한다. 여러 후보는 envelope 적분 또는 posterior predictive Monte Carlo와 비용을 비교한다. pending 관측이 만드는 조건부 사후 분포 위에서 다음 배치의 **추가** 이득을 평가한다. 아직 관측하지 않은 값을 결과처럼 넣지 않는다. 이미 예약한 계산의 비용·정보를 다시 세지 않는다. myopic KG=0이 모든 다단계 계산의 가치=0이라는 정지 정리는 아니다.

**검증과 명령.** 현재 존재하는 `python docs/research/beyond_alphazero_20260907/test_research_math.py`는 작은 수학·source probe 실행 명령이다. W3의 추가 테스트는 새 `test_computation_value.py`로 만들고 unittest로 실행한다. 적분과 폐형식, leader-only 정보, 같은 gap variance지만 다른 update variance, pending 중복, 독립 noisy 반복, 잘못된 공분산, 한 단계 가치가 작아도 두 단계 후 개선되는 경우를 포함한다. `quartz/experiments/forked_voc.py::voc_proxy`는 정책 TV 움직임의 합이므로 실현 효용 증가의 label로 그대로 쓰지 않는다. 새 label은 동일 prefix에서 fork한 계산/무계산의 최종 선택을 외부 기준으로 평가한 **부호 있는** 효용 차이다.

**통과/중단.** exact Gaussian의 수치 일치는 미리 정한 부동소수점 허용오차 내에서 성립해야 한다. 게임에서는 예측 gain 구간별 실현 gain과 순위를 out-of-family에서 검증한다. uniform뿐 아니라 known-noise Neyman 배분(고예산 점근 기준선), 대각 KG, stale-batch KG, pending-count/WU, Sezener–Dayan식 상관 KG와 비교한다. uniform만 이기면 “배분의 필요성”까지만 인정한다. 계산 overhead를 포함한 이득이 없으면 KG를 amortize하는 후속 가설로 분리하거나 저렴한 기준선을 유지한다.

## W4 — 영역을 지우지 않고 휴면시켰다가 다시 연다

**실제 소스 연결점.** `src/mcts/policy/reservoir.rs::Reservoir`, `src/mcts/foundry/search.rs`의 A07/A11, `src/mcts/foundry/types.rs::MetaAction`, `src/mcts/mod.rs`의 `n_visible`/EdgeView 조립을 사용한다. 현 reservoir cooldown bookkeeping, `ResampleMode` 제안, activation 문구만으로 실제 재개방을 했다고 볼 수 없다. `n_visible=n_children`도 materialized 후보와 전체 legal 후보의 차이를 지울 수 있다.

**제안 인터페이스.** `basin_reservoir.py`의 `BasinState(id, members, generation, active, completed, pending, upper_utility, last_probe)`와 `hibernate`, `propose_reopen`, `commit_reopen`, `invalidate_membership`를 만든다. 행동은 active/hibernated 상태 사이에서 되돌릴 수 있고, terminal proof로 지운 후보와 예산상 잠시 닫은 후보를 구별한다. 변화 증거가 있으면 cooldown을 우회할 수 있다. basin별 quota와 외부 후보 upper bound를 로그에 남긴다. `REOPEN`은 새 제안 인터페이스 이름이며 현재 production에 존재하는 CLI/실행 동작이 아니다.

Wang–Landau의 density-aware proposal은 후보 생성에 한정해 실험한다. 고정 epoch에서 상태 공간·bin·목표 measure·가중치 보정을 정의한다. 최종 효용을 bin의 평탄함이나 경로 개수로 대체하지 않는다. coarse energy를 사용하면 각 basin의 `−τ log Σ m(x)exp(−E(x)/τ)`를 작은 exact tree에서 계산해 근사를 비교한다. 상대의 응답을 모두 긍정 가중치로 더하면 minimax 목적이 바뀔 수 있으므로 backup 목적은 별도로 유지한다.

**검증과 명령.** W4에서 생성할 `test_basin_reservoir.py`를 unittest로 실행한다. 낮은 prior의 결정적 반박, 같은 basin에 숨은 희귀 승리, 많은 열등 중복 경로, 단봉 문제, 재개방 직전 epoch 교체, pending 후보의 휴면을 시험한다. 비교는 고정 reservoir, 무작위 재개방, density-only, VOC-only, 결합으로 나누되 모두 한 번의 작은 기전 screen에 한정한다.

**통과/중단.** 누락 회복률·회복까지의 계산량뿐 아니라 같은 비용에서의 최종 regret를 본다. 회복된 결정의 가치가 다른 후보 탐색을 포기한 손실보다 커야 한다. density-only가 방문 다양성만 높이고 regret를 줄이지 못하면 그 가설을 중단한다. 어떤 observable도 희귀 winning branch를 구분하지 못하는 needle 반례에서는 일반적인 지수 탐색 절감을 주장하지 않는다.

## W5 — 좁은 live 루프와 고정 NN 파일럿

**실제 소스 연결점.** `FoundrySearchPolicy::{observe, should_halt, score_adjustment}`, `src/mcts/foundry/coordinator.rs::{FoundryCoordinator, GuardedMetaActionExecutor}`, `quartz/phase15_online.py::run_online_readout`, `quartz/phase15_trace.py::build_trace_artifact`가 연결점이다. 현재 live bridge는 A01 STOP/Noop 중심이며 SAMPLE·REOPEN이 실제로 탐색을 바꾼다는 검증이 필요하다. 기존 Foundry PR 계획이 정의한 freshness/action 계약을 우회하지 않고, 새 진단 adapter에서 `SAMPLE / REOPEN / STOP`만 실행한다. 연구 proposal 번호를 기존 A축 evidence 승격으로 매핑하지 않는다.

**제안 실행물.** W5에서 `run_protocol.py`, `protocols/fixed_nn_pilot.json`, `test_live_adapter.py`를 새로 만든다. driver는 manifest의 source/checkpoint hash, opening family, tree seed, 비용 제약과 controller identity를 캡처한다. 다음 명령은 **W5가 해당 파일과 CLI를 만든 뒤** 사용할 연구 명령이다. 현재 production 플래그가 아니다.

```bash
python docs/research/beyond_alphazero_20260907/run_protocol.py --manifest docs/research/beyond_alphazero_20260907/protocols/fixed_nn_pilot.json --output /path/to/new-run
```

unit test는 shadow에서 탐색이 안 바뀌는지, live SAMPLE/REOPEN이 정해진 한 계산만 바꾸는지, candidate/model epoch가 다른 proposal은 실행되지 않는지, pending accounting 후 STOP의 비용 정산이 맞는지를 본다. 실제 queue enqueue/start/complete, batch size, evaluator별 distinct call, cache hit, outstanding jobs, CPU controller time, peak memory를 수집한다.

**실행 규모 제안.** 사용자가 알려 준 Ryzen 5900X·RAM 96GB·RTX 3090은 계획상의 워크스테이션 가정이다. 이번 환경에서 측정한 사양·처리량이 아니다. 첫 측정은 1시간 이내의 shipped checkpoint latency/메모리 probe로 제한한다. 학습된 NN 하나를 고정하고 12 opening family × family당 2 root × 3 tree seed에서, screen으로 선택한 3개 정책을 `32,128,512,2048,8192` 완료 평가 예산으로 비교한다. 모두 독립 실행하면 상한은 **2,356,992 평가 slot**이다. slot·실제 NN call·방문 수는 따로 기록한다. budget-dependent 정책의 최대 예산 prefix를 짧은 예산의 독립 실행처럼 재사용하지 않는다.

serial cold-root로 시작하고, 같은 root의 재분석 및 실제 successor/transposition reuse를 별도 층으로 추가한다. 반복 검색 자체와 새 증거의 추가를 구별한다. pending 검증은 `width=1,8,32`의 선정된 부분 집합에서만 수행해 전체 조합 폭증을 피한다. 각 비교에서 method·budget·width를 사전에 고정하고, 동일 family/tree-seed 난수의 pairing을 보존한다. pilot latency로 시간을 산정한 뒤 최대 24시간/원시 자료 20GB의 제안 envelope 안에 들어오지 않으면 문제 수나 별도 batch 층을 줄여 **실행 전** manifest를 확정한다. 빠른 처리량을 가정해 실행 시간을 확약하지 않는다.

**통과/중단.** accounting/freshness 오류 0, 실패·timeout 기록 누락 0을 요구한다. 학습은 하지 않는다. 높은 budget에서 이득이 사라지면 저예산 결과와 분리하며, controller overhead가 wall-clock 이득을 없애면 실제 비용에서 실패로 처리한다. 관측 손실이나 잘못된 call budget은 그 셀을 과학적 비교에 넣지 않고 재현 가능한 오류로 남긴다.

## W6 — 고예산 확인과 총비용 재투자

W5는 효과 크기와 분산을 알아보는 탐색 실험이다. W6는 새 held-out opening family와 새 tree seed를 사용한다. PUCT, 고정 NN에서 충분히 검증한 현대 기준선 하나, surviving Quartz 하나의 최대 3개 arm으로 좁힌다. full covariance oracle은 작은 모형의 진단 기준선이지 실제 게임의 공정한 성능 경쟁자로 넣지 않는다.

**비용·품질의 서로 다른 추정 대상.** 다음 세 가지를 분리한다.

| 실험 | 맞추는 자원 | 주요 질문 |
|---|---|---|
| 호출 수 일치 | 실제 완료 NN calls; 이 단계는 평가기 하나 고정 | 같은 평가 정보 예산에서 선택이 좋아지는가? |
| 시간 일치 | 엔드투엔드 wall time; belief·proposal·queue 비용 포함 | 사용자에게 같은 시간 안에 더 좋은 결정을 주는가? |
| workload 재투자 | 전체 root 묶음의 고정 call 또는 고정 시간 한도 | 쉬운 root에서 절약한 자원을 어려운 root에 실제로 써서 평균·tail 손실을 낮추는가? |

`32,64,128,256,512,1024,2048,4096,8192`를 기본 곡선으로 사용한다. `32768`은 8192에서 tree 포화가 심하지 않고 자원 probe가 허용하는 별도 stress 층이다. 작은 게임에서 이미 exact solved root가 되면 32768의 실제 배분 효과를 그 게임에서 검증할 수 없다고 보고하고, 더 큰 board/더 어려운 위치를 새 protocol로 정의한다. 8–64 visit에서 작동한 `1/N` 보정만으로 이 고예산 가설을 대변하지 않는다.

**관리 가능한 시작 규모.** 제안 확인 root suite는 48 opening family × 2 root × 3 tree seed × 3 arm으로 시작한다. 기본 예산을 각각 독립 실행할 경우 상한 **14,128,128 평가 slot**이다. 32768 stress는 12 추가 family × 1 root × 2 seed × 3 arm = **2,359,296 slot**로 분리한다. wall-clock 확인은 모든 조합을 반복하지 않고 512/2048/8192 기준선에 대응하는 세 시간 한도로 축소한다. 첫 pilot 이후 측정된 marginal cost와 아래 CI precision 규칙으로 총 root 수·게임 수를 확정한다. 이 숫자는 성능 검증에 충분하다는 보장이 아니다.

**분석 단위와 효과 크기.** root·색상 반전·budget prefix·반복 재생을 독립 표본으로 세지 않는다. family 안의 paired method 차이를 먼저 집계하고 family를 bootstrap/resample한다. 고정 NN의 tree seed는 family 내부 반복이다. 별도의 학습 결과에서는 training seed가 상위 단위다. 주 지표는 exact utility가 있는 root의 simple regret 또는 paired game score다. 같은 NN의 고예산 reference와 KL이 줄어든 결과는 참고 지표다. exact truth가 없으면 서로 다른 강한 reference와 전술 proof, paired games를 함께 사용하되 reference uncertainty를 남긴다.

제안 사전등록은 다음과 같다. 높은 구간 `2048,4096,8192`의 평균 absolute regret 감소를 하나의 주 estimand로 삼고, 낮은 budget과 각 점별 결과는 보조로 공개한다. utility를 `[-1,1]`로 정했을 때 실용적 effect 후보는 0.01의 평균 regret 감소, 또는 baseline regret가 충분히 클 때 10% 상대 감소다. 게임에서는 +3%p score-rate 개선을 검토할 최소 효과, −2%p를 비열등성 margin 후보로 둔다. 비용 효율은 품질 비열등성을 먼저 통과한 뒤 10% actual cost 절감을 별도로 시험한다. 모두 **pilot 전에 검토할 제안 기준**이며 달성된 숫자나 역사적 P2 ≥30% 목표의 대체 결과가 아니다.

CI는 95% paired cluster interval을 사용한다. 파일럿의 family-level paired 차이 표준편차를 `s_d`라 하면 `t_(K−1,.975)·s_d/sqrt(K)`는 평균 효과 CI 반폭의 초기 근사다. 예컨대 목표 반폭 `h`에 필요한 `K≈(1.96s_d/h)^2`는 **정밀도 산정**이지 “80% power”가 아니다. heavy tail·family 크기 불균형을 반영해 pilot cluster 재표집으로 CI 폭과 선언한 효과에서의 검정력을 시뮬레이션한다. 현재 pilot에서 관측한 유리한 평균을 참 효과로 놓고 낙관적인 power를 만들지 않는다.

paired game 확인은 48 새 opening family × 2 game seed × 색상 교환 pair, 즉 96 pair/192 game부터 시작할 수 있다. 이 수로 +3%p를 구분할 수 있다고 가정하지 않는다. W5 분산에 따른 사전 산정에서 필요하면 최대 384 family의 768 pair/1536 game까지 계획하고, 정해진 compute cap으로 그 정밀도를 얻을 수 없으면 결과를 **불충분**으로 끝낸다. 중간 유의성에 따라 유리한 시점에 멈추지 않는다. 늘릴지 여부는 blind variance/resource 정보 또는 사전에 정의한 sequential CI 규칙으로만 결정한다.

**재투자와 GM 관측.** 고정 workload에서 root 간 배분을 수행한 뒤 candidate 집중도, 깊이, 수정 시점과 offline 독립 oracle의 계산 가치에 대한 배분 상관을 평가한다. runtime은 자체 VOC 추정치를 사용할 수 있다. 금지해야 할 것은 held-out oracle label을 runtime/tuning에 유출하는 것이다. 총비용을 줄인 정책은 절약량을 기록한 뒤 추가 root 또는 어려운 root의 추가 search에 배분한다. 사용하지 않은 예산을 “동일 총비용에서의 성능 향상”으로 계산하지 않는다.

**음성 결과 경로.** 고예산 NI 실패 → 저예산 전용 가설로 범위 축소. NN calls는 절약하지만 wall time 악화 → 시스템 비용 실패로 보고. oracle covariance만 이김 → 학습/보정 문제를 남기고 실용적 기여를 주장하지 않음. known-noise 또는 Sezener–Dayan식 기준선과 동률 → 익숙한 기전 재현으로 정리. 재투자 전후 동률 → 적응 정지의 workload 이점을 입증하지 못한 결과로 남긴다. 어느 경우에도 다음 단계의 대규모 학습을 결과 복구 수단으로 자동 실행하지 않는다.

## W7 — 작은·큰 평가기와 학습 효과를 따로 검증한다

**실제 소스 연결점.** `docs/idea_foundry/04_representation_training.md`, `scripts/idea_foundry/a23_cpu_incremental_pattern_student.py`, `scripts/idea_foundry/a18_diffusion_regularized_evaluator.py`, `quartz/idea_foundry/a18_ablation.py`는 관련 진단/학습 경로다. 현재 axis 이름이나 prototype 존재가 MPV 전환기 또는 지속 믿음 evaluator를 완성했다는 뜻은 아니다.

**제안 인터페이스.** W3 `Computation.evaluator_id`를 `small`, `large`, `deepen`, `reuse`로 확장하되 각 관측 likelihood와 오차 상관을 따로 둔다. 먼저 학습된 small/large checkpoint를 모두 동결하고 작은 네트워크 단독, 큰 네트워크 단독, 고정 혼합 MPV-style, VOC 선택을 비교한다. 작은 모델이 제안한 특징을 큰 모델 평가로 검증하고 잘못된 prior·공유 bias를 포함한다. 서로 다른 evaluator의 call 한 개를 동등 비용으로 합산하지 않는다. 호출 벡터 `(n_small,n_large)`, 측정된 service cost, wall time을 모두 보고한다. 공정한 call 제약 실험이면 평가기별 quota를 같이 고정하고, 자유로운 evaluator 선택은 시간/비용 일치 실험으로 평가한다.

**독립 학습 단계.** 고정 모델에서 효과가 확인된 배분/재투자 규칙만 동결한 뒤 학습한다. W7에서 `protocols/training_pilot.json`과 같은 별도 manifest를 새로 만들고 controller/checkpoint lineage를 보존한다. 초기 3 training seed/family는 분산·오류를 확인하는 pilot이며 정식 training claim에 충분하다고 가정하지 않는다. 고정 총시간 또는 총 evaluator 비용에서 baseline, search 개선만, 재투자 포함, MPV 포함을 순차적으로 최소 비교한다. 남길 arm을 pilot에서 고르고 새 최소 6 independent training seed의 확인 규모를 분산/compute cap으로 확정한다. 필요량이 cap을 넘으면 확인 불충분으로 남긴다.

학습이 checkpoint를 바꿀 때마다 W1 model/representation epoch를 갱신한다. 비교군은 reset-on-change, 검증된 raw-evidence 재해석, 허용된 transport다. stale posterior를 유지하는 군은 정상 후보가 아니라 의도적인 실패 통제군이다. 같은 root의 재분석이 새 sample을 만든 것인지, 새 target을 만든 것인지 구분한다. self-play score·학습 곡선 AUC·최종 held-out paired game score를 training-seed 단위로 평가하고, GPU 시간·NN calls·생성한 유효 target 수를 함께 공개한다.

**통과/중단.** fixed mixture보다 선택기 overhead를 포함한 품질/비용 이점이 없으면 MPV-VOC 선택을 중단한다. fixed NN search는 좋아졌지만 총 학습 성능은 같거나 나쁘면 두 결론을 별도로 유지한다. CPU pattern student의 저비용만으로 고성능 GPU 반복 search의 우월성을 주장하지 않는다. Physics/GM라는 설명은 이 실험을 통과시키는 장식이 아니라, 공통 모드 소거·다봉성 복구·선택적 계산의 사전 예측이 맞았는지 판단하는 근거로 사용한다.

## 각 작업의 완료 기록

각 W 단위는 source/input/config hash, 실제 실행 명령, 환경·서비스 측정, raw 완료/실패 기록, 독립 단위 정의, 변환 코드, effect/CI, 남은 반례를 한 receipt로 남긴다. 새 신뢰 인프라를 이 연구 안에서 다시 만들지 않고 현재 Foundry PR 계획이 제공하는 기능을 재사용한다. 실행하지 않은 명령에는 `PLANNED`, 소스만 있는 기전에는 `IMPLEMENTED_UNVALIDATED`, 작은 수학 모형에는 그 모형 범위의 결과만 붙인다. 구현 성공, 수학적 타당성, 게임의 성능, 학습의 성능은 각각 별개의 완료 항목이다.

## 실행 체크와 열린 결정

- [ ] W1 관측 identity·epoch·완료 블록 계약
- [ ] W2 held-out 효용차 belief calibration
- [ ] W3 계산가치·pending 배치 배분
- [ ] W4 별도 기전 증거가 있는 후보 재개방
- [ ] W5 고정 NN live 파일럿
- [ ] W6 고예산 확인·총예산 재투자
- [ ] W7 다중 평가기·독립 학습

실행 전 고정할 미정 항목은 checkpoint artifact, 실제 latency에 따른 작업 수와 wall-time 한도, pilot cluster variance에 따른 확인 실험 규모다. 방법·집단을 결과에 맞춰 고르는 자유도로 남겨두지 않는다. 현재 문서와 작은 계산 결과의 게시에는 이 미정 항목의 결정을 요구하지 않는다.
