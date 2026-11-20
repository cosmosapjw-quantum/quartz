# QUARTZ COG-01B Connect4 Recognizer 06: CPU 정책·가치 인식기 학습 및 실제 MCTS 효용 검증

Date: 2026-11-20. Branch: `research/quartz-cog01b-connect4-recognizer-20261120`. Parent baseline `c4e219f930065c4e4b56eeb4f779f473a6c5b007`. GPU: unavailable / **not used**.

**종합 판정:** 192개 표준 7×6 중력 Connect Four의 정확 교사 라벨 생성, 소형 CPU policy/value 인식기의 학습, Python–Rust f32 inference parity, 실제 QUARTZ Rust same-root MCTS의 12개 고정 국면 비교까지 `IMPLEMENTATION_VERIFIED` 및 `NUMERICALLY_CHECKED`. **일반화와 계산가치 제어의 우위는 `UNRESOLVED`**. 비단조 결과를 긍정적으로 포장하지 않는다.

## 1. 원래 연구의 의미와 수학적 목표

이전 축소 7×7 비중력 4목이 아닌 표준 **Connect Four 7×6 중력 적용 4목**을 사용한다. 총 7개 열 행동 `a=0..6`. Pascal Pons 공개 완전게임 solver (https://github.com/PascalPons/connect4)의 signed integer score `S(s,a)`를 외부, 독립 정답원으로 사용한다. `S=-1000`은 불법 열 sentinel, 아니면 `U(s,a)=sign S(s,a) ∈ {-1,0,+1}`가 게임이론적 승패 효용이다. 정수 점수의 절댓값은 최적 승리/패배 속도 기준으로만 해석한다. 양쪽 모델은 외부 oracle 결과를 Rust inference에 입력받지 않는다.

참조행동 집합 `A*(s)={a legal | U(s,a)=max_b U(s,b)}`; 학습 policy target은 `q(a|s)=1/|A*(s)|` for a∈A*(s), else 0. Value target `V*(s)=max_b U(s,b)`. 완전한 7열 평가를 훈련 라벨로 사용하되 같은 결괏값을 갖는 모든 수를 동등히 인정한다. 비학습 비교군은 QUARTZ의 기존 `UniformEval`, NN 그룹은 새 JSON-f32 MLP evaluator다. GameState와 `MctsEngine`, 고정 seed17, 탐색 방문량 8→16→32→64, 단일 CPU process/rust thread 조건을 유지한다.

## 2. 원자료와 데이터 누출 방지

- 학습 대상은 `COHORT_FROZEN.json`의 192개 독립 root 기록: 12개 mirror-canonical 5-ply family×16 descendant, 각 9/10/13/14/17/18/21/22 ply의 2개 변형, 현재 수 흑/백 각각 96개.
- Family 단위 분할 8/2/2: train=128, validation=32, test=32. 같은 기초 prefix의 유래·좌우 반전은 split 내에만 존재. 훈련 샘플에는 좌우 반전을 추가해 256개로 증강, validation과 test는 증강하지 않음.
- 이전 연구 `standard_external_04/POSITIONS_FIXED.json`의 12개 root는 완전히 다른 5-ply opening-family로 분리된 **원래부터 동결된 최종 평가**. 학습, epoch 결정, 보정에 사용하지 않았다. 이 12개는 모두 흑 차례라는 제한이 있다.
- 원 teacher source `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`, 실행 파일 `fca2b665...`, book `38f9834317c37d95...`; 정확 전체 SHA는 `INPUT_LOCK.json`.
- `evidence/teacher_20261120/labels.json`: 모든 192개에서 정확 solver 7열 점수 획득, **timeout=0, 실패=0** (13.62초). Train WDL win/draw/loss 89/10/29, val 26/0/6, test 20/0/12. 독립 6개 mirrored requery의 순서 반전 점수가 일치. 최적행동 집합 크기 1개인 root 64개, 7개 전부인 root 40개. 일부는 매우 쉬운/경계에 민감한 목표 분포이다.
- 단순 무작위 합법수 선택의 root별 기대 regret를 정확 solver labels로 계산한 내부 진단: train≈0.84193, val≈1.01875, test≈0.74821. 학습 모델의 greedy top1 regret를 비교한 통계이지 실제 대전 성능이 아니다.

## 3. 신경망·학습 조건

- `TinyPolicyValue` 126(3×6×7)→64(ReLU)→8 (policy logits 7, value tanh 1), float32.
- 입력은 현재 플레이어 상대 좌표계의 자신/상대/흑 차례 indicator 평면 3개. Connect4 Rust `Connect4::encode_planes_into`와 동일.
- Adam lr=0.003, weight_decay=1e-4, batch=64, 80 epochs, seed=20261120, Torch CPU 2 threads. Loss `L_policy_set + 0.5 MSE(value,U_star)`, illegal 열은 masked logits. Validation loss가 최소인 epoch를 선택; test는 선택 후 읽음. 수정/재탐색 없음.
- 선택된 epoch **23**. 80까지 훈련했으나 train loss는 계속 감소하고 validation은 악화되므로 모델 일반화 격차가 뚜렷하다.
- `evidence/nn_train_20261120/tiny_mlp_weights.json` (SHA256 `1660b450d4ab948afc71d0aeebf1abe1f31120093849e224bc28f777605aff60`)는 PyTorch에서 내보낸 f32 가중치이며 모델 체크포인트 자체가 재현 가능한 사용자 연구 자산이다.
- Train top-1 WDL-optimal fraction **0.921875**, mean regret **0.140625**, value MSE **0.30057**.
- Validation top-1 **0.5625**, regret **0.8125**, value MSE **0.58922**.
- Test top-1 **0.625**, regret **0.71875**, value MSE **1.18660**. Test의 단순 random action 기대 regret 0.74821에 비해 개선 폭은 작다. 일반화 강건성을 입증하지 못했다.
- 모델 선택 횟수 1개(동결된 구조·lr·epoch)이며 테스트 셋을 보며 추가 조정을 하지 않았다. 이는 이번 단위의 `negative result`이다.

## 4. Rust 실제 추론 연동 및 의미론적 검증

- 새 `src/games/connect4_nn.rs`: JSON weights의 strict format `quartz_connect4_tiny_mlp_v1`, 126×64 및 64×8 matmul, ReLU, tanh(value), 합법 열 마스킹 policy softmax. Rust MCTS runtime에서 exact solver/book/label은 전혀 호출하지 않는다.
- 기존 연구 전용 `research_connect4_cpu` endpoint는 `eval_mode=uniform|tiny_mlp`과 `model_path`를 추가. 모델 파일은 root 시작 때 한 번 로드한다. 일반 QUARTZ 검색 경로·uniform 기본값은 불변.
- Python PyTorch와 Rust의 실제 동일 모델 출력을 12개 다양한 history에서 비교하여 policy/value absolute tolerance `3e-5` 통과, invalid JSON/shape 차단 테스트 통과.
- MCTS evaluator counter는 Rust in-process MLP의 evaluator 호출에 대응한다. 별도 Python NN cache가 없어 캐시 재사용으로 인한 가짜 call 차이는 없다. 단 terminal/TT 경로는 eval를 생략할 수 있다. 관측된 actual `evaluator_calls`와 순수 `MctsEngine::run` wall time을 각각 기록했다. 모델 JSON 파싱·root 생성은 step_wall_us 안에 포함되지 않으므로 end-to-end 서비스 시간 측정치가 아니다.

## 5. 사전 동결 12개 root: 실제 MCTS 실험 결과

**정확한 승패효용을 비교한 짝 자료:** `evidence/heldout12_nn_vs_uniform_20261120/{rust_raw_responses.json,paired_heldout.json,summary.json,evaluation_manifest.json,INDEPENDENT_READBACK.json,verification_receipt.json}`. 모델·Rust binary·source·oracle SHA는 manifest에 귀속. 출처가 다른 원래 uniform 실험의 12개 행동·평가기 누적 횟수를 현재 uniform과 비교해 차이 **0건** 확인. 동일 실제 root의 방문은 8→16→32→64. Fresh16 동일성도 각 root에서 PASS.

| 누적 visits | uniform mean U / regret | trained mean U / regret | trained-uniform mean ΔU | uniform mean eval calls | trained mean eval calls |
|---|---|---|---:|---:|---:|
| 8 | 0.5000 / 0.2500 | 0.3333 / 0.4167 | **-0.1667** | 4.333 | 4.667 |
| 16 | 0.5000 / 0.2500 | 0.6667 / 0.0833 | **+0.1667** | 8.000 | 7.167 |
| 32 | 0.6667 / 0.0833 | 0.6667 / 0.0833 | 0 | 13.750 | 12.500 |
| 64 | 0.6667 / 0.0833 | 0.6667 / 0.0833 | 0 | 25.833 | 19.000 |

- 8 visits: `C4_001` 한 root에서 학습 인식기 선택이 uniform의 강제승(+1)보다 나쁜 강제패(-1). 나머지 11개 WDL 같음.
- 16 visits: `C4_003` 한 root에서 학습 인식기가 강제승을 uniform보다 먼저 발견(+2의 국면별 차이). 나머지 11개 WDL 같음.
- 32/64 visits: **12개 WDL이 모두 같음**. 일정한 성능 우위가 아니다.
- 각 모델의 방문별 실효 비용은 달랐다. 16 방문의 평균 `evaluator_calls`는 model 7.167 vs uniform 8.000이지만 model 평균 단계 wall time은 **2.1515ms vs uniform 2.05025ms**; 모델이 조금 느렸다. 8 방문에서는 2.2485ms vs 1.9982ms. 작은 처리시간에는 측정 노이즈와 모델 로드 배제 영향이 크다. **방문 수만 일치하는 비교**이며 동일 CPU 비용 비교가 아니다.
- 모델의 동일-root 8→16 추가 계산은 **2개** 국면에서 강제패→강제승으로 변화, uniform은 개선 0. 16→32에서 uniform 1개가 개선, model은 0. 실제로 발생한 signed gain과 모델 상대 우위는 서로 다른 estimand.
- 12개 자료 모두 흑 차례, 인위적 선정이고 시드 17 하나다. No generalizable p-value or Elo; 정규분포 CI를 48 observations에 억지로 붙이지 않았다.
- 여기서 관찰된 것은 고정 NN 평가기로 실행한 MCTS 행동의 실제 게임결과 변화다. **어떤 추가 계산을 선택해야 하는지를 예측하거나 중단하는 VOC controller는 아직 없다.**

## 6. 독립 검산·보존

독립 `audit_recognizer06.py`가 게임 solver에 12개 원본 root와 12개 대칭 root를 재질의, 모든 점수/좌우 반전 확인, Rust 24개 응답의 진위·합법성·visited root·evaluator call·각 action U의 부호를 별도로 합산했다. `INDEPENDENT_READBACK.json`에서 8 방문 비교 {-2:1,0:11}, 16 방문 {0:11,+2:1}, 32/64 방문 {0:12} 재현.

최종 테스트: Rust 전체 `cargo test --offline --locked -- --test-threads=4` → **582 passed, 0 failed, 89 ignored**. Python 연구 테스트 **70 passed, 0 failed** (9+4+7+7+7+9+27). 학습 첫 시험에서 Python 테스트는 미구현 import, Rust 테스트는 실제 모델 미구현을 확인한 후 구현해 green을 확인했다. Python 훈련/예측 일치 및 Rust cross-check 모두 포함한다.

연구 구간마다 commit → 로컬 Dropbox `tar.gz` 백업 및 SHA256 → push → `git ls-remote` readback:
- A `2bf64f2fc7994b8d8ed06a2be0ac8e021508c06e`: cohort / protocol lock.
- B1 `72f0e79b95ce6499aea801e07c55c894ec19509d`: exact teacher runner.
- B2 `7a87ff3bfa10a3a742b102224212932e7b1657af`: 192 solver label results.
- B3 `abbde64477e251b99877c754078712696ba1e74d`: deterministic CPU trainer and export.
- C0 `23f7d6c84a08ffbe09098ecb24e664d335f13283`: trained weights+metrics.
- C1 `27ce17543a0e5bbc17a5bf787cd6b20c56c2b0f7`: Rust MLP runtime + 12 source crosscheck.
- C2 `cbaf14b72bee5a7c0b29321065ac86c72cf420cd`: fixed benchmark code.
- C3 `629cd7521d295135d07c58a43fa00ddf36fa9bb1`: raw exact WDL comparison + independent audit.
- D: 본 과학적 판정과 후속 실행 계약을 단일 최종 커밋에 별도로 게시하며, 정확한 commit/tree 및 archive hash는 최종 실행 영수증과 대화 인계에서 확인한다.

각 저장 snapshot은 로컬 Dropbox 경로 `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups/`에 있다. Dropbox cloud ACK 및 복원검증은 하지 않았고, GitHub 게시 SHA는 단계별로 읽어 확인했다. 원래의 QUARTZ 개발 브랜치, 다른 worktree, 기존 외부 solver/book, 15×15 오목 및 7×7 축소 연구 자료는 변경하지 않았다. 외부 AGPL solver 바이너리·source·book은 Git history/backup archive에 포함하지 않았다.

## 7. 다음 단일 연구 노드

**`QUARTZ-COG-01B-RECOGNIZER-GENERALIZATION-07`**

현재는 인식기를 사용할 수 있게 되었지만, 그 인식기의 validation/test 성능이 약하다. 다음 단계는 이 불확실성을 줄이는 하나의 명확한 실험을 권고한다.

1. 새 40개 이상 서로 다른 opening-family에서 1000개 안팎의 Connect4 solver label을 CPU 예산 안에서 생성하고, **승패 원인/전술 유형/ply/현재 플레이어** 균형 및 별도 color-balanced 완전 고정 test roots를 확보. Timeouts는 원인/범위까지 보존.
2. 작은 geometry-aware convolution 인식기 혹은 equivariant feature와 현재 126-64-8 MLP를 **동일 총 훈련·추론 CPU 비용**에서 비교한다. Holdout splitting은 family와 mirror를 모두 봉쇄한다. 현재 12개 MCTS root는 최종 evaluation으로 보존하되 오히려 이 자료에 대한 최적화는 금지한다.
3. 새 모델은 root의 game-theoretic optimal-action *set* + WDL value를 동시에 예측하며, val/test 평균 regret가 단순 무작위 합법수보다 의미 있게 개선되는지를 먼저 확인한다. 모형이 실패하면 학습진단과 실패례를 보존하고 더 넓은 과학적 결론을 만들지 않는다.
4. 실제 MCTS 비교는 **완료한 평가기 호출 횟수뿐 아니라 같은 CPU wall-time 단위로도** 수행하고, 모델 로드·초기 root 평가 비용을 포함하는 end-to-end 시간도 명시한다. 여러 seed가 동일 궤적을 만들면 독립으로 세지 않는다.
5. 이후, 실행 가능한 충분한 인식기가 확보되어야 원래 인지적 목표인 `QUARTZ-COG-02` selective computation/attention reopening으로 이관한다. 수치적 비교는 실제 `ΔU`와 비용의 결합으로 판단하며 단순 policy flip 수를 utility로 간주하지 않는다.

선행 문헌: Kong, Ram & Yu, *AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision* (2026), https://arxiv.org/abs/2607.08984, oracle-optimal action set으로 정책을 보조지도하는 방법을 설명한다. 기술적 아이디어 유사성의 근거이지 QUARTZ 과학적 novelty 증명은 아니다. Willemsen et al., *Value Targets in Off-policy AlphaZero* (2021), https://doi.org/10.1007/s00521-021-05928-5, value target 선택의 의의를 보여준다.

**Native subagent:** `NOT_RUN` (현재 호스트에서 실제 native delegation 도구가 없었음). 직렬 직접 구현, Rust/Python parity 및 별도 실행된 solver·독립 프로그램 검산으로 수행했다.
