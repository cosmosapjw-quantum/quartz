# QUARTZ COG-02 COST-MATCHED-VOC-08 | CPU 선택적 계산 연구 최종 보고서

Date: 2026-11-21 (Asia/Seoul). Parent commit: `4a4d1381635393715c29d672124cfe32c08d20d5`. Dedicated branch: `research/quartz-cog02-cost-matched-voc-20261121`.

**Scoped scientific outcome:** 실제 Rust MCTS에서 현재 root의 관측 특징으로 추가 계산을 선택하고, 동일한 CPU 시간 범위의 고정 중단 정책과 정확 게임이론적 승패 효용을 비교했다. Uniform baseline의 유한 시험집합에서 비용 대비 의사결정 개선을 관측했다. 다만 이 구현은 **Bayesian preposterior KG/VOC가 아니라 signed realized-gain surrogate**이며, 새 오프닝 모집단으로의 일반화와 신경망 평가기의 메타제어 효과는 미입증이다.

## 1. 대상, 정의, 원전

게임은 standard Connect Four 7×6 gravity four-in-a-row. Pascal Pons standalone exact solver의 합법 행동 점수를 `S(s,a)`라 놓고 `u(s,a)=sgn S(s,a)`를 정의한다. `-1000`은 불법 수 sentinel이고 승패 효용이 아니다. `Regret_b=max_a u(s,a)-u(s,a_b)` 및 `G_b=u(s,a_{2b})-u(s,a_b)`이다. `G_b`에는 음수가 허용된다. 실제 posterior belief 및 적분으로 정의하는 nonnegative Bayesian expected KG와 구별한다.

학습 목표는 `E[G_b | X_b]`의 선형 Ridge surrogate(α=1, intercept unregularized)이다. root에서 직접 측정한 특징 9개는 intercept, b/64, normalized visit entropy, top1 visit probability, top1−top2 gap, legal fraction, completed evaluator calls/(b+1), ply/42, black indicator이다. 현재 탐색시점 이외의 다음 단계 방문결과와 외부 정답점수는 Rust 정책 입력에 없다. 이어서 계산할 조건은 `predicted_signed_gain > λ × predicted_next_search_ms`이다.

원 논문: Hay et al. (2012) https://proceedings.mlr.press/r10/hay12a.html ; Tolpin & Shimony (2012) https://ojs.aaai.org/index.php/AAAI/article/view/8126 ; Sezener & Dayan (2020) https://proceedings.mlr.press/v124/sezener20a.html ; Kong, Ram & Yu (2026) https://arxiv.org/abs/2607.08984 . 이들 문헌은 방법론적 선행연구이며 본 실험의 보편적 효과나 novelty를 인증하지 않는다.

## 2. 권위·범위·조건

- Base/input SSOT: `INPUT_LOCK.json` SHA256 `93fcad5f6d48262882419bd75fd4166915d37bb42821fe622c1f7e2aa5020025`; 원본 16개 source/model/oracle/binary/book 파일의 byte identity 고정.
- 이전 `COHORT_1024_FROZEN.json`: 40개 train family/640 roots, 12개 validation family/192 roots, 12개 test family/192 roots. 1가족당 16개 descendant roots는 iid가 아니며 192를 독립표본으로 취급하지 않는다. Test는 흑/백 각각 96, oracle root win/loss/draw 144/43/5.
- Evaluator는 Uniform, 이전에 학습된 126→64→8 new MLP, 공간 ConvNet이다. 시드17, serial MCTS, 방문예산 8/16/32/64, Dirichlet off. 외부 게임이론적 정답은 탐색 완료 뒤에만 결합한다.
- Rust `src/mcts/cog02.rs` + `src/mcts_server.rs`: 학습 weight는 프로세스별로 캐시, 단일 source root에서 실제 방문을 연장하고 각 checkpoint의 stop/continue 판단을 관측한다. `model_load_wall_us`, `root_setup_wall_us`, 누적 evaluator calls, stage `step_wall_us`, 실시간 `time_budget_ms` checkpoint stop과 overshoot를 분리한다.
- Model-cache key는 mode+canonical path+**호출자가 제공한 64자리 SHA256 문자열**이며, 학습·실험 러너가 실제 weight bytes를 실험 전/후 해시로 확인했다. Rust 캐시가 모든 lookup에서 파일 자체의 SHA를 재계산하는 일반적 변조방지 보장은 아니다.
- CPU 연구는 Rust DEBUG binary로 실행했으며 모든 실행시간은 wall-clock이다. 실제 CPU process-time, release 성능 및 동시 사용자 부하는 검증하지 않았다. Model load는 상주화되나 setup과 탐색시간을 따로 기록한다.

## 3. Train/validation: 실제 2496번 full resident MCTS

3 evaluators × (640 train+192 val) = **2496 real Rust searches**, 소요 약 59.59s. 원시 `evidence/train_val_2496_20261121/{full_rust_trajectories.json,joined_train_val.json,execution_manifest.json,INDEPENDENT_READBACK.json}`. New MLP/ConvNet 모델 캐시는 각각 831 hits+1 miss였고, 독립 재계산에서 실제 전환 WDL의 부호와 모델계측이 일치했다.

Uniform train family의 8→16행동효용 변화는 `G=0:576, +2:31, +1:17, -1:7, -2:9`였다. New MLP는 `0:617,+2:17,+1:1,-1:4,-2:1`; Conv는 `0:614,+2:16,+1:7,-1:2,-2:1`. 즉 계산을 더 한다고 실제 최종 선택효용이 항상 개선되지는 않는다.

학습된 stagewise Ridge와 평균 추가비용 forecast는 train640만 사용. Validation192는 고정 후보 λ={0,.01,.03,.05,.10} 중 `mean WDL−0.04×mean search_ms`를 가장 높이는 중단 규칙을 선택했다.

| evaluator | 선택 λ (WDL/ms) | Val stop8 | Val stop16 | Val stop32 | Val mean search ms |
|---|---:|---:|---:|---:|---:|
| Uniform | .05 | 107 | 5 | 80 | 4.027 |
| New MLP | .10 | 192 | 0 | 0 | 2.443 |
| Conv | .03 | 163 | 29 | 0 | 5.900 |

**MLP 8회 일괄 중단은 degenerate negative result**이며 사후 임계값 조정으로 감추지 않았다. Validation mean adaptive 비용에 맞추도록 전날 정의한 grid 0.25..40ms에서 solver WDL을 보지 않고 사전 clock-cap을 결정: Uniform 2.25ms, MLP 0.25ms, Conv 3.75ms. Clock cap은 매 8/16/32 checkpoint **완료 이후** 확인하므로 per-iteration hard deadline이 아니라 overshoot 가능한 실제 checkpoint stop이다. 별도 `audit_surrogate08.py`가 Ridge 정규방정식과 검증 전용 임계값 순위를 독립 재계산했다.

## 4. 완전히 보존된 test12 family에서 실제 1728 Rust MCTS 실행

192 test roots ×3 evaluators ×[full resident, live adaptive, live fixed-clock] = **1728 실제 Rust 서버 실행**, 약 24.01s. 학습/λ/clock cap은 시험 family를 보기 전 커밋 및 remote 게시했다. 외부 oracle 테스트 값은 1728 MCTS 검색이 끝난 뒤에만 결합했다.

| Evaluator | Live adaptive regret | Fixed-clock regret | Adaptive search ms | Fixed-clock search ms | mean CPU cost mismatch |
|---|---:|---:|---:|---:|---:|
| Uniform | **.21875** | .36458 | 4.1684 | 3.8823 | adaptive +6.86% |
| New MLP | .30208 | .30208 | 2.5607 | 2.5484 | adaptive +0.48% |
| Conv | .19792 | .21875 | 5.5597 | 6.0278 | adaptive −8.42% |

계약상 평균 탐색시간 불일치≤10%는 세 평가기 모두 통과했지만 **per-root 비용 일치는 주장하지 않는다**. Uniform adaptive−clock WDL regret 평균 차이=−.145833, fixed trained policy의 12-family cluster-bootstrap 95% 구간 `[−.239583,−.046875]`. 그 대신 adaptive는 clock보다 평균 6.86% 더 많은 검색시간을 썼다는 점을 함께 보고한다.

더 강한 조건부 Pareto 사례: Uniform adaptive는 평균 regret=.21875와 search=4.1684ms를 기록했고, fixed16 실제 full resident checkpoint prefix는 regret=.317708과 search=4.4941ms를 기록했다. Adaptive−fixed16 regret=−.098958, family bootstrap `[−.182292,−.015625]`. **fixed16 비용/행동은 full trajectory의 정확 prefix를 사용한 posthoc 재구성**이며 별도 live fixed16 중단을 192회 재실행한 것은 아니다. 실제 adaptive/full prefix는 root/action/policy/eval-call identity가 일치했다.

Conv adaptive−clock regret 차이=−.020833, 95% `[−.072917,+.026042]`: 0 포함, 개선 미입증. Conv adaptive가 fixed16보다 평균 regret .0625 나쁘고 비용은 더 낮은 trade-off. MLP는 adaptive/clock 모두 8회에서 전원 중단해 결과 동일.

Uniform adaptive 실제 stop distribution=8:109,16:7,32:76,64:0. 이미 실행하기로 결정한 continuation 단계의 실제 signed gain은 `-2:7,0:126,+1:8,+2:18`; **7회의 악화**도 보존. Conv stop8:170/stop16:22, 유료 다음 계산의 gain은 `-2:1,-1:1,0:16,+1:1,+2:3`. 개별 근사 MCTS 결과의 비단조성과 모형의 교정 실패 위험을 남겨둔다.

## 5. 실제 검증과 보존

- `audit_train08.py`: train/validation full 2496 source labels, negative gain, 실제 Rust 누적 eval calls, per-process model cache hit/miss를 별도 재검산.
- `audit_surrogate08.py`: train-only Ridge regularized normal equations 및 train cost mean, validation-only λ/stop 선택 재계산. Test 정보 사용하지 않음.
- `audit_live08.py`: test 1728 Rust full/online-adaptive/live clock, 모든 root hash·합법행동·WDL·time/eval 계산을 재검산. 새 test family 12개와 좌우반전 12개를 외부 Pons solver에 재질의해 정확 점수 일치 확인.
- Source test batch commit `8682f805cb8c74deef967e3ed9595bc41bea474b`; Rust binary SHA256 `f261061a5798c6f47a5b0e94b906ae96fa9fab296b470b069aaea064c94cff48`. 증거 `evidence/live_test_1728_20261121/{raw_1728_rust_requests.json,paired_192_livetest.json,summary.json,execution_manifest.json,INDEPENDENT_READBACK.json,verification_receipt.json}`.
- 실제 fresh Rust `cargo test --offline --locked -- --test-threads=4`: **592 PASS, 0 FAIL, 89 ignored**. 관련 Python 9 suites **87 PASS, 0 FAIL**. Tests confirm cached evaluator, Rust↔Python nine features, online halt, negative signed gains, time cap stop and legacy regressions.
- Machine/native subagent: actual native dispatch unsupported, `native_subagent=NOT_RUN`; tools were serial direct Rust/Python CPU, full source manifest and external independent solver readback. GPU not used.

## 6. 백업 및 GitHub 게시

사용자 승인에 따라 각 단위 commit → local Dropbox `tar.gz` archive → sha256sum → GitHub push → `git ls-remote` equality readback:

- A contract/input: `d2800d00b6ed38780e306c6ef659549119586e87`, `QUARTZ_COG02_VOC08_A_d2800d00b6ed.tar.gz`.
- B model-resident and online stop: `62ab5cfccdc467e6b68b24cd202401a78a2581fc`, `QUARTZ_COG02_VOC08_B_62ab5cfccdc4.tar.gz`.
- C1 2496 true CPU trajectories: `1027ed1cbe6d0cfe198d4a570aa43429d6093be3`, `QUARTZ_COG02_VOC08_C1_1027ed1cbe6d.tar.gz`.
- C2 fitted/validated surrogate: `8772ad02b6f51740453929713b350470c2774c35`, `QUARTZ_COG02_VOC08_C2_8772ad02b6f5.tar.gz`.
- D1 actual checkpoint-clock stop: `984624903a3f5cdc77ef4c2113fbf5bd22ea9c8c`, `QUARTZ_COG02_VOC08_D1_984624903a3f.tar.gz`.
- D2 validation-only clock cap: `4d1a2a7ffab9a94eb37748cf5e563bd6e2c2791f`, `QUARTZ_COG02_VOC08_D2_4d1a2a7ffab9.tar.gz`.
- D3 source code live experiment: `8682f805cb8c74deef967e3ed9595bc41bea474b`, `QUARTZ_COG02_VOC08_D3_8682f805cb8c.tar.gz`.
- D4 1728 live test and independent audit: `9938ad4d44513695b2d2c593a7d022b15fd5846d`, `QUARTZ_COG02_VOC08_D4_9938ad4d4451.tar.gz`.
- E: 본 종료 보고서 및 다음 ID 인계; final publication commit/tree SHA and archive checksum are verified separately upon publishing.

Backup local folder `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups/`. **Dropbox cloud synchronization acknowledgment and restoration test NOT_RUN**. Original developer worktree `/home/cosmosapjw/Dropbox/personal_projects/quartz` at `agent/local-experiment-foundry` SHA `65e7d3e9596ca259ebfe523b0c74058692b6a755` and its existing untracked `QUARTZ_CLAUE_SCIENTIST_AGENT_CONTEXT.md` are preserved. Main/other branches unchanged; external AGPL solver/book not redistributed.

## 7. 다음 연구 단위: QUARTZ-COG-02-VOC-CALIBRATION-09

1. **새로운 별도 5-ply family heldout 집합**으로 지금 봉인된 Uniform adaptive 정책·lambda·models를 반복한다. seed17이 무작위 효과 없는 경로일 수 있으므로 다른 seed 문자열만 바꾼 것을 독립 표본이라고 주장하지 않는다. 실제 다른 PRNG/전술 정보 상태를 확인한 후 독립 복제한다.
2. 최적화된 Rust release build, 따뜻한 in-process model cache, 모델 초기화 vs root setup vs completed step time, OS 간섭을 분리해 평가한다. 개별 실행의 준실시간 비교 및 **actual live fixed16**을 실행해 현재 deterministic full-prefix comparator 한계를 해결한다.
3. `E[G_b|X_b]`의 예측값·비용 forecast를 train-family 기반 group calibration/reliability와 asymmetric negative-gain risk로 검산하고, false-stop과 과다계산을 비교한다. Calibration 성립 전 새 theory novelty를 주장하지 않는다.
4. One-step nonmyopic synergy를 가진 **조건부 두 단계 option**을 별도로 구성·검증한다. 기존 R1–R10의 XOR counterexample을 엄밀한 설계 참고로 사용하며, 단순 ridge를 Bayesian belief/model이라고 이름 바꾸지 않는다.
5. 검증력과 비용 절약이 새로운 집합에서 반복 확인된 뒤에야 `Einstellung`·주의 고착·후보 재개방을 동일 비용으로 평가한다. 15×15 Gomoku/Rapfi는 별개의 전이 연구다.

**Stop-condition:** 이 단위는 실제 signed-gain 기반의 적응적 계산과 비용 비교에 필요한 구현·고정 검증까지 완료한 것으로 종료한다. 강한 인지적 일반화 및 진정한 Bayesian VOC 우위는 unresolved다.
