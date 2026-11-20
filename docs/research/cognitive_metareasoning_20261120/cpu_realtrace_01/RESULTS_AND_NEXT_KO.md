# QUARTZ COG-01B-CPU-REALTRACE-01: 실측 CPU 연구 결과 및 다음 단계

**2026-11-20.** 기준 `cc3aa1771460f5a0f608375d985fd02998861364`; 실행 코드 `e8b1e944e9564725ae7252220297c239af506cbb`. 사용 GPU 0, PyTorch 2.9.1+cpu, CPU torch threads 2. 연구대상은 **이미 학습된 6개 네트워크의 초기 인식**과 48개 국면의 **독립 재시작** 검색 정책이다. 완성된 Rust MCTS의 실제 추가 계산에 대한 causal VOC 실험이 아니다.

## 직접 실행한 실험

- 296개의 고정 입력과 SHA256을 `INPUT_LOCK.json`에 기록: 체크포인트 6개, trace 288개, A4 포함 6912행 assay, 국면 모음 48개. 실험 외부 입력 checkout의 당시 HEAD는 `65e7d3e9596ca259ebfe523b0c74058692b6a755`; 이는 원래 결과를 생성한 커밋을 보증하지 않는다.
- 네트워크 6개를 CPU에서 `torch.load(weights_only=True,map_location='cpu')`, `AlphaZeroNet` 및 17-channel Gomoku7 encoder로 실행. 모든 **288/288** 모델·국면에서 저장된 `argmax_prior`와 원시(합법수 마스크 적용 전) NN logits의 argmax가 일치했다. 이중 **47/288**은 raw argmax가 이미 놓인 칸이므로 합법수 적용 후 인식과 혼동하면 안 된다. 검색 자체가 불법수를 선택한다는 주장은 아니다.
- 288개 root와 세 전환(8→16, 16→32, 32→64)을 조합한 총 **864**개 관측. 목적변수 `Y_b=1[a_{2b}=a_ref]-1[a_b=a_ref]`는 {-1,0,1}; 강한 reference action은 minimax truth가 아니다. trace가 독립 재시작이므로 같은 탐색에 추가 계산을 했을 때의 효과는 아니다.
- 위치 ID 48개 기준 6-fold 교차검증. 같은 위치의 여섯 checkpoint는 같은 fold에 할당. 고정 ridge 정규화, 각 훈련 fold 안에서만 feature scaling. `M0`은 budget-only, `M1`은 현재 탐색 policy(최댓값·격차·entropy), `M2`는 M1 + CPU NN 첫 prior(최댓값·격차·entropy·value·prior agreement) + tie signal. 원전의 `trace_p_flips`, oracle class, 참고 답, `position_bucket`은 예측입력에 넣지 않았다.
- 2000번 position-cluster bootstrap은 이미 고정된 OOF 예측의 **조건부** 불확실성 추정이며 학습 재적합의 변동은 포함하지 않는다. 같은 위치의 체크포인트/예산 전환의 상관을 보존한다.

## 원자료 수치

| 전환 | 개선(+1) | 악화(-1) | 불변(0) | 평균 signed |
|---|---:|---:|---:|---:|
| 8→16 | 31 | 29 | 228 | +0.0069444 |
| 16→32 | 36 | 41 | 211 | -0.0173611 |
| 32→64 | 43 | 28 | 217 | +0.0520833 |
| 전체 | **110** | **98** | **656** | +0.0138889 |

총 864 중 **708/864=81.944%** 전환은 두 독립 정책 가운데 적어도 하나가 최상위 확률 동률이었다. 동률이 없는 전환만 남기면 156개다. 방문횟수 기반 `argmax`는 동률 해소 순서에 민감하므로 수정 전후의 모든 변화가 숙련된 탐색을 뜻하지 않는다.

## 비교 결과: 음성 확인

| Cross-fitted predictor | OOF signed-MSE |
|---|---:|
| M0 budget-only | **0.2428780864** |
| M1 search morphology | **0.2427567000** |
| M2 search + CPU NN recognition | **0.2434453955** |

M2–M1 MSE 차이 **+0.0006886955**(양수는 악화), 조건부 위치 bootstrap 95% 구간 `[-0.0006835504,0.0021372914]`이다. 따라서 이 입력·평가 계약에서 인식기 특징의 추가 예측 이득이 확인되지 않는다. 이는 신경망 특성을 이용하는 일반적 메타제어의 no-go가 아니다.

참고용 retrospective 8→16 *동일한 예상 명목 시뮬레이션 비용*의 25% 선택 실험: OOF M1이 고른 72/288의 total mean reference-alignment change는 **+0.0173611**, 무작위 동률 기대는 **+0.0017361**. 차이 **+0.015625**이나, 별도 표준 라이브러리 구현의 위치 bootstrap 95% 구간은 **[-0.0052083,0.0381944]**이므로 0을 포함한다. M2도 M1보다 좋아지지 않았다(M2-M1 선택 이득 구간 `[-0.03125,0.0277778]`). 이 선택은 미래 policy를 생성하지 않고 **기존 서로 독립적인 기록에서 재가중만** 하므로 실제 절약된 NN 호출 수가 아니다.

보조 진단: reference checkpoint C06_latest 자체를 제외한 720행의 M2 MSE 0.2479535 vs M1 0.2475153. 다만 이는 학습을 다시 하지 않은 subset score이고 완전한 leave-checkpoint-out 실험은 아니다. 동률이 없는 156행에서는 M2 0.2662753 vs M1 0.2647804. 역시 이전부터 frozen인 OOF predictor의 부분집합 점수다.

## 결과 admission

**IMPLEMENTATION-VERIFIED, exact identity:** CPU checkpoint load, cfg(17,7,49), 288 NN prior argmax identity, 288 trace ↔ A4 joins, 864 rows, stable SHA, source+output input lock and manifest.

**OBSERVATIONALLY CHECKED:** 110/+98/-656/zero, M0/M1/M2 crossfit MSE, conditionally bootstrapped uncertainty. The predictor improvement claim is `NOT ESTABLISHED`, not positive.

**SCIENTIFICALLY BLOCKED for causal claims:** independent budget restarts; high-budget reference not exact; mined position suite; tie contamination; same checkpoint reference leakage; no live NN call accounting or controlled costs. These are result *interpretation limits*, not input hash or runtime failures.

**RUNTIME error resolved:** initial direct runner launch failed `ModuleNotFoundError: quartz`, because nested script directory was first on Python import path. Import worktree path explicitly; fix published separately. Initial TDD test fixtures initially had two defects (changed policy but not assay, lexical string ordering); both corrected without dropping corresponding assertions. Only green tests after correction are counted as PASS.

## Verification / preservation

- New realtrace tests 7/7, COG01 exact tests 9/9, W1 reference tests 27/27. Script compiles with local PyTorch+CPU.
- The runner directly hashes all 296 input files before and after; it records source commit `e8b1e944e9564725ae7252220297c239af506cbb`, model inputs and output checksums.
- Evidence folder `evidence/20261120_cpu_full/` includes `records.jsonl`, `predictions_oof.jsonl`, `summary.json`, `cpu_model_checks.json`, `run_manifest.json` and `independent_readback.json`.
- Independent `stdlib` readback recomputed row cardinalities, all signed labels and budget counts, MSE with tolerance `1e-14`, all output hashes, and selected M1 vs uniform with separate 2000-draw position bootstrap.
- Commit A `c9193ca16c4868079b5e78c3249e683639b79de9`: source/input contract, local archive and remote ref checked.
- Commit B `3c1b7a8a0251ad38cabc62e41c050bca3a5d613b`: offline predictor/code/test, local archive and remote ref checked.
- Commit Bfix `e8b1e944e9564725ae7252220297c239af506cbb`: import repair, local archive and remote ref checked.
- Commit C `9c9820cf8e6a150c00408ab7ecc9abcd7e1689a4`: raw CPU evidence and independent review, local archive and remote ref checked.
- Archive directory: `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups`, locally present Dropbox-synced *folder*. Cloud sync acknowledgment, restore validation and `main` merge were NOT performed. An isolated worktree was used; original dirty developer branch left untouched.

## 다음 연구 단위: QUARTZ-COG-01B-RESIDENT-FORK-02

**선택:** 신경망 posterior 특징을 더 추가하기보다 **동일 root·동일 random stream에서 실제 계산을 연장한 경우의 증분 효과를 측정**하는 것부터 해야 한다. 근거는 이번 82% tie, 656/864 불변, M2 음성 결과와 causal label 부재다.

1. `src/mcts/mod.rs`의 engine 같은 root에 대한 `run_search` 의미와 `src/mcts_server.rs`의 engine pool 호출을 좁게 조사하라. `advance_root`는 선택된 수 이후 *다음 root*로 subtree를 옮기는 경로여서, 지금 요구하는 **동일 root의 8→16 시뮬레이션**을 보증하지 않는다. 정확한 실행/seed/paired prefix 재생 계약을 확정하라.
2. 국면 8–12개, 체크포인트 2개, 독립 seed 3개 정도에서 CPU 8번 공통 prefix 이후 중단 vs 8번 실제 완료 계산을 더 수행하여 페어를 구성한다. 새로운 결과가 같은 input/pending/model epoch를 사용하는지 기록하고, 필요하면 두 경로에서 같은 prefix를 동일하게 재생하는 대조 실험을 먼저 수행한다.
3. 두 경로에서 실제 완료 NN 평가 수, 시간, 목적 행동 및 독립 fixed stronger reference에 대한 signed loss `L(a_stop)-L(a_continue)`를 동시에 기록한다. reference는 높은 검색예산의 surrogate일 뿐이라는 한계를 유지하라. 특히 **동률 tie-breaking 고정**과 합법 후보 집합의 identity 검사를 명시하라.
4. 이후에야 COG-01B의 채널 인식형 계산가치 예측을 baseline (fixed extra calls, PUCT/Gumbel)과 CPU 동일비용으로 비교한다. 긍정적 결과가 없으면 STOP, COG-02 attention-reopening으로 자동 승격하지 않는다.
5. 이 단계에서도 각 성공한 변경마다 commit → local backup → GitHub push → `ls-remote` readback을 병행하며, `main` 및 기존 연구 브랜치는 직접 변경하지 않는다.

정본 산출물은 본 `RESULTS_AND_NEXT_KO.md`, raw frozen evidence, 입력 잠금 및 실행 소스의 immutable SHA다. 특정 학습 checksum, checkpoint 별 결과가 새로운 출판가능한 심리기제를 증명하지는 않는다.

원전 비교: Sezener & Dayan, *Static and Dynamic Values of Computation in MCTS* (UAI 2020), https://proceedings.mlr.press/v124/sezener20a.html; Callaway et al., *Learning to Select Computations* (UAI 2018), https://proceedings.mlr.press/r16/callaway18a.html .
