# W1 완료 보고와 다음 연구 계획 — 2026-09-15

**기존 연구 계획의 W1을 독립 Python 참조 구현으로 완료했다.** 중복·transposition·pending·epoch를 다루는 관측 계약 21개 검사와 Gaussian 진단 6개 검사가 통과했다. 이어 정확 계산 15행을 생성했다. 핵심 결과는 **중복 제거와 공유 모델 오차 보정은 서로 다른 문제**라는 것이다. 이 단계는 고예산 연구에 필요한 입력의 의미를 확정하며, 실제 검색 성능 향상은 아직 입증하지 않는다.

원 연구 평가와 전체 W1–W7 계획은 각각 [RESEARCH_UPGRADE_KO.md](RESEARCH_UPGRADE_KO.md), [IMPLEMENTATION_PLAN_KO.md](IMPLEMENTATION_PLAN_KO.md)에 있다. 9월 7일의 Gaussian 배분 실험에서 알려진 강한 기준선 대비 유용한 고예산 우위가 없었던 결과는 그대로 유지한다. 이번 결과로 그 판정을 바꾸지 않는다.

## 1. 무엇을 실제로 끝냈는가

| 항목 | 결과와 근거 |
|---|---|
| 출발점 | 게시된 연구 커밋 `9c3add3b5b28136bd9153d2650ff02bdf049fb65`; 복구 시 이후 변경 없음 |
| 확정 실행 소스 | `0c72bba36a7114108582161f921c7a1724808db3`, tree `8b66df2e4ba67da3f61da2a975bd812fccba9d26` |
| 관측 계약 | [observation_contract.py](observation_contract.py), [21개 계약 검사](tests_research/test_observation_contract.py) |
| 수학 진단 | [진단 코드](duplicate_evidence_diagnostic.py), [6개 경계·유리수 oracle 검사](tests_research/test_duplicate_evidence_diagnostic.py) |
| 확정 소스 검증 | 깨끗한 커밋에서 총 27개 통과, exit 0; [실행 기록](evidence/w1_20260915/execution.json) |
| 계산 결과 | 3개 관측 설계 × 5개 고정 예산 = [정확 계산 15행](evidence/w1_20260915/exact_diagnostic.json), exit 0 |
| 독립 검토 | [유리수 행렬 검산](W1_INDEPENDENT_REVIEW.md), [코드 검토와 수정 내역](W1_CODE_REVIEW.md) |
| 범위 | 연구 디렉터리만 변경; Rust 연결·실제 NN·GPU·self-play·학습은 실행하지 않음 |

실행 환경은 CPython 3.12.14이다. 테스트 프로세스는 약 0.114초, 정확 진단 프로세스는 약 0.067초였다. 이는 이 작은 진단의 실행 시간이며 엔진 처리량·CPU 효율의 벤치마크가 아니다. 이전 파일럿이나 전체 저장소 테스트는 재실행하지 않았다.

### 관측 계약의 구체적 동작

| 상황 | 구현된 처리 | 과학적 의미 |
|---|---|---|
| 같은 결과 재전달 | 같은 receipt를 재사용 | 메시지 수를 새 증거로 세지 않음 |
| 다른 경로의 같은 deterministic leaf | evaluator·leaf·epoch 기준으로 중복 제거 | transposition 경로 수가 likelihood 곱셈 수가 되지 않음 |
| 같은 leaf 재평가, 새 evaluation ID | 동일한 값이면 중복, 모순된 값이면 거부 | ID 변경만으로 독립 관측을 만들 수 없음 |
| 새로운 stochastic block | RNG·생성 정책·블록 식별을 요구 | 생성 과정 선언을 보존하며 독립성 자체는 인증하지 않음 |
| 실행 중인 작업 | 완료 evidence와 별도 보관 | pending의 미관측 값을 posterior에 넣지 않음 |
| root·후보 세대·model·representation·target 변경 | visible view를 비우고 기존 receipt를 보존 | 다른 추정 대상·모델의 증거가 조용히 섞이지 않음 |
| 늦게 도착한 이전 작업 | quarantine | A→B→A로 돌아와도 이전 pending은 다시 유효해지지 않음 |
| 블록 병합 | centered-moment merge | N/W/M2를 일관된 완료 블록으로 계산; ESS로 해석하지 않음 |

동일 상태라도 player 관점, 평가기 입력의 역사, continuation target이 다르면 같은 leaf identity를 쓰면 안 된다. 호출자가 완전한 evaluator 입력과 epoch 의미를 정확히 제공해야 한다. 서로 다른 leaf의 `moments` 합계는 회계용 통계이며 하나의 잠재 효용에 대한 posterior가 아니다.

초기 계획의 `CompletedBlock`은 구현에서 `Moments(n, mean, m2)`를 포함한다. `n_completed`, `sum_value`, `centered_sum_sq`는 이를 읽는 속성이다. deterministic block에는 정확히 한 값만 허용한다. stochastic 집계 블록의 n은 원시 값 수이며 블록 내부 상관은 별도 likelihood가 필요하다.

검토 중 발견된 실제 오류 두 개도 보존했다. 첫째, identity가 되돌아오면 오래된 pending 결과가 수용되던 오류를 고쳤다. 둘째, 같은 RNG에 붙인 별칭 블록 ID를 다른 RNG에 다시 붙일 수 있던 모순을 고쳤다. [증거 디렉터리](evidence/w1_20260915/)의 해당 `red`·`green` 로그와 코드 리뷰에서 실패와 수정 결과를 확인할 수 있다. 초기 수학 테스트는 모듈 미생성에 따른 collection error였으며, 이를 여섯 개 개별 assertion 실패로 표현하지 않는다.

이 구현은 단일 소유자 reducer다. 병렬 Rust의 원자적 갱신을 고친 것이 아니며, 저장된 원시 기록과 인덱스의 메모리는 작업 수에 따라 증가한다. 모델 변경 뒤 증거를 다시 해석하는 transport도 아직 없다. 이전 deterministic evidence를 새 root에서 자동 재주입하지 않는 보수적 동작은 W2에서 명시적인 전달 규칙과 비용을 검증해야 한다. `pending`에는 아직 물리적으로 끝나지 않은 이전 세대 작업도 남으므로, W3는 현재 의사결정의 pending과 전체 outstanding 비용을 구분해야 한다.

## 2. 수학 연구 결과: 많은 관측이 없애는 오차와 남기는 오차

고정된 스칼라 추정 대상에 대해 다음 모형을 사용했다.

\[
\theta\sim\mathcal N(0,v_0),\quad
Y_j=\theta+b+\epsilon_j,\quad
b\sim\mathcal N(0,\tau^2),\quad
\epsilon_j\overset{\mathrm{iid}}\sim\mathcal N(0,\sigma^2).
\]

사전에서 서로 독립이며, b는 같은 관측 집단 안에서 공유한다. 계산에는 \(v_0=1,\tau^2=1/4,\sigma^2=3/4\)를 사용했다. 이 값은 실제 Quartz NN 오차를 측정하거나 추정한 값이 아니다. n은 결과를 보고 정하는 stopping time이 아닌 고정 설계다.

관측평균의 오차 분산과 올바른 posterior는

\[
R_n=\tau^2+\sigma^2/n,\qquad
v_n=\frac{v_0R_n}{v_0+R_n},\qquad
\mu_n=\frac{v_0}{v_0+R_n}\bar Y.
\]

반면 상관을 무시하면 \(\widetilde R_n=(\tau^2+\sigma^2)/n\)과
\(\widetilde v_n=v_0\widetilde R_n/(v_0+\widetilde R_n)\)을 쓰게 된다.
\(k_n=v_0/(v_0+\widetilde R_n)\)라 할 때 그 posterior 평균의 실제 사전예측 MSE는

\[
q_n=(1-k_n)^2v_0+k_n^2R_n
=\widetilde v_n+k_n^2\tau^2(1-1/n).
\]

따라서 잘못된 모형이 제시하는 명목 95% 구간의 실제 사전예측 포함률은

\[
C_n=\operatorname{erf}\!\left(z_{.975}
\sqrt{\frac{\widetilde v_n}{2q_n}}\right).
\]

| 관측 수 n | 공유 오차를 반영한 분산 | iid로 오해한 분산 | 공유 오차 무시 시 포함률 | 같은 결과를 n번 복제하고 iid로 오해한 포함률 |
|---:|---:|---:|---:|---:|
| 1 | 0.500000000 | 0.500000000 | 95.000000% | 95.000000% |
| 8 | 0.255813953 | 0.111111111 | 77.981721% | 53.418846% |
| 64 | 0.207430341 | 0.015384615 | 37.047629% | 19.499269% |
| 512 | 0.200936403 | 0.001949318 | 13.726993% | 6.909270% |
| 4096 | 0.200117170 | 0.000244081 | 4.882709% | 2.443393% |

동일 결과 복제의 올바른 posterior 분산은 모든 n에서 **0.5**다. 완전히 독립적인 새 오차의 대조군에서는 올바른 분산과 iid 분산이 일치하며 모든 n에서 포함률은 95%다. 위 표는 시뮬레이션 추정치가 아니라 주어진 Gaussian 모형의 정확한 기대값이다. 개별 고정 theta에 대한 빈도주의 coverage나 적응적 정지 보장을 뜻하지 않는다.

여기에는 서로 다른 두 바닥값이 있다. \(\operatorname{Var}(\bar Y-\theta)\to.25\)이고, prior shrinkage까지 포함한 \(v_n\to.2\)다. 반복 계산으로 epsilon은 평균화할 수 있지만 공유한 b는 제거하지 못한다. W1의 중복 제거로 첫 번째 복제 문제를 막아도, 서로 다른 RNG의 관측에 남는 b를 무시하면 두 번째 과신 문제는 남는다.

독립 검토자는 원 관측 벡터의 공분산을 구성하고 유리수 Gauss–Jordan 소거로 n=1,2,3,8을 계산했다. 그 결과가 스칼라 식과 정확히 일치했다. Wolfram도 \(v_n=(n+3)/(5n+3)\), \(q_n=(n^2+3n+4)/(4(n+1)^2)\)를 반환했다. 첫 Wolfram 호출에는 내부 경고가 있었고 좁힌 대수 검산에는 경고가 없었다. [외부 검산 기록](evidence/w1_20260915/external_checks.json)에 둘 다 남겼다. 9월 7일의 Wolfram 접속 실패 기록은 당시 사실로 유지한다.

이것은 알려진 상관 Gaussian 조건부 추론을 이용한 반례다. 새 수학 정리의 novelty로 내세우지 않는다. [EMCTS](https://proceedings.iclr.cc/paper_files/paper/2025/file/bcbd670951a6dede2123961de19e5ea3-Paper-Conference.pdf)의 모델 epistemic uncertainty 전파와 [Sezener–Dayan](https://proceedings.mlr.press/v124/sezener20a.html)의 상관 belief·VOC가 직접적인 선행 비교 대상이라는 기존 판정도 유지한다. SciSpace는 보충 발견에 사용했고 최종 연결은 원 논문으로 확인했다.

## 3. 원 동기를 살리는 고예산 연구 방향

[원 동기 기록](MOTIVATION_PROVENANCE.md)의 작은 NN·잠재 효용·선택적 계산·여러 대안 영역 유지라는 직관을 유지한다. 이번 결과가 제안하는 초점은 **더 많은 계산을 어디에 해야 최종 효용차에 관한 새로운 정보를 얻는가**이다. 단순히 visit를 줄이거나 검증 장치를 추가하는 것을 연구의 기여로 삼지 않는다.

다음 다섯 수준을 연결하는 연구 조직이 적절하다. 통계역학적 표현은 잠재 효용의 공동 분포와 다봉성을 정의하고, 표본·근사 알고리즘은 그 분포를 계산하며, 순차 배분은 다음 계산의 결정 가치를 평가한다. 병렬 population·wave는 pending과 중복을 다루고, 이종 실행 시스템은 실제 비용과 지연을 측정한다. W1은 이 사이를 흐르는 관측의 의미만 확정했다. MCTS 효용, SMC의 적분·분포 추정 대상, BO의 잠재 목적함수는 각각 별도로 정의해야 하며 하나의 알고리즘으로 동일시하지 않는다.

고예산에서 중요한 추가 가설은 **같은 오차 원천의 재평가보다 효용차를 왜곡하는 오차를 다르게 관측하는 계산이 유리할 수 있는가**이다. 예를 들어 동일 target을 관측하는 독립적인 G개 오차 집단에 각각 m회 관측하고 N=Gm이라면, 단순 평균의 오차 분산은

\[
\operatorname{Var}(\bar Y-\theta)=\tau^2/G+\sigma^2/N
\]

이다. 이 식은 가정하에서의 유도이며 이번 코드 실험에 새 조건으로 추가하지 않았다. 새 NN, 다른 rollout, 다른 fidelity가 실제로 독립적인 b를 제공하거나 같은 target을 유지한다고 가정할 수는 없다. **계산 종류를 바꿀 때 줄어드는 효용차 오차와 추가 비용을 함께 측정하는 것**이 검증할 가설이다. 작은·큰 평가기 선택은 이 가설이 살아남은 뒤 기존 W7과 MPV 기준선에서 다룬다.

노벨티 후보는 중복 제거, covariance, KG 각각의 이름이 아니다. 검증된 저차 관계 표현이 여러 root와 계산 파동을 거쳐 재사용되고, 효용차에 유효한 새로운 계산을 선택하며, 그 결합이 강한 기준선보다 실제 비용 대비 결정 품질을 높이는지가 기여 후보다. W1은 그 필요조건의 일부만 제공했다.

## 4. 다음에 수행할 W2의 구체적 인수인계

W2–W7은 아직 미수행이다. 다음 작업은 전체 엔진 개조에 앞서 **두 행동 효용차에 공유 모델 오차를 넣은 W2-A**로 좁힌다. 아래는 실행 전 계획이며 이번 결과가 아니다.

### W2-A: 정확 oracle과 작은 잠재 belief

새 `latent_belief.py`와 `tests_research/test_latent_belief.py`를 만들고, W1의 visible block을 관측 연산자에 연결한다. 원 계획의 `UtilityBelief`·`predict_contrast`·`observe`·`reset_epoch`를 유지한다. 첫 구현은 동결한 두 행동의 Gaussian 모델로 한정한다.

\[
\theta\sim\mathcal N(\mu,\Sigma_\theta),\quad
Y_j=\theta+b+\epsilon_j,\qquad
\Sigma_{\theta\mid Y}
=\Sigma_\theta-\Sigma_\theta
(\Sigma_\theta+\Sigma_b+\Sigma_\epsilon/n)^{-1}\Sigma_\theta.
\]

고정 설계의 전체 관측 벡터 또는 증대 잠재 상태 \((\theta,b)\)에 조건부 추론하는 것을 oracle로 사용한다. 재관측 때 매번 b를 새로 독립 추출하는 잘못된 갱신을 대조군으로 둔다. 위 단순 식을 적응적 비대칭 표본 수에 그대로 적용하지 않는다.

| 구성 | 실행 전 고정할 값·조건 |
|---|---|
| theta prior | \(\mu=(.1,0)\), \(\Sigma_\theta=.16\begin{pmatrix}1&.5\\.5&1\end{pmatrix}\) |
| 신규 관측 noise | \(\Sigma_\epsilon=\operatorname{diag}(.04,.25)\) |
| 공유 bias 층 | 0; \(.04\mathbf1\mathbf1^\top\); \(.04I\); \(.04\begin{pmatrix}1&-.6\\-.6&1\end{pmatrix}\) |
| 오지정 층 | 마지막 음의 상관을 양의 .6으로 오해; true 조건은 그대로 유지 |
| 관측 예산 | 각 행동 쌍을 함께 관측하는 n=1,8,64,512,4096; 실제 스칼라 관측 수는 2n |
| 기준선 | 전체 증대 Gaussian oracle, 공유 bias를 반영하는 대각/저차 모델, iid 오류 대조군 |
| 물리적 확인 | 완전 공통 bias가 관측 효용차에서 상쇄되는지, contrast bias floor가 남는지 |
| epoch·재사용 | 모델 변경은 기존 posterior 무효화; target/operator 대응이 없는 root transport는 거부 |

통과 조건은 이론 identity가 작은 dense oracle과 절대 오차 \(10^{-10}\) 이내로 일치하고, 공통 모드의 상쇄와 반복 관측의 bias floor를 모두 재현하는 것이다. 이것은 코드·모형 정확성 gate이며 실제 학습된 uncertainty의 calibration gate가 아니다.

공통 모드 상쇄는 \(d=(1,-1)^\top\)에 대해 \(d^\top\Sigma_b d=0\)이라는 관측 contrast의 검사다. 공통 채널과 contrast 채널의 관측 noise가 상관되면, 전체 관측에 조건부 추론한 contrast posterior가 bias 없는 경우와 항상 같다는 뜻은 아니다. 위 비대칭 noise에서도 그 차이를 dense oracle로 검증한다.

### W2-B: 알려진 모형에서 calibration을 확인한다

W2-A 통과 뒤 고정 seed `20260915 + scenario_index`, 각 true 조건마다 4,096개의 독립 prior world를 생성하는 CPU 탐색 파일럿을 제안한다. 조건·예산·방법 사이에는 같은 theta·bias·noise-prefix pairing을 보존한다. 별도 예산 셀을 독립 표본으로 세지 않는다. Gaussian에서는 noise 평균의 분포가 정확하므로 4,096개 개별 draw를 전부 생성하는 대신 독립 budget increment의 충분통계로 같은 nested prefix를 만들 수 있다. 이 최적화도 adaptive sampling에는 그대로 적용하지 않는다.

모든 방법의 효용차 50/80/95% 구간 포함률·폭, posterior 평균 이동의 예측, 최종 행동의 정확 Bayes regret 또는 같은 truth에 대한 paired regret를 저장한다. oracle과 일치하는 것은 구현 확인이며 새 성능 기여가 아니다. 최초 runtime probe와 전체 파일럿의 합을 **CPU 10분·메모리 1GB 이내**로 제한하는 제안이다. 초과하면 실행 전 크기를 줄여 새 manifest에 기록한다. 수행 시간은 아직 측정하지 않았다.

### W2-C 이후: novelty와 significance를 판정한다

1. 동결한 실제 작은 NN 특징의 저차 모델을 도입한 뒤, 별도 held-out 문제군에서 contrast calibration과 폭을 함께 평가한다. 실제 최소최대값을 모르면 stronger reference와의 일치를 진실 coverage라고 부르지 않는다.
2. W3에서 새 계산의 **posterior 평균 이동**으로 KG를 산출한다. 현재 uncertainty 크기나 정책 TV 이동을 실현 효용 이득으로 쓰지 않는다. pending의 추가 정보와 비용을 조건부로 계산한다.
3. W5에서 실제 완료 NN call과 엔드투엔드 시간을 따로 맞춘다. W6에서 쉬운 root의 절약 비용을 어려운 root에 실제 재투자해 전체 workload의 평균·tail 손실을 비교한다. 기존의 독립 문제군·고정 예산·강한 기준선 계획을 유지한다.
4. 중복 제거만으로 개선되는지, bias/상관 모델의 추가 이득이 있는지, pending·영역 재개방이 추가 이득을 주는지 각각 ablation한다. 보정만 좋아지고 결정 품질이 나빠지거나 overhead가 이득을 지우면 그 구성은 실패로 기록한다.

W2에서 학습된 상관의 이득이 없으면 보수적 대각 모델로 축소한다. 높은 예산에서 강한 기준선 대비 이득이 사라지면 저예산 적응 계산의 장점과 고예산 우위의 미입증을 함께 보고한다. 이 판정이 원 직관을 폐기하는 것은 아니다. 어떤 관계 표현과 계산 종류가 실제로 새 정보를 주는지 원인을 더 좁히는 결과다.

## 5. 재현과 인수인계

코드의 확정 실행 SHA는 위 `0c72bba…`이며 결과·문서 커밋의 조상으로 보존된다. 다음 명령은 저장소 루트에서 실행한다. 기존 결과를 덮어쓰지 않도록 새 출력 경로를 사용한다.

```bash
python -m unittest discover -s docs/research/beyond_alphazero_20260907/tests_research -p 'test_*.py' -v
python docs/research/beyond_alphazero_20260907/duplicate_evidence_diagnostic.py --output /tmp/quartz-w1-new-result.json
python docs/research/beyond_alphazero_20260907/evidence/w1_20260915/independent_matrix_oracle.py
```

[W1 계약](W1_CONTRACT_20260915.md)은 가정·실패 조건·실행 범위와 모델별 하네스 선택 상태를 기록한다. 원시 stderr 한 줄의 trailing space는 실패 출력을 그대로 보존한 것이다. 해당 로그를 제외한 소스·문서의 `git diff --check`를 확인했다. 이것은 테스트 실패나 결과 수정이 아니다.

현재 상태: **W1_REFERENCE=COMPLETE; W1_EXACT_DIAGNOSTIC=COMPLETE; W2–W7=PLANNED; ENGINE_HIGH_BUDGET_GAIN=NOT_ESTABLISHED.**
