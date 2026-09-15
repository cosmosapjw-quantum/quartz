# W1 수학 진단 독립 검토

검토일: 2026-09-15. 계획·수학 문서의 기준 커밋:
`9c3add3b5b28136bd9153d2650ff02bdf049fb65`.

검토 범위는 W1의 중복·공유 오차 Gaussian 진단에 필요한 수학과 주장 범위다.
W1 관측 계약 구현의 코드 검토는 별도이며, 이 문서는 그 구현을 검토하거나
실제 엔진 연결·고예산 성능을 확인했다는 기록이 아니다. 기존 9월 7일 파일럿을
재실행하지 않았다.

## 1. 정확한 모형과 coverage

고정된 관측 수 n에 대해 서로 독립인
`theta ~ N(0,v0)`, `b ~ N(0,tau2)`, `epsilon_j ~ N(0,sigma2)`와
`Y_j = theta + b + epsilon_j`를 가정한다. b는 모든 반복에 공유된다.
관측 이외 정보로 b를 식별하는 추가 채널은 없다. 적응적 정지, target 변화,
모형 학습·추정 오차는 이 진단에 포함되지 않는다.

\[
R_n=\tau^2+\sigma^2/n,\qquad
v_n=\frac{v_0R_n}{v_0+R_n},\qquad
\mathbb E[\theta\mid Y]=\frac{v_0}{v_0+R_n}\bar Y.
\]

iid로 잘못 간주하면
`R_iid=(tau2+sigma2)/n`, `k=v0/(v0+R_iid)`와
`v_iid=v0*R_iid/(v0+R_iid)`를 사용한다. 잘못된 posterior 중심의
실제 사전예측 평균제곱오차는

\[
q_n=(1-k)^2v_0+k^2R_n
=v_{\rm iid}+k^2\tau^2(1-1/n).
\]

따라서 명목 95% 구간 `k*Ybar +/- z_0.975*sqrt(v_iid)`의 정확한
**사전예측(prior-predictive) coverage**는

\[
C_n=\operatorname{erf}\!\left(
z_{.975}\sqrt{\frac{v_{\rm iid}}{2q_n}}\right).
\]

이는 theta, b, epsilon을 모두 생성모형에서 평균한 확률이다. 고정된 각
theta에 대한 빈도주의 95% coverage 보장으로 읽으면 안 된다. 올바른 posterior
구간은 이 모형에서 조건부 및 사전예측 coverage 95%를 갖는다.

## 2. 수치 oracle과 두 종류의 floor

`v0=1`, `tau2=.25`, `sigma2=.75`일 때 다음과 같다. coverage는 표에서만
백분율로 표현한다. 동일 Y0 복제 열은 실제 관측 하나를 n번 iid 관측으로
오인하는 별도 경우다.

| n | 올바른 posterior 분산 | iid 오해 분산 | 공유 오차 무시 coverage | 동일 Y0 복제의 iid 오해 coverage |
|---:|---:|---:|---:|---:|
| 1 | .500000000 | .500000000 | 95.000000% | 95.000000% |
| 8 | .255813953 | .111111111 | 77.981721% | 53.418846% |
| 64 | .207430341 | .015384615 | 37.047629% | 19.499269% |
| 512 | .200936403 | .001949318 | 13.726993% | 6.909270% |
| 4096 | .200117170 | .000244081 | 4.882709% | 2.443393% |

관측평균 오차 `Var(Ybar-theta)`의 한계는 .25이고, prior shrinkage를
포함한 올바른 posterior 분산의 한계는 `v0*tau2/(v0+tau2)=.2`다.
이 둘은 같은 floor가 아니다. iid 오해 분산은 0으로 가지만 실제 MSE는 .25로
가므로 사전예측 coverage는 0으로 간다.

동일 Y0 복제의 올바른 posterior 분산은 모든 n에서 .5다. 오해한 추정량의
실제 MSE는 `q_dup=(1-k)^2*v0+k^2*(tau2+sigma2)`이며, 위 coverage 식에
q_dup를 넣는다. 같은 수치가 우연히 관측된 별개의 확률적 평가와 동일 관측의
재전달을 값의 일치만으로 구별해서는 안 된다.

## 3. 구현과 독립적인 정확 행렬 검산

원 관측 벡터의 사전예측 공분산

\[
C=\sigma^2I+(v_0+\tau^2)\mathbf1\mathbf1^T
\]

을 구성하고 유리수 Gauss–Jordan 소거로 `C*w=v0*1`을 풀었다.
조건부 분산은 `v0-v0*sum(w)`로, iid 오해의 MSE는
`v0-2*v0*sum(w_iid)+w_iid^T*C*w_iid`로 계산했다.
폐형식 함수나 W1 구현을 import하지 않는다. n=1,2,3,8에서 다음 유리수
폐형식과 오차 없이 일치했다.

\[
v_n=\frac{n+3}{5n+3},\qquad
v_{\rm iid}=\frac1{n+1},\qquad
q_n=\frac{n^2+3n+4}{4(n+1)^2}.
\]

저장된 [검산 코드](evidence/w1_20260915/independent_matrix_oracle.py),
[표준 출력](evidence/w1_20260915/independent_matrix_oracle.stdout),
[실행 기록](evidence/w1_20260915/independent_matrix_oracle.execution.json)을 참조한다.
이 행렬 검산은 분산과 MSE를 검사한다. coverage 공식은 위 정규분포 오차의
해석적 확률 계산이며, 별도 Monte Carlo coverage 실험을 했다는 뜻은 아니다.

## 4. 주장 범위와 W2 인수인계

이 결과는 알려진 상관 Gaussian conditioning의 정확한 반례다. **증거 중복과
공유 오차를 무시하면 큰 계산량이 잘못된 확신을 강화할 수 있다**는 주장까지
지지한다. 새로운 Gaussian 이론, 실제 NN likelihood의 적합성, CPU 비용 절감,
GPU 확장 성능, 게임 성능 또는 학습 성능 향상은 이 진단으로 입증되지 않는다.

독립 RNG block 식별은 중복 방지에 필요하지만 공유 모델 오차까지 제거하지
않는다. 실제 NN 출력은 자동으로 theta에 대한 noisy measurement가 되지 않는다.
target, player perspective, evaluator, model/representation epoch를 고정하고
likelihood를 검증해야 한다.

W2의 작은 인수인계는 2행동 효용차다. `theta ~ N(mu,Sigma_theta)`,
`Y_j=theta+b+epsilon_j`, `b ~ N(0,Sigma_b)`에서

\[
\Sigma_{\theta\mid Y}=\Sigma_\theta-
\Sigma_\theta(\Sigma_\theta+\Sigma_b+\Sigma_\epsilon/n)^{-1}\Sigma_\theta
\]

를 oracle로 두고, 완전 공통 행동 bias, 행동별 bias, 잘못된 상관 부호,
model epoch 교체를 각각 검증한다. 완전 공통 bias는 행동 차이에서 상쇄되지만,
차이를 왜곡하는 bias는 같은 평가기의 반복만으로 사라지지 않아야 한다.
held-out 생성 조건에서 효용차 coverage와 구간 폭, posterior 이동 예측을
함께 비교한다. 넓은 구간만으로 coverage를 확보한 것을 배분 성능으로 부르지
않고, W3에서 선택 regret와 계산비용을 별도로 평가한다.

관계 표현과 불확실성을 사용해 다음 계산을 선택한다는 일반적 adaptive
computation 동기는 유지할 수 있다. MCTS의 효용, SMC의 추정 대상, BO의 잠재
목적함수는 각각 정의해야 하며, 이 Gaussian 계약이 세 방법을 동일한 알고리즘으로
만들거나 프로젝트 전체를 하나의 배분기로 축소한다는 주장은 하지 않는다.
