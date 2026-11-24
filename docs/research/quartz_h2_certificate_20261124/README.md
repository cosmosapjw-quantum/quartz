# Quartz 유한 H2 계산의 정확한 가지치기

직접 부모는 사용자 지정 C26 `87c15dc828d6eea519c219fd8a412fa21dfc464d`다. `THEORY.md`에 R01–R10의 유도와 독립 검토, `CODING.md`에 뒤이은 C01–C10 실행 기록, `C08_benchmark.json`에 원 측정값이 있다.

## 결과

완전정보 상계는 모든 잠재 효용을 아는 경우 얻을 수 있는 최대 기대 이득보다 첫 질의가 비쌀 때 계산을 생략한다. 마지막 질의에서는 그 질의의 대상 효용을 완전히 공개한 결과로 더 엄격한 상계를 얻는다. 이 상계는 해당 효용을 통제하고도 신호가 다른 세계 상태의 정보를 전달할 때 적용할 수 없다. 코드에서는 그러한 채널을 입력 계약에서 제외한다. 실제 게임 rollout에서 이 계약이 성립하는지는 아직 확인하지 않았다.

독립 normalized Fraction oracle에 대해 240개 세계 모형의 2,160개 H0/H1/H2 점수와 전체 조건부 정책이 정확히 일치했다. 별도 native reviewer가 비재귀 정책 열거로 1,440개의 점수를 교차 확인했다.

| Synthetic fixture | Full → 두 상계 전개 수 | Full → 두 상계 중앙 시간 |
|---|---:|---:|
| Saturated | 600 → 0 | 13.61 → 0.10 ms |
| Expensive | 600 → 0 | 13.41 → 0.10 ms |
| Competitive | 600 → 432 | 13.75 → 16.13 ms |
| Free | 600 → 507 | 14.22 → 17.90 ms |
| Weak signal | 42 → 42 | 0.22 → 0.32 ms |

두 상계는 계산이 가치 없는 두 사례를 큰 폭으로 단축했다. Competitive와 Free에서는 전개 감소를 exact-reveal 상계 비용이 넘었다. 이 5개 fixture는 세계 열거 Python reference의 비용 진단이다. C26·C27·C29의 game evidence는 그대로 유지된다.

## 재실행

```bash
python3 theory_check.py
python3 certificate.py fixture.json --bounds both
python3 check_certificate.py
python3 benchmark_certificate.py
```

## 다음 연구

C29의 factor posterior에 완전정보 상계를 Rust로 붙인다. 유틸리티가 finite support이고 잠재 조건부 독립일 때
`P(max(U,b) <= t) = 1[t >= b] sum_z q_z product_a P(U_a <= t | z,h)`
로 세계를 열거하지 않고 완전정보 기대 효용을 support 구간별로 계산할 수 있다. exact-world oracle과 stop/tie 동등성을 맞춘다. 이어서 residual rollout 정보가 후보 효용 외의 공통 상태에 의존하는지 점검하고, 그 검증 전에는 exact-reveal 상계를 게임 경로에서 빼 둔다. 원래 목적의 테스트는 신규 root와 사전 고정 비용으로 controller 전 시간을 차감한 paired utility다. C26–C29에서 이미 관측한 root로 확증 평가를 구성하지 않는다.
