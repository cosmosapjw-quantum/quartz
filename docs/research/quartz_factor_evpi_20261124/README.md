# Quartz의 잠재요인 완전정보 상계

기준은 사용자 지정 C26 87c15dc828d6eea519c219fd8a412fa21dfc464d다. 직전 exact-world 결과 45722dcc 및 C29 corrected native 576cbe554ea0을 사용했다.

고정 효용이 posterior 잠재변수 조건부로 독립이면 완전정보 효용의 CDF는 조건부 후보 CDF의 곱을 혼합한 값이다. 두 support 경계만 필요하므로 C29의 ternary support에서는 후보당 상수 작업을 한다. 최소 질의 비용을 뺀 완전정보값이 현재 STOP보다 작을 때 남은 H2를 생략한다. 논증, 의존성 반례, float 방어 범위는 [THEORY.md](THEORY.md)에 있다.

| 소급 root | Full H2 | Guarded H2 | Initial-only |
|---|---:|---:|---:|
| 전체 300 | 15.651 µs | 7.946 µs | 2.986 µs |
| CAL 60 | 8.772 µs | 4.514 µs | 1.943 µs |
| Guard 발동 190 | 15.030 µs | 2.759 µs | 2.674 µs |
| Guard 미발동 110 | 16.723 µs | 16.905 µs | 3.527 µs |

Ryzen 9 5900X, root당 교대 실행 23회 중앙값의 root 평균이다. 양쪽 H2 모두 C29의 초기 계획 중복 계산을 포함한다. 300개 루트에서 C29의 저장된 행동, 관측, 추가 노드, utility 및 plan value와 정확히 일치했다. CAL 추가 평균 utility는 0이고 63 rollout node를 추가 사용했다. 일치와 시간 절감은 기존 H2 정책을 위한 관측 결과다. initial-only를 능가하는 게임 효능은 관측되지 않았다. 새 TEST 및 전 입력의 binary64 formal bound는 없다.

채널 진단에서 (ply, focal minimax utility) 6개 strata 중 5개는 같은 stratum 안에서도 정확한 random-rollout law가 달랐다. Sibling-best로 적합한 조건부 채널은 held-back retrospective 117 roots에서 root 평균 +0.00558 bit를 얻었다. 이는 사용할 수 없는 sibling oracle label의 설명력 검사이며 native 입력에는 포함되지 않는다.

## 재현

```bash
d=docs/research/quartz_factor_evpi_20261124
q=docs/research/quartz_c29_noisy_query_20261123/evidence/C09_CORRECTED_BENCHMARK
git show "576cbe5:$q/requests.jsonl.gz" > /tmp/quartz_c29_requests.jsonl.gz
git show "576cbe5:$q/responses.jsonl.gz" > /tmp/quartz_c29_responses.jsonl.gz
cargo test --locked --offline --release --bin quartz_factor_evpi
cargo build --locked --offline --release --bin quartz_factor_evpi
python3 "$d/theory_exact.py"
python3 "$d/verify_native.py" target/release/quartz_factor_evpi
python3 "$d/game_replay.py" /tmp/quartz_c29_requests.jsonl.gz /tmp/quartz_replay.json \
  --binary target/release/quartz_factor_evpi \
  --baseline-responses /tmp/quartz_c29_responses.jsonl.gz
```

## 다음 단계

먼저 두 H2 경로에 공통인 round-0 계획 중복을 없앤다. 경계 판단은 정규화 모델에 대한 directed interval 또는 증명 가능한 forward bound로 바꾼다. 다음 채널 후보는 runtime에서 얻을 수 있는 cue와 Z를 넣은 P(Y|U_i,Z,history)다. sibling oracle을 대체할 cue는 TRAIN에서 정하고 held-out root로 calibration을 확인한다. Conditional factor가 유지될 때만 CDF를 사용한다. 이후 새 루트에서 종료 판단과 실제 전체 CPU 예산을 함께 비교한다.
