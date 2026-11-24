# 잠재요인 posterior의 완전정보 STOP

## R01 — SSOT와 질문

직접 부모는 C26 87c15dc828d6eea519c219fd8a412fa21dfc464d, tree 021a61b574557378ed5917322e10c9884d3bdc24다. 직전 45722dcc의 세계열거 oracle 및 C29 576cbe554ea0의 수정된 noisy filter는 비교·이식 자료다. 이번 질문은 고정 효용에 대한 완전정보 상계를 C29의 2성분 posterior에서 선형 비용으로 계산해 H2 STOP을 재현하며 전체 시간을 줄일 수 있는지다. 실제 게임 효용 우위는 새로운 root 및 비용 보정이 필요한 별도 주장이다.

## R02 — 선행 근거

Hay et al., UAI 2012 정의 1–2 및 정리 4–5는 유료 metalevel MDP와 완전정보 비용 상계를 정식화한다. Callaway et al., UAI 2018 §3 식 (2)–(4)는 myopic VOI, VPI, 부분 VPI를 나눈다. 이번 기여는 C29 고정 효용 posterior에서 VPI를 선형 비용으로 계산하고 native 경로에서 검사하는 것이다. C24의 SPD 응답 오차 certificate와 적용 대상이 다르다.
- https://www.auai.org/uai2012/papers/123.pdf
- https://cocosci.princeton.edu/papers/callawayLearningToSelect.pdf
