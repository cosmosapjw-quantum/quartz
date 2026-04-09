# Phase 1a 새 스레드 시작 프롬프트

아래 내용을 새 Claude Code 스레드에 붙여넣으세요:

---

```
프로젝트: QUARTZ AlphaZero (Rust MCTS 엔진 + Python 학습 루프)

## 작업 요약

docs/HANDOFF_PACKET_PHASE1A_GLOBAL_BROKER.md 를 읽고 Phase 1a (GlobalBroker 구현)를 실행하라.

## 맥락

이전 세션에서 adversarial audit를 수행하여 20개 병목을 식별했고, 그 중 13개를 이미 해결했다 (335 Rust + 107 Python 테스트 통과).

남은 핵심 병목은 B1+B5: "글로벌 추론 broker 부재"이다.

현재 `src/mcts/eval.rs`의 `BatchStdioEval`은 인스턴스별 collector thread를 생성한다. 이를 **단일 GlobalBroker**로 교체하여 프로세스당 하나의 eval I/O 소유자를 만들어야 한다.

## 필수 참조 문서

1. **docs/HANDOFF_PACKET_PHASE1A_GLOBAL_BROKER.md** — 전체 설계, 구현 단계, 현재 아키텍처, 변경 계획
2. **docs/CONCURRENT_MCTS_CRAG_2026-04.md** — 외부 문헌 기반 CRAG 분석
3. **docs/PERFORMANCE_WORKLOG_2026-04.md** — 성능 워크로그

## 구현 순서 (handoff 문서 §4 참조)

1. `collector_loop` → 독립 `broker_loop` 함수로 추출
2. `GlobalBroker` + `GlobalBrokerShared` 구조체 생성
3. `BatchStdioEval`이 `GlobalBroker`를 참조하도록 리팩터링
4. `mcts_server.rs`의 `serve()`에서 broker 생성 및 전달
5. End-to-end 검증

## 변경 금지 사항

- AsyncEvalTicket API (try_take, recv_blocking)
- QIPC 프레임 포맷 (Python 호환성)
- SHM 전송 동작
- BatchRequest / EvalResult 타입
- BatchBrokerStats 텔레메트리 스키마
- Evaluator<G> 트레이트 구현

## 검증 명령

cargo test --release --quiet
venv/bin/python -m pytest tests/ -q
venv/bin/python -m quartz.train --game gomoku7 --iterations 2 --retune --search-profile quartz
```

---

## 참고: 이전 세션에서 이미 완료된 항목 (재작업 불필요)

handoff 문서 §7에 전체 목록이 있음. 주요 항목:
- edges Mutex→RwLock (14곳)
- CAS backoff, Welford race fix
- TT eviction, PW depth awareness, depth-extended scoring
- ENVAR_CONST O(1/√N), TimeManager elapsed_ms
- IPC resync, load shedding, adaptive idle backoff
- per-slot results Mutex, thread cap hard bound
