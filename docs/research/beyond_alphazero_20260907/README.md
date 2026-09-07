# Beyond-AlphaZero 연구 평가 — 2026-09-07

**먼저 읽기:** [한국어 연구 평가](RESEARCH_UPGRADE_KO.md) → [구체적 구현·연구 계획](IMPLEMENTATION_PLAN_KO.md) → [수학적 유도](MATHEMATICAL_UPGRADE_KO.md).

원 직관은 보존하지만 현재 beyond-AlphaZero 성능은 미입증이다. 이번에는 수학·소스 반례 12개 검사와 240조건 Gaussian 실험을 실제 수행했다. 고예산 배분 효과는 알려진 강한 배분법에서도 재현되어 새 성능 우위로 판정하지 않았다. 생산 엔진·기존 캠페인·claim ledger는 수정하지 않았다.

- [선행연구](LITERATURE_REVIEW.md), [최신 소스 감사](SOURCE_AUDIT.md), [원 동기 provenance](MOTIVATION_PROVENANCE.md)
- [수행 기록](RESEARCH_RECORD.md), [실행 receipt](evidence/execution.json)
- [원시 결과 CSV.gz](evidence/pilot/raw.csv.gz), [결과 JSON](evidence/pilot/summary.json), [분석 검증](evidence/analysis/validation.json)
- [계산 독립 검토](PILOT_INDEPENDENT_REVIEW.md), [최종 독립 검토](FINAL_INDEPENDENT_REVIEW.md)
- [재현·identity 안내](REPRODUCIBILITY.md)

![계산 예산과 calibration](evidence/analysis/pilot_budget_and_calibration.png)
