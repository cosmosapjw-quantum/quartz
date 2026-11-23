# C24 재개와 재현

## 저장된 원시 결과의 정확 복원

이 문서의 코드·실험 identity는 [BASELINE.json](BASELINE.json), [C07_PRE_TEST_FREEZE.json](evidence/C07_PRE_TEST_FREEZE.json), [C24_TEST_TIMING.manifest.json](evidence/C24_TEST_TIMING.manifest.json)에 있다. 실제 복원 검사는 [check_c09_restore.py](coding/check_c09_restore.py)와 [C09_RESTORE.json](evidence/C09_RESTORE.json)에 보존했다.

백업 위치:

```text
/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta24_backups
```

`STAGE_RECEIPTS.jsonl`의 원하는 stage 항목에서 base archive와 누적 overlay의 이름·SHA-256을 읽는다. 새 빈 디렉터리에 **base 먼저, 해당 stage overlay 다음** 순서로 풀어야 한다. C24 overlay 단독으로는 선행 프로젝트를 구성하지 않는다. 서버 동기화 acknowledgment는 별도로 관측되지 않았으며 이 문서는 파일의 실제 byte 검사와 복원만 보장한다.

Git에서 C24 자료가 필요한 경우, 최초 완결 실험 snapshot은 아래의 정확 commit이다.

```text
df34b7ae1b867f4917c2d302ea97aca40cc55077
```

이후 C09/C10에는 독립 검토·복원·최종 설명과 인계 기록이 추가되며 과학 코드와 입력은 그대로다. 최종 branch는 `research/quartz-general-c24-cheap-certificates-20261123`이고 최종 C10 commit/readback은 외부 receipt로 확인한다. 원래 개발 worktree를 덮어쓰지 않는다.

큰 trace는 원래 gzip의 연속된 bytes를 최대45MiB 조각으로 저장했다. 별도 재압축·변환을 하지 않았다. 복원된 checkout에서 whole gzip이 아직 없을 때 다음을 실행한다. 기존 출력은 덮어쓰지 않는다.

```bash
venv/bin/python docs/research/quartz_general_cheap_certificates_20261123/coding/reassemble_c24_trace.py assemble --split TRAIN
venv/bin/python docs/research/quartz_general_cheap_certificates_20261123/coding/reassemble_c24_trace.py assemble --split CAL
venv/bin/python docs/research/quartz_general_cheap_certificates_20261123/coding/reassemble_c24_trace.py assemble --split TEST
```

각 part hash/size와 전체 hash/size를 모두 검사한다. 원래 파일을 보존한 별도 검증에는 `--output /tmp/새로운_이름.jsonl.gz`를 지정한다.

| split | whole bytes | SHA-256 |
|---|---:|---|
| TRAIN | 65,708,367 | `b23180121487c18caf32edd717f5f8cc983514641905d95a83d95f35e2c814cf` |
| CAL | 65,633,648 | `63208ecc5fa1de8ebce34ef919652734997ee01a853ed9af138861a2ea38a24f` |
| TEST | 131,402,222 | `1a225dd982f6b5cd512971880340c2d7d22cf71c4d930b119914f3fe7066cd87` |

실제 측정 native executable도 별도로 압축해 보존했다.

```text
C24_NATIVE_RELEASE_2cef68fda2bdacae.gz
compressed SHA256 b46d74685a8aef857b2cb801e75e0d85e83fbb8c0665d733212ed8083bee6736
uncompressed SHA256 2cef68fda2bdacae138efe7928531d9cfebc26bff8fbc13e895a7ad7d41f45b6
```

실행 파일의 대상은 기록된 Linux x86_64 환경이다. 원래 byte snapshot 복원과 다른 환경에서의 재빌드는 다른 재현 작업이다.

## 환경과 구현 검사

실제 환경은 Python3.12.3, NumPy2.4.2, SciPy1.17.0, mpmath1.3.0, Matplotlib3.10.8, Rust1.99.0이다. 정확 compiler 문자열·CPU·flags는 C04_RUN_CONTRACT와 manifest에 있다. C24의 ignored venv는 system-site-packages를 사용해 만들었으며 패키지를 설치하지 않았다. 새 환경에서는 같은 라이브러리가 제공되는지 먼저 확인한다.

repo root에서 실제 사용한 명령:

```bash
venv/bin/python -m unittest discover -s tests -p test_c24_sources.py -v
cargo test --offline --locked --bin quartz_c24_cheap_certificate -- --nocapture
cargo test --release --offline --locked --bin quartz_c24_cheap_certificate -- --nocapture
cargo build --release --offline --locked --bin quartz_c24_cheap_certificate
```

`--offline` 재빌드는 Cargo dependency cache가 있는 환경을 전제로 한다. `Cargo.lock`과 source를 보존했으며, 새 빌드의 binary hash는 실제로 계산해 기록해야 한다. 원래 compiler/runtime과 다른 빌드를 원래 측정 executable이라고 표시하지 않는다.

## 같은 입력으로 새 실행하기

이미 저장한 raw output과 verifier report는 덮어쓰지 않는다. 같은 입력의 재실행에는 새 evidence 디렉터리를 만든 뒤 공개 입력·별도 oracle·source manifest 세 파일을 복사한다. 기존 source를 다시 manufacture하면 libm/Python 환경에 따른 새 input identity가 생길 수 있으므로, 동일 입력 재현에는 저장된 입력 자체를 사용한다.

```bash
C24_DOC=docs/research/quartz_general_cheap_certificates_20261123
C24_REPLAY_DIR=$(mktemp -d /tmp/quartz_c24_replay_XXXXXX)
cp "$C24_DOC/evidence/C24_PUBLIC_TEST.jsonl.gz" "$C24_REPLAY_DIR/"
cp "$C24_DOC/evidence/C24_ORACLE_TEST.jsonl.gz" "$C24_REPLAY_DIR/"
cp "$C24_DOC/evidence/C24_SOURCES_TEST.json" "$C24_REPLAY_DIR/"
venv/bin/python "$C24_DOC/coding/c24_run.py" --split TEST --mode trace --evidence-dir "$C24_REPLAY_DIR"
venv/bin/python "$C24_DOC/coding/c24_verify.py" --split TEST --mode trace --evidence-dir "$C24_REPLAY_DIR"
venv/bin/python "$C24_DOC/coding/c24_run.py" --split TEST --mode timing --evidence-dir "$C24_REPLAY_DIR"
venv/bin/python "$C24_DOC/coding/c24_verify.py" --split TEST --mode timing --evidence-dir "$C24_REPLAY_DIR"
venv/bin/python "$C24_DOC/coding/c24_analyze.py" --split TEST --evidence-dir "$C24_REPLAY_DIR"
venv/bin/python "$C24_DOC/coding/render_c24_report_figure.py" --split TEST --evidence-dir "$C24_REPLAY_DIR"
```

생산 runner는 과학 코드·계약이 Git에 추적되고 HEAD와 같아야 한다. `--allow-dirty`는 UNIT 전용이다. 입력을 확인한 다음 JSON wire는 과학값을 decimal string으로 보내며 trace가 실제 native parsed bits를 반환한다.

통계 분석기는 matching verification PASS와 입력 hash를 확인한 뒤 실행한다. TEST의 정확 잔차 vector audit은 C04에서 지정한9개 parent의 예산32에만 적용한다. q interval과 fallback 검사는 전체 request를 대상으로 한다.

새 wall-time 측정은 원래 시간값과 같지 않으며, 따라서 timing raw bytes의 SHA도 달라지는 것이 정상이다. 비교할 것은 검증된 source·binary identity, 정확 utility/계수·선택·enclosure의 의미, 측정 조건과 새로운 시간 통계다. 같은 raw snapshot의 재조립에는 원본 SHA와 정확히 같은 bytes를 요구한다.

## 다음 사이클의 인계

[NEXT_STAGE.json](NEXT_STAGE.json)은 C25 제안이며 실행 기록이 아니다. 다음 실행에서도 먼저 현재 C24 최종 branch/receipt/보호 파일을 확인하고, 실제 이론10단계 이후 코딩10단계를 수행한다. 각 단계의 commit·archive·remote readback은 계속 남긴다.
