# C24 Rust C5-C6 handoff

Source: `src/bin/quartz_c24_cheap_certificate.rs`; SHA256 `5a7cc595c267d5337f268045d1df8bf5909093ed7920ee50293da76da94cd381`.
Release binary SHA256 `2cef68fda2bdacae138efe7928531d9cfebc26bff8fbc13e895a7ad7d41f45b6`.

Implemented ordinary stencil/dot/max scalar envelopes from DERIVATION R5, upward gamma/F budgets, positive-square-sum inversion, raw clipping, common supported-domain validation, and sqrt-then-divide certified residual uppers. All six policy labels, exact input bits, full final_x traces, counts, and through-destruction timing follow CODE_CONTRACT.

## Actual verification

| Record | Command | Exit | stdout SHA256 | stderr SHA256 |
|---|---|---:|---|---|
| C05_RUST_RED | `cargo test --offline --locked --bin quartz_c24_cheap_certificate c24_behavior_tests -- --nocapture` | 101 | `4f2fa7636fef4254c709c8afe80472a79805d60eaf68bae4051acb558a4c49c4` | `c1dce40e6e4866218e3519350ecd626e8ad3aad6dd66af05aff3bddd6097d887` |
| C05_RUST_GREEN | `cargo test --offline --locked --bin quartz_c24_cheap_certificate c24_behavior_tests -- --nocapture` | 0 | `aa80b04170db75bbf668d04f14d67b0c2de98cacbaae1b79e817c701d0bcc4aa` | `10b2244c407806fafe7a245fd2799e6341c10be172843ec7bc2a9dfec019f964` |
| C06_RUST_FORMAT | `rustfmt --edition 2021 src/bin/quartz_c24_cheap_certificate.rs` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C06_RUST_GREEN_FINAL | `cargo test --offline --locked --bin quartz_c24_cheap_certificate c24_behavior_tests -- --nocapture` | 0 | `aeb97f5d8cfd06babd146319eef8970cb114904975b17507f6a43b32a473c5ca` | `a5efc8f72669039d59175f74d47c5a5b44459b48c5b98cae0b395df6c25b436c` |
| C06_RUST_DEBUG_FINAL | `cargo test --offline --locked --bin quartz_c24_cheap_certificate -- --nocapture` | 0 | `58a779d49c7d6a7cc7f8342bb2092b709934d10d7cd6f214941a051426b56c02` | `67be2a488f7476c8889b3ae100bb30378998550cc26a70e7e34dd8ffd1c438f2` |
| C06_RUST_RELEASE_TEST | `cargo test --release --offline --locked --bin quartz_c24_cheap_certificate -- --nocapture` | 0 | `2d3c864eda70694bc5a467edbe231228519244269b62894cfcbe6a1642d3b8d3` | `152531b2a87a89d9465c238922af05ed74f091a8dc741fe60710fdface2ab896` |
| C06_RUST_RELEASE_BUILD | `cargo build --release --offline --locked --bin quartz_c24_cheap_certificate` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `681e607a9d87a09c7b1b03c053d014bbc425a3e39b07aa98d30289c535b39733` |
| C06_RUST_NATIVE | `target/release/quartz_c24_cheap_certificate` | 0 | `ff5e451cca923567cf508bd8b11ab167723b95ac8cf974bbbcc2077ba1006544` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The exact focused behavioral command had 6 intended failures, then 6 passes. The complete isolated binary suite passed 27/27 in debug and release. Release build passed. The native JSON-lines smoke executed one real fixture and two expected structured errors, verified exact parsed input bits, explicit null plain diagnostics and all six labels. The strong fixture used 64 solver matvecs under PLAIN_ALWAYS and 8 under FAST_ONCE/FAST_BLOCK/OUTWARD_BLOCK; these are functional saved-work evidence, not a timing-speedup claim.

The known b=2^-600,S=2^-1000 fixture now returns UNSUPPORTED_RHS_NORM_RANGE. The RED log preserves the old false success with zero iterations and residual_goal_met=true.

## Test provenance

Six explicit changed behaviors had observed behavioral RED then GREEN; numerical enclosure replacement reuses intended-correctness characterizations and added focused checks, not claimed as additional observed RED.

The inherited rational example was rescaled from S=1 to S=2 and its exact oracle divided by2, because common validation requires S>=the rigorous computed norm upper. The retired ALWAYS comparator was mapped to an uncertified OUTWARD_BLOCK tie fixture. Assertions were preserved under the new policy/Option diagnostics interface; no assertion was weakened to obtain green.

## Skill context used

- PhysMath Research Loop 1.3.0 SKILL.md
- references/CODING_LOOP.md C5-C6
- references/POLICY_INTEGRATION.md
- references/SUBAGENT_MODE.md
- test-driven-development SKILL.md

Supporting writing-good-tests.md failed actual skills.read; the main TDD behavioral contract was read and applied. No recursive delegation or review.

## Scope and remaining work

Only the new C24 binary, owned C05_RUST/C06_RUST evidence, and /tmp/quartz_c24_rust* staging were written. C23 source and Cargo files remain byte-identical to the base. No cohort benchmark, commit or push was performed. Parent owns publication; c24_data/parent own the independent Fraction/native-source integration and subsequent scientific/measurement phases.
