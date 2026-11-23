# R07 — executed exact finite audit

Mathbox4.0.0 bounded computation actually ran the supplied Python3.12.14 Fraction oracle; final execution and manifest validation both exited0. Runtime6.042682s, one core/thread, wall120s, CPU100s, address-space512MiB. The final run made53851 successful exact assertions without RNG or tolerance.

The336 joint models include ternary, binary and degenerate distributions. There were3024 initial H2 comparisons and5076 conditional-history comparisons, of which4740 have nonempty histories;336 empty histories are included in the latter total.120 information-only cases check the formula/boundary.81 cases check value/policy transfer bounds.

The109/200 pure-information fixture, -1/5 moving-incumbent correction and reused69/100 general-joint fixture were reproduced. The last fixture is not claimed to belong to the restricted positive common-factor family.

Source/runner distinctions: the theory execution occurred in local scratch, so its runner manifest has no repository commit. The source authority is independently the user-selected87c15dc828d6eea519c219fd8a412fa21dfc464d. The remote handoff script is byte-identical to the final executed input, SHA256420149645d7f482cb908f8d319fa09017bdb817b8b15173789fc2812d1d6f7a9. Its final raw result SHA256 is450f9f8351affc811eb7d876bbe106f5902e7435f687b20dc67bbca634142c47.

All raw contracts/manifests/results/logs are embedded in evidence/THEORY_EXACT_AUDIT.json. Finite checking supports implementation of the oracle and can expose errors; the universal statements rely on the written proofs.

WolframContext returned internal error-32603; the independent WolframLanguageEvaluator then succeeded. This is a failed context endpoint plus successful symbolic verification, not a mathematical blocker. External PDF-fetch failures were resolved through source extraction/verified earlier PDF as recorded in R02; none is silently counted as a successful fetch.

No game TRAIN fit or CAL/TEST was run in the theory loop. Coding handoff must exercise the actual Rust controller against independent exact predictions and measure full runtime.
