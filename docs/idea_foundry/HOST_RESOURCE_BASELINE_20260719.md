# Host resource baseline — 2026-07-19

This is an execution-readiness snapshot, not scientific evidence for any
ablation effect. Measurements were taken after the old QUARTZ environment and
campaign process were stopped.

## Hardware and scheduler topology

- CPU: AMD Ryzen 9 5900X, 12 physical cores, 24 logical CPUs, 2 threads/core
- online CPUs and effective cgroup cpuset: `0-23`
- NUMA: one node containing `0-23`
- kernel-isolated CPU list: empty
- kernel command line: no `isolcpus`, `nohz_full`, or `rcu_nocbs` reservation
- GPU: NVIDIA RTX 3080 Ti, 12 GiB, compute capability 8.6
- NVIDIA driver: 595.71.05

Therefore the host currently provides no kernel-level exclusive CPU. A
`taskset` or `sched_setaffinity` result is process pinning only.

## Live load and affinity findings

At 2026-07-19 18:17 KST, load average was `2.38 / 2.24 / 2.74`. Several
unrelated processes were eligible to run on every logical CPU:

- VS Code renderer processes: affinity `0-23`
- Chrome Remote Desktop host: affinity `0-23`
- a surviving `htt_base` multiprocessing worker: affinity `0-23`

The end-to-end A15 diagnostic selected logical CPU 8; its SMT sibling is CPU
20. The selected sibling pair was quiet at sampling time, but three competing
processes exceeded the configured CPU threshold while retaining affinity to
that pair. The recorded outcome was therefore:

- `guard_passed=false`
- `isolation_level=pinned_contended`
- diagnostic allowed, because it is non-promotional
- full A15 would fail closed before importing PyTorch or allocating VRAM

This distinction prevents a low instantaneous core-utilization sample from
being misreported as an isolated CPU comparison when broad-affinity work can
migrate onto the measured core.

## Implemented experiment guard

`quartz.host_resources.prepare_host_resources` now performs the following
before A15 measurement:

1. samples all allowed logical CPUs;
2. chooses the quietest CPU after considering its SMT sibling;
3. pins the experiment to exactly that logical CPU and verifies the mask;
4. records 1/5/15-minute load and normalized one-minute load;
5. inventories high-CPU processes whose affinity overlaps the sibling pair;
6. distinguishes `kernel_isolated`, `pinned_quiescent`, and
   `pinned_contended` evidence;
7. enforces the preregistered thresholds for the `full` profile.

The thresholds are versioned in
`configs/a15_matched_service_curve.v1.json`. Diagnostic runs retain failed
guard evidence but cannot support controlled wall-clock conclusions.

## Operator precondition for full A15

Before restarting full A15, stop unrelated compute campaigns or give them a
non-overlapping affinity set. Re-run the diagnostic and require:

- exactly one selected logical CPU in `affinity_after`;
- no threshold-exceeding process eligible on the selected SMT pair;
- sibling utilization and normalized host load below the registered limits;
- `guard_passed=true`.

Kernel-level isolation requires a separately planned boot configuration and
reboot. It is not inferred from user-space pinning and was not changed here.
