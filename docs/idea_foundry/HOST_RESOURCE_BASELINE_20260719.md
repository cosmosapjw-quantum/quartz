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

That 2026-07-19 result used the original conservative affinity-overlap guard.
It remains historical evidence, but broad affinity alone is no longer a
blocking observation after the 2026-07-22 refinement. The replacement guard
measures repeated thread residence and retains affinity only as inventory.

## Implemented experiment guard

`quartz.host_resources.prepare_host_resources` now performs the following
before A15 measurement:

1. samples all allowed logical CPUs five times;
2. chooses the quietest CPU after considering its SMT sibling;
3. pins the experiment to exactly that logical CPU and verifies the mask;
4. records 1/5/15-minute load and normalized one-minute load;
5. combines `/proc/<pid>/task/<tid>/stat` CPU-time deltas with each thread's
   observed processor and blocks only sustained activity on the selected pair;
6. retains broad-affinity high-CPU processes as non-blocking inventory;
7. samples `nvidia-smi pmon`, separating graphics contexts/VRAM reservations
   from sustained external CUDA SM activity;
8. distinguishes `kernel_isolated`, `pinned_quiescent`, and
   `pinned_contended` evidence;
9. enforces complete CPU and GPU process-accounting evidence for `full`.

The thresholds are versioned in
`configs/a15_matched_service_curve.v1.json`. Diagnostic runs retain failed
guard evidence but cannot support controlled wall-clock conclusions.

## Operator precondition for full A15

Before restarting full A15, unrelated compute campaigns must be inactive or
measured away from the selected resources. Desktop applications do not need to
be closed solely because their affinity mask spans all CPUs. Run the explicit
preflight and require:

```bash
venv/bin/python scripts/a15_matched_service_curve.py \
  --profile full --cuda-device 0 --host-preflight-only
```

- exactly one selected logical CPU in `affinity_after`;
- no process with threshold-exceeding measured residence on the selected SMT pair;
- no external process with sustained CUDA SM activity;
- complete five-sample CPU-residency and GPU-process evidence;
- sibling utilization and normalized host load below the registered limits;
- `guard_passed=true`.

Kernel-level isolation requires a separately planned boot configuration and
reboot. It is not inferred from user-space pinning and was not changed here.
