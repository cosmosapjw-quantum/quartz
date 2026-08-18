# A15: Service-Curve Dynamic GPU Scheduler Estimand Redesign Protocol

## 1. Executive Summary & Audit Correction

- **Axis ID**: `A15` (`A15.service_curve_scheduler_v1`)
- **Previous Estimand**: `cuda_throughput_ratio` ($+4.9171$, $I^2 = 99.9\%$).
- **Audit Defect**: The previous estimand compared raw CUDA execution vs single-threaded CPU execution. This measured NVIDIA RTX GPU hardware acceleration over AMD Ryzen CPU, NOT the algorithmic quality of the service-curve dynamic queue batching policy.
- **Corrected Estimand**: **Latency-Bounded Scheduling Gain ($\Delta \text{QPS}_{p99 \le \tau}$)** and **Dynamic Scheduling Regret ($G_{\rm sched}$)** under realistic asynchronous MCTS query arrival processes on shipped neural networks.

---

## 2. Mathematical Definition of Redesigned Estimands

### 1. Latency-Bounded Throughput Gain ($\Delta \text{QPS}_{p99 \le \tau}$)
For a strict latency deadline $\tau$ (e.g., $\tau = 5.0\,\text{ms}$ per MCTS search step):
$$\Delta \text{QPS}_{p99 \le \tau} = \frac{\text{Throughput}_{\rm DynamicServiceCurve}(p99 \le \tau)}{\text{Throughput}_{\rm StaticBatchBaseline}(p99 \le \tau)} - 1$$

### 2. Scheduling Regret ($G_{\rm sched}$)
Let $T_{\rm arrival}(i)$ be the arrival timestamp of query $i$, and $T_{\rm eval}(i)$ be the completion timestamp.
Under Poisson or empirical MCTS burst arrival rates $\lambda \in [10^2, 10^5]\,\text{queries/sec}$:
$$G_{\rm sched} = \mathbb{E}\left[ \text{QueueWaitTime} \right] + \lambda \cdot \text{BatchPaddingWaste}$$

---

## 3. Experimental Design

1. **Hardware & Model Environment**:
   - Fixed hardware: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, TensorRT / CUDA cu128).
   - Fixed workload: Production Gomoku15 / Chess 128x10 ResNet policy-value networks.
2. **Compared Schedulers**:
   - **Baseline**: Static fixed-timeout batching ($\tau_{\rm timeout} \in \{100\,\mu\text{s}, 500\,\mu\text{s}, 2000\,\mu\text{s}\}$).
   - **Treatment (A15)**: Dynamic convex service-curve queue scheduler adapting batch dispatch thresholds based on active tree search depth and in-flight GPU streams.
3. **Primary Contract Invariant**:
   - Both baseline and treatment execute on the identical GPU hardware backend.
   - Any measured gain reflects queueing policy efficiency, not hardware backend disparities.

---

## 4. Promotion Criteria

- **Throughput Efficiency**: $\Delta \text{QPS}_{p99 \le 5\text{ms}} \ge +0.10$ ($\ge 10\%$ throughput gain at identical tail latency).
- **Tail Latency Reduction**: $p99 \text{ Latency} \le 0.85 \times \text{Baseline}$.
- **Zero Starvation**: Maximum per-query wait time $< 15\,\text{ms}$.
