# 1) Is your current NN size appropriate? How far can we shrink it?

### What you have right now (from your code)

* Your **“AlphaZeroNet” default** is a ResNet trunk with `hidden_channels=256, num_blocks=20`, policy/value heads, optional SE/ECA attention. That’s a “256×20” network—very AlphaZero-style and heavy.
* The same file’s docstring also mentions a **192×15 (~10 M params)** variant as a typical Gomoku-size build.

### FLOPs + parameters (15×15 board, 36 input planes)

Rough single-forward estimates (3×3 convs, policy 1×1 + small FC, value 1×1 + tiny FC; attention adds only a small extra):

| Model                   | Params (≈) | FLOPs / fwd (≈) | Max pps on RTX 3060 Ti* |
| ----------------------- | ---------: | --------------: | ----------------------: |
| **256×20**              |     23.8 M | **5.35 GFLOPs** |           **4.2–6.1 k** |
| **192×15**              |     10.2 M | **2.27 GFLOPs** |         **10.0–14.3 k** |
| **128×12 (proposal)**   |      3.7 M | **0.82 GFLOPs** |         **27.8–39.7 k** |
| **96×12 (ultra-light)** |      2.2 M | **0.46 GFLOPs** |         **49.1–70.1 k** |

*Assumes FP16 Tensor Cores at ~**64.8 TFLOPs dense** on RTX 3060 Ti (Wikipedia table; 64.8 dense / 129.6 sparse) and a realistic **35–50%** kernel-level utilization for small 15×15 convs. ([Wikipedia][1])  (Also, FLOPs ≠ speed—memory access and small-kernel launch overheads matter; see ShuffleNetV2 guidelines.) ([arXiv][2])

**Takeaway:**

* For 15×15 Gomoku, **192×15 is already comfortable** and usually superhuman with decent MCTS.
* On an **8 GB 3060 Ti**, you can **shrink to 128×12** safely for a *big* speedup with minor strength hit (MCTS recovers a lot).
* **96×12** is feasible if you really need speed (e.g., big batch, many threads), but I’d start at **128×12** as the “balanced” sweet spot.

---

# 2) Lighter NNs with near-same strength (best pick + pseudocode)

Gomoku is pattern-local; MCTS covers long-horizon tactics. You can keep a full-board receptive field with fewer blocks (rf ≈ 1+2B ≥ 21 for B=10+; full board is 15). These two families are excellent on mid-spec GPUs:

### A) **ResNet-ECA (my top pick)**

Replace SE with **ECA** (param-free 1D attention) to keep accuracy with negligible cost; drop channels/blocks to **128×12**. ECA is known to recover SE-like gains with tiny overhead. ([CVF Open Access][3])
**Why this is best for you:** very simple drop-in; keeps classic AZ training; fast and memory-light; perfect on 8 GB.

**Pseudocode (PyTorch-style):**

```python
class ResBlockECA(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(C)
        # ECA: 1D conv over channel descriptor, no FC/reduction
        self.eca   = ECALayer(C, k=3)   # tiny param count

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.eca(y) + x
        return F.relu(y)

class AlphaZeroECA(nn.Module):
    def __init__(self, in_ch=36, C=128, B=12, board=15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResBlockECA(C) for _ in range(B)])
        # Policy head (2 planes -> FC to 225)
        self.p_head = nn.Conv2d(C, 2, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2*board*board, board*board)
        # Value head
        self.v_head = nn.Conv2d(C, 1, 1, bias=False)
        self.v_bn   = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(board*board, 256)
        self.v_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        z = self.body(self.stem(x))
        p = self.p_fc(F.relu(self.p_bn(self.p_head(z))).flatten(1))
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(
                F.relu(self.v_bn(self.v_head(z))).flatten(1)))))
        return p, v
```

(**Throughput**: ~28–40 k positions/s peak on your 3060 Ti, with batches ≥64. See table above.)

### B) **Ghost-ResNet-ECA (ultra-light)**

Swap each 3×3 conv with a **Ghost bottleneck** (cheap intrinsic conv + linear “ghost” maps), keep shallow depth **96×12** + ECA. GhostNet consistently matches baselines with ~half compute. ([arXiv][4])

**Pseudocode (skeleton for a Ghost bottleneck):**

```python
class GhostModule(nn.Module):
    def __init__(self, in_c, out_c, ratio=2, kernel_size=1, dw_size=3):
        super().__init__()
        init_c = math.ceil(out_c / ratio)
        new_c  = init_c * (ratio - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, init_c, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_c), nn.ReLU(inplace=True))
        self.cheap = nn.Sequential(
            nn.Conv2d(init_c, new_c, dw_size, padding=dw_size//2,
                      groups=init_c, bias=False),
            nn.BatchNorm2d(new_c), nn.ReLU(inplace=True))
    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)[:, :self.out_c, :, :]

class GhostBlockECA(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.g1 = GhostModule(C, C, ratio=2, kernel_size=1, dw_size=3)
        self.g2 = GhostModule(C, C, ratio=2, kernel_size=1, dw_size=3)
        self.eca = ECALayer(C, k=3)
    def forward(self, x):
        y = self.g2(self.g1(x))
        return F.relu(self.eca(y) + x)
```

(**Throughput**: ~49–70 k positions/s peak.)

### C) **RepVGG-style “plain” trunk (re-param at inference)**

Train with multi-branch blocks; re-param to single 3×3 at inference, giving very fast conv stacks. Good if you want maximum kernel fusion on TensorRT later. ([CVF Open Access][5])

---

# 3) Scrutinizing your async inference worker (what to fix)

I read your worker carefully. Highlights:

* **Dynamic micro-batching**: target ≥32; timeout ≤3 ms; utilization target 80%. Good start. 
* **Pinned-memory fast paths** for H2D/D2H are implemented. Nice. 
* **Mixed precision (autocast)** with fallback logic is in place. 
* **Warmup** (1, 8, 16, 32, ≤64) and **NVML** sampling are present.  

### Critical issues / bottlenecks

1. **Misleading units + too-tight time windows**
   `timeout_ms` is stored as seconds; `max_timeout_ms` name suggests ms but holds seconds (0.003). Easy to mis-tune. Also, 0.5–3 ms is *very* short for 8+ threads under Python scheduling and can under-fill batches. (Industry dynamic batching uses **microseconds–tens of ms**, tuned by SLA.) Fix the naming and expose a **range like 2–10 ms**, auto-tuned by traffic.  ([NVIDIA Docs][6])

2. **Pinned buffer is hard-coded for Go (361 actions)**
   For Gomoku it should be **225**, otherwise you waste host RAM and add copy work on D2H. Also, it must support all three types of games - gomoku, chess, go(9*9/19*19)

3. **Pinned input buffer is `float32` but you run FP16**
   You allocate **fp32** pinned input then cast after H2D/autocast. Sending **fp16** from host halves H2D bandwidth and reduces GPU casting. (PyTorch pinned + `non_blocking=True` is the recipe.)  ([PyTorch Docs][7])

4. **Autocast + tiny batches = launch overhead bound**
   At 15×15, kernels are small; you’ll be **CPU/launch-overhead bound** without capture/fusion. Use **CUDA Graphs** to eliminate per-launch overhead once the shapes are fixed. PyTorch has first-class support and big wins on small-compute graphs. ([PyTorch][8])

5. **OOM recovery recursion**
   `batch_inference` recursively retries on OOM; better to loop (tail recursion risk, harder to reason). 

6. **Micro-batch target is fixed at ≥32**
   Great when traffic is high, but under load variance it may under-utilize the GPU. You already keep a performance deque; extend it to **auto-tune min_batch in [8,64]** based on observed throughput & NVML GPU util.  

### Upgrades that will move the needle (concrete)

**(A) Make shapes truly match Gomoku/Chess/Go**

* Set different **policy buffer** for each game and board size.

**(B) Half-precision I/O path**

```python
# host: create pinned input as float16 and transfer non_blocking
batch_data_np = np.stack(positions).astype(np.float16)
inp = torch.from_numpy(batch_data_np).pin_memory()
batch_tensor = inp.to(self.device, non_blocking=True)
```

Use pinned D2H for outputs too; you already have that path.   (Docs & rationale.) ([PyTorch Docs][7])

**(C) Traffic-aware dynamic batching (2–10 ms window)**
Replace the hard 3 ms with an adaptive controller (NVML GPU util + recent throughput trend) and a **floor of 8-16** on low traffic; raise to ≥32 when util <70%. You already have perf history—extend it.

```python
def choose_batch_window(self):
    util = self._get_gpu_utilization()   # 0..1
    # adapt 2–10ms window
    base = 0.002 + (1.0 - min(util, 0.9)) * 0.008
    # smooth with recent trend
    return clamp(smooth(base, hist=self._performance_history), 0.002, 0.010)

def _collect_batch(...):
    deadline = now() + self.choose_batch_window()
    target   = self._get_optimal_batch_size()      # you already compute
    while len(batch) < target and now() < deadline:
        try: batch.append(input_queue.get(timeout=deadline-now()))
        except Empty: break
```

(Aligns with Triton’s micro-batching idea.) ([NVIDIA Docs][6])

**(D) Capture with CUDA Graphs**
For fixed `(C,H,W)` and a small menu of batch sizes (e.g., 8,16,32,64,128,256), **pre-warm** and **graph-capture** the forward to nuke Python/kernel-launch overhead:

```python
# once per batch_size choice
static = torch.zeros(bs, *self.input_shape, device=self.device, dtype=torch.float16)
g     = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
with torch.cuda.graph(g):
    p_out, v_out = self.model(static)

# inference path
static.copy_(batch_tensor)     # non_blocking copy from pinned host
g.replay()
# p_out, v_out now hold outputs
```

(See PyTorch CUDA Graphs post/docs.) ([PyTorch][8])

**(E) Try Torch-TensorRT for a free 1.5–2×**
Once the model stabilizes (especially RepVGG or ResNet-ECA), compile via **Torch-TensorRT**; on Ampere RTX this typically halves latency / doubles throughput versus eager PyTorch. ([NVIDIA Developer][9])

**(F) Streams + double-buffering (careful!)**
Overlap H2D copy (pinned, non-blocking) with compute via a dedicated transfer stream and a compute stream, using double-buffers. PyTorch streams can help, but concurrency quirks exist—profile with Nsight. ([PyTorch Docs][10])

**(G) Fix tiny stuff**

* Rename `max_timeout_ms` → `max_timeout_s` to avoid confusion. 
* Replace recursive OOM retry with a loop. 
* Make `min_batch_size` adaptive (8–64). 

---

## Final recommendations (actionable)

1. **Switch trunk to ResNet-ECA 128×12** first. Expect ~**3×** the pps of 256×20 with minimal Elo loss (MCTS will cover it). If you still want more speed, try the **Ghost-ECA 96×12** variant. ([CVF Open Access][3])
2. **Worker:**

   * Change policy buffer to **225**, input pinned dtype to **fp16**, and **adaptive 2–10 ms** batching.  
   * **Pre-capture CUDA Graphs** for batch sizes {8,16,32,64,128}. ([PyTorch][8])
   * Try **Torch-TensorRT** once graphs are stable. ([NVIDIA Developer][9])
3. **Throughput to expect (3060 Ti):**

   * **256×20**: ~**4–6 k** pps
   * **192×15**: ~**10–14 k** pps
   * **128×12 ResNet-ECA (best pick)**: ~**28–40 k** pps
   * **96×12 Ghost-ECA**: ~**49–70 k** pps
     (FP16, well-filled batches, graphs on.)

---

### Why I’m confident in the shrink

AlphaGo Zero used 256×20 and 256×40 on 19×19 Go; Leela/KataGo do similar. Gomoku (15×15) is much smaller spatially; you don’t need that trunk width/depth for a superhuman bot once MCTS is strong. (See AlphaGo Zero paper + community engine docs for typical 20b/40b setups.) ([augmentingcognition.com][11])

[1]: https://en.wikipedia.org/wiki/GeForce_RTX_30_series "GeForce RTX 30 series - Wikipedia"
[2]: https://arxiv.org/abs/1807.11164?utm_source=chatgpt.com "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
[3]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf?utm_source=chatgpt.com "ECA-Net: Efficient Channel Attention for Deep ..."
[4]: https://arxiv.org/abs/1911.11907?utm_source=chatgpt.com "GhostNet: More Features from Cheap Operations"
[5]: https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf?utm_source=chatgpt.com "RepVGG: Making VGG-Style ConvNets Great Again"
[6]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html?utm_source=chatgpt.com "Dynamic Batching & Concurrent Model Execution"
[7]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html?utm_source=chatgpt.com "A guide on good usage of non_blocking and pin_memory() ..."
[8]: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/?utm_source=chatgpt.com "Accelerating PyTorch with CUDA Graphs"
[9]: https://developer.nvidia.com/blog/double-pytorch-inference-speed-for-diffusion-models-using-torch-tensorrt/?utm_source=chatgpt.com "Double PyTorch Inference Speed for Diffusion Models ..."
[10]: https://docs.pytorch.org/docs/stable/notes/cuda.html?utm_source=chatgpt.com "CUDA semantics — PyTorch 2.9 documentation"
[11]: https://augmentingcognition.com/assets/Silver2017a.pdf?utm_source=chatgpt.com "Mastering the game of Go without human knowledge"
