# 05. Representation, Evaluator, Distillation, and Training Control

이 문서는 A18–A21을 다룬다. controller 실험과 evaluator 구조 변경은 처음부터 분리한다.

```text
Campaign R: frozen evaluator, search/controller only
Campaign E: fixed search contract, evaluator architecture only
Campaign T: training-state allocation only
```

두 축을 동시에 바꾸면 search improvement인지 representation improvement인지 식별할 수 없다.

---

## A18. Diffusion-Regularized Deterministic Evaluator

### 핵심 계약

MCTS inference는 direct하고 deterministic하다.

```python
features = encoder(board)
policy = policy_head(features)
value = value_head(features)
```

다음은 inference path에서 금지한다.

```text
VAE reparameterization
random timestep
noise sampling
denoising loop
board reconstruction
generated-state transition
```

Diffusion/denoising은 training-only auxiliary다.

### 권장 구조

```text
board planes
  → deterministic multiscale encoder/U-Net
      ├─ full-resolution policy map
      ├─ pooled multiscale value
      └─ training-only denoising head
```

15×15는 16×16 padding 후 policy를 15×15로 crop하거나 skip size에 맞춰 interpolate한다.

### Latent denoising loss

\[
h=E_\theta(s),
\qquad
h_t=\sqrt{\bar\alpha_t}h+
\sqrt{1-\bar\alpha_t}\epsilon.
\]

\[
L_{denoise}
=\|\epsilon-\epsilon_\phi(h_t,t)\|^2.
\]

Policy/value는 clean `h`에서 계산한다.

### Discrete masked denoising alternative

보드 cell을 `{empty, current, opponent, mask}`로 두고 masked cell의 원 범주를 예측한다. Gaussian pixel noise보다 board semantics에 맞다.

### Python skeleton

```python
from quartz.idea_foundry.representation import DiffusionRegularizedEvaluatorSpec

spec = DiffusionRegularizedEvaluatorSpec(
    input_channels=20,
    base_channels=64,
    board_size=15,
    latent_denoising=True,
    masked_board_denoising=False,
    diffusion_weight=0.1,
    inference_uses_diffusion=False,
)
spec.validate()
```

Skeleton은 architecture/training contract다. 실제 `torch.nn.Module`은 별도 implementation commit에서 추가한다.

### Training schedule

각 self-play iteration 내부에서:

```text
first 70–80% epochs:
  policy + value + small denoise
last 20–30% epochs:
  policy + value only
```

self-play generation 중 diffusion head는 호출하지 않는다.

### Baseline/metric

- current ResNet
- parameter/FLOP-matched plain U-Net
- U-Net + latent denoise
- U-Net + masked-board denoise

지표:

- policy KL/cross entropy
- value MSE/Brier/calibration
- immediate win/block accuracy
- batch-1 and actual-batch latency
- positions/s
- same-NN-eval Elo
- same-wall-clock Elo

---

## A19. RW-ResT-AZ-40/144-Lite

### 목적

RandWire/DDW/ResT 아이디어를 evaluator architecture 실험으로 분리한다.

### 권장 skeleton spec

```text
stem: input → 144
cell A: 20 sparse random one-conv residual nodes
axial/global attention
cell B: 20 sparse random one-conv residual nodes
axial/global attention
policy: spatial conv (+ separate pass logit if needed)
value: mean + max + global feature
```

모든 node에 2-conv BasicBlock을 넣지 않는다. random topology보다 convolution cost가 지배하기 때문이다.

### Graph contract

```text
mandatory chain edges
WS-dominant sparse DAG
average out-degree 3.5–4
max degree 6
small fraction of degree-capped long edges
fixed graph seed in manifest
```

### Routing

training 중 random skip edge에 soft sigmoid gate를 적용할 수 있다. 배포 시 dataset-average gate로 static pruning한다.

sample마다 execution graph가 달라지는 hard routing은 batched MCTS inference에 부적합하므로 기본값으로 금지한다.

### Python skeleton

```python
from quartz.idea_foundry.representation import RwRestLiteSpec

spec = RwRestLiteSpec(
    channels=144,
    random_nodes=40,
    cells=2,
    average_out_degree=4.0,
    max_degree=6,
    attention_blocks=2,
    static_prune_for_inference=True,
)
spec.validate()
```

### Graph-seed funnel

1. 8개 graph seed 생성
2. frozen replay supervised proxy training
3. throughput/validation Pareto 상위 2–3개 선택
4. full self-play

단일 random seed 결과를 architecture claim으로 사용하지 않는다.

### Baseline/metric

- current 96f/6b ResNet
- FLOP-matched residual FCN
- local/global ResT hybrid
- static RandWire
- soft-gated RandWire
- large 40×144 BasicBlock upper bound

지표:

- graph-seed variance
- policy/value quality
- boards/s
- MCTS nodes/s
- peak VRAM
- same-time Elo
- held-out board-size degradation if size-transfer is tested

---

## A20. CPU Incremental Pattern Student

### 목적

GPU teacher의 지식을 소비자 CPU에서 빠르게 평가하는 별도 deployment lane이다.

소형 CNN + dynamic quantization만으로 `Rapfi-like`라고 부르지 않는다. 핵심은 **local pattern codebook과 make/unmake incremental accumulator**다.

### State model

Gomoku 예:

```text
line patterns:
  horizontal
  vertical
  main diagonal
  anti-diagonal

move update:
  only windows intersecting the move are invalidated/recomputed
```

### Skeleton spec

```python
from quartz.idea_foundry.representation import CpuIncrementalStudentSpec

spec = CpuIncrementalStudentSpec(
    pattern_lengths=(5, 6, 7, 9),
    codebook_dim=64,
    accumulator_dim=256,
    quantization="int8",
    simd_target="avx2",
    incremental_updates=True,
)
spec.validate()
```

### Rust deployment shape

```rust
pub trait IncrementalEvaluator<G: GameState> {
    type Accumulator: Clone;

    fn init(&self, state: &G) -> Self::Accumulator;
    fn apply_move(&self, acc: &mut Self::Accumulator,
                  before: &G, mv: G::Move);
    fn undo_move(&self, acc: &mut Self::Accumulator,
                 after: &G, undo: &G::Undo);
    fn evaluate(&self, acc: &Self::Accumulator,
                state: &G) -> EvalResult;
}
```

`GameState`의 compact make/unmake path와 함께 benchmark한다.

### Distillation targets

- teacher policy logits / search policy
- teacher value and uncertainty
- tactical motif auxiliary labels
- optional intermediate feature projection

### Baseline/metric

- full teacher GPU
- teacher CPU/ONNX
- small quantized CNN
- incremental pattern student
- classical pattern evaluator

지표:

- incremental update ns
- full recompute ns
- batch-1 move latency
- NPS
- policy/value distillation error
- CPU-only Elo
- memory footprint

Generic student와 game-specialized student의 결과를 분리한다.

---

## A21. Regret / Instability State Archive

### 목적

한 position 안의 계산 배분뿐 아니라 self-play 전체에서 어떤 state를 다시 학습·재검색할지도 선택한다.

### Archive triggers

```text
high oracle regret
late argmax revision
H1 instability
large epistemic/bias radius
large prior–search mismatch
large residual candidate mass
hidden tactical discovery
rare regime / rare path cluster
external-anchor failure
```

### Record schema

```python
from quartz.idea_foundry.representation import ArchiveRecord

record = ArchiveRecord(
    schema_version=1,
    game="gomoku7",
    position_id="...",
    board_payload={"board": []},
    checkpoint_id="...",
    source="self_play",
    oracle_regret=0.2,
    h1_instability=0.3,
    epistemic_error=0.1,
    prior_q_js=0.2,
    residual_mass_upper=0.15,
    revision_count=2,
    tactical_tags=("late_flip",),
)
```

필수 provenance:

```text
game rules/version
side to move
board/FEN/state metadata
teacher/evaluator checkpoint hash
search contract hash
root oracle/ref budget
trigger values
inserted_at / last_sampled_at
```

### Deduplication

- exact state hash
- D4 canonical hash for symmetric board games
- optional semantic cluster, but exact states는 유지
- model-version-specific label은 overwrite하지 않고 history로 저장

### Sampling

priority-only replay는 training distribution을 왜곡한다. 혼합:

```text
uniform self-play
priority archive
recent failures
rare regimes
```

importance correction 또는 최소한 source-aware batch weighting을 기록한다.

### Python skeleton

```python
from quartz.idea_foundry.representation import RegretStateArchiveSkeleton

archive_policy = RegretStateArchiveSkeleton(uniform_mix=0.25)
priority = archive_policy.priority(record)
sampling_probs = archive_policy.normalized_priorities([record])
```

### Compute fairness

standard self-play와 비교할 때 다음을 고정한다.

- total NN evaluations
- GPU seconds
- learner updates
- replay positions consumed

게임 수만 고정하면 archive restart의 비용효율을 왜곡한다.

### Baseline/metric

- standard initial-state self-play
- uniform historical restart
- Go-Exploit-like archive
- regret/instability archive

지표:

- Elo per total compute
- learning progress on archived states
- replay diversity/effective sample size
- catastrophic forgetting
- external anchor/adversarial suite

---

## Training artifact contract

각 evaluator/training campaign은 다음을 저장한다.

```text
model architecture spec + hash
parameter/FLOP count
training data manifest
search manifest used for targets
loss weights and schedules
hardware/runtime contract
throughput benchmarks
checkpoint hashes
paired arena matrix
```

Evaluator architecture의 improvement를 controller improvement로 보고하지 않는다.
