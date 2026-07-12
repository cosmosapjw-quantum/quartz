# Omoknuni: Production Requirements Document
## AlphaZero-Style Game AI with Thermodynamically-Efficient MCTS

### Version 2.0 | December 2024

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Core Components & File Structure](#core-components)
5. [Algorithm Specifications](#algorithms)
6. [Application Flowchart](#flowchart)
7. [Technology Stack](#tech-stack)
8. [Development Phases](#phases)
9. [Claude Development Guidelines](#claude-guidelines)
10. [Scope Definition](#scope)

---

## 1. Executive Summary {#executive-summary}

**Omoknuni** is an AlphaZero-style game AI engine that incorporates Thermodynamics & Active Inference (TAI) principles into Monte Carlo Tree Search (MCTS) to achieve 30-50% computational efficiency gains while maintaining competitive playing strength.

### Key Innovations
- **Information-Theoretic Selection**: Prioritizes moves that maximize information gain
- **Entropy-Aware Search**: Tracks configuration and policy entropy during search
- **Active Inference Integration**: Uses free energy minimization for principled exploration
- **Self-Play Training**: AlphaZero-style iterative improvement through self-play

### Target Metrics
- 30-50% reduction in simulations required for equal playing strength
- <200ms move generation latency on consumer hardware (Python implementation)
- Memory footprint <2GB during search
- Convergence to expert-level play within 100,000 self-play games

---

## 2. Project Overview {#project-overview}

### 2.1 Architecture Overview
```
┌─────────────────────────────────────────────────────┐
│                   Omoknuni Engine                    │
├─────────────────────────┬───────────────────────────┤
│      Game Logic         │      Python AI Core       │
│  ┌─────────────────┐   │  ┌─────────────────┐     │
│  │ C++ Game Engine │   │  │   TAI-MCTS      │     │
│  │   (Existing)    │◄──┼──┤  (Python)        │     │
│  └─────────────────┘   │  └────────┬────────┘     │
│   Via Python Binding    │           │               │
│                        │  ┌────────▼────────┐     │
│                        │  │  Neural Network  │     │
│                        │  │   (PyTorch)     │     │
│                        │  └────────┬────────┘     │
│                        │           │               │
│                        │  ┌────────▼────────┐     │
│                        │  │  Training Loop   │     │
│                        │  │   (Python)       │     │
│                        │  └─────────────────┘     │
└─────────────────────────┴───────────────────────────┘
```

### 2.2 Language Strategy
- **C++**: Game logic only (existing implementation)
- **Python**: Everything else (MCTS, neural networks, training, analysis)
- **Interface**: Python bindings for C++ game engine (pybind11)

### 2.3 Neural Network Input Specification
- **20 channels total**:
  - Channel 0: Current board state
  - Channel 1: Current player indicator (0/1 plane)
  - Channels 2-9: Previous 8 moves for player 1
  - Channels 10-17: Previous 8 moves for player 2
  - Channel 18: Attack score plane
  - Channel 19: Defense score plane

---

## 3. System Architecture {#system-architecture}

### 3.1 Core System Design
```
┌────────────────────────────────────────────────────┐
│                  Python Application                 │
├────────────────────────────────────────────────────┤
│              Omoknuni Main Controller              │
├──────────────┬──────────────┬─────────────────────┤
│   Game API   │  Search API  │   Training API      │
├──────────────┴──────────────┴─────────────────────┤
│                Python Core Engine                   │
├─────────────┬──────────────┬──────────────────────┤
│  TAI-MCTS   │   PyTorch    │   Self-Play          │
│   Module    │   Networks   │   Manager            │
├─────────────┴──────────────┴──────────────────────┤
│              Python Infrastructure                  │
├─────────────┬──────────────┬──────────────────────┤
│   NumPy     │  Threading   │    Storage           │
│  Operations │    Pool      │    (HDF5)            │
└─────────────┴──────────────┴──────────────────────┘
                        │
                   ┌────▼────┐
                   │   C++   │
                   │  Game   │
                   │  Logic  │
                   └─────────┘
```

### 3.2 Data Flow Architecture
```
C++ Game State → Python Wrapper → Encoder → Neural Network → TAI-MCTS → Move
        ↑                                                         │
        └────────────────── Self-Play Loop ──────────────────────┘
```

---

## 4. Core Components & File Structure {#core-components}

### 4.1 File Organization
```
omoknuni/
├── src/
│   ├── game_wrapper/
│   │   ├── __init__.py
│   │   ├── game_binding.cpp         # pybind11 wrapper for C++ game
│   │   ├── game_interface.py        # Python interface to C++ game
│   │   └── position.py              # Position representation
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── tai_mcts.py             # TAI-MCTS implementation
│   │   ├── node.py                 # Tree node structure
│   │   ├── search_tree.py          # Tree management
│   │   └── config.py               # MCTS configuration
│   ├── tai_components/
│   │   ├── __init__.py
│   │   ├── entropy_tracker.py      # Entropy computation & tracking
│   │   ├── info_selector.py        # Information-theoretic selection
│   │   ├── active_inference.py     # Active inference components
│   │   └── thermodynamic_metrics.py # Metrics computation
│   ├── neural/
│   │   ├── __init__.py
│   │   ├── network.py              # Neural network architecture
│   │   ├── encoder.py              # Position encoder (20 channels)
│   │   ├── model_manager.py        # Model loading/saving
│   │   └── inference.py            # Batch inference optimization
│   ├── training/
│   │   ├── __init__.py
│   │   ├── self_play.py            # Self-play game generation
│   │   ├── replay_buffer.py        # Experience replay storage
│   │   ├── trainer.py              # Neural network training
│   │   ├── evaluator.py            # Model evaluation
│   │   └── distributed.py          # Distributed training support
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py               # Logging configuration
│   │   ├── profiler.py             # Performance profiling
│   │   ├── visualization.py        # Entropy/metric visualization
│   │   └── parallel.py             # Parallel computation utilities
│   └── main.py                     # Entry point
├── config/
│   ├── default.yaml                # Default configuration
│   ├── training.yaml               # Training parameters
│   └── search.yaml                 # Search parameters
├── scripts/
│   ├── train.py                    # Training script
│   ├── play.py                     # Play against AI
│   ├── analyze.py                  # Game analysis
│   └── benchmark.py                # Performance benchmarking
├── notebooks/
│   ├── entropy_analysis.ipynb      # Entropy pattern analysis
│   ├── training_progress.ipynb     # Training visualization
│   └── thermodynamic_metrics.ipynb # Thermodynamic analysis
├── tests/
│   ├── test_mcts.py                # MCTS unit tests
│   ├── test_entropy.py             # Entropy calculation tests
│   ├── test_active_inference.py    # Active inference tests
│   └── test_integration.py         # Integration tests
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
└── CMakeLists.txt                  # For building C++ game wrapper
```

### 4.2 Component Specifications

#### 4.2.1 tai_mcts.py
**Purpose**: Core TAI-MCTS search implementation
```python
class TAI_MCTS:
    """Thermodynamically-Aware Monte Carlo Tree Search"""
    
    def __init__(self, config: MCTSConfig, neural_net: NeuralNetwork):
        self.config = config
        self.neural_net = neural_net
        self.entropy_tracker = EntropyTracker()
        self.info_selector = InfoSelector(config.lambda_info)
        self.active_inference = ActiveInference(config.temperature)
        
    def search(self, root_position: Position, 
               time_limit: float) -> Tuple[np.ndarray, float]:
        """Main search method returning policy and value"""
        
    def _select_leaf(self, node: Node) -> Tuple[Node, List[Node]]:
        """Tree traversal with information gain"""
        
    def _expand_and_evaluate(self, node: Node) -> float:
        """Expand node and get neural network evaluation"""
        
    def _backup(self, path: List[Node], value: float):
        """Propagate values with entropy tracking"""
```

#### 4.2.2 entropy_tracker.py
**Purpose**: Track and compute entropy metrics
```python
class EntropyTracker:
    """Tracks configuration and policy entropy during search"""
    
    def __init__(self, use_neural_approx: bool = True):
        self.config_entropy_model = self._load_entropy_model() if use_neural_approx else None
        self.entropy_history = deque(maxlen=1000)
        self.pattern_cache = LRUCache(maxsize=10000)
        
    def compute_config_entropy(self, position: Position) -> float:
        """Calculate position complexity"""
        
    def track_trajectory(self, trajectory: List[Node]):
        """Record entropy along search path"""
        
    def compute_entropy_production(self) -> float:
        """Calculate entropy production rate σ"""
```

#### 4.2.3 info_selector.py
**Purpose**: Information-theoretic action selection
```python
class InfoSelector:
    """Selects actions based on information gain"""
    
    def __init__(self, lambda_info: float):
        self.lambda_info = lambda_info
        self.info_gain_cache = {}
        
    def select_action(self, node: Node, c_puct: float) -> int:
        """Select action maximizing UCT + information gain"""
        
    def approximate_info_gain(self, node: Node, action: int) -> float:
        """Fast approximation of information gain"""
        
    def _variance_method(self, child_values: List[float]) -> float:
        """Use value variance as uncertainty proxy"""
        
    def _visit_ratio_method(self, visits: int, parent_visits: int) -> float:
        """Use visit ratio as confidence proxy"""
```

#### 4.2.4 active_inference.py
**Purpose**: Active inference components
```python
class ActiveInference:
    """Active inference for principled exploration"""
    
    def __init__(self, temperature: float):
        self.temperature = temperature
        self.surprise_threshold = 2.0
        self.belief_tracker = BeliefTracker()
        
    def compute_free_energy(self, node: Node, action: int) -> float:
        """Compute expected free energy F̃"""
        
    def minimize_expected_free_energy(self, node: Node) -> np.ndarray:
        """Return action distribution minimizing F̃"""
        
    def update_beliefs(self, node: Node, outcome: float):
        """Update beliefs based on surprise"""
```

#### 4.2.5 network.py
**Purpose**: Neural network architecture
```python
class PolicyValueNetwork(nn.Module):
    """AlphaZero-style policy-value network"""
    
    def __init__(self, input_channels: int = 20, board_size: int = 15):
        super().__init__()
        self.input_channels = input_channels
        self.board_size = board_size
        
        # Convolutional tower
        self.conv_tower = self._make_conv_tower()
        
        # Policy head
        self.policy_head = self._make_policy_head()
        
        # Value head
        self.value_head = self._make_value_head()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value"""
```

#### 4.2.6 self_play.py
**Purpose**: Self-play game generation
```python
class SelfPlayManager:
    """Manages parallel self-play game generation"""
    
    def __init__(self, game_interface: GameInterface, 
                 num_workers: int = 8):
        self.game = game_interface
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        
    def generate_games(self, model: PolicyValueNetwork, 
                      num_games: int) -> List[GameRecord]:
        """Generate self-play games in parallel"""
        
    def _play_game(self, model: PolicyValueNetwork) -> GameRecord:
        """Play a single self-play game"""
```

---

## 5. Algorithm Specifications {#algorithms}

### 5.1 TAI-MCTS Main Algorithm

```
Algorithm: TAI_MCTS_Search
Input: root_position, time_limit, config
Output: action_probabilities, value_estimate

1. Initialize:
   - tree ← SearchTree(root_position)
   - entropy_tracker ← EntropyTracker()
   - start_time ← time.time()

2. While time.time() - start_time < time_limit:
   a. path ← []
   b. leaf ← SELECT_LEAF(tree.root, path)
   c. value ← EVALUATE_AND_EXPAND(leaf)
   d. BACKUP(path, value)
   e. entropy_tracker.track_trajectory(path)
   f. If len(path) % 10 == 0:
      - UPDATE_SELECTION_PARAMS(entropy_tracker.metrics)

3. action_probs ← EXTRACT_POLICY(tree.root, config.temperature)
4. Return (action_probs, tree.root.value_sum / tree.root.visit_count)
```

### 5.2 Information-Theoretic Selection

```
Algorithm: INFO_SELECT_ACTION
Input: node, c_puct, lambda_info, use_active_inference
Output: selected_action

1. Initialize scores ← {}
2. sqrt_total ← sqrt(node.visit_count)

3. For each action in node.legal_actions:
   a. child ← node.children.get(action)
   b. If child is None:
      - prior ← node.priors[action]
      - value_score ← 0
      - visit_count ← 0
   c. Else:
      - prior ← child.prior
      - value_score ← child.value_sum / child.visit_count
      - visit_count ← child.visit_count
   
   d. # UCT score
      exploration ← c_puct * prior * sqrt_total / (1 + visit_count)
      uct_score ← value_score + exploration
   
   e. # Information gain
      info_gain ← approximate_info_gain(node, action, child)
   
   f. # Combined score
      scores[action] ← uct_score + lambda_info * info_gain

4. If use_active_inference:
   a. free_energies ← compute_free_energies(node)
   b. For each action:
      - scores[action] -= 0.1 * free_energies[action]

5. Return argmax(scores)
```

### 5.3 Entropy Computation

```
Algorithm: COMPUTE_CONFIG_ENTROPY
Input: position, method='neural'
Output: entropy_value

1. If method == 'neural' and entropy_model exists:
   a. features ← extract_features(position)
   b. with torch.no_grad():
      - entropy ← entropy_model(features).item()
   c. Return entropy

2. Else (exact computation):
   a. patterns ← extract_patterns(position, sizes=[3, 5, 7])
   b. pattern_counts ← Counter(patterns)
   c. total ← sum(pattern_counts.values())
   d. probabilities ← [count/total for count in pattern_counts.values()]
   e. entropy ← -sum(p * log(p) for p in probabilities if p > 0)
   f. Return entropy

Algorithm: TRACK_ENTROPY_PRODUCTION
Input: trajectory (list of nodes)
Output: entropy_metrics

1. entropies ← []
2. For each node in trajectory:
   a. s_config ← compute_config_entropy(node.position)
   b. # Policy entropy from visit distribution
      visits ← [child.visit_count for child in node.children.values()]
      if sum(visits) > 0:
         probs ← visits / sum(visits)
         h_policy ← -sum(p * log(p) for p in probs if p > 0)
      else:
         h_policy ← log(len(node.legal_actions))
   c. entropies.append((s_config, h_policy))

3. # Compute entropy production
   sigma ← 0
   For i in range(1, len(entropies)):
      ds_config ← entropies[i][0] - entropies[i-1][0]
      dh_policy ← entropies[i][1] - entropies[i-1][1]
      sigma += max(0, ds_config - dh_policy)

4. Return {
      'avg_config_entropy': mean([e[0] for e in entropies]),
      'avg_policy_entropy': mean([e[1] for e in entropies]),
      'entropy_production': sigma / len(trajectory)
   }
```

### 5.4 Active Inference Integration

```
Algorithm: COMPUTE_FREE_ENERGY
Input: node, action
Output: free_energy

1. # Get child node or create virtual child
   child ← node.children.get(action)
   If child is None:
      expected_value ← node.value_sum / max(1, node.visit_count)
      expected_entropy ← log(len(node.legal_actions))
      epistemic_value ← 1.0  # Maximum uncertainty
   Else:
      expected_value ← child.value_sum / max(1, child.visit_count)
      # Entropy from child's action distribution
      visits ← [gc.visit_count for gc in child.children.values()]
      if sum(visits) > 0:
         probs ← visits / sum(visits)
         expected_entropy ← -sum(p * log(p) for p in probs if p > 0)
      else:
         expected_entropy ← log(len(child.legal_actions))
      # Epistemic value from visit count
      epistemic_value ← 1.0 / (1 + sqrt(child.visit_count))

2. # Free energy calculation
   free_energy ← -expected_value + temperature * expected_entropy - 0.1 * epistemic_value

3. Return free_energy

Algorithm: UPDATE_BELIEFS
Input: node, actual_value
Output: updated_node

1. # Compute surprise
   expected_value ← node.value_sum / max(1, node.visit_count)
   surprise ← abs(actual_value - expected_value)

2. # Update exploration bonus based on surprise
   If surprise > surprise_threshold:
      node.exploration_bonus *= 1.2
      node.info_gain_weight *= 1.1
   Else:
      node.exploration_bonus *= 0.98
      node.info_gain_weight *= 0.99

3. # Update value estimates
   node.value_sum += actual_value
   node.visit_count += 1

4. Return node
```

### 5.5 Self-Play Training Loop

```
Algorithm: SELF_PLAY_TRAINING
Input: initial_model, config
Output: trained_model

1. Initialize:
   - current_model ← initial_model
   - best_model ← initial_model
   - replay_buffer ← ReplayBuffer(max_size=1e6)
   - iteration ← 0

2. While iteration < max_iterations:
   a. # Generate self-play games
      games ← []
      with ProcessPoolExecutor(max_workers=num_workers) as executor:
         futures ← [executor.submit(play_game, current_model) 
                   for _ in range(games_per_iteration)]
         games ← [f.result() for f in futures]
   
   b. # Add to replay buffer
      for game in games:
         replay_buffer.add_game(game)
   
   c. # Train network
      for epoch in range(epochs_per_iteration):
         batch ← replay_buffer.sample(batch_size)
         loss ← train_step(current_model, batch)
   
   d. # Evaluate new model
      If iteration % eval_frequency == 0:
         win_rate ← evaluate_models(current_model, best_model, num_games=100)
         If win_rate > 0.55:
            best_model ← current_model.copy()
            save_model(best_model, f"model_iter_{iteration}.pt")
   
   e. iteration += 1

3. Return best_model

Algorithm: PLAY_SELF_PLAY_GAME
Input: model, config
Output: game_record

1. Initialize:
   - position ← GameInterface.initial_position()
   - game_record ← []
   - move_count ← 0

2. While not GameInterface.is_terminal(position):
   a. # Run MCTS
      mcts ← TAI_MCTS(config, model)
      action_probs, value ← mcts.search(position, time_limit=5.0)
   
   b. # Temperature-based action selection
      If move_count < 30:
         temperature ← 1.0
      Else:
         temperature ← 0.1
      
      If temperature == 0:
         action ← argmax(action_probs)
      Else:
         adjusted_probs ← action_probs ** (1/temperature)
         adjusted_probs /= sum(adjusted_probs)
         action ← random.choice(actions, p=adjusted_probs)
   
   c. # Record position
      encoded_position ← encode_position(position)
      game_record.append({
         'position': encoded_position,
         'action_probs': action_probs,
         'value': value
      })
   
   d. # Make move
      position ← GameInterface.make_move(position, action)
      move_count += 1

3. # Get final outcome
   outcome ← GameInterface.get_outcome(position)
   
4. # Assign rewards
   For i, record in enumerate(game_record):
      If i % 2 == 0:  # Player 1's move
         record['outcome'] = outcome
      Else:  # Player 2's move
         record['outcome'] = -outcome

5. Return game_record
```

---

## 6. Application Flowchart {#flowchart}

```mermaid
graph TB
    Start([Start Omoknuni]) --> Init[Initialize Python Environment]
    Init --> LoadConfig[Load YAML Configuration]
    LoadConfig --> LoadGame[Load C++ Game Module<br/>via pybind11]
    LoadGame --> LoadModel[Load PyTorch Model]
    
    LoadModel --> MainLoop{Main Mode?}
    
    MainLoop -->|Play| PlayMode[Interactive Play Mode]
    MainLoop -->|Train| TrainMode[Self-Play Training]
    MainLoop -->|Analyze| AnalyzeMode[Game Analysis]
    MainLoop -->|Benchmark| BenchMode[Performance Testing]
    
    %% Play Mode Flow
    PlayMode --> GameInit[Initialize Game Position]
    GameInit --> GameLoop{Game Over?}
    GameLoop -->|No| GetPosition[Get Current Position<br/>from C++ Engine]
    GetPosition --> EncodePosition[Encode Position<br/>20 channels NumPy]
    EncodePosition --> RunMCTS[Run TAI-MCTS<br/>(Python)]
    
    RunMCTS --> TreeSearch[Tree Search Loop]
    TreeSearch --> SelectLeaf[Select Leaf Node<br/>Info Gain + UCT]
    SelectLeaf --> NeuralEval[PyTorch Neural<br/>Evaluation]
    NeuralEval --> Expand[Expand Node<br/>in Python Tree]
    Expand --> Backup[Backup Values]
    Backup --> TrackEntropy[Track Entropy<br/>Metrics]
    TrackEntropy --> CheckTime{Time Limit<br/>Reached?}
    CheckTime -->|No| TreeSearch
    CheckTime -->|Yes| ExtractPolicy[Extract Policy<br/>Distribution]
    
    ExtractPolicy --> SelectMove[Select Move<br/>from Policy]
    SelectMove --> SendMove[Send Move to<br/>C++ Engine]
    SendMove --> UpdateGame[C++ Updates<br/>Game State]
    UpdateGame --> GameLoop
    
    GameLoop -->|Yes| ShowResult[Display Result]
    ShowResult --> End([End])
    
    %% Train Mode Flow
    TrainMode --> InitTraining[Initialize Training<br/>Components]
    InitTraining --> TrainLoop{Training<br/>Complete?}
    TrainLoop -->|No| ParallelGames[Generate Self-Play<br/>Games in Parallel]
    ParallelGames --> CollectData[Collect Training<br/>Data to Buffer]
    CollectData --> SampleBatch[Sample Training<br/>Batch]
    SampleBatch --> TrainNetwork[Train PyTorch<br/>Network]
    TrainNetwork --> CheckEval{Time to<br/>Evaluate?}
    CheckEval -->|Yes| EvaluateModel[Play vs Previous<br/>Best Model]
    CheckEval -->|No| ContinueTrain
    EvaluateModel --> UpdateBest{Win Rate<br/>> 55%?}
    UpdateBest -->|Yes| SaveModel[Save New<br/>Best Model]
    UpdateBest -->|No| ContinueTrain[Continue Training]
    SaveModel --> ContinueTrain
    ContinueTrain --> TrainLoop
    TrainLoop -->|Yes| End
    
    %% Analyze Mode Flow
    AnalyzeMode --> LoadGameFile[Load Game<br/>Record]
    LoadGameFile --> ReplayGame[Replay Each<br/>Position]
    ReplayGame --> AnalyzePos[Run TAI-MCTS<br/>Analysis]
    AnalyzePos --> ShowMetrics[Display Entropy<br/>& Thermodynamic<br/>Metrics]
    ShowMetrics --> PlotGraphs[Plot Entropy<br/>Evolution]
    PlotGraphs --> End
    
    %% Benchmark Mode
    BenchMode --> SetupBench[Setup Benchmark<br/>Positions]
    SetupBench --> RunBench[Run TAI-MCTS<br/>vs Baseline]
    RunBench --> CollectMetrics[Collect Performance<br/>Metrics]
    CollectMetrics --> GenerateReport[Generate Benchmark<br/>Report]
    GenerateReport --> End
    
    %% TAI-MCTS Components
    subgraph Python TAI-MCTS
        InfoGain[Information<br/>Gain Calc]
        EntropyCalc[Entropy<br/>Computation]
        ActiveInf[Active<br/>Inference]
        FreeEnergy[Free Energy<br/>Minimization]
    end
    
    SelectLeaf -.-> InfoGain
    SelectLeaf -.-> EntropyCalc
    SelectLeaf -.-> ActiveInf
    ActiveInf -.-> FreeEnergy
    
    %% Data Storage
    subgraph Storage
        ReplayBuf[(Replay Buffer<br/>HDF5)]
        ModelStore[(Model<br/>Checkpoints)]
        GameLogs[(Game<br/>Records)]
    end
    
    CollectData --> ReplayBuf
    SaveModel --> ModelStore
    ShowResult --> GameLogs
```

---

## 7. Technology Stack {#tech-stack}

### 7.1 Core Technologies

#### Programming Languages
- **Python 3.9+**: All AI components
  - Type hints for code clarity
  - Async/await for concurrent operations
  - Dataclasses for configuration
- **C++**: Game logic only (existing)
  - No optimization required
  - Simple pybind11 interface

#### Python Libraries
- **PyTorch 2.0+**: Neural networks
  - TorchScript for production inference
  - Mixed precision training
  - Distributed data parallel
- **NumPy**: Array operations
- **pybind11**: C++ bindings
- **Ray**: Distributed self-play (optional)
- **Weights & Biases**: Experiment tracking
- **HDF5/h5py**: Replay buffer storage
- **PyYAML**: Configuration
- **tqdm**: Progress bars
- **matplotlib/seaborn**: Visualization
- **Jupyter**: Interactive analysis

#### Development Tools
- **Poetry**: Dependency management
- **Black**: Code formatting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **pytest-benchmark**: Performance testing
- **Sphinx**: Documentation
- **pre-commit**: Git hooks

### 7.2 Performance Considerations

#### Python Optimization Strategies
- **Numba**: JIT compilation for hot loops
- **Cython**: Optional for critical paths
- **Vectorization**: NumPy operations
- **Multiprocessing**: Parallel self-play
- **Caching**: LRU cache for patterns
- **Profile-guided optimization**: cProfile/line_profiler

#### Memory Management
- **Object pooling**: Reuse node objects
- **Weak references**: For tree cleanup
- **Memory mapping**: Large replay buffers
- **Garbage collection tuning**: For latency

---

## 8. Development Phases {#phases}

### Phase 1: Foundation & Integration (Weeks 1-3)
**Goal**: Set up project and integrate C++ game engine

#### Tasks:
1. **Project Setup**
   - [ ] Initialize Python project with Poetry
   - [ ] Configure development environment
   - [ ] Set up testing framework
   - [ ] Create project documentation

2. **C++ Game Integration**
   - [ ] Create pybind11 wrapper for game engine
   - [ ] Define Python game interface
   - [ ] Implement position encoder (20 channels)
   - [ ] Test game state synchronization

3. **Basic MCTS**
   - [ ] Implement tree node structure
   - [ ] Create basic MCTS with UCT
   - [ ] Add simple neural network stub
   - [ ] Benchmark baseline performance

4. **Infrastructure**
   - [ ] Set up logging system
   - [ ] Create configuration management
   - [ ] Implement basic profiling
   - [ ] Add unit tests for core components

**Deliverables**: Working Python MCTS connected to C++ game

### Phase 2: TAI Components (Weeks 4-7)
**Goal**: Implement thermodynamic enhancements

#### Tasks:
1. **Entropy System**
   - [ ] Implement pattern extraction
   - [ ] Create configuration entropy calculator
   - [ ] Build entropy tracking system
   - [ ] Train neural entropy approximator

2. **Information Selection**
   - [ ] Implement info gain computation
   - [ ] Add variance-based approximation
   - [ ] Create visit ratio heuristics
   - [ ] Integrate with MCTS selection

3. **Measurement Tools**
   - [ ] Build entropy visualization
   - [ ] Create metric dashboards
   - [ ] Implement trajectory analysis
   - [ ] Add real-time monitoring

4. **Optimization**
   - [ ] Profile and identify bottlenecks
   - [ ] Add caching for patterns
   - [ ] Optimize NumPy operations
   - [ ] Implement parallel evaluation

**Deliverables**: TAI-MCTS with measurable efficiency gains

### Phase 3: Active Inference (Weeks 8-10)
**Goal**: Add active inference components

#### Tasks:
1. **Free Energy Framework**
   - [ ] Implement free energy computation
   - [ ] Add temperature control
   - [ ] Create belief tracking
   - [ ] Build surprise detection

2. **Integration**
   - [ ] Combine with info selection
   - [ ] Tune hyperparameters
   - [ ] Add adaptive exploration
   - [ ] Test stability

3. **Validation**
   - [ ] Measure thermodynamic quantities
   - [ ] Compare selection distributions
   - [ ] Run ablation studies
   - [ ] Document findings

**Deliverables**: Complete TAI-MCTS implementation

### Phase 4: Neural Network & Training (Weeks 11-14)
**Goal**: Implement full training pipeline

#### Tasks:
1. **Neural Architecture**
   - [ ] Design network architecture
   - [ ] Implement in PyTorch
   - [ ] Add regularization
   - [ ] Test inference speed

2. **Self-Play System**
   - [ ] Create game generator
   - [ ] Implement replay buffer
   - [ ] Add distributed workers
   - [ ] Build data pipeline

3. **Training Loop**
   - [ ] Implement training script
   - [ ] Add model evaluation
   - [ ] Create checkpointing
   - [ ] Set up tensorboard

4. **Evaluation**
   - [ ] Build tournament system
   - [ ] Track ELO ratings
   - [ ] Create analysis tools
   - [ ] Generate reports

**Deliverables**: Working training pipeline

### Phase 5: Analysis & Optimization (Weeks 15-16)
**Goal**: Analyze results and optimize performance

#### Tasks:
1. **Performance Analysis**
   - [ ] Profile full system
   - [ ] Identify bottlenecks
   - [ ] Optimize critical paths
   - [ ] Reduce memory usage

2. **Scientific Analysis**
   - [ ] Analyze entropy patterns
   - [ ] Validate thermodynamic metrics
   - [ ] Create publication figures
   - [ ] Write technical report

3. **Tools & Documentation**
   - [ ] Create analysis notebooks
   - [ ] Build visualization tools
   - [ ] Write user documentation
   - [ ] Prepare demo

**Deliverables**: Optimized system with full analysis

---

## 9. Claude Development Guidelines {#claude-guidelines}

### 9.1 General Python Guidelines

```
You are developing Omoknuni, a Python-based game AI that uses a C++ game engine.
The system combines AlphaZero-style self-play with Thermodynamic & Active 
Inference (TAI) principles.

Key principles:
1. Write clean, typed Python code with clear docstrings
2. Use NumPy for numerical operations, avoid loops where possible
3. Implement comprehensive error handling and logging
4. Profile before optimizing - correctness first
5. Use descriptive variable names and add type hints
6. Write unit tests for all new functionality
7. Follow PEP 8 style guidelines

Architecture notes:
- The C++ game engine is accessed via pybind11 bindings
- All AI logic is in Python (MCTS, neural networks, training)
- Use PyTorch for neural networks
- Prefer composition over inheritance for components
```

### 9.2 TAI-MCTS Implementation Guidelines

```
When implementing TAI-MCTS components:

1. Entropy Tracking:
   - Cache pattern computations to avoid recalculation
   - Use incremental updates where possible
   - Ensure entropy values are non-negative
   - Default to neural approximation for speed
   - Fall back to exact computation for analysis

2. Information Selection:
   - Balance exploration vs exploitation with tunable λ
   - Start with simple variance-based approximation
   - Monitor selection diversity
   - Log selection statistics for analysis
   - Avoid expensive computations in the hot path

3. Active Inference:
   - Keep free energy computation simple initially
   - Make temperature easily adjustable
   - Track surprise for adaptive exploration
   - Ensure numerical stability (avoid log(0))
   - Document the theoretical mapping clearly

4. Performance:
   - Profile with cProfile before optimizing
   - Use NumPy vectorization over loops
   - Consider numba for critical functions only
   - Implement caching for repeated computations
   - Monitor memory usage during search
```

### 9.3 Code Style Examples

```python
# Good: Clear, documented, typed
class EntropyTracker:
    """Tracks configuration and policy entropy during MCTS.
    
    This class maintains a history of entropy measurements and
    computes thermodynamic metrics like entropy production rate.
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 use_neural: bool = True) -> None:
        """Initialize entropy tracker.
        
        Args:
            window_size: Size of history window for metrics
            use_neural: Whether to use neural approximation
        """
        self.window_size = window_size
        self.use_neural = use_neural
        self.history: deque[EntropyMeasurement] = deque(maxlen=window_size)
        self._pattern_cache: Dict[str, float] = {}
        
    def compute_config_entropy(self, 
                             position: Position) -> float:
        """Compute configuration entropy of a position.
        
        Args:
            position: Game position to analyze
            
        Returns:
            Configuration entropy in bits
        """
        # Implementation here
        pass

# Avoid: Unclear, untyped, undocumented
class ET:
    def __init__(self, ws=1000, un=True):
        self.ws = ws
        self.un = un
        self.h = deque(maxlen=ws)
        self._pc = {}
```

### 9.4 Testing Guidelines

```
When writing tests:

1. Test both correctness and performance
2. Use pytest fixtures for common setups
3. Mock the C++ game engine for unit tests
4. Test edge cases (empty positions, terminal states)
5. Verify thermodynamic quantities are physical
6. Include integration tests for full search
7. Add regression tests for bug fixes

Example test structure:
- test_mcts.py: Core MCTS functionality
- test_entropy.py: Entropy computation correctness
- test_info_selection.py: Information gain calculations
- test_active_inference.py: Free energy minimization
- test_integration.py: Full system tests
```

---

## 10. Scope Definition {#scope}

### 10.1 In Scope

#### Core Features
- ✓ Python TAI-MCTS implementation
- ✓ PyTorch neural network (policy + value)
- ✓ Self-play training pipeline
- ✓ 20-channel position encoding
- ✓ Configuration and policy entropy tracking
- ✓ Information-theoretic action selection
- ✓ Active inference with free energy
- ✓ Python-C++ game interface via pybind11
- ✓ Performance profiling tools
- ✓ Analysis and visualization notebooks

#### Infrastructure
- ✓ Poetry-based dependency management
- ✓ Comprehensive test suite
- ✓ Benchmark framework
- ✓ Logging and monitoring
- ✓ Configuration via YAML
- ✓ Replay buffer with HDF5
- ✓ Model checkpointing
- ✓ Distributed self-play support

#### Documentation
- ✓ API documentation
- ✓ Algorithm explanations
- ✓ Jupyter notebooks for analysis
- ✓ Usage examples
- ✓ Training guides

### 10.2 Out of Scope

#### Not Included
- ✗ C++ optimization (using existing implementation as-is)
- ✗ GUI/graphical interface (command line only)
- ✗ Web deployment (local execution only)
- ✗ Mobile optimization
- ✗ Alternative games (Omok-specific)
- ✗ Real-time online play
- ✗ Production deployment infrastructure
- ✗ Advanced parallelization (beyond multiprocessing)

#### Future Considerations
- ✗ Rust implementation for performance
- ✗ GPU-accelerated MCTS
- ✗ Quantum-inspired algorithms
- ✗ Neuromorphic hardware support
- ✗ Multi-agent variants
- ✗ Transfer learning to other games

### 10.3 Success Criteria

#### Performance Metrics
- Search speed: ≥400 simulations/second (Python)
- Memory usage: <2GB during search
- Training convergence: <100,000 games
- Efficiency gain: 30-50% over baseline

#### Quality Metrics
- Test coverage: >85%
- Type coverage: 100% of public APIs
- Documentation: All public functions
- Reproducible results

---

## Appendices

### A. Configuration Schema
```yaml
# config/default.yaml
search:
  simulations_per_move: 800
  time_limit: 5.0
  cpuct: 1.0
  temperature: 1.0
  
tai_mcts:
  lambda_info: 0.3
  lambda_entropy: 0.1
  use_active_inference: true
  exploration_temperature: 0.5
  use_neural_entropy: true
  
neural_network:
  model_path: "models/latest.pt"
  input_channels: 20
  board_size: 15
  batch_size: 8
  device: "cuda"  # or "cpu"
  
training:
  games_per_iteration: 1000
  epochs_per_iteration: 10
  replay_buffer_size: 100000
  batch_size: 2048
  learning_rate: 0.001
  weight_decay: 0.0001
  
self_play:
  num_workers: 8
  games_per_worker: 125
  reuse_mcts_tree: true
  add_noise: true
  noise_epsilon: 0.25
```

### B. Expected Performance
On reference hardware (Intel i7-10700K, RTX 3070, 32GB RAM):
- Baseline MCTS: 400 sims/sec (Python)
- TAI-MCTS (MVP): 350 sims/sec with 25% fewer sims needed
- TAI-MCTS (Full): 300 sims/sec with 40% fewer sims needed
- Neural network batch inference: 20ms (batch_size=8)
- Self-play generation: 100 games/hour/worker

### C. Development Milestones
1. **M1**: C++ game integrated with Python
2. **M2**: Baseline MCTS working
3. **M3**: Entropy tracking functional
4. **M4**: Information selection integrated
5. **M5**: Active inference complete
6. **M6**: Self-play training working
7. **M7**: Full system optimized