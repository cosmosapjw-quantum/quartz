# Dynamic Prior Revision under Bounded Deliberation
## Self-Contained Experimental Design Document

Version: draft v1  
Scope: AlphaZero-style board-game AI / prior refresh ablation redesign  
Primary aim: 구조 식별(structure identification)과 최소 검증(minimal verification)을 먼저 수행하고, survivor만 training contract 검증으로 올린다.

Implementation note:

- The old phase-1 runner has been retired.
- The repository now uses the phase-1.5 clean-split runner
  `scripts/phase15_ablation_study.py`.
- This plan remains useful as historical context for why the prior-revision
  framing failed, but the active in-tree experiment contract is now the
  substrate/controller/refresh split from `phase15_strategy_revision_v2.md`.

---

## 1. 문제 설정

현재 prior refresh 실험 결과는 적어도 현 구현 형태에선 기대보다 약하거나 부정적이다. 그러나 이 결과만으로 “탐색 중 prior가 동적으로 갱신될 수 있다”는 아이디어 자체를 폐기하면 안 된다. 현재 구현은 실제로는 `prior` 슬롯에 search-derived quantity를 다시 섞는 쪽에 가까우며, 그 실패는 **belief revision 패러다임의 실패**가 아니라 **refresh operator 정의의 실패**일 수 있다.

이 문서는 이 점을 분리하기 위해 설계되었다. 핵심 질문은 다음과 같다.

> 탐색 중의 동적 신념 갱신은, 정적 prior를 덮어쓰는 heuristic이 아니라, bounded deliberation 하에서 실제로 유의미한 belief revision architecture로 작동하는가?

이 문서는 두 개의 새 아키텍처를 중심으로 설계한다.

1. **Dual-Channel Refresh (N1)**  
   neural prior는 기저 신념으로 고정하고, 탐색 중 생긴 새 증거는 별도의 posterior 채널로 유지한다.

2. **Root-Only Posterior Snapshot (N2)**  
   belief revision의 효과를 non-root 전체로 퍼뜨리지 않고, root에서의 policy improvement로 제한한다.

이 두 구조는 각각 기존 MCTS 문헌의 다른 직관과 정렬된다.

- AlphaZero류는 search를 root policy-improvement operator로 이해한다.
- Gumbel AlphaZero는 특히 적은 simulation에서 root policy-improvement를 더 principled하게 구성한다.
- implicit minimax backup 계열은 heuristic 정보와 empirical search 정보를 separate channels로 유지하는 것이 유리할 수 있음을 보여준다.

---

## 2. 목표

이 문서의 목표는 아래 네 가지다.

1. 잘못된 prior에 대해 실제 correction이 일어나는지 본다.
2. 좋은 prior에 대해 do-no-harm가 성립하는지 본다.
3. bounded deliberation의 이득이 root decision quality에 집중되는지 본다.
4. deploy-time search improvement가 self-play training contract와도 양립하는지 본다.

즉 단순 승률 실험이 아니라, 아래 네 축을 동시에 본다.

- wrong-prior correction
- do-no-harm
- few-simulation root improvement
- train–deploy consistency

---

## 3. 중심 가설

### H1. Dynamic correction hypothesis
탐색 중 생긴 search evidence는 정적 prior를 덮어쓰지 않고도 별도의 posterior 채널을 통해 더 나은 선택으로 연결될 수 있다.

### H2. Do-no-harm hypothesis
좋은 prior에 대해선 belief revision이 sparse해야 하며, 어려운 포지션이나 늦게 드러나는 증거가 있을 때만 revision이 일어나야 한다.

### H3. Root concentration hypothesis
bounded deliberation의 실제 이득은 트리 전체의 동적 재배치보다, root에서의 policy improvement에 더 강하게 나타난다.

### H4. Contract hypothesis
좋은 deploy-time 구조가 반드시 좋은 training-time target 구조는 아니다. 따라서 inference gain과 train–deploy consistency는 분리해서 봐야 한다.

---

## 4. 비교 시스템

모든 실험에서 비교군은 아래 네 개로 고정한다.

- **B0**: no-refresh baseline
- **B1**: current refresh
- **N1**: dual-channel refresh
- **N2**: root-only posterior snapshot

이 네 시스템을 고정해야 “현재 refresh가 왜 실패하는가”와 “새 구조가 실제로 baseline을 넘는가”를 동시에 판정할 수 있다.

---

## 5. 핵심 정의

### 5.1 정책 관련 객체

- `prior_base`  
  expansion 시점 neural network가 제공한 원래 prior. immutable로 취급한다.

- `posterior_search`  
  탐색 과정에서 accumulated evidence로 형성되는 별도 posterior 채널.

- `effective_policy`  
  selection 또는 최종 move choice에 실제로 쓰이는 정책.

- `oracle_policy`  
  exact solver, deep-budget search, 또는 strongest available reference가 주는 참조 정책.

### 5.2 사건(event) 정의

- `revision`  
  `argmax(effective_policy)`가 `argmax(prior_base)`와 달라지는 사건.

- `wrong-prior`  
  `argmax(prior_base)`가 oracle best move와 다름.

- `easy-case`  
  `argmax(prior_base)`가 oracle best move와 일치하고 confidence도 충분함.

- `late-evidence`  
  작은 budget에서는 A가 좋아 보이지만, 더 깊은 budget 또는 oracle에선 B가 옳은 포지션.

- `root-conflict`  
  갈등의 대부분이 root candidate ranking에서 발생하는 포지션.

- `deep-conflict`  
  root보다는 deeper branch에서 tactical divergence가 생기는 포지션.

---

## 6. 공통 평가 규약

### 6.1 평가 층위

평가는 두 층으로 분리한다.

1. **Frozen-checkpoint assay**  
   구조 자체를 먼저 본다.  
   training noise를 최대한 배제한다.

2. **Self-play training assay**  
   구조가 self-play target / deploy search와 어떤 계약을 이루는지 본다.

### 6.2 체크포인트 구성

최소 3개 checkpoint를 사용한다.

- weak
- mid
- strong

이유는 단순하다. belief revision 구조가 정말 의미가 있다면, prior 품질이 낮은 weak/mid regime에서 더 큰 이득이 나타나야 한다.

Implementation constraint:

- do not substitute lexical directory order for weak/mid/strong semantics
- repository runners should use explicit curated checkpoint paths for this layer

### 6.3 예산(budget) 스윕

Frozen-checkpoint assay 권장 budget:

- 8
- 16
- 32
- 64

필요시 `4`를 추가한다. 특히 root-only 구조는 low-budget에서 효과가 집중될 가능성이 크다.

### 6.4 공통 1차 지표

- root best-move accuracy
- top-k recall
- wrong-prior correction rate
- easy-case regret
- candidate undercoverage
- KL / cross-entropy to oracle policy

### 6.5 공통 2차 지표

- revision rate
- revision timing
- oscillation count
- entropy change
- gain per simulation

### 6.6 공통 로그 필드

아래 로그는 모든 실험에서 공통으로 남긴다.

- `checkpoint_id`
- `position_id`
- `position_bucket`
- `budget`
- `system`
- `argmax_prior`
- `argmax_effective`
- `oracle_best`
- `revision_occurred`
- `revision_step`
- `num_revisions`
- `posterior_entropy_t`
- `kl_prior_to_effective_t`
- `candidate_set_contains_oracle_best`
- `topk_recall`
- `compute_time_ms`

---

## 7. 포지션 셋 구성 원칙

포지션 셋은 반드시 bucketized suite로 구성한다.

### 7.1 기본 bucket

- `wrong_top1`
- `wrong_confident`
- `wrong_top2swap`
- `easy_good_prior`
- `ambiguous`
- `late_evidence`
- `root_conflict`
- `deep_conflict`

### 7.2 bucket 생성 원칙

- 가능하면 exact solver 또는 deep-budget no-refresh를 oracle로 사용한다.
- 자동 생성보다 deep-search mining 기반 분류를 우선한다.
- 한 bucket당 최소 150개, 가능하면 200개 이상을 권장한다.

---

## 8. 아키텍처 정의

## 8.1 Dual-Channel Refresh (N1)

핵심 원칙은 `prior_base`를 절대 덮어쓰지 않는 것이다.

### 개념

- `prior_base`는 expansion 시점 네트워크가 준 기저 신념이다.
- `posterior_search`는 탐색 중 생긴 evidence를 담는다.
- selection은 `effective_policy = combine(prior_base, posterior_search)`를 사용한다.
- `prior_base`는 끝까지 immutable이다.

### pseudocode

```python
class DualChannelNode:
    def __init__(self, prior_base):
        self.prior_base = normalize(prior_base)
        self.posterior_search = None
        self.visit_counts = zeros_like(prior_base)
        self.q_values = zeros_like(prior_base)


def infer_search_posterior(prior_base, visit_counts, q_values, evidence):
    # heuristic placeholder: exact form is intentionally left minimal
    posterior = combine_evidence(prior_base, visit_counts, q_values, evidence)
    return normalize(posterior)


def update_dual_channel(node, evidence):
    node.posterior_search = infer_search_posterior(
        prior_base=node.prior_base,
        visit_counts=node.visit_counts,
        q_values=node.q_values,
        evidence=evidence,
    )


def effective_policy_dual(node, gate):
    if node.posterior_search is None:
        return node.prior_base
    return normalize((1.0 - gate) * node.prior_base + gate * node.posterior_search)


def select_action_dual(node, c_puct, gate):
    p_eff = effective_policy_dual(node, gate)
    return argmax_over_actions(
        q=node.q_values,
        u=exploration_bonus(p_eff, node.visit_counts, c_puct),
    )
```

### 핵심 검증 포인트

- wrong-prior correction이 실제로 되는가
- easy-case에서 unnecessary intervention이 낮은가
- oscillation이 current refresh보다 낮은가

---

## 8.2 Root-Only Posterior Snapshot (N2)

핵심 원칙은 belief revision을 root에서만 정의하는 것이다.

### 개념

- non-root에서는 baseline search semantics를 유지한다.
- root에서만 challenger set을 추출한다.
- root posterior snapshot을 계산한다.
- 최종 move selection 또는 self-play target에서 이 snapshot을 사용한다.

### pseudocode

```python
class RootPosteriorController:
    def __init__(self, challenger_k):
        self.challenger_k = challenger_k


def build_root_challenger_set(prior_base, q_estimates, budget, challenger_k):
    logits = score_root_candidates(prior_base, q_estimates)
    return top_k(logits, k=min(challenger_k, adaptive_k(budget)))


def compute_root_posterior(prior_base, root_stats, challenger_set):
    posterior = restricted_policy_improvement(
        prior_base=prior_base,
        q_values=root_stats.q_values,
        visit_counts=root_stats.visit_counts,
        candidate_set=challenger_set,
    )
    return normalize(posterior)


def run_root_only_search(root_state, evaluator, budget, challenger_k):
    root_prior = evaluator.prior(root_state)
    root_stats = run_standard_tree_search_without_nonroot_refresh(
        root_state=root_state,
        evaluator=evaluator,
        budget=budget,
    )
    challenger_set = build_root_challenger_set(
        prior_base=root_prior,
        q_estimates=root_stats.q_values,
        budget=budget,
        challenger_k=challenger_k,
    )
    root_posterior = compute_root_posterior(root_prior, root_stats, challenger_set)
    return root_posterior, challenger_set, root_stats
```

### 핵심 검증 포인트

- low-budget root best-move accuracy 향상
- candidate undercoverage 억제
- train–deploy target consistency 여부

---

# Part I. Frozen-Checkpoint Structure Assays

## 9. 실험 1 — Dual-Channel Wrong-Prior Correction Assay

### 9.1 질문

초기 prior가 체계적으로 틀렸을 때, dual-channel 구조는 prior 자체를 훼손하지 않으면서 더 나은 belief revision을 만들 수 있는가?

### 9.2 가설

- N1은 B0와 B1보다 wrong-prior correction rate가 높아야 한다.
- 이 이득은 weak/mid checkpoint에서 더 커야 한다.
- very-low / low budget에서 특히 커야 한다.

### 9.3 포지션 셋

- `wrong_top1`
- `wrong_confident`
- `wrong_top2swap`

### 9.4 조작

두 종류의 prior corruption만 사용한다.

- `swap_top12`
- `inflate_wrong_confidence`

### 9.5 비교군

- B0
- B1
- N1
- N2

### 9.6 1차 지표

- wrong-prior correction rate
- oracle best-move recovery
- top-3 recall
- KL / CE to oracle policy

### 9.7 2차 지표

- false revision rate
- revision latency
- overcorrection rate
- posterior entropy drop

### 9.8 kill criterion

- biased-prior regime에서도 N1이 B0를 못 이기면 종료
- false revision rate가 baseline보다 의미 있게 높으면 종료

### 9.9 pseudocode

```python
def experiment1_wrong_prior_correction(checkpoints, position_set, budgets, corruptions, systems):
    rows = []
    for ckpt in checkpoints:
        evaluator = load_frozen_evaluator(ckpt)
        for pos in position_set:
            oracle = compute_oracle_policy(pos)
            for corruption in corruptions:
                prior0 = evaluator.prior(pos)
                prior_corrupted = apply_prior_corruption(prior0, corruption)
                for budget in budgets:
                    for system in systems:
                        out = run_search(
                            position=pos,
                            evaluator=evaluator,
                            injected_prior=prior_corrupted,
                            budget=budget,
                            system=system,
                        )
                        rows.append({
                            "experiment": "E1",
                            "checkpoint_id": ckpt.id,
                            "position_id": pos.id,
                            "position_bucket": pos.bucket,
                            "corruption": corruption,
                            "budget": budget,
                            "system": system.name,
                            "argmax_prior": argmax(prior_corrupted),
                            "argmax_effective": argmax(out.final_policy),
                            "oracle_best": oracle.argmax(),
                            "revision_occurred": argmax(prior_corrupted) != argmax(out.final_policy),
                            "revision_step": out.revision_step,
                            "num_revisions": out.num_revisions,
                            "correction_success": int(
                                argmax(prior_corrupted) != oracle.argmax()
                                and argmax(out.final_policy) == oracle.argmax()
                            ),
                            "false_revision": int(
                                argmax(prior_corrupted) == oracle.argmax()
                                and argmax(out.final_policy) != oracle.argmax()
                            ),
                            "topk_recall": topk_recall(out.final_policy, oracle),
                            "kl_to_oracle": kl_divergence(out.final_policy, oracle),
                        })
    return rows
```

---

## 10. 실험 2 — Dual-Channel Do-No-Harm + Late-Evidence Assay

### 10.1 질문

prior가 이미 좋은 경우엔 개입을 억제하고, evidence가 늦게 드러나는 경우엔 필요한 시점에만 revision을 일으킬 수 있는가?

### 10.2 가설

- N1은 `easy_good_prior`에서 revision rate가 낮아야 한다.
- N1은 `late_evidence`에서 필요한 시점에만 뒤집혀야 한다.
- N1은 B1보다 oscillation이 낮아야 한다.

### 10.3 포지션 셋

- `easy_good_prior`
- `ambiguous`
- `late_evidence`

### 10.4 비교군

- B0
- B1
- N1
- optional: N2 as reference only

### 10.5 1차 지표

- easy-case regret
- unnecessary intervention rate
- late-evidence recovery
- oscillation count

### 10.6 2차 지표

- revision step distribution
- entropy slope
- posterior stability
- confidence overshoot

### 10.7 kill criterion

- easy-case regret가 의미 있게 증가하면 실패
- oscillation이 B0보다 높아지면 실패

### 10.8 pseudocode

```python
def experiment2_do_no_harm(checkpoints, bucketed_positions, budgets, systems):
    rows = []
    for ckpt in checkpoints:
        evaluator = load_frozen_evaluator(ckpt)
        for bucket_name, positions in bucketed_positions.items():
            for pos in positions:
                oracle = compute_oracle_policy(pos)
                prior0 = evaluator.prior(pos)
                for budget in budgets:
                    for system in systems:
                        out = run_search(
                            position=pos,
                            evaluator=evaluator,
                            injected_prior=prior0,
                            budget=budget,
                            system=system,
                        )
                        rows.append({
                            "experiment": "E2",
                            "checkpoint_id": ckpt.id,
                            "position_id": pos.id,
                            "position_bucket": bucket_name,
                            "budget": budget,
                            "system": system.name,
                            "argmax_prior": argmax(prior0),
                            "argmax_effective": argmax(out.final_policy),
                            "oracle_best": oracle.argmax(),
                            "regret_easy": int(
                                bucket_name == "easy_good_prior"
                                and argmax(out.final_policy) != oracle.argmax()
                            ),
                            "late_recovery": int(
                                bucket_name == "late_evidence"
                                and argmax(out.final_policy) == oracle.argmax()
                            ),
                            "revision_occurred": argmax(prior0) != argmax(out.final_policy),
                            "revision_step": out.revision_step,
                            "num_revisions": out.num_revisions,
                            "oscillation_count": out.oscillation_count,
                            "entropy_final": entropy(out.final_policy),
                        })
    return rows
```

---

## 11. 실험 3 — Root-Only Posterior Snapshot Few-Sim Assay

### 11.1 질문

bounded deliberation의 주된 이득은 트리 전체를 동적으로 재배치하는 데서가 아니라, root decision quality를 개선하는 데서 오는가?

### 11.2 가설

- N2는 low-budget에서 B0와 B1보다 root best-move accuracy가 높아야 한다.
- 이득은 medium budget 이상에서 줄어들어도 괜찮다.
- candidate undercoverage가 낮아야 한다.

### 11.3 포지션 셋

- `root_conflict`
- `generic`
- `shallow_trap`

### 11.4 비교군

- B0
- B1
- N1
- N2

### 11.5 1차 지표

- root best-move accuracy
- top-k recall
- challenger recall
- gain per simulation

### 11.6 2차 지표

- root entropy reduction
- rank swap count
- candidate undercoverage
- compute-normalized advantage

### 11.7 kill criterion

- low-budget에서도 N2가 B0를 못 이기면 실패
- candidate undercoverage가 의미 있게 높으면 실패

### 11.8 pseudocode

```python
def experiment3_root_only_few_sim(checkpoints, positions, budgets, systems):
    rows = []
    for ckpt in checkpoints:
        evaluator = load_frozen_evaluator(ckpt)
        for pos in positions:
            oracle = compute_oracle_policy(pos)
            prior0 = evaluator.prior(pos)
            for budget in budgets:
                for system in systems:
                    out = run_search(
                        position=pos,
                        evaluator=evaluator,
                        injected_prior=prior0,
                        budget=budget,
                        system=system,
                    )
                    rows.append({
                        "experiment": "E3",
                        "checkpoint_id": ckpt.id,
                        "position_id": pos.id,
                        "position_bucket": pos.bucket,
                        "budget": budget,
                        "system": system.name,
                        "argmax_prior": argmax(prior0),
                        "argmax_effective": argmax(out.final_policy),
                        "oracle_best": oracle.argmax(),
                        "root_accuracy": int(argmax(out.final_policy) == oracle.argmax()),
                        "topk_recall": topk_recall(out.final_policy, oracle),
                        "candidate_set_contains_oracle_best": int(
                            oracle.argmax() in out.root_candidate_set
                        ),
                        "rank_swap_count": out.rank_swap_count,
                        "gain_per_sim": out.search_gain / max(1, budget),
                    })
    return rows
```

---

## 12. 실험 4 — Root-Only Posterior Train–Deploy Contract Assay

### 12.1 질문

root-only posterior는 deploy-time reranking heuristic인가, 아니면 self-play target까지 바꿔야 하는 genuine policy-improvement mechanism인가?

### 12.2 설계

2×2 매트릭스로 제한한다.

- Train target:
  - old visit-count target
  - root-posterior target

- Deploy search:
  - old baseline search
  - root-only posterior search

조건 네 개:

1. Train old / Deploy old
2. Train old / Deploy root-only
3. Train root-posterior / Deploy old
4. Train root-posterior / Deploy root-only

### 12.3 1차 지표

- arena win rate / Elo
- learning-curve AUC
- held-out fixed-position accuracy
- training stability

### 12.4 2차 지표

- self-play diversity
- target entropy
- mode-collapse signal
- overfitting gap

### 12.5 kill criterion

- Train+Deploy가 Deploy-only보다 일관되게 못하면 종료
- training instability가 baseline보다 크게 악화되면 종료

### 12.6 pseudocode

```python
def experiment4_train_deploy_contract(train_configs, deploy_configs, seeds):
    rows = []
    for seed in seeds:
        for train_cfg in train_configs:
            model = train_selfplay_model(train_cfg, seed=seed)
            for deploy_cfg in deploy_configs:
                arena = run_arena(model, deploy_cfg, seed=seed)
                heldout = run_fixed_position_eval(model, deploy_cfg)
                rows.append({
                    "experiment": "E4",
                    "seed": seed,
                    "train_target": train_cfg.name,
                    "deploy_search": deploy_cfg.name,
                    "arena_winrate": arena.winrate,
                    "elo": arena.elo,
                    "auc_learning_curve": model.training_auc,
                    "heldout_accuracy": heldout.root_accuracy,
                    "target_entropy": model.target_entropy,
                    "stability_flag": model.training_stable,
                })
    return rows
```

---

## 13. 실험 5 — Structure Identification Assay
### Root-Conflict vs Deep-Conflict

### 13.1 질문

문제가 root conflict인가, 아니면 deeper tactical conflict인가?

### 13.2 가설

- N2는 `root_conflict`에서 상대적으로 강해야 한다.
- N1은 `deep_conflict`에서 상대적으로 강해야 한다.

### 13.3 포지션 셋

- `root_conflict`
- `deep_conflict`

### 13.4 비교군

- B0
- B1
- N1
- N2

### 13.5 1차 지표

- bucket별 root accuracy
- bucket별 correction gain
- bucket별 KL / CE to oracle

### 13.6 2차 지표

- revision timing histogram
- candidate undercoverage by bucket
- entropy drop by bucket

### 13.7 kill criterion

- 두 bucket 모두에서 N1, N2가 차별 패턴을 거의 못 보이면 실패

### 13.8 pseudocode

```python
def experiment5_structure_identification(checkpoints, bucketed_positions, budgets, systems):
    rows = []
    for ckpt in checkpoints:
        evaluator = load_frozen_evaluator(ckpt)
        for bucket_name, positions in bucketed_positions.items():
            for pos in positions:
                oracle = compute_oracle_policy(pos)
                prior0 = evaluator.prior(pos)
                for budget in budgets:
                    for system in systems:
                        out = run_search(
                            position=pos,
                            evaluator=evaluator,
                            injected_prior=prior0,
                            budget=budget,
                            system=system,
                        )
                        rows.append({
                            "experiment": "E5",
                            "checkpoint_id": ckpt.id,
                            "position_id": pos.id,
                            "position_bucket": bucket_name,
                            "budget": budget,
                            "system": system.name,
                            "accuracy": int(argmax(out.final_policy) == oracle.argmax()),
                            "correction_gain": policy_gain(out.final_policy, prior0, oracle),
                            "kl_to_oracle": kl_divergence(out.final_policy, oracle),
                            "revision_step": out.revision_step,
                            "candidate_set_contains_oracle_best": int(
                                oracle.argmax() in out.root_candidate_set
                            ),
                        })
    return rows
```

---

## 14. 실험 6 — Human-Like Bounded Deliberation Dynamics Assay

### 14.1 질문

새 구조가 깊은 brute-force의 저예산 근사일 뿐인가, 아니면 적은 탐색에서 선택적 belief revision이라는 인간형 bounded deliberation 패턴을 보이는가?

### 14.2 budget

매우 작은 budget만 사용한다.

- 4
- 8
- 16

### 14.3 비교군

- B0
- B1
- N1
- N2

### 14.4 1차 지표

- first revision step
- number of revisions
- revision sparsity
- flip-flop rate
- final confidence after revision

### 14.5 2차 지표

- entropy trajectory
- revision concentration by bucket
- agreement with deeper-search oracle

### 14.6 kill criterion

- revision이 거의 모든 포지션에서 무차별적으로 일어나면 실패
- flip-flop이 높으면 실패
- deeper-search agreement가 개선되지 않으면 실패

### 14.7 pseudocode

```python
def experiment6_bounded_deliberation_dynamics(checkpoints, positions, tiny_budgets, systems):
    rows = []
    for ckpt in checkpoints:
        evaluator = load_frozen_evaluator(ckpt)
        for pos in positions:
            prior0 = evaluator.prior(pos)
            oracle = compute_oracle_policy(pos)
            for budget in tiny_budgets:
                for system in systems:
                    trace = run_search_with_trace(
                        position=pos,
                        evaluator=evaluator,
                        injected_prior=prior0,
                        budget=budget,
                        system=system,
                    )
                    rows.append({
                        "experiment": "E6",
                        "checkpoint_id": ckpt.id,
                        "position_id": pos.id,
                        "position_bucket": pos.bucket,
                        "budget": budget,
                        "system": system.name,
                        "first_revision_step": trace.first_revision_step,
                        "num_revisions": trace.num_revisions,
                        "flip_flop_rate": trace.flip_flop_rate,
                        "revision_sparsity": trace.num_revisions / max(1, budget),
                        "final_confidence": max(trace.final_policy),
                        "oracle_match": int(argmax(trace.final_policy) == oracle.argmax()),
                        "entropy_path": trace.entropy_series,
                    })
    return rows
```

---

# Part II. Minimal Implementation Contracts

## 15. 실행 순서

### 15.1 1차 구조 식별 단계

가장 먼저 아래 세 실험을 수행한다.

1. 실험 1 — wrong-prior correction
2. 실험 3 — root-only few-sim
3. 실험 5 — root-vs-deep conflict identification

이 셋이 가장 적은 비용으로 가장 큰 구조 정보를 준다.

### 15.2 2차 정밀 검증 단계

위 단계에서 살아남은 구조만 아래로 보낸다.

4. 실험 2 — do-no-harm
5. 실험 4 — train–deploy contract
6. 실험 6 — human-like dynamics

---

## 16. survivor 판정 규칙

### 16.1 N1을 mainline 후보로 올리는 조건

아래 세 조건을 모두 만족해야 한다.

- 실험 1에서 wrong-prior correction이 B0/B1보다 우세
- 실험 2에서 easy-case regret가 낮음
- 실험 5에서 deep-conflict bucket에서 상대 우세

### 16.2 N2를 mainline 후보로 올리는 조건

아래 세 조건을 모두 만족해야 한다.

- 실험 3에서 low-budget root accuracy 우세
- 실험 4에서 train–deploy consistency가 유지됨
- 실험 5에서 root-conflict bucket에서 상대 우세

### 16.3 hybrid 후보 검토 조건

아래 조건이면 hybrid 가능성을 검토한다.

- N2는 root-conflict에서 강함
- N1은 deep-conflict에서 강함
- 둘이 서로 다른 bucket에서 분명한 우세를 보임

이 경우 “root-local improvement + selective deeper posterior” 같은 2단 구조를 2차 후보로 올릴 수 있다.

---

## 17. 미리 박아둘 사전 등록 문장

실험 전에 아래 세 문장을 고정한다.

1. 우리는 refresh 자체를 증명하려는 것이 아니라, **bounded deliberation 하의 belief revision architecture**를 비교한다.
2. 성공 기준은 단순 arena 승률이 아니라, wrong-prior correction, do-no-harm, few-sim root improvement, train–deploy consistency의 동시 만족이다.
3. 새 구조가 current refresh를 이겨도 no-refresh baseline을 못 넘으면 mainline 승격은 보류한다.

---

## 18. 최소 결과 테이블 형식

권장 결과 테이블은 아래와 같다.

### 18.1 구조 식별 요약표

| Experiment | Metric | B0 | B1 | N1 | N2 | Winner | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| E1 | wrong-prior correction |  |  |  |  |  |  |
| E2 | easy-case regret |  |  |  |  |  |  |
| E3 | low-budget root accuracy |  |  |  |  |  |  |
| E4 | train–deploy consistency |  |  |  |  |  |  |
| E5 | root-conflict gain |  |  |  |  |  |  |
| E5 | deep-conflict gain |  |  |  |  |  |  |
| E6 | flip-flop rate |  |  |  |  |  |  |

### 18.2 kill criterion 체크표

| Experiment | Kill criterion | Passed? | Comment |
|---|---|---|---|
| E1 | N1 beats B0 in biased-prior regime |  |  |
| E2 | easy-case regret not significantly worse |  |  |
| E3 | N2 beats B0 in low-budget regime |  |  |
| E4 | Train+Deploy not worse than Deploy-only |  |  |
| E5 | architecture separation visible by bucket |  |  |
| E6 | revision is selective, not indiscriminate |  |  |

---

## 19. 구현 전에 반드시 넣을 로깅 훅

새 구조를 구현하기 전에 아래 로깅이 먼저 들어가야 한다.

### 19.1 공통 로깅 훅

- `revision_occurred`
- `revision_step`
- `num_revisions`
- `entropy_path`
- `argmax_path`
- `candidate_set`
- `candidate_contains_oracle`
- `compute_time_ms`

### 19.2 N1 전용 훅

- `prior_base`
- `posterior_search`
- `effective_policy`
- `dual_gate`
- `posterior_norm`

### 19.3 N2 전용 훅

- `root_candidate_set`
- `root_candidate_scores`
- `root_posterior`
- `root_rank_swap_count`
- `undercoverage_flag`

---

## 20. 지금 하지 말아야 할 것

아래는 당장 미룬다.

1. uncertainty head 추가
2. DAG / transposition consensus posterior
3. generalized refresh trait hierarchy
4. trust-region formal shell
5. theorem-first writeup
6. full family explosion

이유는 간단하다. 지금 필요한 건 새로운 family가 아니라 **belief revision의 객체**와 **belief revision의 위치**를 고정하는 것이다.

---

## 21. 최종 권고

가장 먼저 할 일은 다음 세 가지다.

1. 포지션 bucket을 먼저 고정한다.
2. 공통 로그 필드를 먼저 박는다.
3. 실험 1, 3, 5를 먼저 돌린다.

그 다음에만 2, 4, 6으로 넘어간다.

이 순서를 지키지 않으면, 다시 architecture effect와 training noise와 hyperparameter noise가 섞인다.

---

## 22. 참고 문헌(개념적 기준선)

1. Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero, 2017).  
   search를 root improved policy 생성 관점으로 이해하는 기본 기준선.

2. Danihelka et al., *Policy Improvement by Planning with Gumbel* (Gumbel AlphaZero / Gumbel MuZero, 2020/2021).  
   few-simulation에서 root policy-improvement를 더 principled하게 재구성하는 기준선.

3. Lanctot et al., *Monte Carlo Tree Search with Heuristic Evaluations using Implicit Minimax Backups* (2014).  
   heuristic evaluation과 search-derived estimate를 separate channels로 유지하는 사고를 지지하는 기준선.

---

## 23. 한 문장 요약

이 문서의 핵심은 이것 하나다.

> 새로운 refresh family를 더 만드는 대신, belief revision의 객체(prior를 바꿀 것인가, posterior를 따로 둘 것인가)와 위치(root에서만 할 것인가, deeper tree까지 내릴 것인가)를 먼저 고정하라.
