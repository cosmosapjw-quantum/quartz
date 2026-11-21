# R2. 선행연구와 실제 저장소 source identity

**Primary original work:** Chaslot, Winands, van den Herik, Uiterwijk & Bouzy (2008), *Progressive Strategies for Monte-Carlo Tree Search*, New Mathematics and Natural Computation 4(3):343–357, DOI https://doi.org/10.1142/S1793005708001094 ; original author/publication record https://cris.maastrichtuniversity.nl/en/publications/progressive-strategies-for-monte-carlo-tree-search/ . **Supported result:** progressive unpruning initially reduces the branching factor and progressively increases it; their positive results were in the program Mango on Go. It is **not** a demonstration of Connect Four resident-root gain or human cognition.

**Methodological limitations:** policy-prior and visited-edge statistics, high branching factor and heuristic reliability govern unpruning value. Winands (2017), *Monte-Carlo Tree Search in Board Games*, Maastricht research record, describes progressive widening and the domain-knowledge cost/tradeoff; Pascal Pons Connect Four exact action score is an independently executed game-theory oracle, not a search policy.

**Related human attention literature, different causal domain:** Bilalić et al. (2008), *Why good thoughts block better ones*, DOI https://doi.org/10.1016/j.cognition.2008.05.005, and Sheridan & Reingold (2013), PLOS ONE https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075796 , used chess eye-movement and familiar-move attention. No claim that computer root restrictions explain human Einstellung.

**Actual source read at parent HEAD:**
- `src/mcts/node.rs::MctsNode`: candidates `OnceLock<Box<[(Move,p)]>>`, fixed-capacity arena edge slab and atomic `edge_cursor`; prepublished root edges are stable. No safe in-place append to the OnceLock candidate array without a distinct storage/TT redesign.
- `src/mcts/expand.rs::publish_candidates`: sorts evaluator full legal prior; `materialize_edges` builds child TT nodes and slabs with fixed original length.
- `src/mcts/select.rs::select_core`: root PUCT loops over current materialized edges, currently no opt-in root-visible action gate.
- `src/mcts/mod.rs::MctsEngine::run`: fixed-visit controller checks *existing root n_total* and is safe to invoke again at a higher visit limit, preserving tree. `advance_root` advances to next game state, **not** suitable for the same-root experiment.
- `src/games/connect4.rs`: COG03-12 root mask lives on the cloned game state and changes its root TT hash; it is suitable for restarted-search experiments but cannot safely mutate its root legal action set in an existing OnceLock tree.

**Novelty gate:** a resident selection-mask over *already existing* full-legal edges is a source-specific implementation and empirical experiment in an established progressive-unpruning family, not a novel general widening algorithm. R2 status **literature-supported / implementation readback**, no scientific success claim.
