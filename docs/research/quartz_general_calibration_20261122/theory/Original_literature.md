# R2. Primary scientific literature and exact overlap

(1) Russell & Wefald, "On Optimal Game-Tree Search using Rational Meta-Reasoning" (IJCAI 1989), https://ijcai.org/proceedings/1989/0053.pdf, and "Principles of Metareasoning" (1991). Computational actions are chosen by their expected effect on physical decisions net deliberation cost. This predates the QUARTZ metareasoning principle and does NOT license a new novelty claim.

(2) Hay, Russell, Tolpin & Shimony, "Selecting Computations: Theory and Applications" (UAI 2012), https://www.auai.org/uai2012/papers/123.pdf. Describes belief-state metalevel selection, cost-aware stopping and bounded rational control. C17's finite posterior/observation framework is a restricted case.

(3) Sezener & Dayan, "Static and Dynamic Values of Computation in MCTS" (2020), https://proceedings.mlr.press/v124/sezener20a.html. Supports interpreting computation worth from change in selected downstream action; dynamic and static future information assumptions differ. It does not establish independent MCTS backup samples.

(4) Malik et al., "Calibrated Model-Based Deep Reinforcement Learning", ICML 2019, https://proceedings.mlr.press/v97/malik19a.html. Conditional predictive distribution calibration and proper scoring rules matter for planning; mere uncertainty scalars may be insufficient.

(5) Fisch et al., "Calibrated Selective Classification", https://arxiv.org/abs/2208.12084 . Distinguishes calibrated selective acceptance from generic uncertainty; includes distribution-shift caveats. Use only as methodological precedent, not a ready coverage guarantee for dependent root trajectories.

(6) Original QUARTZ docs/THESIS.md, BQ_PLUS_PLUS_DESIGN.md, RESEARCH_PLAN_PARTB.md, METACOGNITIVE_EXPERIMENTS.md; C17's mathematical reference; C16's live Rust endpoint. Production SearchPolicy has no calibrated likelihood-of-revision guarantee. Neither old synthetic Beta-Bernoulli lab nor previous Freestyle15 proof-source tests estimate the actual conditional law of root revision under real adaptive MCTS.

Scope synthesis: independent parent-group/trajectory split, source-provenance, proper scoring rule and cost-measured shadow forecast are the smallest scientifically meaningful advance. Predicting revisions and predicting *positive decision value* are distinct estimands. This study contributes a project-specific measured calibration study, not a novel mathematical proper scoring rule or automatically production-safe stopping algorithm.