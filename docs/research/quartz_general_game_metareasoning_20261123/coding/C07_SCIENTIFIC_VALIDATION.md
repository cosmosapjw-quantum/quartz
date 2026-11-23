# C07 — TRAIN and CAL execution

Prepared the frozen300source parents with TRAIN90/CAL60/TEST150 and no D4 orbit overlap. The model-free native TRAIN survey measured296nonterminal candidates:183losses,65draws,48wins, total6435exact traversal nodes. Every candidate value, cost and cue was independently checked.

Reconstructed every fitted coefficient from raw TRAIN sufficient statistics. Node-cost means by child remaining count2/3/4 are4.280701754385965,11.02803738317757,37.96212121212121. The +1-smoothed feature/remaining/global model has SHA2560843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126. A separate model backup and time-stamped freeze were written before CAL. All prepare/train/audit-model commands exited0.

CAL executed60parents,720timing requests,38880recorded timing policy runs,720separate trace requests and4320trace policy runs. Native exit0; full run+analysis command exit0 in4.5196s. Every logical signature, query choice, exact reveal, charged node count and timing/trace identity was checked. CAL did not change the model or any frozen price, temperature, source split, cap or contrast.

Diagnostic full-cap PANDORA minus INITIAL_STOP node-priced gain is0.06865234375, paired parent95%interval[0.027977294921875,0.11551188151041666]. This is CAL diagnostic evidence; it does not satisfy the separate frozen TEST gate by itself.

CAL nonterminal-candidate multiclass Brier is0.32894153973575585 (equal-parent0.30668826867181986). Actual versus predicted mean query nodes are21.17241379310345 and21.274657801448512; MAE3.721184902020087 and RMSE5.809366207027223. Descriptive within-root residual correlation is0.27849935555404665 over57parents with at least two nonterminal candidates. This illustrates why empirical product-prior calibration must be assessed and why independent-model optimality is not automatically a game theorem.

The model hash remains identical after CAL and TEST has not run at this stage. Detailed source/model/run identities, finite Gibbs ranges, raw measurements, calibration rows and all comparisons are retained in coding/data/, coding/results/CAL/ and evidence/C07_SCIENTIFIC_CHECKS.json. Proceed to the previously fixed TEST without tuning.
