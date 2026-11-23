# C03 — numerical design and classified errors

Finite support is exact categorical discretization of this declared model, not a discretized PDE. There are no spacetime units or boundary conditions. Initial law has equal latent masses, fieldzero; game boundary is terminal exact utility. Domain: legal nonterminal roots,0<=rho<1, positive finite fitted nonterminal probabilities, nonnegative querycost.

Use full log_odds rather than sequential products. Predictive p=p_base*(1+rho*tanh(log_odds/2)*f); update with log1p(rho*f)-log1p(-rho*f). Each actual observation occurs once. With atmost5rootcandidates, saturation is bounded; nonetheless validate finite values and reject unsupported observation. Never silently reset beliefs. Stable log-sum-exp computes TRAIN likelihood; specialrho0 may returnexact0.

Compute H2 by looping each candidate,3outcomes and remaining candidates, then H1. For fixedsupport complexityO(m^2), storagemodelO(m), excluding recordedtraces. Actual performance remains measured; no complexity-to-speed inference.

Classify error:
- Closure/model error: positivecommonfactor maymissnegative/heterogeneousdependence; remains empirical.
- Horizon truncation: cap2 is exact for thedeclaredfiniteproblem, approximateforlargeroriginalsearch.
- Costmodelerror: learnednodeprediction and omittedfutureplannercharge differfromfullruntimeaccounting.
- Sampling/statisticalerror:90TRAINrootfit,24CAL,120TEST; repeatedseeds/timingarenotrootreplicates.
- Floatingroundoff: probabilities/scores testedwithin1e-12 (allow2e-12accumulation); strictlyknownvalues/nodes exact.
- Decisionboundary: explicitSTOP/tietolerance1e-12; analyticalzerothresholdcasesmuststop.
- Measurementerror: monotonic wallclock includesinstrumentation;5measuredrepeats,1warmupforcampaign; priceprotocol5nowarmup. Store dispersion and allrawvalues.
- Infrastructure: missingTorch, disk/network/processexit are reportedseparately.

Tests use exact Rational fullworldoracle ontoyfixtures and independent fulljoint conditional arithmetic for realgametraces. Decisive fixtures: pureinformation109/200, rho0product, b1STOP, negativecorrection-1/5, posteriororder, projectedcontrol, no repeatedquery, cap2. Do not claim53851theoryassertions proveproduction correctness.
