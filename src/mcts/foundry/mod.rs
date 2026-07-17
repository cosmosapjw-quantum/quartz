//! QUARTZ idea-foundry meta-control skeletons.
//!
//! ## Wiring (deliberately not applied by this scaffold)
//!
//! 1. Add `pub mod foundry;` beside `pub mod policy;` in `src/mcts/mod.rs`.
//! 2. Extend the root observation boundary with `FoundryRootExtras` using an
//!    explicit telemetry/schema bump.
//! 3. Run modules in `observe`/checkpoint time, collect `MetaProposal`s, and let
//!    one arbiter choose at most one explicit `MetaAction`.
//! 4. Translate score/readout proposals through the existing immutable
//!    `PolicyCache`; translate SAMPLE/WIDEN/etc. through a separate root-session
//!    executor.  Do not make `score_adjustment()` perform scheduling work.
//! 5. Keep this feature gated until the Python trace-replay skeletons and
//!    counterfactual labels pass their promotion gates.

pub mod control;
pub mod search;
pub mod systems;
pub mod types;

pub use control::*;
pub use search::*;
pub use systems::*;
pub use types::*;
