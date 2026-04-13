use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::game::{EvalResult, Evaluator, GameState};
use crate::games::gomoku15::{Gomoku15, GomokuVariant};
use crate::mcts::parallel::VlMode;
use crate::mcts::quartz::QuartzConfig;
use crate::mcts::{MctsConfig, PwConfig};

#[cfg(feature = "onnx")]
use ort::{session::Session, value::TensorRef};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct GomocupAbout {
    pub name: String,
    pub version: String,
    pub author: String,
    pub country: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct GomocupSearchConfig {
    pub search_profile: Option<String>,
    pub vl_mode: Option<String>,
    pub tt_enabled: Option<bool>,
    pub c_puct: Option<f32>,
    pub sigma_0: Option<f32>,
    pub min_visits: Option<u32>,
    pub check_interval: Option<u32>,
    pub budget_ms: Option<u64>,
    pub max_visits: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct GomocupSource {
    pub condition: Option<String>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct GomocupManifest {
    pub format_version: Option<u32>,
    pub game: Option<String>,
    pub gomocup_rule: Option<String>,
    pub onnx_model: Option<String>,
    pub checkpoint_copy: Option<String>,
    pub about: GomocupAbout,
    pub search: GomocupSearchConfig,
    pub source: GomocupSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchProfile {
    Quartz,
    Baseline,
    BaselineStrict,
}

pub struct LoadedGomocupBundle {
    pub root: PathBuf,
    pub manifest_path: Option<PathBuf>,
    pub manifest: GomocupManifest,
    pub model_path: Option<PathBuf>,
    #[cfg(feature = "onnx")]
    evaluator: Option<Arc<Gomoku15OnnxEvaluator>>,
}

impl LoadedGomocupBundle {
    fn from_root(root: &Path) -> Option<Self> {
        let root_dir = if root.is_dir() {
            root.to_path_buf()
        } else {
            root.parent()?.to_path_buf()
        };
        let manifest_path = if root.is_file() {
            root.to_path_buf()
        } else {
            root_dir.join("gomocup_manifest.json")
        };

        let mut manifest = if manifest_path.exists() {
            let text = fs::read_to_string(&manifest_path).ok()?;
            serde_json::from_str::<GomocupManifest>(&text).ok()?
        } else {
            GomocupManifest::default()
        };
        if manifest.game.is_none() {
            manifest.game = Some("gomoku15".to_string());
        }
        if manifest.about.name.is_empty() {
            manifest.about.name = "QUARTZ-Gomocup".to_string();
        }
        if manifest.about.version.is_empty() {
            manifest.about.version = "0.2".to_string();
        }
        if manifest.about.author.is_empty() {
            manifest.about.author = "cosmosapjw+Codex".to_string();
        }
        if manifest.about.country.is_empty() {
            manifest.about.country = "KR".to_string();
        }

        let model_rel = manifest
            .onnx_model
            .clone()
            .unwrap_or_else(|| "gomocup_model.onnx".to_string());
        let candidate_model = root_dir.join(model_rel);
        if !manifest_path.exists() && !candidate_model.exists() {
            return None;
        }

        let model_path = candidate_model.exists().then_some(candidate_model.clone());
        #[cfg(feature = "onnx")]
        let evaluator = if let Some(path) = model_path.as_ref() {
            match Gomoku15OnnxEvaluator::load(path) {
                Ok(model) => Some(Arc::new(model)),
                Err(err) => {
                    eprintln!("[gomocup] ONNX load failed for {}: {}", path.display(), err);
                    None
                }
            }
        } else {
            None
        };

        Some(Self {
            root: root_dir,
            manifest_path: manifest_path.exists().then_some(manifest_path),
            manifest,
            model_path,
            #[cfg(feature = "onnx")]
            evaluator,
        })
    }

    pub fn supports_variant(&self, variant: GomokuVariant) -> bool {
        match self.manifest.game.as_deref().unwrap_or("gomoku15") {
            "gomoku15" | "gomoku15_free" => variant == GomokuVariant::Freestyle,
            "gomoku15_std" => variant == GomokuVariant::Standard,
            "gomoku15_renju" => variant == GomokuVariant::Renju,
            "gomoku15_caro" => variant == GomokuVariant::Caro,
            "gomoku15_omok" => variant == GomokuVariant::Omok,
            _ => false,
        }
    }

    pub fn about_line(&self) -> String {
        format!(
            "name={}, version={}, author={}, country={}",
            self.manifest.about.name,
            self.manifest.about.version,
            self.manifest.about.author,
            self.manifest.about.country
        )
    }

    pub fn budget_ms(&self) -> Option<u64> {
        self.manifest.search.budget_ms
    }

    pub fn max_visits(&self) -> Option<u32> {
        self.manifest.search.max_visits
    }

    pub fn uses_quartz_controller(&self) -> bool {
        parse_search_profile(self.manifest.search.search_profile.as_deref()) == SearchProfile::Quartz
    }

    #[cfg(feature = "onnx")]
    pub fn evaluator_for_variant(
        &self,
        variant: GomokuVariant,
    ) -> Option<Arc<dyn Evaluator<Gomoku15> + Send + Sync>> {
        if self.supports_variant(variant) {
            self.evaluator
                .clone()
                .map(|ev| ev as Arc<dyn Evaluator<Gomoku15> + Send + Sync>)
        } else {
            None
        }
    }
}

pub fn load_bundle(search_roots: &[PathBuf]) -> Option<LoadedGomocupBundle> {
    for root in search_roots {
        if let Some(bundle) = LoadedGomocupBundle::from_root(root) {
            return Some(bundle);
        }
    }
    None
}

fn parse_search_profile(raw: Option<&str>) -> SearchProfile {
    match raw.unwrap_or("quartz") {
        "baseline" => SearchProfile::Baseline,
        "baseline_strict" => SearchProfile::BaselineStrict,
        _ => SearchProfile::Quartz,
    }
}

fn apply_search_profile(mut cfg: MctsConfig, profile: SearchProfile) -> MctsConfig {
    match profile {
        SearchProfile::Quartz => {}
        SearchProfile::Baseline => {
            cfg.quartz = None;
            cfg.gvoc = None;
            cfg.vl_mode = VlMode::Disabled;
        }
        SearchProfile::BaselineStrict => {
            cfg.quartz = None;
            cfg.gvoc = None;
            cfg.vl_mode = VlMode::Disabled;
            cfg.root_forced_win = false;
            cfg.exact_terminal_value = false;
            cfg.fpu_reduction = 0.0;
        }
    }
    cfg
}

pub fn apply_bundle_search_config(mut cfg: MctsConfig, bundle: Option<&LoadedGomocupBundle>) -> MctsConfig {
    let Some(bundle) = bundle else {
        return cfg;
    };
    cfg = apply_search_profile(
        cfg,
        parse_search_profile(bundle.manifest.search.search_profile.as_deref()),
    );
    if let Some(ref vl) = bundle.manifest.search.vl_mode {
        cfg.vl_mode = match vl.as_str() {
            "disabled" => VlMode::Disabled,
            "fixed" => VlMode::Fixed,
            "adaptive" => VlMode::Adaptive,
            "vvisit_only" => VlMode::VvisitOnly,
            "vvalue_only" => VlMode::VvalueOnly,
            _ => cfg.vl_mode,
        };
    }
    if let Some(tt_enabled) = bundle.manifest.search.tt_enabled {
        cfg.tt_enabled = tt_enabled;
    }
    if let Some(c_puct) = bundle.manifest.search.c_puct {
        cfg.c_puct = c_puct;
    }
    if let Some(ref mut q) = cfg.quartz {
        if let Some(sigma_0) = bundle.manifest.search.sigma_0 {
            q.sigma_0 = sigma_0;
        }
        if let Some(min_visits) = bundle.manifest.search.min_visits {
            q.min_visits = min_visits;
        }
        if let Some(check_interval) = bundle.manifest.search.check_interval {
            q.check_interval = check_interval;
        }
    }
    cfg
}

pub fn default_gomoku15_config() -> MctsConfig {
    MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5)).with_quartz(QuartzConfig {
        sigma_0: 0.3,
        min_visits: 30,
        check_interval: 50,
        ..Default::default()
    })
}

#[cfg(feature = "onnx")]
pub struct Gomoku15OnnxEvaluator {
    session: std::sync::Mutex<Session>,
}

#[cfg(feature = "onnx")]
impl Gomoku15OnnxEvaluator {
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut builder = Session::builder().map_err(|err| err.to_string())?;
        let session = builder.commit_from_file(path).map_err(|err| err.to_string())?;
        Ok(Self {
            session: std::sync::Mutex::new(session),
        })
    }

    fn predict(&self, state: &Gomoku15) -> Result<(Vec<f32>, f32), String> {
        let input = state.encode_planes();
        let tensor = TensorRef::from_array_view(([1usize, 17usize, 15usize, 15usize], input.as_slice()))
            .map_err(|err| err.to_string())?;
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|err| err.to_string())?;
        if outputs.len() < 2 {
            return Err("onnx output missing value head".to_string());
        }
        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|err| err.to_string())?;
        let (_, values) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|err| err.to_string())?;
        let value = *values
            .first()
            .ok_or_else(|| "onnx value tensor is empty".to_string())?;
        Ok((logits.to_vec(), value.clamp(-1.0, 1.0)))
    }
}

#[cfg(feature = "onnx")]
impl Evaluator<Gomoku15> for Gomoku15OnnxEvaluator {
    fn evaluate(&self, state: &Gomoku15) -> EvalResult<u16> {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult {
                policy: vec![],
                value: 0.0,
            };
        }
        let Ok((logits, value)) = self.predict(state) else {
            return EvalResult::uniform(&legal, 0.0);
        };
        if logits.len() < state.num_actions() {
            return EvalResult::uniform(&legal, value);
        }

        let mut max_logit = f32::NEG_INFINITY;
        for &mv in &legal {
            max_logit = max_logit.max(logits[state.move_to_idx(mv)]);
        }

        let mut sum = 0.0_f32;
        let mut policy = Vec::with_capacity(legal.len());
        for &mv in &legal {
            let prior = (logits[state.move_to_idx(mv)] - max_logit).exp();
            sum += prior;
            policy.push((mv, prior));
        }
        if !sum.is_finite() || sum <= 0.0 {
            return EvalResult::uniform(&legal, value);
        }
        for (_, prior) in policy.iter_mut() {
            *prior /= sum;
        }
        EvalResult { policy, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "quartz_gomocup_bundle_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_load_bundle_uses_default_onnx_name_without_manifest() {
        let root = temp_root("default_onnx");
        let onnx = root.join("gomocup_model.onnx");
        fs::write(&onnx, b"stub").unwrap();

        let bundle = load_bundle(&[root.clone()]).expect("bundle");
        assert_eq!(bundle.manifest.game.as_deref(), Some("gomoku15"));
        assert_eq!(bundle.model_path.as_deref(), Some(onnx.as_path()));
    }

    #[test]
    fn test_apply_bundle_search_config_respects_manifest_overrides() {
        let bundle = LoadedGomocupBundle {
            root: PathBuf::new(),
            manifest_path: None,
            manifest: GomocupManifest {
                game: Some("gomoku15".to_string()),
                search: GomocupSearchConfig {
                    search_profile: Some("baseline_strict".to_string()),
                    vl_mode: Some("fixed".to_string()),
                    tt_enabled: Some(false),
                    c_puct: Some(1.5),
                    ..Default::default()
                },
                ..Default::default()
            },
            model_path: None,
            #[cfg(feature = "onnx")]
            evaluator: None,
        };

        let cfg = apply_bundle_search_config(default_gomoku15_config(), Some(&bundle));
        assert!(cfg.quartz.is_none());
        assert_eq!(cfg.vl_mode, VlMode::Fixed);
        assert!(!cfg.tt_enabled);
        assert_eq!(cfg.c_puct, 1.5);
        assert!(!cfg.root_forced_win);
    }

    #[test]
    fn test_bundle_supports_exact_variant_only() {
        let bundle = LoadedGomocupBundle {
            root: PathBuf::new(),
            manifest_path: None,
            manifest: GomocupManifest {
                game: Some("gomoku15_renju".to_string()),
                ..Default::default()
            },
            model_path: None,
            #[cfg(feature = "onnx")]
            evaluator: None,
        };
        assert!(bundle.supports_variant(GomokuVariant::Renju));
        assert!(!bundle.supports_variant(GomokuVariant::Freestyle));
    }
}
