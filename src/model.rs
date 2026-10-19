//! Language-model promotion over existing decoder candidates.
use crate::{Candidate, CandidateSource, session::SessionSnapshot};
use chinese_ime_lm::{Reranker, SentenceModel};
use std::{error::Error, path::Path, sync::Arc};

#[cfg(feature = "accelerator")]
mod openvino;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Model {
    backend: Backend,
}

enum Backend {
    Cpu(Reranker),
    #[cfg(feature = "accelerator")]
    Accelerator(openvino::Ranker),
}

impl Model {
    /// `device` is `cpu`, or `gpu`/`npu` with the OpenVINO runtime directory of an accelerator build.
    pub fn load(path: &Path, device: &str, runtime_dir: Option<&Path>) -> Result<Self> {
        let backend = match (device, runtime_dir) {
            ("cpu", None) => {
                let load = || -> Result<_> { Ok(SentenceModel::load(&std::fs::read(path)?)?) };
                let model = load().map_err(|error| format!("{}: {error}", path.display()))?;
                Backend::Cpu(Reranker::new(Arc::new(model)))
            }
            ("cpu", Some(_)) | (_, None) => {
                return Err("gpu/npu need an OpenVINO runtime directory; cpu takes none".into());
            }
            #[cfg(feature = "accelerator")]
            (device, Some(runtime_dir)) => {
                Backend::Accelerator(openvino::Ranker::load(path, runtime_dir, device)?)
            }
            #[cfg(not(feature = "accelerator"))]
            (_, Some(_)) => {
                return Err(
                    "GPU/NPU ranking requires a binary built with --features accelerator".into(),
                );
            }
        };
        Ok(Self { backend })
    }

    /// Return the index of at most one full-input candidate to move to the front.
    pub fn promotion(&mut self, session: &SessionSnapshot) -> Result<Option<usize>> {
        let SessionSnapshot {
            input,
            context,
            candidates,
            learning,
            ..
        } = session;
        let personal = learning.as_deref().filter(|learning| learning.contextual());
        let considered = shortlist(input, context, candidates, personal.is_some());
        if considered.len() < 2 {
            return Ok(None);
        }
        let considered_candidates = || considered.iter().map(|&i| &candidates[i]);
        let texts: Vec<&str> = considered_candidates().map(|c| c.text.as_str()).collect();
        // Enabling personal learning alone must not remove the no-context leader protection.
        let mut scores = if context.is_empty()
            && matches!(
                candidates[0].source,
                CandidateSource::ExactDictionary | CandidateSource::User
            ) {
            vec![0.0; texts.len()]
        } else {
            self.language_scores(context, &texts)?
        };
        // Dictionary words and explicit choices provide a prior. Composed beam scores
        // include segmentation costs, so their gaps are not word-frequency evidence.
        if matches!(
            candidates[0].source,
            CandidateSource::ExactDictionary | CandidateSource::User
        ) {
            for (score, &index) in scores.iter_mut().zip(&considered) {
                *score -=
                    STATIC_GAP_PENALTY * (candidates[0].score - candidates[index].score).max(0.0);
            }
        }
        if let Some(learning) = personal {
            let personal = learning.personal_scores(input, context, considered_candidates());
            let counts = learning.counts(input, considered_candidates());
            for ((score, personal), count) in scores.iter_mut().zip(personal).zip(counts) {
                *score += personal;
                *score += 0.15 * (count as f32).ln_1p();
            }
        }
        Ok(best_promotion(&considered, &scores))
    }

    fn language_scores(&mut self, context: &str, texts: &[&str]) -> Result<Vec<f32>> {
        let window = match &self.backend {
            Backend::Cpu(reranker) => reranker.model().context_length(),
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(_) => openvino::CONTEXT,
        };
        // A prior must cover the whole candidate, including long dictionary macros, and the
        // character vocabulary maps letters in entries such as B站 to UNK. Either way the
        // group prior stays neutral rather than favouring a distorted likelihood.
        if texts.iter().any(|text| {
            text.chars().count() >= window || text.bytes().any(|b| b.is_ascii_alphabetic())
        }) {
            return Ok(vec![0.0; texts.len()]);
        }
        let scores = match &mut self.backend {
            Backend::Cpu(reranker) => reranker.log_probabilities(context, texts),
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(ranker) => ranker.log_probabilities(context, texts)?,
        };
        if !scores.iter().all(|score| score.is_finite()) {
            return Err("nonfinite language score for full-input candidate".into());
        }
        Ok(scores)
    }
}

const PROMOTION_MARGIN: f32 = 0.25;
const STATIC_GAP_PENALTY: f32 = 2.0;

fn best_promotion(indices: &[usize], scores: &[f32]) -> Option<usize> {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    (best != 0 && scores[best] > scores[0] + PROMOTION_MARGIN).then(|| indices[best])
}

fn answers_input(input: &str, candidate: &Candidate) -> bool {
    candidate.consumed == input.len() && candidate.source != CandidateSource::Completion
}

fn shortlist(
    input: &str,
    context: &str,
    candidates: &[Candidate],
    // Contextual learning may overrule the leader that is otherwise protected.
    personal: bool,
) -> Vec<usize> {
    let Some(leader) = candidates.first() else {
        return Vec::new();
    };
    if !answers_input(input, leader)
        || (!personal
            && (leader.source == CandidateSource::User
                || (context.is_empty() && leader.source == CandidateSource::ExactDictionary)))
    {
        return Vec::new();
    }
    let mut eligible = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| answers_input(input, c).then_some(i));
    let mut selected: Vec<usize> = eligible.by_ref().take(7).collect();
    for i in eligible.clone() {
        if selected.len() == 9 {
            break;
        }
        let length = candidates[i].text.chars().count();
        if !selected
            .iter()
            .any(|&index| candidates[index].text.chars().count() == length)
        {
            selected.push(i);
        }
    }
    for i in eligible {
        if selected.len() == 9 {
            break;
        }
        if !selected.contains(&i) {
            selected.push(i);
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decoder, Dictionary, Session, learning::Learning};
    use std::io::Cursor;

    fn candidate(text: &str, consumed: usize, source: CandidateSource) -> Candidate {
        Candidate {
            text: text.into(),
            consumed,
            source,
            score: 1.0,
        }
    }

    fn snapshot(input: &str, context: &str, candidates: &[Candidate]) -> SessionSnapshot {
        SessionSnapshot {
            revision: 0,
            input: input.into(),
            context: context.into(),
            candidates: candidates.to_vec(),
            learning: None,
        }
    }

    fn model() -> Model {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
        Model::load(&path, "cpu", None)
            .expect("download the pinned model with scripts/fetch_assets.py")
    }

    #[test]
    fn common_dictionary_spelling_survives_a_weak_language_prior() {
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n开源\tkai yuan\t75955\n开元\tkai yuan\t2\n",
            ))
            .unwrap(),
        ));
        let mut model = model();
        for context in ["", "这个项目准备", "代码仓库已经公开，项目正式"] {
            for contextual in [false, true] {
                let learning = contextual.then(|| Arc::new(Learning::new(true)));
                let mut session = Session::new(decoder.clone(), learning);
                session.set_context(context);
                session.insert("kaiyuan").unwrap();
                assert_eq!(session.candidates()[0].text, "开源");
                assert_eq!(model.promotion(&session.snapshot()).unwrap(), None);
            }
        }
    }

    #[test]
    fn neutral_language_prior_does_not_revoke_an_explicit_choice() {
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n避战\tbi zhan\t1000000\nB站\tbi zhan\t1\n",
            ))
            .unwrap(),
        ));
        let learning = Arc::new(Learning::new(true));
        let mut session = Session::new(decoder, Some(learning));
        session.set_context("视频");
        session.insert("bizhan").unwrap();
        let chosen = session
            .candidates()
            .iter()
            .position(|candidate| candidate.text == "B站")
            .unwrap();
        session.select(chosen).unwrap();
        session.reset();
        session.set_context("视频");
        session.insert("bizhan").unwrap();
        assert_eq!(session.candidates()[0].text, "B站");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        assert_eq!(model().promotion(&session.snapshot()).unwrap(), None);
    }

    #[test]
    fn facts_use_consumed_input_instead_of_character_count() {
        assert!(answers_input(
            "xian",
            &candidate("现", 4, CandidateSource::Composed)
        ));
        assert!(answers_input(
            "xian",
            &candidate("西安", 4, CandidateSource::Composed)
        ));
        assert!(!answers_input(
            "xian",
            &candidate("西", 2, CandidateSource::Composed)
        ));
        assert!(!answers_input(
            "xian",
            &candidate("西安", 4, CandidateSource::Completion)
        ));
    }

    #[test]
    fn shortlist_protects_leader_and_reserves_length_diversity() {
        let mut candidates = vec![candidate("现", 4, CandidateSource::Composed); 12];
        candidates[10].text = "西安".into();
        assert_eq!(
            shortlist("xian", "我去", &candidates, false),
            vec![0, 1, 2, 3, 4, 5, 6, 10, 7]
        );
        candidates[0].source = CandidateSource::ExactDictionary;
        assert!(shortlist("xian", "", &candidates, false).is_empty());
        assert_eq!(shortlist("xian", "我去", &candidates, false).len(), 9);
        candidates[0].source = CandidateSource::User;
        assert!(shortlist("xian", "我去", &candidates, false).is_empty());
        assert_eq!(shortlist("xian", "我去", &candidates, true).len(), 9);
        candidates[0].source = CandidateSource::Composed;
        candidates[0].consumed = 2;
        assert!(shortlist("xian", "我去", &candidates, true).is_empty());
    }

    #[test]
    fn letters_and_overlong_candidates_neutralize_the_language_prior() {
        let mut model = model();
        assert_eq!(
            model
                .language_scores("视频", &["避战", "必占", "B站"])
                .unwrap(),
            vec![0.0; 3]
        );
        let long = "字".repeat(342);
        assert_eq!(
            model.language_scores("前文", &[&long, "你"]).unwrap(),
            vec![0.0; 2]
        );
        assert!(
            model
                .language_scores("视频", &["避战", "必占"])
                .unwrap()
                .iter()
                .all(|score| *score < 0.0)
        );
        let candidates = [
            candidate("避战", 6, CandidateSource::Composed),
            candidate("必占", 6, CandidateSource::Composed),
            candidate("B站", 6, CandidateSource::ExactDictionary),
        ];
        assert_eq!(
            model
                .promotion(&snapshot("bizhan", "视频", &candidates))
                .unwrap(),
            None
        );
    }

    #[test]
    fn previous_inference_matches_a_fresh_model() {
        let mut used = model();
        let previous = [
            candidate("现", 4, CandidateSource::Composed),
            candidate("西安", 4, CandidateSource::Composed),
        ];
        assert_eq!(
            used.promotion(&snapshot("xian", "我去", &previous))
                .unwrap(),
            Some(1)
        );
        let input = "wojintianqushangban";
        let candidates = [
            candidate("我今天去上班", input.len(), CandidateSource::Composed),
            candidate("我今天去伤斑", input.len(), CandidateSource::Composed),
        ];
        assert_eq!(
            used.promotion(&snapshot(input, "", &candidates)).unwrap(),
            model()
                .promotion(&snapshot(input, "", &candidates))
                .unwrap()
        );
    }

    #[test]
    fn loading_reports_the_model_path_and_rejects_runtime_mismatch() {
        let missing = Path::new("/nonexistent/yanliu-model.safetensors");
        let error = Model::load(missing, "cpu", None).err().unwrap().to_string();
        assert!(
            error.contains("/nonexistent/yanliu-model.safetensors"),
            "{error}"
        );
        for (device, runtime) in [("cpu", Some(Path::new("runtime"))), ("npu", None)] {
            assert!(Model::load(missing, device, runtime).is_err());
        }
    }

    #[test]
    fn real_model_proposes_existing_full_candidate_and_protects_leader() {
        let mut model = model();
        let mut candidates = vec![
            candidate("现", 4, CandidateSource::Composed),
            candidate("西", 2, CandidateSource::Composed),
            candidate("西安", 4, CandidateSource::Composed),
        ];
        let promote = |model: &mut Model, context: &str, candidates: &[Candidate]| {
            model
                .promotion(&snapshot("xian", context, candidates))
                .unwrap()
        };
        assert_eq!(promote(&mut model, "我去", &candidates), Some(2));
        candidates.rotate_right(1);
        assert_eq!(promote(&mut model, "我去", &candidates), None);
        assert_eq!(promote(&mut model, "我们去", &candidates), None);
        candidates[0].source = CandidateSource::ExactDictionary;
        assert_eq!(promote(&mut model, "", &candidates), None);
        candidates[0].source = CandidateSource::User;
        assert_eq!(promote(&mut model, "我去", &candidates), None);
    }
}
