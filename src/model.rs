//! Bounded pinyin recall and language-model promotion.
use crate::{Candidate, CandidateSource, Decoder, session::SessionSnapshot};
use chinese_ime_lm::{Reranker, SentenceModel};
use serde::Serialize;
use std::{error::Error, path::Path, sync::Arc};

#[cfg(feature = "accelerator")]
mod openvino;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Model {
    backend: Backend,
}

#[derive(Serialize)]
pub struct CandidateScore {
    pub index: usize,
    pub language: f32,
    pub dictionary: f32,
    pub personal: f32,
    pub count: u32,
    pub frequency: f32,
    pub total: f32,
}

#[derive(Serialize)]
pub struct Promotion {
    pub prior_scale: f32,
    pub margin: f32,
    pub index: Option<usize>,
    pub scores: Vec<CandidateScore>,
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

    /// Enrich a snapshot with proven pinyin paths, then score its actual menu.
    /// A cancelled result must not be applied to the live session.
    pub fn refine(
        &mut self,
        session: &mut SessionSnapshot,
        decoder: &Decoder,
        mut still_current: impl FnMut() -> bool,
    ) -> Result<Option<Promotion>> {
        if !still_current() {
            return Ok(None);
        }
        let additional = match &mut self.backend {
            Backend::Cpu(reranker) => {
                let Some(candidates) = decoder.neural_candidates(
                    &session.input,
                    &session.context,
                    reranker,
                    &mut still_current,
                )?
                else {
                    return Ok(None);
                };
                candidates
            }
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(_) => Vec::new(),
        };
        let included = session.include_candidates(&additional);
        if !still_current() {
            return Ok(None);
        }
        let promotion = self.score_candidates(session, &included)?;
        Ok(still_current().then_some(promotion))
    }

    fn score_candidates(
        &mut self,
        session: &SessionSnapshot,
        included: &[usize],
    ) -> Result<Promotion> {
        let SessionSnapshot {
            input,
            context,
            candidates,
            learning,
            ..
        } = session;
        let personal = learning.as_deref().filter(|learning| learning.contextual());
        let limit = match &self.backend {
            Backend::Cpu(_) => 16,
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(_) => 9,
        };
        let mut considered = shortlist(input, context, candidates, personal.is_some(), limit);
        if !considered.is_empty() {
            for &index in included {
                let candidate = &candidates[index];
                if answers_input(input, candidate)
                    && candidate.source.is_correction() == candidates[0].source.is_correction()
                    && !considered.contains(&index)
                {
                    considered.push(index);
                }
            }
        }
        if !considered.is_empty() && matches!(&self.backend, Backend::Cpu(_)) {
            // CPU single characters share the prefix's next-character distribution and need
            // no tail inference. The fixed accelerator IR keeps its nine-candidate batch.
            for (index, _) in candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    answers_input(input, c)
                        && c.source.is_correction() == candidates[0].source.is_correction()
                        && c.text.chars().count() == 1
                        && c.text
                            .chars()
                            .all(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
                })
                .take(64)
            {
                if !considered.contains(&index) {
                    considered.push(index);
                }
            }
        }
        if considered.len() < 2 {
            return Ok(Promotion {
                prior_scale: 1.0,
                margin: PROMOTION_MARGIN,
                index: None,
                scores: Vec::new(),
            });
        }
        let considered_candidates = || considered.iter().map(|&i| &candidates[i]);
        let texts: Vec<&str> = considered_candidates().map(|c| c.text.as_str()).collect();
        // A bare initial has no syllable evidence for a public language-model prior.
        // Personal learning also keeps the existing no-context leader protection.
        let language = if single_initial(input)
            || (context.is_empty()
                && matches!(
                    candidates[0].source,
                    CandidateSource::ExactDictionary | CandidateSource::User
                )) {
            vec![0.0; texts.len()]
        } else {
            self.language_scores(
                context,
                &texts,
                candidates[0].source == CandidateSource::Composed,
            )?
        };
        let dictionary_prior = matches!(
            candidates[0].source,
            CandidateSource::ExactDictionary | CandidateSource::User
        );
        let mut scores: Vec<_> = considered
            .iter()
            .zip(language)
            .map(|(&index, language)| {
                // Composed beam scores contain segmentation costs, not word-frequency evidence.
                let dictionary = if dictionary_prior {
                    -STATIC_GAP_PENALTY * (candidates[0].score - candidates[index].score).max(0.0)
                } else {
                    0.0
                };
                CandidateScore {
                    index,
                    language,
                    dictionary,
                    personal: 0.0,
                    count: 0,
                    frequency: 0.0,
                    total: 0.0,
                }
            })
            .collect();
        if let Some(learning) = personal {
            let personal = learning.personal_scores(input, context, considered_candidates());
            let counts = learning.counts(input, considered_candidates());
            for ((score, personal), count) in scores.iter_mut().zip(personal).zip(counts) {
                score.personal = personal;
                score.count = count;
                score.frequency = 0.15 * (count as f32).ln_1p();
            }
        }
        // A larger LM can produce much wider likelihood gaps for the same alternatives.
        // Personal updates use fixed feature units, so compare them to a bounded prior range.
        // Scaling its margin too preserves the base decision before personal feedback exists.
        let prior_scale = if scores
            .iter()
            .any(|score| score.personal != 0.0 || score.count > 0)
        {
            let low = scores
                .iter()
                .map(|score| score.language + score.dictionary)
                .fold(f32::INFINITY, f32::min);
            let high = scores
                .iter()
                .map(|score| score.language + score.dictionary)
                .fold(f32::NEG_INFINITY, f32::max);
            (high - low).max(1.0)
        } else {
            1.0
        };
        for score in &mut scores {
            score.total = (score.language + score.dictionary) / prior_scale
                + score.personal
                + score.frequency;
        }
        let margin = PROMOTION_MARGIN / prior_scale;
        Ok(Promotion {
            index: best_promotion(&scores, margin),
            prior_scale,
            margin,
            scores,
        })
    }

    fn language_scores(
        &mut self,
        context: &str,
        texts: &[&str],
        composed_leader: bool,
    ) -> Result<Vec<f32>> {
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
        let known: Vec<_> = texts
            .iter()
            .enumerate()
            .filter_map(|(index, text)| {
                text.chars()
                    .all(|character| match &self.backend {
                        Backend::Cpu(reranker) => reranker.model().contains_character(character),
                        #[cfg(feature = "accelerator")]
                        Backend::Accelerator(ranker) => ranker.contains_character(character),
                    })
                    .then_some(index)
            })
            .collect();
        // UNK describes every missing character together, not the specific candidate. Without
        // a representable leader there is no comparable language prior for a promotion.
        if known.first() != Some(&0) || known.len() < 2 {
            return Ok(vec![0.0; texts.len()]);
        }
        let known_texts: Vec<_> = known.iter().map(|&index| texts[index]).collect();
        let known_scores = match &mut self.backend {
            Backend::Cpu(reranker) => reranker.log_probabilities(context, &known_texts),
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(ranker) => ranker.log_probabilities(context, &known_texts)?,
        };
        if !known_scores.iter().all(|score| score.is_finite()) {
            return Err("nonfinite language score for full-input candidate".into());
        }
        // An unrepresented candidate gets no language advantage over the leader; dictionary
        // evidence and explicit personal choices still apply to it.
        let mut scores = vec![known_scores[0]; texts.len()];
        let leader_length = texts[0].chars().count() as f32;
        for (&index, score) in known.iter().zip(known_scores) {
            // A shared denominator keeps extra segmentation from diluting its token costs.
            scores[index] = if composed_leader {
                score * (texts[index].chars().count() as f32 / leader_length)
            } else {
                score
            };
        }
        Ok(scores)
    }
}

const PROMOTION_MARGIN: f32 = 0.25;
const STATIC_GAP_PENALTY: f32 = 2.0;

fn best_promotion(scores: &[CandidateScore], margin: f32) -> Option<usize> {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i].total > scores[best].total {
            best = i;
        }
    }
    (best != 0 && scores[best].total > scores[0].total + margin).then(|| scores[best].index)
}

fn answers_input(input: &str, candidate: &Candidate) -> bool {
    candidate.consumed == input.len() && candidate.source != CandidateSource::Completion
}

fn single_initial(input: &str) -> bool {
    input.len() == 1 && b"bpmfdtnlgkhjqxrzcsyw".contains(&input.as_bytes()[0].to_ascii_lowercase())
}

fn shortlist(
    input: &str,
    context: &str,
    candidates: &[Candidate],
    // Contextual learning may overrule the leader that is otherwise protected.
    personal: bool,
    limit: usize,
) -> Vec<usize> {
    let Some(leader) = candidates.first() else {
        return Vec::new();
    };
    if !answers_input(input, leader)
        || (!personal
            && (single_initial(input)
                || leader.source == CandidateSource::User
                || (context.is_empty() && leader.source == CandidateSource::ExactDictionary)))
    {
        return Vec::new();
    }
    let mut eligible = candidates.iter().enumerate().filter_map(|(i, c)| {
        (answers_input(input, c) && c.source.is_correction() == leader.source.is_correction())
            .then_some(i)
    });
    let mut selected: Vec<usize> = eligible.by_ref().take(limit - 2).collect();
    for i in eligible.clone() {
        if selected.len() == limit {
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
        if selected.len() == limit {
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
    fn empty_personal_state_preserves_base_promotions() {
        let mut model = model();
        let candidates = [
            candidate("现", 4, CandidateSource::Composed),
            candidate("西安", 4, CandidateSource::Composed),
            candidate("线", 4, CandidateSource::Composed),
        ];
        for context in ["", "我去", "此时此刻", "乘火车前往"] {
            let mut session = snapshot("xian", context, &candidates);
            let base = model.score_candidates(&session, &[]).unwrap();
            session.learning = Some(Arc::new(Learning::new(true)));
            let personal = model.score_candidates(&session, &[]).unwrap();
            assert_eq!(base.index, personal.index, "context={context}");
            assert_eq!(personal.prior_scale, 1.0);
            assert_eq!(
                serde_json::to_value(base).unwrap(),
                serde_json::to_value(personal).unwrap()
            );
        }
    }

    #[test]
    fn composed_candidates_compete_under_the_same_language_and_static_prior() {
        let mut model = model();
        let mut candidates = [
            candidate("现", 4, CandidateSource::ExactDictionary),
            candidate("西安", 4, CandidateSource::Composed),
        ];
        let raw = model
            .language_scores("我去", &["现", "西安"], false)
            .unwrap();
        assert!(raw[1] > raw[0] + PROMOTION_MARGIN);
        let result = model
            .score_candidates(&snapshot("xian", "我去", &candidates), &[])
            .unwrap();
        assert_eq!(result.index, Some(1));
        assert_eq!(result.scores[1].language, raw[1]);
        assert_eq!(result.scores[1].dictionary, 0.0);
        candidates[1].score = 0.5;
        let result = model
            .score_candidates(&snapshot("xian", "我去", &candidates), &[])
            .unwrap();
        assert_eq!(result.scores[1].language, raw[1]);
        assert_eq!(result.scores[1].dictionary, -1.0);
    }

    #[test]
    fn correction_candidates_do_not_steal_an_ordinary_session_promotion() {
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n现\txian\t100\n西安\txi an\t100\n轩\txuan\t1000000\n",
            ))
            .unwrap(),
        ));
        let mut session = Session::new(decoder.clone(), None);
        session.set_context("我去");
        session.insert("xian").unwrap();
        let mut snapshot = session.snapshot();
        assert!(
            snapshot
                .candidates
                .iter()
                .any(|c| { c.text == "轩" && c.source.is_correction() })
        );
        let decision = model()
            .refine(&mut snapshot, &decoder, || true)
            .unwrap()
            .unwrap();
        assert!(
            decision
                .scores
                .iter()
                .all(|s| { !snapshot.candidates[s.index].source.is_correction() })
        );
        let index = decision.index.unwrap();
        assert_eq!(snapshot.candidates[index].text, "西安");
        assert!(session.apply_candidates(snapshot.revision, snapshot.candidates, Some(index)));
        assert_eq!(session.candidates()[0].text, "西安");
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
                let mut snapshot = session.snapshot();
                let result = model
                    .refine(&mut snapshot, &decoder, || true)
                    .unwrap()
                    .unwrap();
                assert!(session.apply_candidates(
                    snapshot.revision,
                    snapshot.candidates,
                    result.index
                ));
                assert_eq!(session.candidates()[0].text, "开源");
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
        let mut session = Session::new(decoder.clone(), Some(learning));
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
        let mut snapshot = session.snapshot();
        let result = model()
            .refine(&mut snapshot, &decoder, || true)
            .unwrap()
            .unwrap();
        assert!(session.apply_candidates(snapshot.revision, snapshot.candidates, result.index));
        assert_eq!(session.candidates()[0].text, "B站");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
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
            shortlist("xian", "我去", &candidates, false, 9),
            vec![0, 1, 2, 3, 4, 5, 6, 10, 7]
        );
        candidates[0].source = CandidateSource::ExactDictionary;
        assert!(shortlist("xian", "", &candidates, false, 9).is_empty());
        assert_eq!(shortlist("xian", "我去", &candidates, false, 9).len(), 9);
        candidates[0].source = CandidateSource::User;
        assert!(shortlist("xian", "我去", &candidates, false, 9).is_empty());
        assert_eq!(shortlist("xian", "我去", &candidates, true, 9).len(), 9);
        candidates[0].source = CandidateSource::Composed;
        candidates[0].consumed = 2;
        assert!(shortlist("xian", "我去", &candidates, true, 9).is_empty());
    }

    #[test]
    fn cpu_word_shortlist_keeps_sixteen_with_two_length_diversity_slots() {
        let mut candidates = vec![candidate("现", 4, CandidateSource::Composed); 20];
        candidates[18].text = "西安".into();
        candidates[19].text = "西安市".into();
        let mut expected: Vec<_> = (0..14).collect();
        expected.extend([18, 19]);
        assert_eq!(shortlist("xian", "我去", &candidates, false, 16), expected);
    }

    #[test]
    fn contextual_single_character_scores_reach_beyond_the_word_shortlist() {
        let mut dictionary = "...\n".to_string();
        for character in "及级即集极急几己既纪机鸡记季继济".chars() {
            dictionary.push_str(&format!("{character}\tji\t1000\n"));
        }
        dictionary.push_str("剂\tji\t1\n");
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(dictionary)).unwrap(),
        ));
        let mut session = Session::new(decoder, None);
        session.set_context("医生调整了药物的");
        session.insert("ji").unwrap();
        let snapshot = session.snapshot();
        let index = snapshot
            .candidates
            .iter()
            .position(|c| c.text == "剂")
            .unwrap();
        assert!(index >= 16);
        let promotion = model().score_candidates(&snapshot, &[]).unwrap();
        let score = promotion
            .scores
            .iter()
            .find(|score| score.index == index)
            .unwrap();
        assert!(score.language.is_finite() && score.language < 0.0);
        assert_eq!(session.candidates(), snapshot.candidates);
    }

    #[test]
    fn single_initials_use_only_personal_scores_even_with_context_and_learned_user_source() {
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new("...\n了\tle\t1000\n来\tlai\t100\n")).unwrap(),
        ));
        let mut model = model();
        assert!(
            model
                .language_scores("我马上", &["了", "来"], false)
                .unwrap()
                .iter()
                .all(|s| *s < 0.0)
        );
        for contextual in [false, true] {
            let learning = Arc::new(Learning::new(contextual));
            let mut session = Session::new(decoder.clone(), Some(learning));
            session.set_context("我马上");
            session.insert("l").unwrap();
            for learned in [false, true] {
                if learned {
                    let index = session
                        .candidates()
                        .iter()
                        .position(|c| c.text == "来")
                        .unwrap();
                    assert_eq!(session.select(index).as_deref(), Some("来"));
                    session.reset();
                    session.set_context("我马上");
                    session.insert("l").unwrap();
                    assert_eq!(session.candidates()[0].source, CandidateSource::User);
                }
                let result = model.score_candidates(&session.snapshot(), &[]).unwrap();
                if contextual {
                    assert_eq!(result.scores.len(), 2);
                    assert!(result.scores.iter().all(|score| score.language == 0.0));
                    if learned {
                        assert!(result.scores.iter().any(|score| score.personal != 0.0));
                    }
                } else {
                    assert!(result.scores.is_empty());
                }
            }
        }
        let candidates = [
            candidate("了", 1, CandidateSource::ExactDictionary),
            candidate("来", 1, CandidateSource::ExactDictionary),
        ];
        for initial in b"bpmfdtnlgkhjqxrzcsywBPMFDTNLGKHJQXRZCSYW" {
            let input = char::from(*initial).to_string();
            assert!(shortlist(&input, "前文", &candidates, false, 9).is_empty());
            assert_eq!(shortlist(&input, "前文", &candidates, true, 9), [0, 1]);
        }
        for vowel in ["a", "e", "o"] {
            assert_eq!(shortlist(vowel, "前文", &candidates, false, 9), [0, 1]);
        }
    }

    #[test]
    fn letters_and_overlong_candidates_neutralize_the_language_prior() {
        let mut model = model();
        assert_eq!(
            model
                .language_scores("视频", &["避战", "必占", "B站"], false)
                .unwrap(),
            vec![0.0; 3]
        );
        let long = "字".repeat(342);
        assert_eq!(
            model
                .language_scores("前文", &[&long, "你"], false)
                .unwrap(),
            vec![0.0; 2]
        );
        assert!(
            model
                .language_scores("视频", &["避战", "必占"], false)
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
                .score_candidates(&snapshot("bizhan", "视频", &candidates), &[])
                .unwrap()
                .index,
            None
        );
    }

    #[test]
    fn unknown_characters_have_no_language_advantage_over_the_leader() {
        let mut model = model();
        let sentence_model = match &model.backend {
            Backend::Cpu(reranker) => reranker.model(),
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(_) => unreachable!("the test fixture loads the CPU model"),
        };
        for character in ['如', '入', '对'] {
            assert!(sentence_model.contains_character(character));
        }
        for character in ['㨎', '𠮷'] {
            assert!(!sentence_model.contains_character(character));
        }
        let known = model.language_scores("出", &["如", "入"], false).unwrap();
        let mixed = model
            .language_scores("出", &["如", "㨎", "入"], false)
            .unwrap();
        assert_eq!(mixed, vec![known[0], known[0], known[1]]);
        assert!(
            known[1] > known[0],
            "known homophones must still use context"
        );
        assert_eq!(
            model
                .language_scores("出", &["㨎", "如", "入"], false)
                .unwrap(),
            vec![0.0; 3]
        );
        let scores = model
            .language_scores("好", &["对", "㨎", "𠮷"], false)
            .unwrap();
        assert_eq!(scores, vec![0.0; 3]);
    }

    #[test]
    fn full_candidate_likelihood_uses_the_leaders_length_and_keeps_oov_tied() {
        let mut model = model();
        let texts = ["手间吗", "手机按摩", "现", "㨎㨎㨎㨎㨎"];
        let raw = match &mut model.backend {
            Backend::Cpu(reranker) => reranker.log_probabilities("可以借用一下洗", &texts[..3]),
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(_) => unreachable!("the test fixture loads the CPU model"),
        };
        let scores = model
            .language_scores("可以借用一下洗", &texts, true)
            .unwrap();
        assert_eq!(scores[0], raw[0]);
        assert_eq!(scores[1], raw[1] * (4.0 / 3.0));
        assert_eq!(scores[2], raw[2] * (1.0 / 3.0));
        assert_eq!(scores[3], scores[0]);

        let input = "shoujianma";
        let candidates: Vec<_> = texts
            .iter()
            .map(|text| candidate(text, input.len(), CandidateSource::Composed))
            .collect();
        let mut session = snapshot(input, "可以借用一下洗", &candidates);
        let composed = model.score_candidates(&session, &[]).unwrap();
        assert_eq!(composed.scores[1].language, scores[1]);
        for source in [CandidateSource::ExactDictionary, CandidateSource::User] {
            session.candidates[0].source = source;
            session.learning = Some(Arc::new(Learning::new(true)));
            let direct = model.score_candidates(&session, &[]).unwrap();
            assert_eq!(direct.scores[1].language, raw[1]);
            assert_eq!(direct.scores[3].language, raw[0]);
        }
    }

    #[test]
    fn an_explicit_unknown_character_choice_still_learns() {
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n如\tru\t4308194\n㨎\tru\t68901\n入\tru\t1000000\n",
            ))
            .unwrap(),
        ));
        let learning = Arc::new(Learning::new(true));
        let mut model = model();
        let sentence_model = match &model.backend {
            Backend::Cpu(reranker) => reranker.model(),
            #[cfg(feature = "accelerator")]
            Backend::Accelerator(_) => unreachable!("the test fixture loads the CPU model"),
        };
        assert!(!sentence_model.contains_character('㨎'));
        let mut session = Session::new(decoder.clone(), Some(learning));
        session.set_context("牛");
        session.insert("ru").unwrap();
        let chosen = session
            .candidates()
            .iter()
            .position(|candidate| candidate.text == "㨎")
            .unwrap();
        session.select(chosen).unwrap();
        session.reset();
        session.set_context("牛");
        session.insert("ru").unwrap();
        assert_eq!(session.candidates()[0].text, "㨎");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        let mut snapshot = session.snapshot();
        let result = model
            .refine(&mut snapshot, &decoder, || true)
            .unwrap()
            .unwrap();
        assert!(session.apply_candidates(snapshot.revision, snapshot.candidates, result.index));
        assert_eq!(session.candidates()[0].text, "㨎");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
    }

    #[test]
    fn previous_inference_matches_a_fresh_model() {
        let mut used = model();
        let previous = [
            candidate("现", 4, CandidateSource::Composed),
            candidate("西安", 4, CandidateSource::Composed),
        ];
        assert_eq!(
            used.score_candidates(&snapshot("xian", "我去", &previous), &[])
                .unwrap()
                .index,
            Some(1)
        );
        let input = "wojintianqushangban";
        let candidates = [
            candidate("我今天去上班", input.len(), CandidateSource::Composed),
            candidate("我今天去伤斑", input.len(), CandidateSource::Composed),
        ];
        assert_eq!(
            used.score_candidates(&snapshot(input, "", &candidates), &[])
                .unwrap()
                .index,
            model()
                .score_candidates(&snapshot(input, "", &candidates), &[])
                .unwrap()
                .index
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
                .score_candidates(&snapshot("xian", context, candidates), &[])
                .unwrap()
                .index
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
