use crate::{
    Candidate, CandidateSource, Decoder,
    decoder::{DecodeCache, MAX_INPUT_BYTES},
    learning::Learning,
};
use std::sync::Arc;

/// Everything a background ranker needs; `revision` rejects a result for changed input.
pub struct SessionSnapshot {
    pub revision: u64,
    pub input: String,
    pub context: String,
    pub candidates: Vec<Candidate>,
    pub learning: Option<Arc<Learning>>,
}

impl SessionSnapshot {
    pub(crate) fn include_candidates(&mut self, additional: &[Candidate]) -> Vec<usize> {
        let mut changed = false;
        for candidate in additional {
            let matching = |existing: &Candidate| {
                existing.text == candidate.text && existing.consumed == candidate.consumed
            };
            if self.candidates.iter().any(|existing| {
                matching(existing)
                    && existing.source != CandidateSource::Completion
                    && !existing.source.is_correction()
            }) {
                continue;
            }
            // A proven verbatim path supersedes an incomplete or corrected reading.
            self.candidates.retain(|existing| !matching(existing));
            self.candidates.push(candidate.clone());
            changed = true;
        }
        if changed {
            self.candidates = rank_candidates(
                &self.input,
                self.learning.as_deref(),
                std::mem::take(&mut self.candidates),
            );
        }
        additional
            .iter()
            .map(|candidate| {
                self.candidates
                    .iter()
                    .position(|existing| {
                        existing.text == candidate.text && existing.consumed == candidate.consumed
                    })
                    .expect("included candidate is present")
            })
            .collect()
    }
}

pub struct Session {
    decoder: Arc<Decoder>,
    decode_cache: DecodeCache,
    learning: Option<Arc<Learning>>,
    input: String,
    cursor: usize,
    revision: u64,
    candidates: Vec<Candidate>,
    context: String,
}

impl Session {
    /// Without `learning`, choices are neither recorded nor applied.
    pub fn new(decoder: Arc<Decoder>, learning: Option<Arc<Learning>>) -> Self {
        Self {
            decoder,
            decode_cache: DecodeCache::default(),
            learning,
            input: String::new(),
            cursor: 0,
            revision: 0,
            candidates: Vec::new(),
            context: String::new(),
        }
    }
    pub fn input(&self) -> &str {
        &self.input
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            revision: self.revision,
            input: self.input.clone(),
            context: self.context.clone(),
            candidates: self.candidates.clone(),
            learning: self.learning.clone(),
        }
    }
    /// Insert at the cursor; invalid text leaves the session unchanged.
    pub fn insert(&mut self, text: &str) -> Result<(), String> {
        if !text.bytes().all(|b| b.is_ascii_alphabetic() || b == b'\'') {
            return Err("input accepts ASCII letters and apostrophes".into());
        }
        if self.input.len() + text.len() > MAX_INPUT_BYTES {
            return Err(format!("input exceeds {MAX_INPUT_BYTES} bytes"));
        }
        if !text.is_empty() {
            self.input.insert_str(self.cursor, text);
            self.cursor += text.len();
            self.refresh();
        }
        Ok(())
    }
    pub fn backspace(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        self.input.remove(self.cursor);
        self.refresh();
        true
    }
    pub fn delete_forward(&mut self) -> bool {
        if self.cursor == self.input.len() {
            return false;
        }
        self.input.remove(self.cursor);
        self.refresh();
        true
    }
    pub fn move_cursor(&mut self, delta: isize) {
        self.cursor = self
            .cursor
            .saturating_add_signed(delta)
            .min(self.input.len());
    }
    /// Commit a candidate, keeping the raw input it did not consume.
    pub fn select(&mut self, index: usize) -> Option<String> {
        let candidate = self.candidates.get(index)?.clone();
        if let Some(learning) = &self.learning
            && !learning.record_choice(&self.input, &self.context, &self.candidates, index)
        {
            return None;
        }
        self.input.drain(..candidate.consumed);
        self.cursor = self.cursor.saturating_sub(candidate.consumed);
        self.append_context(&candidate.text);
        Some(candidate.text)
    }
    pub fn commit_raw(&mut self) -> Option<String> {
        if self.input.is_empty() {
            return None;
        }
        let text = std::mem::take(&mut self.input);
        self.cursor = 0;
        self.append_context(&text);
        Some(text)
    }
    pub fn reset(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.context.clear();
        self.refresh();
    }
    pub fn set_context(&mut self, context: &str) -> bool {
        let start = context
            .char_indices()
            .rev()
            .nth(63)
            .map_or(0, |(index, _)| index);
        let context = &context[start..];
        if self.context != context {
            self.context = context.to_owned();
            self.refresh();
            true
        } else {
            false
        }
    }
    /// Move the candidate a ranker proposed for `revision`; false when the input has changed.
    fn apply_promotion(&mut self, revision: u64, index: usize) -> bool {
        if revision != self.revision {
            return false;
        }
        let preferred = &self.candidates[index];
        let raw_len = self.input.len();
        let contextual = self.learning.as_ref().is_some_and(|l| l.contextual());
        let protected = |candidate: &Candidate| {
            candidate.source == CandidateSource::Completion
                || (!contextual && candidate.source == CandidateSource::User)
        };
        if self
            .candidates
            .iter()
            .any(|c| c.consumed == raw_len && c.source.is_correction())
        {
            if preferred.consumed != raw_len || protected(preferred) {
                return true;
            }
            let correction = preferred.source.is_correction();
            let mut slot = index;
            for previous in (0..index).rev() {
                let candidate = &self.candidates[previous];
                if candidate.consumed == raw_len
                    && !protected(candidate)
                    && candidate.source.is_correction() == correction
                {
                    self.candidates.swap(previous, slot);
                    slot = previous;
                }
            }
        } else {
            self.candidates[..=index].rotate_right(1);
        }
        true
    }
    /// Apply a background result without changing input or advancing its revision.
    pub fn apply_candidates(
        &mut self,
        revision: u64,
        candidates: Vec<Candidate>,
        preferred: Option<usize>,
    ) -> bool {
        if revision != self.revision {
            return false;
        }
        self.candidates = candidates;
        if let Some(index) = preferred {
            self.apply_promotion(revision, index);
        }
        true
    }
    fn append_context(&mut self, text: &str) {
        let context = format!("{}{text}", self.context);
        if !self.set_context(&context) {
            self.refresh();
        }
    }
    fn refresh(&mut self) {
        self.revision += 1;
        let Some(mut candidates) = self
            .decoder
            .decode_cached(&self.input, &mut self.decode_cache)
        else {
            self.candidates.clear();
            return;
        };
        if let Some(learning) = &self.learning {
            candidates.extend(learning.recall(&self.input, &candidates));
        }
        self.candidates = rank_candidates(&self.input, self.learning.as_deref(), candidates);
    }
}

fn rank_candidates(
    input: &str,
    learning: Option<&Learning>,
    candidates: Vec<Candidate>,
) -> Vec<Candidate> {
    let counts = match learning {
        Some(learning) => learning.counts(input, &candidates),
        None => vec![0; candidates.len()],
    };
    let mut ranked: Vec<_> = candidates.into_iter().zip(counts).collect();
    for (candidate, count) in &mut ranked {
        if *count > 0 {
            candidate.source = CandidateSource::User;
        }
    }
    let class = |candidate: &Candidate| {
        if candidate.consumed == input.len() {
            2
        } else if candidate.consumed > 0 && input.as_bytes()[candidate.consumed - 1] == b'\'' {
            1
        } else {
            0
        }
    };
    ranked.sort_by(|(a, ac), (b, bc)| {
        class(b)
            .cmp(&class(a))
            .then(b.consumed.cmp(&a.consumed))
            .then(bc.cmp(ac))
            .then(b.score.total_cmp(&a.score))
    });
    let mut candidates: Vec<_> = ranked.into_iter().map(|(candidate, _)| candidate).collect();
    if !candidates
        .iter()
        .any(|c| c.consumed == input.len() && c.source.is_correction())
    {
        return candidates;
    }
    // Corrections share only full-conversion slots; learned, partial and
    // Completion candidates keep the positions established above.
    let slots: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, c)| {
            (c.consumed == input.len()
                && !matches!(
                    c.source,
                    CandidateSource::User | CandidateSource::Completion
                ))
            .then_some(index)
        })
        .collect();
    for next in 1..slots.len() {
        let mut position = next;
        while position > 0
            && candidates[slots[position - 1]].source.is_correction()
            && !candidates[slots[position]].source.is_correction()
        {
            candidates.swap(slots[position - 1], slots[position]);
            position -= 1;
        }
    }
    candidates
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::Dictionary;
    use std::io::Cursor;

    fn decoder() -> Arc<Decoder> {
        Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n你\tni\t10\n年\tnian\t99\n好\thao\t10\n你好\tni hao\t99\n",
            ))
            .unwrap(),
        ))
    }
    fn assert_fresh(session: &Session) {
        let mut fresh = Session::new(session.decoder.clone(), session.learning.clone());
        fresh.insert(session.input()).unwrap();
        assert_eq!(session.candidates().len(), fresh.candidates().len());
        for (a, b) in session.candidates().iter().zip(fresh.candidates()) {
            assert_eq!(
                (&a.text, a.consumed, a.source.clone(), a.score.to_bits()),
                (&b.text, b.consumed, b.source.clone(), b.score.to_bits())
            );
        }
    }
    #[test]
    fn refined_menu_preserves_existing_paths_and_learns_the_selected_conversion() {
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(decoder(), Some(learning.clone()));
        session.insert("nihao").unwrap();
        session.set_context("请先");
        let mut snapshot = session.snapshot();
        let existing = snapshot.candidates[0].clone();
        let mut duplicate = existing.clone();
        duplicate.source = CandidateSource::Composed;
        duplicate.score = -99.0;
        let added = Candidate {
            text: "拟好".into(),
            consumed: 5,
            source: CandidateSource::Composed,
            score: -4.0,
        };
        let mut completion = added.clone();
        completion.source = CandidateSource::Completion;
        snapshot.candidates.push(completion);
        let included = snapshot.include_candidates(&[duplicate, added.clone()]);
        assert_eq!(snapshot.candidates[included[0]], existing);
        assert_eq!(snapshot.candidates[included[1]], added);
        assert_eq!(
            snapshot
                .candidates
                .iter()
                .filter(|c| c.text == "拟好")
                .count(),
            1
        );
        let revision = session.revision();
        let cursor = session.cursor();
        assert!(session.apply_candidates(revision, snapshot.candidates, Some(included[1])));
        assert_eq!((session.input(), session.context()), ("nihao", "请先"));
        assert_eq!((session.revision(), session.cursor()), (revision, cursor));
        assert_eq!(session.select(0).as_deref(), Some("拟好"));
        assert_eq!(learning.counts("nihao", [&added]), vec![1]);
        assert_eq!(session.input(), "");
    }

    #[test]
    fn stale_candidate_set_does_not_replace_the_current_menu() {
        let mut session = Session::new(decoder(), None);
        session.insert("ni").unwrap();
        let snapshot = session.snapshot();
        session.insert("hao").unwrap();
        let current = session.snapshot();
        assert!(!session.apply_candidates(snapshot.revision, snapshot.candidates, Some(0)));
        assert_eq!(session.candidates(), current.candidates);
        assert_eq!(session.revision(), current.revision);
    }

    #[test]
    fn identical_bounded_context_keeps_the_refined_menu_and_revision() {
        let mut session = Session::new(decoder(), None);
        let context = "前".repeat(64);
        assert!(session.set_context(&context));
        session.insert("ni").unwrap();
        let snapshot = session.snapshot();
        session.apply_candidates(snapshot.revision, snapshot.candidates, Some(1));
        let ranked = session.snapshot();
        for same in [context.clone(), format!("忽略的旧前文{context}")] {
            assert!(!session.set_context(&same));
            assert_eq!(session.candidates(), ranked.candidates);
            assert_eq!(session.revision(), ranked.revision);
        }
    }

    #[test]
    fn commit_refreshes_input_even_when_the_bounded_context_is_unchanged() {
        let decoder = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new("...\n前\tqian\t100\n")).unwrap(),
        ));
        let mut session = Session::new(decoder, None);
        let context = "前".repeat(64);
        session.set_context(&context);
        session.insert("qian").unwrap();
        let revision = session.revision();
        assert_eq!(session.select(0).as_deref(), Some("前"));
        assert_eq!(session.context(), context);
        assert!(session.input().is_empty());
        assert!(session.candidates().is_empty());
        assert!(session.revision() > revision);
    }

    #[test]
    fn context_updates_drop_old_neural_paths_but_keep_explicit_choices() {
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(decoder(), Some(learning));
        session.insert("nihao").unwrap();
        let candidate = |text: &str| Candidate {
            text: text.into(),
            consumed: 5,
            source: CandidateSource::Composed,
            score: -4.0,
        };
        let mut snapshot = session.snapshot();
        let indices = snapshot.include_candidates(&[candidate("拟好")]);
        session.apply_candidates(snapshot.revision, snapshot.candidates, Some(indices[0]));
        assert_eq!(session.select(0).as_deref(), Some("拟好"));
        session.insert("nihao").unwrap();
        let baseline_len = session.candidates().len();
        for (context, text) in [("请先", "倪好"), ("正在", "泥好")] {
            let previous = session.snapshot();
            session.set_context(context);
            assert_eq!(session.candidates().len(), baseline_len);
            assert!(
                session
                    .candidates()
                    .iter()
                    .any(|c| { c.text == "拟好" && c.source == CandidateSource::User })
            );
            assert!(!session.apply_candidates(previous.revision, previous.candidates, None));
            let mut snapshot = session.snapshot();
            snapshot.include_candidates(&[candidate(text)]);
            assert!(session.apply_candidates(snapshot.revision, snapshot.candidates, None));
            assert_eq!(session.candidates().len(), baseline_len + 1);
            assert_eq!((session.input(), session.cursor()), ("nihao", 5));
        }
        assert!(!session.candidates().iter().any(|c| c.text == "倪好"));
        assert!(session.candidates().iter().any(|c| c.text == "泥好"));
    }

    #[test]
    fn completion_selection_commits_raw_consumption_without_learning() {
        let decoder = decoder();
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(decoder.clone(), Some(learning.clone()));
        session.set_context("问");
        session.insert("ni'h").unwrap();
        session.move_cursor(-1);
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "你好" && c.source == CandidateSource::Completion)
            .unwrap();
        let candidate = session.candidates()[index].clone();
        assert_eq!(candidate.consumed, "ni'h".len());
        let revision = session.revision();
        let learning_revision = learning.revision();
        assert_eq!(
            learning.counts("ni'h", std::slice::from_ref(&candidate)),
            [0]
        );

        assert_eq!(session.select(index).as_deref(), Some("你好"));
        assert_eq!(session.input(), "");
        assert_eq!(session.cursor(), 0);
        assert_eq!(session.context(), "问你好");
        assert!(session.revision() > revision);
        assert_eq!(learning.revision(), learning_revision);
        assert_eq!(
            learning.counts("ni'h", std::slice::from_ref(&candidate)),
            [0]
        );

        let mut fresh = Session::new(decoder, Some(learning));
        fresh.insert("ni'h").unwrap();
        assert!(fresh.candidates().contains(&candidate));
    }
    #[test]
    fn non_completion_partial_selection_learns_only_raw_consumed_prefix() {
        let decoder = decoder();
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(decoder.clone(), Some(learning.clone()));
        session.insert("ni'hao").unwrap();
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "你" && c.consumed == 3)
            .unwrap();
        let selected = session.candidates()[index].clone();
        assert_ne!(selected.source, CandidateSource::Completion);
        let revision = learning.revision();
        assert_eq!(session.select(index).as_deref(), Some("你"));
        assert_eq!(session.input(), "hao");
        assert_eq!(session.cursor(), 3);
        assert_eq!(session.context(), "你");
        assert_eq!(learning.revision(), revision + 1);
        assert_eq!(learning.counts("ni'", std::slice::from_ref(&selected)), [1]);
        let mut normalized = selected.clone();
        normalized.consumed = 2;
        assert_eq!(learning.counts("ni", &[normalized]), [0]);
        let mut full = selected;
        full.consumed = 6;
        assert_eq!(learning.counts("ni'hao", &[full]), [0]);

        let mut fresh = Session::new(decoder, Some(learning));
        fresh.insert("ni'").unwrap();
        assert!(
            fresh.candidates().iter().any(|c| {
                c.text == "你" && c.consumed == 3 && c.source == CandidateSource::User
            })
        );
    }
    #[test]
    fn promotion_preserves_candidate_bits_input_cursor_context_and_revision() {
        let dictionary = Dictionary::from_rime_reader(Cursor::new(
            "...\n你好\tni hao\t99\n拟好\tni hao\t90\n你\tni\t10\n好\thao\t10\n",
        ))
        .unwrap();
        let mut session = Session::new(Arc::new(Decoder::new(dictionary)), None);
        session.insert("nihao").unwrap();
        session.set_context("上下文");
        session.move_cursor(-2);
        let original = session.snapshot();
        let index = original
            .candidates
            .iter()
            .position(|c| {
                c.text == "拟好" && c.consumed == 5 && c.source != CandidateSource::Completion
            })
            .unwrap();
        assert!(index > 0);
        assert!(session.apply_promotion(original.revision, index));
        let mut expected = original.candidates.clone();
        expected[..=index].rotate_right(1);
        assert_eq!(session.candidates(), expected);
        for (candidate, expected) in session.candidates().iter().zip(&expected) {
            assert_eq!(candidate.score.to_bits(), expected.score.to_bits());
        }
        assert_eq!(session.revision(), original.revision);
        assert_eq!(session.input(), original.input);
        assert_eq!(session.context(), original.context);
        assert_eq!(session.cursor(), 3);
        assert!(!session.apply_promotion(original.revision.wrapping_add(1), usize::MAX));
        assert_eq!(session.candidates(), expected);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn current_promotion_with_invalid_index_exposes_internal_violation() {
        let mut session = Session::new(decoder(), None);
        session.insert("ni").unwrap();
        session.apply_promotion(session.revision(), session.candidates().len());
    }
    #[test]
    fn append_edit_selection_and_reset_match_fresh_sessions() {
        let decoder = decoder();
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(decoder.clone(), Some(learning));
        for text in ["n", "i", "ha", "o"] {
            session.insert(text).unwrap();
            assert_fresh(&session);
        }
        let previous = session.snapshot();
        session.set_context("上下文");
        assert!(!session.apply_promotion(previous.revision, 0));
        session.move_cursor(-3);
        session.insert("NI").unwrap();
        assert_fresh(&session);
        session.delete_forward();
        assert_fresh(&session);
        session.backspace();
        assert_fresh(&session);
        session.move_cursor(128);
        session.insert("'").unwrap();
        assert_fresh(&session);
        session.insert("an").unwrap();
        assert_fresh(&session);
        session.commit_raw();
        assert_fresh(&session);
        session.reset();
        session.insert("nihao").unwrap();
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "你" && c.consumed == 2)
            .unwrap();
        assert_eq!(session.select(index).as_deref(), Some("你"));
        assert_fresh(&session);
        session.reset();
        session.insert("ni").unwrap();
        assert_fresh(&session);
        let mut other = Session::new(decoder, None);
        other.insert("ni").unwrap();
        assert!(
            other
                .candidates()
                .iter()
                .all(|c| c.source != CandidateSource::User)
        );
        let stale = session.snapshot();
        session.reset();
        assert!(!session.apply_promotion(stale.revision, 0));
        assert!(session.candidates().is_empty());
    }
    #[test]
    fn prefix_edits_repeated_letters_case_and_full_clear_boundaries_match_fresh() {
        let d = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new("...\n你\tni\t10\n好\thao\t10\n")).unwrap(),
        ));
        let mut s = Session::new(d, None);
        s.insert("nnnnnnnn").unwrap();
        assert_fresh(&s);
        s.move_cursor(-4);
        s.insert("N").unwrap();
        assert_eq!(s.input(), "nnnnNnnnn");
        assert_fresh(&s);
        s.backspace();
        assert_fresh(&s);
        s.delete_forward();
        assert_fresh(&s);
        s.insert("'").unwrap();
        assert_fresh(&s);
        s.backspace();
        assert_fresh(&s);
        s.insert("h").unwrap();
        assert_fresh(&s);
        s.set_context("上下文");
        assert_fresh(&s);
        let original = s.snapshot();
        assert!(s.insert("中").is_err());
        assert_eq!(s.input(), original.input);
        assert_eq!(s.revision(), original.revision);
        assert_fresh(&s);
        let text = s.input().to_owned();
        assert_eq!(s.commit_raw().as_deref(), Some(text.as_str()));
        assert!(s.input().is_empty());
        assert_fresh(&s);
        s.insert("nihaox").unwrap();
        let index = s
            .candidates()
            .iter()
            .position(|c| c.consumed < s.input().len())
            .unwrap();
        let consumed = s.candidates()[index].consumed;
        let before = s.input().to_owned();
        assert!(s.select(index).is_some());
        assert_eq!(s.input(), &before[consumed..]);
        assert_fresh(&s);
        s.reset();
        assert!(s.context().is_empty());
        assert_fresh(&s);
        s.insert(&"n".repeat(128)).unwrap();
        assert_fresh(&s);
        let revision = s.revision();
        assert!(s.insert("n").is_err());
        assert_eq!(s.revision(), revision);
        assert_fresh(&s);
        s.backspace();
        assert_fresh(&s);
        s.reset();
        assert!(s.input().is_empty());
        assert_fresh(&s);
    }
}

#[cfg(test)]
mod transposition_tests {
    use super::*;
    use crate::Dictionary;
    use std::io::Cursor;

    fn session() -> Session {
        Session::new(Arc::new(Decoder::new(Dictionary::from_rime_reader(Cursor::new(
            "...\n那\tna\t100\n拿\tna\t90\n安\ta n\t1000000\n按\ta n\t900000\n纳\tnax\t80\n嗯\tn\t70\n",
        )).unwrap())), Some(Arc::new(Learning::new(false))))
    }

    #[test]
    fn transposition_ranking_protects_slots_and_full_candidate_groups() {
        let mut s = session();
        s.insert("na").unwrap();
        assert_eq!(
            s.candidates
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            ["那", "拿", "安", "按", "纳", "安", "按", "嗯", "那", "拿"]
        );
        assert_eq!(
            s.candidates[5].source,
            CandidateSource::MissingLetter {
                at: 0,
                to: b'a',
                ends: vec![1, 2]
            }
        );
        assert_eq!(
            s.candidates[6].source,
            CandidateSource::MissingLetter {
                at: 0,
                to: b'a',
                ends: vec![1, 2]
            }
        );
        assert!(s.candidates[5..].iter().all(|c| c.consumed == 1));
        assert_eq!(s.candidates[7].source, CandidateSource::ExactDictionary);
        assert!(s.candidates[7].score > s.candidates[8].score);
        assert!(
            s.candidates[8..]
                .iter()
                .all(|c| c.source == CandidateSource::Completion)
        );
        let before = s.snapshot();
        let correction = s
            .candidates
            .iter()
            .position(|c| c.text == "按" && c.consumed == 2)
            .unwrap();
        assert!(s.apply_promotion(before.revision, correction));
        let mut expected = before.candidates;
        expected.swap(2, 3);
        assert_eq!(s.candidates, expected);
        assert_eq!(s.candidates[0].text, "那");
        assert_eq!(s.candidates[1].text, "拿");
        let correction = s
            .candidates
            .iter()
            .position(|c| c.text == "安" && c.consumed == 2)
            .unwrap();
        assert!(s.apply_promotion(s.revision(), correction));
        for learned in [false, true] {
            if learned {
                let i = s.candidates.iter().position(|c| c.text == "安").unwrap();
                assert_eq!(s.select(i).as_deref(), Some("安"));
                s.insert("na").unwrap();
                assert_eq!(s.candidates[0].source, CandidateSource::User);
            }
            let before = s.snapshot();
            let index = s.candidates.iter().position(|c| c.text == "拿").unwrap();
            assert!(s.apply_promotion(before.revision, index));
            let mutable = |c: &Candidate| {
                c.consumed == 2
                    && c.source != CandidateSource::User
                    && c.source != CandidateSource::Completion
            };
            for (old, new) in before.candidates.iter().zip(&s.candidates) {
                if !mutable(old) {
                    assert_eq!(old, new);
                }
            }
            let groups = s
                .candidates
                .iter()
                .filter(|c| mutable(c))
                .map(|c| c.source.is_correction())
                .collect::<Vec<_>>();
            assert!(groups.windows(2).all(|pair| pair[0] <= pair[1]));
            let ordinary = s
                .candidates
                .iter()
                .filter(|c| mutable(c) && !c.source.is_correction())
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>();
            assert_eq!(ordinary, ["拿", "那"]);
        }
    }

    #[test]
    fn transposition_partial_selection_learning_and_stale_revision() {
        let mut s = session();
        s.insert("na").unwrap();
        let original = s.snapshot();
        s.insert("q").unwrap();
        assert!(!s.apply_promotion(original.revision, 0));
        let i = s
            .candidates
            .iter()
            .position(|c| c.text == "安" && c.consumed == 2)
            .unwrap();
        assert_eq!(s.select(i).as_deref(), Some("安"));
        assert_eq!(s.input(), "q");
        assert_eq!(s.context(), "安");
        s.reset();
        s.insert("na").unwrap();
        assert_eq!(s.candidates[0].text, "安");
        assert_eq!(s.candidates[0].source, CandidateSource::User);
    }
}

#[cfg(test)]
mod neighbor_tests {
    use super::*;
    use crate::Dictionary;
    use std::io::Cursor;
    #[test]
    fn neighbor_identity_menu_selection_learning_and_revision() {
        let d = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n你好\tni hao\t100000\n拟好\tno hao\t1\n你\tni\t90\n",
            ))
            .unwrap(),
        ));
        let mut s = Session::new(d, Some(Arc::new(Learning::new(false))));
        s.insert("nohao").unwrap();
        assert_eq!(s.candidates[0].text, "拟好");
        let before = s.snapshot();
        let i = s
            .candidates
            .iter()
            .position(|c| {
                matches!(c.source, CandidateSource::NeighborSubstitution { .. }) && c.text == "你好"
            })
            .unwrap();

        assert!(s.apply_promotion(before.revision, i));
        assert_eq!(s.candidates[0].text, "拟好");
        for (old, new) in before.candidates.iter().zip(&s.candidates) {
            if old.consumed < 5
                || matches!(
                    old.source,
                    CandidateSource::User | CandidateSource::Completion
                )
            {
                assert_eq!(old, new);
            }
        }
        s.reset();
        s.insert("no'haoq").unwrap();
        assert!(!s.apply_promotion(before.revision, 0));
        let i = s
            .candidates
            .iter()
            .position(|c| c.text == "你好" && c.consumed == 6)
            .unwrap();
        assert_eq!(s.select(i).as_deref(), Some("你好"));
        assert_eq!(s.input(), "q");
        s.reset();
        s.insert("no'hao").unwrap();
        assert_eq!(s.candidates[0].text, "你好");
        assert_eq!(s.candidates[0].source, CandidateSource::User);
        s.reset();
        s.insert("nihao").unwrap();
        assert_ne!(
            s.candidates[0].source,
            CandidateSource::User,
            "learning uses raw typo, not corrected reading"
        );
    }
}

#[cfg(test)]
mod missing_letter_tests {
    use super::*;
    use crate::Dictionary;
    use std::io::Cursor;
    fn session() -> Session {
        Session::new(
            Arc::new(Decoder::new(
                Dictionary::from_rime_reader(Cursor::new(
                    "...\n你好\tni hao\t100000\n您好\tni hao\t9000\n拟好\ti hao\t1\n你火\tni huo\t90\n",
                ))
                .unwrap(),
            )),
            Some(Arc::new(Learning::new(false))),
        )
    }
    #[test]
    fn missing_identity_ranking_stale_partial_selection_and_raw_learning() {
        let mut s = session();
        s.insert("ihao").unwrap();
        let before = s.snapshot();
        assert_eq!(s.candidates[0].text, "拟好");
        let i = s
            .candidates
            .iter()
            .position(|c| {
                c.text == "你好" && matches!(c.source, CandidateSource::MissingLetter { .. })
            })
            .unwrap();

        assert!(s.apply_promotion(before.revision, i));
        assert_eq!(s.candidates[0].text, "拟好");
        for (a, b) in before.candidates.iter().zip(&s.candidates) {
            if a.consumed < 4
                || matches!(
                    a.source,
                    CandidateSource::Completion | CandidateSource::User
                )
            {
                assert_eq!(a, b);
            }
        }
        s.insert("q").unwrap();
        assert!(!s.apply_promotion(before.revision, 0));
        let i = s
            .candidates
            .iter()
            .position(|c| {
                c.text == "你好"
                    && c.consumed == 4
                    && matches!(c.source, CandidateSource::MissingLetter { .. })
            })
            .unwrap();
        assert_eq!(s.select(i).as_deref(), Some("你好"));
        assert_eq!(s.input(), "q");
        s.reset();
        s.insert("ihao").unwrap();
        assert_eq!(s.candidates[0].text, "你好");
        assert_eq!(s.candidates[0].source, CandidateSource::User);
        let before = s.snapshot();
        let index = s
            .candidates
            .iter()
            .position(|c| {
                c.consumed == 4
                    && c.source != CandidateSource::User
                    && c.source != CandidateSource::Completion
            })
            .unwrap();
        assert!(s.apply_promotion(before.revision, index));
        assert_eq!(s.candidates[0], before.candidates[0]);
        s.reset();
        s.insert("nihao").unwrap();
        assert_ne!(s.candidates[0].source, CandidateSource::User);
        s.reset();
        s.insert("n'haoq").unwrap();
        let i = s
            .candidates
            .iter()
            .position(|c| c.text == "你好" && c.consumed == 5)
            .unwrap();
        assert_eq!(s.select(i).as_deref(), Some("你好"));
        assert_eq!(s.input(), "q");
        s.reset();
        s.insert("n'hao").unwrap();
        assert_eq!(s.candidates[0].text, "你好");
        assert_eq!(s.candidates[0].source, CandidateSource::User);
    }
    #[test]
    fn missing_tail_prediction_is_completion_and_never_learned() {
        let mut s = session();
        s.insert("niha").unwrap();
        let i = s
            .candidates
            .iter()
            .position(|c| c.text == "你好" && c.consumed == 4)
            .unwrap();
        assert_eq!(s.candidates[i].source, CandidateSource::Completion);
        assert_eq!(s.learning.as_ref().unwrap().revision(), 0);
        assert_eq!(s.select(i).as_deref(), Some("你好"));
        assert_eq!(s.learning.as_ref().unwrap().revision(), 0);
        s.reset();
        s.insert("niha'").unwrap();
        let i = s
            .candidates
            .iter()
            .position(|c| c.text == "你好" && c.consumed == 5)
            .unwrap();
        assert_eq!(
            s.candidates[i].source,
            CandidateSource::MissingLetter {
                at: 4,
                to: b'o',
                ends: vec![2, 5]
            }
        );
        assert_eq!(s.select(i).as_deref(), Some("你好"));
        assert_eq!(s.learning.as_ref().unwrap().revision(), 1);
        s.insert("niha'").unwrap();
        assert_eq!(s.candidates[0].source, CandidateSource::User);
    }
}

#[cfg(test)]
mod explicit_boundary_tests {
    use super::*;
    use crate::Dictionary;
    use std::io::Cursor;
    #[test]
    fn boundary_cursor_edits_keep_fresh_tuple_bits() {
        let d = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new("...\n前\tni\t10\n后\thao\t10\n")).unwrap(),
        ));
        let mut s = Session::new(d.clone(), None);
        let check = |s: &Session| {
            let mut fresh = Session::new(d.clone(), None);
            fresh.insert(s.input()).unwrap();
            let tuples = |cs: &[Candidate]| {
                cs.iter()
                    .map(|c| {
                        (
                            c.text.clone(),
                            c.consumed,
                            c.source.clone(),
                            c.score.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                tuples(s.candidates()),
                tuples(fresh.candidates()),
                "{}",
                s.input()
            );
        };
        for ch in ["n", "i", "'", "h", "a", "o"] {
            s.insert(ch).unwrap();
            check(&s);
        }
        s.backspace();
        check(&s);
        s.insert("o").unwrap();
        check(&s);
        s.move_cursor(-3);
        s.backspace();
        check(&s); // Delete the interior apostrophe.
        s.insert("'").unwrap();
        check(&s);
        s.move_cursor(-1);
        s.delete_forward();
        check(&s);
        s.insert("'").unwrap();
        check(&s);
        s.move_cursor(128);
        s.insert("'").unwrap();
        check(&s);
        s.backspace();
        check(&s);
    }
    #[test]
    fn boundary_partial_survives_static_learning_and_valid_model_proposals() {
        let mut rows = String::from("...\n前\tni\t10\n后\thao\t10\n候\tni hao\t9\n");
        for i in 0..40 {
            rows.push_str(&format!("长{i:02}\tni ha\t{}\n", 1000 - i));
        }
        let d = Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(rows)).unwrap(),
        ));
        let learning = Arc::new(Learning::new(false));
        let mut s = Session::new(d.clone(), Some(learning.clone()));
        s.insert("ni'hao").unwrap();
        let partial = s
            .candidates()
            .iter()
            .position(|c| c.text == "前" && c.consumed == 3)
            .unwrap();
        let first_partial = s.candidates().iter().position(|c| c.consumed < 6).unwrap();
        assert_eq!(partial, first_partial);
        let before = s.candidates().to_vec();
        let index = s
            .candidates()
            .iter()
            .rposition(|c| c.consumed == 6 && c.source != CandidateSource::Completion)
            .unwrap();
        assert!(index > 0);
        assert!(s.apply_promotion(s.revision(), index));
        assert!(s.candidates().iter().any(|c| c == &before[partial]));
        let index = s
            .candidates()
            .iter()
            .position(|c| c.text == "前" && c.consumed == 3)
            .unwrap();
        assert_eq!(s.select(index).as_deref(), Some("前"));
        assert_eq!(s.input(), "hao");
        let mut fresh = Session::new(d, Some(learning));
        fresh.insert("ni'hao").unwrap();
        let partial = fresh
            .candidates()
            .iter()
            .position(|c| c.text == "前" && c.consumed == 3)
            .unwrap();
        assert_eq!(fresh.candidates()[partial].source, CandidateSource::User);
        assert!(
            fresh.candidates()[..partial]
                .iter()
                .all(|c| c.consumed == 6)
        );
    }
}
