use crate::dictionary::{Dictionary, Entry};
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CandidateSource {
    ExactDictionary,
    Composed,
    Completion,
    User,
    /// One lowercase byte inserted before a typed raw byte; ends is the recalled reading.
    MissingLetter {
        at: usize,
        to: u8,
        ends: Vec<u16>,
    },
    /// One raw ASCII byte replaced by a same-row QWERTY neighbor.
    NeighborSubstitution {
        at: usize,
        to: u8,
        ends: Vec<u16>,
    },
    /// Adjacent raw ASCII bytes swapped; ends records the recalled reading.
    Transposition {
        at: usize,
        ends: Vec<u16>,
    },
}
impl CandidateSource {
    pub(crate) fn is_correction(&self) -> bool {
        matches!(
            self,
            Self::Transposition { .. }
                | Self::NeighborSubstitution { .. }
                | Self::MissingLetter { .. }
        )
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub consumed: usize,
    pub source: CandidateSource,
    pub score: f32,
}

pub const MAX_INPUT_BYTES: usize = 128;
const BEAM: usize = 12;
const FULL_LIMIT: usize = 24;
const CANDIDATE_LIMIT: usize = 32;
/// Dictionary entries offered per (start, end) span.
const SPAN_ENTRIES: usize = 64;
/// Verbatim readings reserved beside the composed beam.
const RESERVED_READINGS: usize = 8;
/// Representatives kept per distinct syllabification among reserved readings.
const READING_REPRESENTATIVES: usize = 2;
/// Extra slots shared by all typo corrections, which only short raw input gets.
const CORRECTION_LIMIT: usize = 8;
const MAX_CORRECTION_BYTES: usize = 24;

#[derive(Debug)]
pub struct Decoder {
    dictionary: Dictionary,
}
#[derive(Default)]
struct State {
    text: String,
    score: f32,
    words: usize,
    completion: bool,
}
/// Session-owned prefix beams. Completion states never become parents.
#[derive(Default)]
pub(crate) struct DecodeCache {
    input: String,
    beams: Vec<Vec<State>>,
}
impl Decoder {
    pub fn new(dictionary: Dictionary) -> Self {
        Self { dictionary }
    }
    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }
    pub fn decode(&self, input: &str) -> Vec<Candidate> {
        self.decode_cached(input, &mut DecodeCache::default())
    }
    pub(crate) fn decode_cached(&self, input: &str, cache: &mut DecodeCache) -> Vec<Candidate> {
        let Some(input) = scan_input(input) else {
            *cache = DecodeCache::default();
            return Vec::new();
        };
        let mut candidates = self.decode_ordinary_cached(&input, cache);
        // A bare initial is still being typed; keep its ordinary completions.
        if input.raw.len() <= MAX_CORRECTION_BYTES
            && input.key.len() > 1
            && !matches!(input.key.as_str(), "zh" | "ch" | "sh")
        {
            let limit = candidates.len() + CORRECTION_LIMIT;
            self.append_swaps_and_neighbors(&input, &mut candidates, limit);
            self.append_missing_letters(&input, &mut candidates, limit);
        }
        self.append_syllable_characters(&input, &mut candidates);
        candidates
    }

    fn decode_ordinary_cached(
        &self,
        input: &ScannedInput<'_>,
        cache: &mut DecodeCache,
    ) -> Vec<Candidate> {
        let reuse = !cache.beams.is_empty()
            && !input.raw.contains('\'')
            && input.raw.len() > cache.input.len()
            && input.raw.starts_with(&cache.input);
        if !reuse {
            cache.beams.clear();
            cache.beams.push(vec![State::default()]);
        }
        let first_end = cache.beams.len();
        cache.beams.resize_with(input.key.len() + 1, Vec::new);
        let mixed = self.fill_beams(&input.key, &input.boundaries, first_end, &mut cache.beams);
        cache.input.clear();
        cache.input.push_str(input.raw);
        let reserved = reserve_readings(self.dictionary.exact(&input.key), input);
        assemble_picks(&mixed, &cache.beams, &reserved, input)
    }

    /// Extends beams for spans the cache does not cover yet, and returns the
    /// final-span beam, the only one that mixes in completions.
    fn fill_beams(
        &self,
        key: &str,
        boundaries: &[usize],
        first_end: usize,
        beams: &mut [Vec<State>],
    ) -> Vec<State> {
        let n = key.len();
        let mut mixed = Vec::new();
        // Preserve each target's original start/entry/parent update order. Only
        // the current final target interleaves exact and completion entries.
        for end in first_end..=n {
            let (parents, targets) = beams.split_at_mut(end);
            let exact = &mut targets[0];
            let mut abbreviate = false;
            for (start, previous) in parents.iter().enumerate() {
                if previous.is_empty() {
                    continue;
                }
                let matched = if start == 0 {
                    let (matched, is_prefix) = self.dictionary.exact_prefix(&key[..end]);
                    abbreviate = !is_prefix;
                    matched
                } else {
                    self.dictionary.exact(&key[start..end])
                };
                for entry in matched
                    .iter()
                    .filter(|entry| allowed(entry, start, end, boundaries))
                    .take(SPAN_ENTRIES)
                {
                    for state in previous {
                        if end == n {
                            offer(&mut mixed, state, entry, false);
                        }
                        offer(exact, state, entry, false);
                    }
                }
                if end == n {
                    for entry in self
                        .dictionary
                        .completions(&key[start..])
                        .filter(|entry| allowed(entry, start, n, boundaries))
                        .take(SPAN_ENTRIES)
                    {
                        for state in previous {
                            offer(&mut mixed, state, entry, true);
                        }
                    }
                }
            }
            // Initials for every syllable form a full abbreviation, not a completion;
            // a span that still prefixes some pinyin stays on the exact path.
            if abbreviate
                && let Some(pieces) = abbreviation(&self.dictionary, &key[..end], boundaries)
            {
                let initials: Vec<u8> = pieces
                    .iter()
                    .map(|(typed, _)| typed.as_bytes()[0])
                    .collect();
                let root = State::default();
                for entry in self
                    .dictionary
                    .by_initials(&initials)
                    .filter(|entry| abbreviation_matches(entry, &pieces))
                {
                    if end == n {
                        offer(&mut mixed, &root, entry, false);
                    }
                    offer(exact, &root, entry, false);
                }
            }
        }
        mixed
    }

    /// One adjacent swap, then one same-row neighbor substitution: at most 276 and
    /// 600 exact lookups for 24 letters, 64 rows each. Earlier tuples, including
    /// completions, win.
    fn append_swaps_and_neighbors(
        &self,
        input: &ScannedInput<'_>,
        candidates: &mut Vec<Candidate>,
        limit: usize,
    ) {
        let ScannedInput {
            key,
            raw_ends,
            boundaries,
            ..
        } = input;
        let key = key.as_bytes();
        // Appends unseen exact readings of an edited prefix; true once the budget is spent.
        let mut push = |edited: &[u8], source: &dyn Fn(Vec<u16>) -> CandidateSource| {
            let consumed = raw_ends[edited.len()];
            let edited = std::str::from_utf8(edited).expect("edits keep ASCII");
            for entry in self.dictionary.exact(edited).iter().take(SPAN_ENTRIES) {
                if allowed(entry, 0, edited.len(), boundaries)
                    && !candidates
                        .iter()
                        .any(|c| c.text == *entry.text && c.consumed == consumed)
                {
                    candidates.push(Candidate {
                        text: entry.text.to_string(),
                        consumed,
                        source: source(entry.ends.to_vec()),
                        score: extended_score(0.0, entry, false),
                    });
                    if candidates.len() == limit {
                        return true;
                    }
                }
            }
            false
        };
        for end in (2..=key.len()).rev() {
            for left in 0..end - 1 {
                let at = raw_ends[left];
                if raw_ends[left + 1] != at + 1 || key[left] == key[left + 1] {
                    continue;
                }
                let mut swapped = key[..end].to_vec();
                swapped.swap(left, left + 1);
                if push(&swapped, &|ends| CandidateSource::Transposition {
                    at,
                    ends,
                }) {
                    return;
                }
            }
        }
        // Spaces separate and pad the keyboard rows.
        const QWERTY: &[u8] = b" qwertyuiop asdfghjkl zxcvbnm ";
        for end in (1..=key.len()).rev() {
            for left in 0..end {
                let at = raw_ends[left];
                let index = QWERTY
                    .iter()
                    .position(|&byte| byte == key[left])
                    .expect("scanner validated lowercase");
                for to in [QWERTY[index - 1], QWERTY[index + 1]] {
                    if to == b' ' {
                        continue;
                    }
                    let mut substituted = key[..end].to_vec();
                    substituted[left] = to;
                    if push(&substituted, &|ends| {
                        CandidateSource::NeighborSubstitution { at, to, ends }
                    }) {
                        return;
                    }
                }
            }
        }
    }

    fn append_missing_letters(
        &self,
        input: &ScannedInput<'_>,
        candidates: &mut Vec<Candidate>,
        limit: usize,
    ) {
        if candidates.len() >= limit {
            return;
        }
        let ScannedInput {
            raw, key, raw_ends, ..
        } = input;
        // A callback order is not candidate priority. Keep only the best remaining
        // K tuples by raw consumption, static score, then insertion order ties.
        // Visit every branch, including later full prefixes.
        type Priority = (Reverse<usize>, usize, u8, usize, usize);
        let remaining = limit - candidates.len();
        let mut picks: Vec<(Priority, Candidate)> = Vec::with_capacity(remaining);
        self.dictionary
            .visit_missing_letter_prefixes(key, |end, p, to, rows| {
                let consumed = raw_ends[end];
                let first_slot = if p == 0 { 0 } else { raw_ends[p - 1] + 1 };
                for at in first_slot..=raw_ends[p] {
                    let mut corrected = raw[..consumed].to_owned();
                    corrected.insert(at, char::from(to));
                    let boundaries = scan_input(&corrected)
                        .expect("insertion preserves scanner invariants")
                        .boundaries;
                    for (row, entry) in rows.iter().take(SPAN_ENTRIES).enumerate() {
                        if !allowed(entry, 0, end + 1, &boundaries)
                            || candidates
                                .iter()
                                .any(|c| c.text == *entry.text && c.consumed == consumed)
                        {
                            continue;
                        }
                        let priority = (Reverse(consumed), p, to, at, row);
                        let completion = at == consumed;
                        let score = extended_score(0.0, entry, completion);
                        let compare = |(old_priority, old): &(Priority, Candidate)| {
                            old_priority
                                .0
                                .cmp(&priority.0)
                                .then_with(|| score.total_cmp(&old.score))
                                .then_with(|| old_priority.cmp(&priority))
                        };
                        if let Some(i) = picks
                            .iter()
                            .position(|(_, c)| c.text == *entry.text && c.consumed == consumed)
                        {
                            if compare(&picks[i]).is_le() {
                                continue;
                            }
                            picks.remove(i);
                        }
                        if picks.len() == remaining && compare(picks.last().unwrap()).is_le() {
                            continue;
                        }
                        let candidate = Candidate {
                            text: entry.text.to_string(),
                            consumed,
                            source: if completion {
                                CandidateSource::Completion
                            } else {
                                CandidateSource::MissingLetter {
                                    at,
                                    to,
                                    ends: entry.ends.to_vec(),
                                }
                            },
                            score,
                        };
                        if picks.len() == remaining {
                            picks.pop();
                        }
                        let i = picks.partition_point(|old| compare(old).is_lt());
                        picks.insert(i, (priority, candidate));
                    }
                }
            });
        candidates.extend(picks.into_iter().map(|(_, candidate)| candidate));
    }

    /// A single syllable, including an explicitly closed first segment, keeps all
    /// its characters browsable beyond the search beam.
    fn append_syllable_characters(
        &self,
        input: &ScannedInput<'_>,
        candidates: &mut Vec<Candidate>,
    ) {
        let end = input.boundaries.first().copied().unwrap_or(input.key.len());
        let consumed = input.raw_ends[end];
        for entry in self.dictionary.exact(&input.key[..end]) {
            if entry.ends.len() != 1
                || entry.text.chars().count() != 1
                || candidates.iter().any(|old| {
                    old.text == *entry.text
                        && old.consumed == consumed
                        && old.source != CandidateSource::Completion
                })
            {
                continue;
            }
            candidates.push(Candidate {
                text: entry.text.to_string(),
                consumed,
                source: CandidateSource::ExactDictionary,
                score: extended_score(0.0, entry, false),
            });
        }
    }
}
/// One validated view shared by ordinary recall, corrections and character browsing.
struct ScannedInput<'a> {
    raw: &'a str,
    /// Lowercase letters without apostrophes.
    key: String,
    /// Raw offset after the first i letters and any apostrophes that follow them,
    /// which is also the raw position of letter i.
    raw_ends: Vec<usize>,
    /// Key offsets of apostrophes.
    boundaries: Vec<usize>,
}
fn scan_input(input: &str) -> Option<ScannedInput<'_>> {
    let valid = !input.is_empty()
        && input.len() <= MAX_INPUT_BYTES
        && input.bytes().all(|b| b.is_ascii_alphabetic() || b == b'\'')
        && !input.starts_with('\'')
        && !input.contains("''");
    if !valid {
        return None;
    }
    let mut key = String::new();
    let mut raw_ends = vec![0];
    let mut boundaries = Vec::new();
    for (i, b) in input.bytes().enumerate() {
        if b == b'\'' {
            boundaries.push(key.len());
            *raw_ends.last_mut().unwrap() = i + 1;
        } else {
            key.push(char::from(b.to_ascii_lowercase()));
            raw_ends.push(i + 1);
        }
    }
    Some(ScannedInput {
        raw: input,
        key,
        raw_ends,
        boundaries,
    })
}
/// Reserve direct lexical readings alongside composition: a different
/// syllabification must not disappear solely because its beam score is lower.
fn reserve_readings<'a>(exact_full: &'a [Entry], input: &ScannedInput<'_>) -> Vec<&'a Entry> {
    let eligible = exact_full
        .iter()
        .filter(|entry| allowed(entry, 0, input.key.len(), &input.boundaries));
    // Two representatives retain alternatives sharing one syllabification.
    let mut readings = HashMap::new();
    let diverse = eligible
        .clone()
        .filter(|entry| {
            let count = readings.entry(entry.ends.as_ref()).or_insert(0usize);
            *count += 1;
            *count <= READING_REPRESENTATIVES
        })
        .take(RESERVED_READINGS);
    eligible.take(RESERVED_READINGS).chain(diverse).collect()
}
struct Pick<'a> {
    text: &'a str,
    consumed: usize,
    source: CandidateSource,
    score: f32,
}
fn assemble_picks(
    mixed: &[State],
    beams: &[Vec<State>],
    reserved: &[&Entry],
    input: &ScannedInput<'_>,
) -> Vec<Candidate> {
    let n = input.key.len();
    let full_len = input.raw.len();
    let mut picks = Vec::new();
    for end in (1..=n).rev() {
        let states = if end == n { mixed } else { &beams[end] };
        picks.extend(states.iter().map(|state| Pick {
            text: &state.text,
            consumed: input.raw_ends[end],
            source: if state.completion {
                CandidateSource::Completion
            } else if state.words == 1 {
                CandidateSource::ExactDictionary
            } else {
                CandidateSource::Composed
            },
            score: state.score,
        }));
    }
    picks.extend(reserved.iter().map(|entry| Pick {
        text: &entry.text,
        consumed: full_len,
        source: CandidateSource::ExactDictionary,
        score: extended_score(0.0, entry, false),
    }));
    picks.sort_by(|a, b| {
        b.consumed
            .cmp(&a.consumed)
            .then(b.score.total_cmp(&a.score))
            .then(a.text.cmp(b.text))
    });
    let mut seen = HashSet::new();
    picks.retain(|pick| seen.insert((pick.text, pick.consumed)));
    // Full conversions leave room for prefix selections, but reserved readings
    // always keep their slots.
    let reserved: HashSet<&str> = reserved.iter().map(|entry| entry.text.as_ref()).collect();
    let (full, partial): (Vec<_>, Vec<_>) = picks
        .into_iter()
        .partition(|pick| pick.consumed == full_len);
    let (mut menu, unreserved): (Vec<_>, Vec<_>) = full
        .into_iter()
        .partition(|pick| reserved.contains(pick.text));
    menu.extend(unreserved.into_iter().take(FULL_LIMIT - menu.len()));
    menu.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.text.cmp(b.text)));
    // Explicit syllable boundaries remain selectable before longer partial
    // readings ending inside the next syllable. Each group keeps pick order.
    let (closed, open): (Vec<_>, Vec<_>) = partial
        .into_iter()
        .partition(|pick| input.raw.as_bytes()[pick.consumed - 1] == b'\'');
    menu.extend(
        closed
            .into_iter()
            .chain(open)
            .take(CANDIDATE_LIMIT - menu.len()),
    );
    menu.into_iter()
        .map(|pick| Candidate {
            text: pick.text.to_owned(),
            consumed: pick.consumed,
            source: pick.source,
            score: pick.score,
        })
        .collect()
}
fn allowed(entry: &Entry, start: usize, end: usize, boundaries: &[usize]) -> bool {
    boundaries
        .iter()
        .all(|&b| b <= start || b > end || entry.ends.contains(&((b - start) as u16)))
}
/// Splits a span into complete syllables and bare initials (zh, ch, sh or one
/// letter). Explicit boundaries must fall between pieces, and at least one piece
/// must be an initial.
fn abbreviation<'a>(
    dictionary: &Dictionary,
    key: &'a str,
    boundaries: &[usize],
) -> Option<Vec<(&'a str, bool)>> {
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < key.len() {
        let rest = &key[start..];
        let (len, complete) = match dictionary.longest_syllable(rest) {
            Some(len) => (len, true),
            None if ["zh", "ch", "sh"]
                .iter()
                .any(|initial| rest.starts_with(initial)) =>
            {
                (2, false)
            }
            None => (1, false),
        };
        if boundaries
            .iter()
            .any(|&boundary| start < boundary && boundary < start + len)
        {
            return None;
        }
        pieces.push((&rest[..len], complete));
        start += len;
    }
    pieces
        .iter()
        .any(|&(_, complete)| !complete)
        .then_some(pieces)
}
fn abbreviation_matches(entry: &Entry, pieces: &[(&str, bool)]) -> bool {
    entry.ends.len() == pieces.len()
        && entry
            .syllables()
            .zip(pieces)
            .all(|(syllable, &(typed, complete))| {
                if complete {
                    syllable == typed
                } else {
                    syllable.starts_with(typed)
                }
            })
}
fn extended_score(parent_score: f32, entry: &Entry, completion: bool) -> f32 {
    parent_score + (entry.weight + 1.0).ln() / 14.0 - 2.0 - if completion { 0.8 } else { 0.0 }
}
fn ranks_before(a: &State, b: &State) -> bool {
    b.score
        .total_cmp(&a.score)
        .then(a.text.cmp(&b.text))
        .is_lt()
}
fn offer(states: &mut Vec<State>, parent: &State, entry: &Entry, completion: bool) {
    let score = extended_score(parent.score, entry, completion);
    // A full beam drops a lower extension before building its text, unless the
    // extension would replace a completion with the same text.
    if states.len() == BEAM
        && score.total_cmp(&states[BEAM - 1].score).is_lt()
        && (completion
            || !states.iter().any(|state| {
                state.completion
                    && state.text.strip_prefix(&parent.text) == Some(entry.text.as_ref())
            }))
    {
        return;
    }
    let mut text = parent.text.clone();
    text.push_str(&entry.text);
    push_state(
        states,
        State {
            text,
            score,
            words: parent.words + 1,
            completion,
        },
    );
}
fn push_state(states: &mut Vec<State>, state: State) {
    if let Some(index) = states
        .iter()
        .position(|existing| existing.text == state.text)
    {
        let existing = &states[index];
        let replace = (existing.completion && !state.completion)
            || (existing.completion == state.completion && state.score > existing.score);
        if !replace {
            return;
        }
        states.remove(index);
    } else if states.len() == BEAM {
        if !ranks_before(&state, &states[BEAM - 1]) {
            return;
        }
        states.pop();
    }
    let index = states.partition_point(|existing| ranks_before(existing, &state));
    states.insert(index, state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Session, learning::Learning};
    use std::{io::Cursor, sync::Arc};

    fn decoder(rows: &str) -> Decoder {
        Decoder::new(Dictionary::from_rime_reader(Cursor::new(format!("...\n{rows}"))).unwrap())
    }
    fn bits(cs: &[Candidate]) -> Vec<(&str, usize, CandidateSource, u32)> {
        cs.iter()
            .map(|c| {
                (
                    c.text.as_str(),
                    c.consumed,
                    c.source.clone(),
                    c.score.to_bits(),
                )
            })
            .collect()
    }
    fn ordinary(d: &Decoder, raw: &str) -> Vec<Candidate> {
        d.decode_ordinary_cached(&scan_input(raw).unwrap(), &mut DecodeCache::default())
    }
    /// The menu just before missing-letter recovery appends its picks.
    fn before_missing(d: &Decoder, raw: &str) -> Vec<Candidate> {
        let mut cs = ordinary(d, raw);
        let limit = cs.len() + CORRECTION_LIMIT;
        d.append_swaps_and_neighbors(&scan_input(raw).unwrap(), &mut cs, limit);
        cs
    }
    fn states(prefix: &str, count: usize) -> Vec<State> {
        (0..count)
            .map(|i| State {
                text: format!("{prefix}{i:02}"),
                score: 100.0 - i as f32,
                words: 1,
                completion: i == 1,
            })
            .collect()
    }
    fn exact_score(d: &Decoder, raw: &str, text: &str) -> u32 {
        d.decode(raw)
            .iter()
            .find(|c| c.text == text)
            .unwrap()
            .score
            .to_bits()
    }

    #[test]
    fn tail_completions_beyond_the_beam_keep_the_completion_score() {
        let rows: String = (0..BEAM + 1)
            .map(|i| {
                let text = char::from_u32(0x4e00 + i as u32).unwrap();
                format!("{text}\tde\t{}\n", 100 - i)
            })
            .collect();
        let d = decoder(&rows);
        let mut candidates = ordinary(&d, "d");
        assert_eq!(candidates.len(), BEAM);
        let limit = candidates.len() + CORRECTION_LIMIT;
        d.append_missing_letters(&scan_input("d").unwrap(), &mut candidates, limit);
        assert_eq!(candidates.len(), BEAM + 1);
        for candidate in &candidates {
            let entry = d
                .dictionary
                .exact("de")
                .iter()
                .find(|entry| *entry.text == candidate.text)
                .unwrap();
            assert_eq!(candidate.consumed, 1);
            assert_eq!(candidate.source, CandidateSource::Completion);
            assert_eq!(
                candidate.score.to_bits(),
                extended_score(0.0, entry, true).to_bits()
            );
        }
    }

    #[test]
    fn single_initials_keep_ordinary_completions_without_learning() {
        let d = Arc::new(decoder(
            "的\tde\t100\n第\tdi\t10\n你\tni\t100\n片\tpian\t100\n人\tren\t100\n我\two\t100\n是\tshi\t100\n中\tzhong\t100\n吃\tchi\t100\n啊\ta\t1e30\n饿\te\t1e30\n哦\to\t1e30\n嗯\ten\t1e30\n",
        ));
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(d.clone(), Some(learning.clone()));
        for (raw, expected) in [
            ("d", "的"),
            ("n", "你"),
            ("p", "片"),
            ("r", "人"),
            ("w", "我"),
            ("s", "是"),
            ("zh", "中"),
            ("ch", "吃"),
            ("sh", "是"),
        ] {
            session.insert(raw).unwrap();
            assert_eq!(
                bits(session.candidates()),
                bits(&ordinary(&d, raw)),
                "{raw}"
            );
            assert_eq!(session.candidates()[0].text, expected, "{raw}");
            assert!(session.candidates().iter().all(|candidate| {
                candidate.consumed == raw.len() && candidate.source == CandidateSource::Completion
            }));
            assert_eq!(session.select(0).as_deref(), Some(expected));
            assert_eq!(session.input(), "");
            assert_eq!(learning.revision(), 0, "{raw}");
        }
        for (raw, expected) in [("a", "啊"), ("e", "饿"), ("o", "哦"), ("de", "的")] {
            session.insert(raw).unwrap();
            assert_eq!(session.candidates()[0].text, expected, "{raw}");
            assert_eq!(
                session.candidates()[0].source,
                CandidateSource::ExactDictionary,
                "{raw}"
            );
            session.reset();
        }
    }

    #[test]
    fn reserves_alternative_readings_when_composition_crowds_the_beam() {
        let mut rows = String::new();
        for i in 0..8 {
            rows.push_str(&format!("单{i}\tlian\t{}\n", 100 - i));
            rows.push_str(&format!("前{i}\tli\t1000000000000000000000000000000\n"));
            rows.push_str(&format!("后{i}\tan\t1000000000000000000000000000000\n"));
        }
        rows.push_str("读法甲\tli an\t50\n读法乙\tli an\t49\n");
        let d = decoder(&rows);
        let candidates = d.decode("lian");
        for text in ["读法甲", "读法乙"] {
            assert!(candidates.iter().any(|c| c.text == text
                && c.consumed == 4
                && c.source == CandidateSource::ExactDictionary));
        }
        let prior = ordinary(&d, "lian");
        assert!(prior.len() <= CANDIDATE_LIMIT);
        assert_eq!(&candidates[..prior.len()], prior);
        assert!(candidates.len() <= prior.len() + CORRECTION_LIMIT);
        assert!(candidates[prior.len()..].iter().all(|c| matches!(
            c.source,
            CandidateSource::MissingLetter { .. } | CandidateSource::Completion
        )));
        let split = d.decode("li'an");
        assert!(split.iter().all(|c| !c.text.starts_with('单')));
        for text in ["读法甲", "读法乙"] {
            assert!(split.iter().any(|c| c.text == text && c.consumed == 5));
        }
    }

    fn cache_decoder() -> Decoder {
        let mut rows = String::from("甲\ta\t1\n甲\taa\t99999999\n乙\tab\t100\n丙\tb\t10\n");
        for i in 0..20 {
            rows.push_str(&format!("词{i}\ta b\t{}\n", 1000 + i));
        }
        decoder(&rows)
    }
    #[test]
    fn appended_frames_keep_only_exact_parents_and_match_fresh() {
        let d = cache_decoder();
        let mut cache = DecodeCache::default();
        for input in [
            "a", "aa", "aab", "aabaa", "aabaab", "a'ab", "a'aba", "a'", "a", "", "A", "''",
        ] {
            assert_eq!(
                bits(&d.decode_cached(input, &mut cache)),
                bits(&d.decode(input)),
                "{input}"
            );
            assert!(cache.beams.iter().flatten().all(|state| !state.completion));
            assert!(cache.beams.iter().all(|beam| beam.len() <= BEAM));
        }
        assert!(cache.input.is_empty());
        assert!(cache.beams.is_empty());
    }
    #[test]
    fn invalid_inputs_decode_to_nothing_and_release_the_cache() {
        let d = cache_decoder();
        let mut cache = DecodeCache::default();
        for input in [
            "".to_owned(),
            "a1".to_owned(),
            "中".to_owned(),
            "'a".to_owned(),
            "a''b".to_owned(),
            "ab''".to_owned(),
            "a".repeat(MAX_INPUT_BYTES + 1),
        ] {
            d.decode_cached("aab", &mut cache);
            assert!(!cache.beams.is_empty());
            assert!(d.decode_cached(&input, &mut cache).is_empty(), "{input}");
            assert_eq!(cache.input.capacity(), 0);
            assert_eq!(cache.beams.capacity(), 0);
        }
    }
    #[test]
    fn beam_replacement_and_equal_score_rules_survive_truncation() {
        let state = |text: &str, score, words, completion| State {
            text: text.into(),
            score,
            words,
            completion,
        };
        let mut states = Vec::new();
        push_state(&mut states, state("same", 9.0, 1, true));
        for i in 0..11 {
            push_state(&mut states, state(&format!("x{i}"), 10.0, 1, false));
        }
        push_state(&mut states, state("same", -1.0, 2, false));
        let same = states.iter().find(|s| s.text == "same").unwrap();
        assert_eq!(same.score.to_bits(), (-1.0f32).to_bits());
        push_state(&mut states, state("same", -1.0, 3, false));
        assert_eq!(states.iter().find(|s| s.text == "same").unwrap().words, 2);
        push_state(&mut states, state("new", 11.0, 1, false));
        assert!(!states.iter().any(|s| s.text == "same"));
        push_state(&mut states, state("new", 99.0, 4, true));
        assert!(!states.iter().find(|s| s.text == "new").unwrap().completion);

        // A full beam skips lower extensions before building text, except one
        // replacing a completion with the same text.
        let mut beam = Vec::new();
        push_state(&mut beam, state("ab", 9.0, 1, true));
        for i in 0..11 {
            push_state(&mut beam, state(&format!("x{i}"), 10.0, 1, false));
        }
        let parent = state("a", -5.0, 1, false);
        let entry = |text: &str| Entry {
            text: text.into(),
            key: "k".into(),
            ends: Box::from([1u16]),
            weight: 0.0,
        };
        offer(&mut beam, &parent, &entry("c"), true);
        offer(&mut beam, &parent, &entry("b"), false);
        assert_eq!(beam.len(), BEAM);
        assert!(!beam.iter().any(|s| s.text == "ac"));
        assert!(
            beam.iter()
                .any(|s| s.text == "ab" && !s.completion && s.words == 2)
        );
    }

    #[test]
    fn transposition_preserves_raw_offsets_boundaries_and_ordinary_tuples() {
        let d = decoder("你好\tni hao\t100\n你\tni\t90\n");
        let score = exact_score(&d, "nihao", "你好");
        for (raw, consumed) in [("inhao", 5), ("in'hao'", 7), ("in'haoq", 6)] {
            let candidates = d.decode(raw);
            let hit = candidates
                .iter()
                .find(|c| c.text == "你好" && c.consumed == consumed)
                .unwrap();
            assert_eq!(
                hit.source,
                CandidateSource::Transposition {
                    at: 0,
                    ends: vec![2, 5]
                }
            );
            assert_eq!(hit.score.to_bits(), score);
        }
        assert!(
            d.decode("i'nhao")
                .iter()
                .all(|c| !matches!(c.source, CandidateSource::Transposition { .. }))
        );
        assert!(d.decode("inh'ao").iter().all(|c| c.text != "你好"));
        for raw in ["nihao", "inh", "inhao", "in'hao'"] {
            let prior = ordinary(&d, raw);
            assert_eq!(&d.decode(raw)[..prior.len()], prior);
        }
        let candidates = decoder("安\tnax\t10\n安\ta n\t100\n").decode("na");
        let hits: Vec<_> = candidates
            .iter()
            .filter(|c| c.text == "安" && c.consumed == 2)
            .collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, CandidateSource::Completion);
    }
    #[test]
    fn transposition_is_bounded_and_cached_results_match() {
        let rows: String = (0..100)
            .map(|i| format!("词{i}\ta n\t{}\n", 100 - i))
            .collect();
        let d = decoder(&rows);
        let swaps = |cs: Vec<Candidate>| {
            cs.iter()
                .filter(|c| matches!(c.source, CandidateSource::Transposition { .. }))
                .count()
        };
        assert_eq!(swaps(d.decode("na")), CORRECTION_LIMIT);
        assert_eq!(swaps(d.decode(&format!("na{}", "q".repeat(23)))), 0);
        let mut cache = DecodeCache::default();
        for raw in ["n", "na", "naq", "na'", "na", "", "na"] {
            assert_eq!(d.decode_cached(raw, &mut cache), d.decode(raw), "{raw}");
        }
    }

    #[test]
    fn neighbor_raw_offsets_boundaries_and_cache() {
        let d = decoder("你好\tni hao\t100\n你\tni\t90\n");
        let score = exact_score(&d, "nihao", "你好");
        for (raw, consumed) in [("nohao", 5), ("no'haoq", 6), ("nohao'", 6), ("nohaoq", 5)] {
            let cs = d.decode(raw);
            let hit = cs
                .iter()
                .find(|c| c.text == "你好" && c.consumed == consumed)
                .unwrap();
            assert_eq!(
                hit.source,
                CandidateSource::NeighborSubstitution {
                    at: 1,
                    to: b'i',
                    ends: vec![2, 5]
                }
            );
            assert_eq!(hit.score.to_bits(), score);
            let prior = ordinary(&d, raw);
            assert_eq!(&cs[..prior.len()], prior);
        }
        for raw in ["n'ohao", "noh'ao"] {
            assert!(d.decode(raw).iter().all(|c| c.text != "你好"), "{raw}");
        }
        assert_eq!(d.decode("Nohao"), d.decode("nohao"));
        let long = format!("nohao{}", "q".repeat(20));
        assert!(d.decode(&long).iter().all(|c| !c.source.is_correction()));
        let mut cache = DecodeCache::default();
        for raw in [
            "n", "no", "noh", "nohao", "no'haoq", "noha", "nihao", "", "nohao",
        ] {
            assert_eq!(d.decode_cached(raw, &mut cache), d.decode(raw), "{raw}");
        }
    }
    #[test]
    fn neighbor_budget_and_existing_tuple_priority() {
        let rows: String = (0..100)
            .map(|i| format!("词{i}\tni hao\t{}\n", 100 - i))
            .collect();
        let corrections = decoder(&rows)
            .decode("nohao")
            .into_iter()
            .filter(|c| c.source.is_correction())
            .count();
        assert_eq!(corrections, CORRECTION_LIMIT);
        let cs = decoder("同文\tno hao\t1\n同文\tni hao\t1000\n").decode("nohao");
        let hits: Vec<_> = cs
            .iter()
            .filter(|c| c.text == "同文" && c.consumed == 5)
            .collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, CandidateSource::ExactDictionary);
        let rows = (0..8)
            .map(|i| format!("换位{i}\ta n\t{}\n", 100 - i))
            .collect::<String>()
            + "邻键\tma\t100000\n";
        let cs = decoder(&rows).decode("na");
        assert_eq!(
            cs.iter()
                .filter(|c| matches!(c.source, CandidateSource::Transposition { .. }))
                .count(),
            CORRECTION_LIMIT
        );
        assert!(
            cs.iter()
                .all(|c| !matches!(c.source, CandidateSource::NeighborSubstitution { .. }))
        );
    }

    #[test]
    fn missing_offsets_slots_boundaries_and_prior_tuple_precedence() {
        let d = decoder("你好\tni hao\t100\n你火\tni huo\t90\n");
        for (raw, text, consumed) in [
            ("ihao", "你好", 4),
            ("n'hao", "你好", 5),
            ("ni'ao", "你好", 5),
            ("niho", "你火", 4),
            ("n'haoq", "你好", 5),
            ("n'hao'", "你好", 6),
            ("niha'", "你好", 5),
        ] {
            let prior = before_missing(&d, raw);
            let cs = d.decode(raw);
            assert_eq!(&cs[..prior.len()], prior);
            let hit = cs
                .iter()
                .find(|c| c.text == text && c.consumed == consumed)
                .unwrap_or_else(|| panic!("{raw}: {cs:?}"));
            if let Some(old) = prior
                .iter()
                .find(|c| c.text == text && c.consumed == consumed)
            {
                assert_eq!(hit, old, "existing tuple wins: {raw}");
                continue;
            }
            let CandidateSource::MissingLetter { at, to, ends } = &hit.source else {
                panic!("{raw}: {hit:?}")
            };
            assert!(*at < consumed);
            let mut corrected = raw[..consumed].to_owned();
            corrected.insert(*at, char::from(*to));
            let corrected = scan_input(&corrected).unwrap();
            assert!(
                d.dictionary
                    .exact(&corrected.key)
                    .iter()
                    .any(|e| e.text.as_ref() == text
                        && e.ends.as_ref() == ends
                        && allowed(e, 0, corrected.key.len(), &corrected.boundaries)),
                "{raw}"
            );
        }
        for (raw, valid) in [("ni'hao", true), ("n'ihao", false), ("nih'ao", false)] {
            let input = scan_input(raw).unwrap();
            assert_eq!(
                d.dictionary
                    .exact(&input.key)
                    .iter()
                    .any(|e| e.text.as_ref() == "你好"
                        && allowed(e, 0, input.key.len(), &input.boundaries)),
                valid,
                "{raw}"
            );
        }
        for raw in ["niha", "nihaq"] {
            let cs = d.decode(raw);
            let hits: Vec<_> = cs
                .iter()
                .filter(|c| c.text == "你好" && c.consumed == 4)
                .collect();
            assert_eq!(hits.len(), 1, "{raw}");
            assert_eq!(
                hits[0].source,
                CandidateSource::Completion,
                "completion cannot be strict conversion: {raw}"
            );
        }
        let hit = d
            .decode("niha'")
            .into_iter()
            .find(|c| c.text == "你好" && c.consumed == 5)
            .unwrap();
        assert_eq!(
            hit.source,
            CandidateSource::MissingLetter {
                at: 4,
                to: b'o',
                ends: vec![2, 5]
            }
        );
    }
    #[test]
    fn missing_priority_full_input_duplicate_gap_budget_and_cache() {
        // Early p=0 produces a short prefix before later p=3 rescues the full raw.
        let mut rows: String = (0..12)
            .map(|i| format!("短{i}\ta i\t{}\n", 100 - i))
            .collect();
        rows.push_str("全\ti h a x o\t10\n");
        let cs = decoder(&rows).decode("ihao");
        assert!(cs.iter().any(|c| c.text == "全" && c.consumed == 4));
        let added = cs.iter().filter(|c| {
            matches!(
                c.source,
                CandidateSource::MissingLetter { .. } | CandidateSource::Completion
            )
        });
        assert!(added.count() <= CORRECTION_LIMIT);
        let mut cs = Vec::new();
        decoder("重复\ta a\t10\n").append_missing_letters(&scan_input("a").unwrap(), &mut cs, 8);
        let hits: Vec<_> = cs
            .iter()
            .filter(|c| c.text == "重复" && c.consumed == 1)
            .collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].source,
            CandidateSource::MissingLetter {
                at: 0,
                to: b'a',
                ends: vec![1, 2]
            }
        );
        let d = decoder("长\ta b c d e f g h i j k l m n o p q r s t u v w x y\t1\n");
        let raw = "bcdefghijklmnopqrstuvwxy";
        assert_eq!(raw.len(), MAX_CORRECTION_BYTES);
        assert!(d.decode(raw).iter().any(|c| c.text == "长"
            && c.consumed == 24
            && matches!(
                c.source,
                CandidateSource::MissingLetter {
                    at: 0,
                    to: b'a',
                    ..
                }
            )));
        assert!(
            d.decode(&format!("{raw}z"))
                .iter()
                .all(|c| !matches!(c.source, CandidateSource::MissingLetter { .. }))
        );
        let d = decoder("你好\tni hao\t100\n你火\tni huo\t90\n");
        assert_eq!(d.decode("Ihao"), d.decode("ihao"));
        let mut cache = DecodeCache::default();
        for raw in [
            "i", "ih", "ihao", "iha", "niho", "nihoq", "ni'ho", "niha'", "", "ihao",
        ] {
            assert_eq!(d.decode_cached(raw, &mut cache), d.decode(raw), "{raw}");
        }
    }
    #[test]
    fn missing_same_consumption_uses_score_before_insertion_letter() {
        // Both insertions consume the same raw prefix. Swap their frequencies to
        // show that selection follows static score rather than letter traversal.
        for (early, late, expected) in [(10, 1000, ["后", "先"]), (1000, 10, ["先", "后"])] {
            let d = decoder(&format!("先\ta b c\t{early}\n后\tz b c\t{late}\n"));
            for limit in [1, 2] {
                let mut cs = Vec::new();
                d.append_missing_letters(&scan_input("bc").unwrap(), &mut cs, limit);
                assert_eq!(
                    cs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
                    expected[..limit]
                );
                assert!(cs.iter().all(|c| c.consumed == 2
                    && matches!(c.source, CandidateSource::MissingLetter { at: 0, .. })));
            }
        }
    }
    #[test]
    #[ignore = "requires downloaded full dictionary assets"]
    fn real_dictionary_corrections() {
        let d = Decoder::new(
            Dictionary::load_dir_with_user_dictionary("data/dictionaries", None).unwrap(),
        );
        for (raw, consumed) in [("nohao", 5), ("no'haoq", 6)] {
            let hit = d
                .decode(raw)
                .into_iter()
                .find(|c| c.text == "你好" && c.consumed == consumed)
                .unwrap();
            assert_eq!(
                hit.source,
                CandidateSource::NeighborSubstitution {
                    at: 1,
                    to: b'i',
                    ends: vec![2, 5]
                }
            );
        }
        let cs = d.decode("ihao");
        assert!(
            cs.iter().any(
                |c| matches!(c.source, CandidateSource::MissingLetter { .. }) && c.consumed == 4
            ),
            "{cs:?}"
        );
        let prior = before_missing(&d, "niho");
        assert_eq!(
            prior.iter().filter(|c| c.source.is_correction()).count(),
            CORRECTION_LIMIT
        );
        assert_eq!(
            d.decode("niho"),
            prior,
            "real neighbor bucket already owns all eight correction slots"
        );
    }

    #[test]
    fn explicit_boundaries_precede_crowded_longer_partials_without_changing_full() {
        let raw = "ab'cdef";
        let input = scan_input(raw).unwrap();
        let full = states("full", FULL_LIMIT);
        let mut beams: Vec<Vec<State>> = (0..=input.key.len()).map(|_| Vec::new()).collect();
        beams[5] = states("long", CANDIDATE_LIMIT);
        beams[2] = states("boundary", 2);
        let menu = assemble_picks(&full, &beams, &[], &input);
        assert_eq!(menu.len(), CANDIDATE_LIMIT);
        let expected: Vec<_> = full
            .iter()
            .map(|s| Candidate {
                text: s.text.clone(),
                consumed: raw.len(),
                score: s.score,
                source: if s.completion {
                    CandidateSource::Completion
                } else {
                    CandidateSource::ExactDictionary
                },
            })
            .collect();
        assert_eq!(bits(&menu[..FULL_LIMIT]), bits(&expected));
        assert_eq!(menu[FULL_LIMIT].text, "boundary00");
        assert_eq!(menu[FULL_LIMIT + 1].text, "boundary01");
        assert_eq!(menu[FULL_LIMIT].consumed, 3);
        assert_eq!(menu[FULL_LIMIT].score.to_bits(), 100.0f32.to_bits());
        assert!(
            menu[FULL_LIMIT + 2..]
                .iter()
                .all(|c| c.text.starts_with("long"))
        );
        // Without a delimiter, the same beams keep longest-first order.
        let menu = assemble_picks(&full, &beams, &[], &scan_input("abcdef").unwrap());
        assert!(
            menu[FULL_LIMIT..]
                .iter()
                .all(|c| c.text.starts_with("long"))
        );
    }
    #[test]
    fn multiple_boundaries_share_one_category_and_tail_apostrophe_is_full() {
        let input = scan_input("ab'cd'ef").unwrap();
        let mut beams: Vec<Vec<State>> = (0..=input.key.len()).map(|_| Vec::new()).collect();
        beams[2] = states("early", 2);
        beams[4] = states("later", 2);
        beams[5] = states("long", 12);
        let menu = assemble_picks(&states("full", 1), &beams, &[], &input);
        assert_eq!(
            menu.iter()
                .take(5)
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            ["full00", "later00", "later01", "early00", "early01"]
        );
        let tail = decoder("前\tab\t10\n后\tcd\t10\n").decode("ab'");
        assert!(tail.iter().any(|c| c.text == "前"
            && c.consumed == 3
            && c.source == CandidateSource::ExactDictionary));
    }
    #[test]
    fn cached_boundary_insert_delete_and_append_equal_fresh_tuples() {
        let d = decoder("前\tni\t10\n后\thao\t10\n中\tshi\t10\n");
        let mut cache = DecodeCache::default();
        for raw in [
            "n",
            "ni",
            "ni'",
            "ni'h",
            "ni'ha",
            "ni'hao",
            "ni'hao'",
            "ni'hao'shi",
            "ni'hao'sh",
            "ni'hao",
            "nihao",
            "ni'hao",
            "ni''hao",
            "ni'hao",
            "nihao'",
            "nihao",
            "",
        ] {
            assert_eq!(
                bits(&d.decode_cached(raw, &mut cache)),
                bits(&d.decode(raw)),
                "{raw}"
            );
        }
    }

    #[test]
    fn single_syllable_characters_are_all_browsable() {
        let chars: Vec<String> = (0..70)
            .map(|i| char::from_u32(0x4e00 + i).unwrap().to_string())
            .collect();
        let mut rows: String = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| format!("{ch}\tji\t{}\n", 1000 - i))
            .collect();
        // One character can have both an exact reading and a longer reading.
        let tail = &chars[69];
        rows.push_str(&format!("{tail}\tjia\t1e30\n"));
        rows.push_str("吕\tv\t10\n词组\tji\t10\n对\tj i\t10\n成\tcheng\t10\n仙\txian\t10\n西\txi\t10\n安\tan\t10\n");
        let d = Arc::new(decoder(&rows));
        for (raw, consumed) in [("ji", 2), ("ji'", 3), ("ji'cheng", 3)] {
            let menu = d.decode(raw);
            assert!(menu.len() > SPAN_ENTRIES, "{raw}");
            for entry in d
                .dictionary
                .exact("ji")
                .iter()
                .filter(|e| e.ends.len() == 1)
            {
                if entry.text.chars().count() != 1 {
                    assert!(menu.iter().all(|c| c.text != *entry.text), "{raw}");
                    continue;
                }
                // An earlier non-Completion tuple wins, such as the tail's missing-letter
                // "jia'" for "ji'"; browsed characters keep their static score.
                let hit = menu
                    .iter()
                    .find(|c| {
                        c.text == *entry.text
                            && c.consumed == consumed
                            && c.source != CandidateSource::Completion
                    })
                    .unwrap_or_else(|| panic!("{raw} {}", entry.text));
                if hit.source == CandidateSource::ExactDictionary {
                    let score = extended_score(0.0, entry, false);
                    assert_eq!(hit.score.to_bits(), score.to_bits(), "{raw} {}", entry.text);
                }
            }
        }
        assert!(
            d.decode("ji")
                .iter()
                .any(|c| c.text == *tail && c.source == CandidateSource::Completion)
        );
        let split = d.decode("xi'an");
        assert!(!split.iter().any(|c| c.text == "仙" && c.consumed == 5));
        assert!(split.iter().any(|c| c.text == "西"
            && c.consumed == 3
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("v").iter().any(|c| c.text == "吕"
            && c.consumed == 1
            && c.source == CandidateSource::ExactDictionary));
        let mut cache = DecodeCache::default();
        for raw in [
            "j", "ji", "ji'", "ji'cheng", "j'i", "jia", "v", "xi'an", "ji", "", "ji'",
        ] {
            assert_eq!(
                bits(&d.decode_cached(raw, &mut cache)),
                bits(&d.decode(raw)),
                "{raw}"
            );
        }

        // A browsed first syllable keeps the remainder and learns only its raw reading.
        let learned = chars[68].clone();
        let learning = Arc::new(Learning::new(false));
        let mut session = Session::new(d, Some(learning.clone()));
        session.insert("jicheng").unwrap();
        session.move_cursor(-5);
        session.insert("'").unwrap();
        assert_eq!(session.input(), "ji'cheng");
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == learned && c.source == CandidateSource::ExactDictionary)
            .unwrap();
        let selected = session.candidates()[index].clone();
        assert_eq!(session.select(index), Some(learned.clone()));
        assert_eq!(selected.consumed, 3);
        assert_eq!(session.input(), "cheng");
        assert_eq!(session.cursor(), 0);
        assert_eq!(learning.counts("ji'", std::slice::from_ref(&selected)), [1]);
        let mut normalized = selected;
        normalized.consumed = 2;
        assert_eq!(learning.counts("ji", &[normalized]), [0]);
        session.reset();
        session.insert("ji'").unwrap();
        assert!(
            session
                .candidates()
                .iter()
                .any(|c| c.text == learned && c.source == CandidateSource::User)
        );
    }
}
