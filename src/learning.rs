//! Explicit-choice counts, optional contextual parameters and their single persistence owner.
use crate::{Candidate, CandidateSource, decoder::MAX_INPUT_BYTES};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread,
    time::Duration,
};

const VERSION: u32 = 1;
const MAX_ROWS: usize = 4096;
const RECALL_LIMIT: usize = 8;
const FEATURES: usize = 4096;
const WEIGHT_LIMIT: f32 = 8.0;
const SAVE_DELAY: Duration = Duration::from_millis(250);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    input: String,
    text: String,
    count: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    rows: Vec<Row>,
    adapter: Option<Vec<f32>>,
    updates: u64,
}

#[derive(Default)]
struct State {
    counts: BTreeMap<(String, String), u32>,
    adapter: Option<Vec<f32>>,
    updates: u64,
    revision: u64,
    stopped: bool,
}

impl State {
    fn record_count(&mut self, input: &str, text: &str) {
        let key = (input.to_owned(), text.to_owned());
        if !self.counts.contains_key(&key) && self.counts.len() == MAX_ROWS {
            // Evict the least chosen row; the smallest key breaks ties deterministically.
            let victim = self
                .counts
                .iter()
                .min_by_key(|(key, count)| (**count, *key))
                .map(|(key, _)| key.clone())
                .expect("full table is nonempty");
            self.counts.remove(&victim);
        }
        let count = self.counts.entry(key).or_default();
        *count = count.saturating_add(1);
    }
}

/// Preferences shared by every session; `contextual` applies and updates the personal parameters.
pub struct Learning {
    contextual: bool,
    state: Mutex<State>,
    ready: Condvar,
}

impl Learning {
    fn with_state(contextual: bool, state: State) -> Self {
        Self {
            contextual,
            state: Mutex::new(state),
            ready: Condvar::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn new(contextual: bool) -> Self {
        Self::with_state(contextual, State::default())
    }

    #[cfg(test)]
    pub(crate) fn revision(&self) -> u64 {
        self.lock().revision
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("learning state poisoned")
    }

    pub(crate) fn contextual(&self) -> bool {
        self.contextual
    }

    /// Recall previously confirmed raw prefixes without changing dictionary candidate identities.
    pub(crate) fn recall(&self, input: &str, candidates: &[Candidate]) -> Vec<Candidate> {
        let state = self.lock();
        if state.counts.is_empty() {
            return Vec::new();
        }
        let mut rows = Vec::new();
        for consumed in 1..=input.len() {
            if input.as_bytes().get(consumed) == Some(&b'\'') {
                continue;
            }
            let prefix = &input[..consumed];
            if consumed == 1
                && input.len() != 1
                && b"bpmfdtnlgkhjqxrzcsyw".contains(&input.as_bytes()[0].to_ascii_lowercase())
            {
                continue;
            }
            for ((raw, text), &count) in state.counts.range((prefix.to_owned(), String::new())..) {
                if raw != prefix {
                    break;
                }
                // In particular, a matching Completion must keep its untyped-text semantics.
                if !candidates
                    .iter()
                    .any(|candidate| candidate.consumed == consumed && candidate.text == *text)
                {
                    rows.push((raw, text, count));
                }
            }
        }
        let class = |raw: &str| {
            if raw.len() == input.len() {
                2
            } else if raw.ends_with('\'') {
                1
            } else {
                0
            }
        };
        rows.sort_by(|(a, at, ac), (b, bt, bc)| {
            class(b)
                .cmp(&class(a))
                .then(b.len().cmp(&a.len()))
                .then(bc.cmp(ac))
                .then(at.cmp(bt))
        });
        rows.into_iter()
            .take(RECALL_LIMIT)
            .map(|(raw, text, count)| Candidate {
                text: text.clone(),
                consumed: raw.len(),
                source: CandidateSource::User,
                // Use the decoder's lexical-frequency scale with explicit choices as frequency.
                score: (count as f32).ln_1p() / 14.0 - 2.0,
            })
            .collect()
    }

    /// Explicit-choice counts for the raw input each candidate consumes. A Completion contains
    /// untyped text, so no recorded choice applies to it.
    pub(crate) fn counts<'a>(
        &self,
        input: &str,
        candidates: impl IntoIterator<Item = &'a Candidate>,
    ) -> Vec<u32> {
        let state = self.lock();
        candidates
            .into_iter()
            .map(|c| {
                if c.source == CandidateSource::Completion {
                    return 0;
                }
                state
                    .counts
                    .get(&(input[..c.consumed].to_owned(), c.text.clone()))
                    .copied()
                    .unwrap_or(0)
            })
            .collect()
    }

    pub(crate) fn personal_scores<'a>(
        &self,
        input: &str,
        context: &str,
        candidates: impl IntoIterator<Item = &'a Candidate>,
    ) -> Vec<f32> {
        let candidates = candidates.into_iter();
        {
            let state = self.lock();
            if state.adapter.is_none()
                || !state
                    .counts
                    .range((input.to_owned(), String::new())..)
                    .next()
                    .is_some_and(|((raw, _), _)| raw == input)
            {
                drop(state);
                return candidates.map(|_| 0.0).collect();
            }
        }
        let rows: Vec<_> = candidates
            .map(|c| features(&input[..c.consumed], context, &c.text, None))
            .collect();
        // A concurrent choice may change parameters or evict this input during feature work.
        let state = self.lock();
        let Some(adapter) = &state.adapter else {
            return vec![0.0; rows.len()];
        };
        // Shared features must not transfer corrections to unconfirmed raw pinyin.
        if !state
            .counts
            .range((input.to_owned(), String::new())..)
            .next()
            .is_some_and(|((raw, _), _)| raw == input)
        {
            return vec![0.0; rows.len()];
        }
        rows.iter()
            .map(|row| {
                row.iter()
                    .map(|&(index, value)| adapter[index] * value)
                    .sum()
            })
            .collect()
    }

    /// Record an explicit choice. Returns false once persistence has stopped, so the caller does
    /// not commit a choice that can no longer be saved.
    pub(crate) fn record_choice(
        &self,
        input: &str,
        context: &str,
        candidates: &[Candidate],
        selected: usize,
        joined: Option<(&str, &str)>,
        prior: Option<&[(usize, f32)]>,
    ) -> bool {
        let chosen = &candidates[selected];
        if chosen.source == CandidateSource::Completion {
            return true;
        }
        let raw = &input[..chosen.consumed];
        // Only candidates shown above the choice for the same raw input and correction class were
        // passed over in its favour.
        let rejected: Vec<_> = if self.contextual {
            candidates[..selected]
                .iter()
                .enumerate()
                .filter_map(|(index, c)| {
                    (c.consumed == chosen.consumed
                        && c.source != CandidateSource::Completion
                        && c.source.is_correction() == chosen.source.is_correction())
                    .then_some(index)
                })
                .collect()
        } else {
            Vec::new()
        };
        let prior = prior.filter(|prior| {
            !rejected.is_empty()
                && chosen.consumed == input.len()
                && !chosen.source.is_correction()
                && prior.iter().any(|&(index, _)| index == selected)
                && rejected
                    .iter()
                    .any(|index| prior.iter().any(|(scored, _)| scored == index))
        });
        let (prior_rows, context_rows): (Vec<_>, Vec<_>) = prior
            .into_iter()
            .flatten()
            .map(|&(index, _)| {
                let mut contextual = Vec::new();
                let forward =
                    features(raw, context, &candidates[index].text, Some(&mut contextual));
                (forward, contextual)
            })
            .unzip();
        let chosen_features = if prior.is_none() && !rejected.is_empty() {
            features(raw, context, &chosen.text, None)
        } else {
            Vec::new()
        };
        let rejected_rows: Vec<_> = if prior.is_none() {
            rejected
                .iter()
                .map(|&index| features(raw, context, &candidates[index].text, None))
                .collect()
        } else {
            Vec::new()
        };
        let mut state = self.lock();
        if state.stopped {
            return false;
        }
        state.record_count(raw, &chosen.text);
        if let Some((raw, text)) = joined {
            state.record_count(raw, text);
        }
        let change = if rejected.is_empty() {
            None
        } else if let Some(prior) = prior {
            let rows = &prior_rows;
            let selected_row = prior
                .iter()
                .position(|&(index, _)| index == selected)
                .expect("selected candidate has a prior");
            let personal: Vec<f32> = rows
                .iter()
                .map(|row| {
                    state.adapter.as_ref().map_or(0.0, |adapter| {
                        row.iter()
                            .map(|&(index, value)| adapter[index] * value)
                            .sum()
                    })
                })
                .collect();
            let counts: Vec<_> = prior
                .iter()
                .map(|&(index, _)| {
                    state
                        .counts
                        .get(&(raw.to_owned(), candidates[index].text.clone()))
                        .copied()
                        .unwrap_or(0)
                })
                .collect();
            // A recorded full choice participates as a User candidate on the next ranking.
            let (scores, scale) = crate::model::fuse_scores(
                candidates,
                prior,
                &personal,
                &counts,
                &CandidateSource::User,
                chosen.score,
            );
            let (opponent, rival) = scores
                .iter()
                .enumerate()
                .filter(|(_, score)| rejected.contains(&score.index))
                .max_by(|(_, a), (_, b)| a.total.total_cmp(&b.total))
                .expect("a rejected candidate has a prior");
            let mut direction = vec![0.0; FEATURES];
            for &(index, value) in &rows[selected_row] {
                direction[index] += value;
            }
            for &(index, value) in &rows[opponent] {
                direction[index] -= value;
            }
            let mut change = vec![0.0; FEATURES];
            for &(index, value) in &context_rows[selected_row] {
                change[index] += value;
            }
            for &(index, value) in &context_rows[opponent] {
                change[index] -= value;
            }
            let slope: f32 = direction.iter().zip(&change).map(|(d, q)| d * q).sum();
            let norm_direction: f32 = direction.iter().map(|value| value * value).sum();
            let norm_context: f32 = change.iter().map(|value| value * value).sum();
            let loss = (crate::model::PROMOTION_MARGIN / scale + rival.total
                - scores[selected_row].total)
                .max(0.0);
            if loss == 0.0 || slope <= 0.0 || norm_context == 0.0 {
                None
            } else {
                let rate = (loss / slope).min(0.5 * (norm_direction / norm_context).sqrt());
                for value in &mut change {
                    *value *= rate;
                }
                Some(change)
            }
        } else {
            let mut change = vec![0.0; FEATURES];
            for &(index, value) in &chosen_features {
                change[index] += 0.5 * value;
            }
            let rate = 0.5 / rejected.len() as f32;
            for row in rejected_rows {
                for (index, value) in row {
                    change[index] -= rate * value;
                }
            }
            Some(change)
        };
        if let Some(change) = change {
            let adapter = state.adapter.get_or_insert_with(|| vec![0.0; FEATURES]);
            for (weight, change) in adapter.iter_mut().zip(change) {
                *weight = (*weight + change).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
            }
            state.updates = state.updates.saturating_add(1);
        }
        state.revision += 1;
        self.ready.notify_one();
        true
    }
}

fn hash(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325, |value, byte| {
        (value ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn grams(text: &[char], maximum: usize) -> Vec<u64> {
    let mut result = Vec::new();
    for length in 1..=maximum.min(text.len()) {
        for window in text.windows(length) {
            result.push(hash(
                window.iter().flat_map(|ch| (*ch as u32).to_le_bytes()),
            ));
        }
    }
    result
}

/// Signed hashing preserves raw ASCII identity and bounds the feature vector. Stored parameters
/// are indexed by these features, so any change here invalidates every persisted adapter.
fn features(
    input: &str,
    context: &str,
    text: &str,
    context_out: Option<&mut Vec<(usize, f32)>>,
) -> Vec<(usize, f32)> {
    let input: Vec<char> = input.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let mut context: Vec<char> = context.chars().rev().take(32).collect();
    context.reverse();
    let input_grams = grams(&input, 3);
    let text_grams = grams(&text, 3);
    let context_grams = grams(&context, 2);
    let input_id = hash(input.iter().flat_map(|ch| (*ch as u32).to_le_bytes()));
    let text_id = hash(text.iter().flat_map(|ch| (*ch as u32).to_le_bytes()));
    let context_id = hash(
        context[context.len().saturating_sub(8)..]
            .iter()
            .flat_map(|ch| (*ch as u32).to_le_bytes()),
    );
    let mut values = [0.0_f32; FEATURES];
    let mut context_values = context_out.as_ref().map(|_| vec![0.0_f32; FEATURES]);
    let mut add = |family: u8, left: u64, right: u64| {
        let value = hash(
            [family]
                .into_iter()
                .chain(left.to_le_bytes())
                .chain(right.to_le_bytes()),
        );
        let index = value as usize % FEATURES;
        let sign = if value >> 63 == 0 { 1.0 } else { -1.0 };
        values[index] += sign;
        if let Some(values) = &mut context_values
            && matches!(family, 2 | 5 | 6)
        {
            values[index] += sign;
        }
    };
    for gram in input_grams {
        add(0, gram, 0);
        add(3, gram, text_id);
    }
    for gram in text_grams {
        add(1, gram, 0);
        add(4, input_id, gram);
        if !context.is_empty() {
            add(6, context_id, gram);
        }
    }
    for gram in context_grams {
        add(2, gram, 0);
        // Pair grams with phrase identity instead of a quadratic gram cross product.
        add(5, gram, text_id);
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    // Context contributions share the full forward norm and the existing weight buckets.
    if let Some((output, values)) = context_out.zip(context_values) {
        *output = values
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (value != 0.0).then_some((index, value / norm)))
            .collect();
    }
    values
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| (value != 0.0).then_some((index, value / norm)))
        .collect()
}

impl State {
    /// The snapshot is an external file that may be damaged; validate it once here.
    fn load(path: &Path) -> io::Result<Self> {
        let invalid = |reason: &str| io::Error::new(io::ErrorKind::InvalidData, reason);
        let read = || -> io::Result<Self> {
            match fs::metadata(path) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
                Err(error) => return Err(error),
                // Opening a FIFO would block startup.
                Ok(metadata) if !metadata.is_file() => {
                    return Err(invalid("must be a regular file"));
                }
                Ok(_) => {}
            }
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
            let bytes = fs::read(path)?;
            // Earlier releases persisted only explicit-choice rows as an array.
            let snapshot = if bytes.iter().find(|byte| !byte.is_ascii_whitespace()) == Some(&b'[') {
                Snapshot {
                    version: VERSION,
                    rows: serde_json::from_slice(&bytes)?,
                    adapter: None,
                    updates: 0,
                }
            } else {
                serde_json::from_slice::<Snapshot>(&bytes)?
            };
            if snapshot.version != VERSION {
                return Err(invalid("unsupported version"));
            }
            if snapshot.adapter.as_ref().is_some_and(|adapter| {
                adapter.len() != FEATURES
                    || !adapter.iter().all(|weight| weight.abs() <= WEIGHT_LIMIT)
            }) {
                return Err(invalid("invalid context model"));
            }
            if snapshot.rows.len() > MAX_ROWS {
                return Err(invalid("too many learning rows"));
            }
            let mut counts = BTreeMap::new();
            for Row { input, text, count } in snapshot.rows {
                if input.is_empty()
                    || input.len() > MAX_INPUT_BYTES
                    || !input.bytes().all(|b| b.is_ascii_alphabetic() || b == b'\'')
                    || text.is_empty()
                    || count == 0
                {
                    return Err(invalid("invalid learning row"));
                }
                if counts.insert((input, text), count).is_some() {
                    return Err(invalid("duplicate learning row"));
                }
            }
            Ok(Self {
                counts,
                adapter: snapshot.adapter,
                updates: snapshot.updates,
                ..Self::default()
            })
        };
        read().map_err(|error| at("load learning snapshot", path, error))
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            version: VERSION,
            rows: self
                .counts
                .iter()
                .map(|((input, text), &count)| Row {
                    input: input.clone(),
                    text: text.clone(),
                    count,
                })
                .collect(),
            adapter: self.adapter.clone(),
            updates: self.updates,
        }
    }
}

fn at(operation: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{operation} {}: {error}", path.display()),
    )
}

/// Atomically replace the snapshot. Callers hold the directory lock, so one temporary name suffices.
fn write_snapshot(path: &Path, snapshot: &Snapshot) -> io::Result<()> {
    let temporary = path.with_extension("json.tmp");
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec(snapshot)?)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(
            path.parent()
                .expect("snapshot is inside the state directory"),
        )?
        .sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(|error| at("save learning snapshot", path, error))
}

fn lock_directory(directory: &Path) -> io::Result<File> {
    fs::create_dir_all(directory)
        .and_then(|()| fs::set_permissions(directory, fs::Permissions::from_mode(0o700)))
        .map_err(|error| at("create learning directory", directory, error))?;
    let path = directory.join("learning.lock");
    let lock = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(&path)
        .map_err(|error| at("open learning lock", &path, error))?;
    lock.try_lock().map_err(|error| {
        io::Error::other(format!(
            "lock learning state {}: {error}; stop yanliu-ibus and retry",
            path.display()
        ))
    })?;
    Ok(lock)
}

// Unlock explicitly: a child forked by another thread may briefly share the descriptor.
fn unlock(lock: File, directory: &Path) -> io::Result<()> {
    lock.unlock().map_err(|error| {
        at(
            "unlock learning state",
            &directory.join("learning.lock"),
            error,
        )
    })
}

/// Read the persisted preferences with the same validation as service startup.
pub fn list(directory: &Path) -> io::Result<serde_json::Value> {
    let path = directory.join("learning.json");
    let snapshot = State::load(&path)?.snapshot();
    Ok(serde_json::json!({
        "path": path,
        "rows": snapshot.rows,
        "context_model": {
            "updates": snapshot.updates,
            "parameters": snapshot.adapter.map_or(0, |adapter| adapter.len()),
        },
    }))
}

/// Replace even a damaged snapshot, excluding any live persistence owner.
pub fn clear(directory: &Path) -> io::Result<()> {
    let lock = lock_directory(directory)?;
    write_snapshot(
        &directory.join("learning.json"),
        &State::default().snapshot(),
    )?;
    unlock(lock, directory)
}

pub fn state_directory() -> io::Result<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from)
        && path.is_absolute()
    {
        return Ok(path.join("yanliu"));
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "absolute HOME required for learning state",
            )
        })?;
    Ok(home.join(".local/state/yanliu"))
}

/// Exclusively owns the state directory and saves the shared state shortly after each change.
pub struct LearningWriter {
    pub learning: Arc<Learning>,
    thread: Option<thread::JoinHandle<io::Result<()>>>,
}

impl LearningWriter {
    pub fn start(
        directory: &Path,
        contextual: bool,
        failure: impl Fn(String) + Send + 'static,
    ) -> io::Result<Self> {
        let lock = lock_directory(directory)?;
        let directory = directory.to_owned();
        let path = directory.join("learning.json");
        let learning = Arc::new(Learning::with_state(contextual, State::load(&path)?));
        let shared = learning.clone();
        let thread = thread::Builder::new()
            .name("yanliu-learning".into())
            .spawn(move || {
                let result = persist(&shared, &path).and(unlock(lock, &directory));
                if let Err(error) = &result {
                    shared.lock().stopped = true;
                    failure(format!("learning writer: {error}"));
                }
                result
            })?;
        Ok(Self {
            learning,
            thread: Some(thread),
        })
    }

    /// Save any remaining change and stop; later choices are refused.
    pub fn finish(&mut self) -> io::Result<()> {
        self.learning.lock().stopped = true;
        self.learning.ready.notify_all();
        match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| io::Error::other("learning writer panicked"))?,
            None => Ok(()),
        }
    }
}

impl Drop for LearningWriter {
    fn drop(&mut self) {
        // Failures were already reported through the writer's failure callback.
        let _ = self.finish();
    }
}

fn persist(learning: &Learning, path: &Path) -> io::Result<()> {
    let mut saved = 0;
    loop {
        let state = learning
            .ready
            .wait_while(learning.lock(), |state| {
                state.revision == saved && !state.stopped
            })
            .expect("learning state poisoned");
        if state.revision == saved {
            return Ok(());
        }
        // Merge a burst of choices into one save unless shutting down.
        let (state, _) = learning
            .ready
            .wait_timeout_while(state, SAVE_DELAY, |state| !state.stopped)
            .expect("learning state poisoned");
        let revision = state.revision;
        let snapshot = state.snapshot();
        drop(state);
        write_snapshot(path, &snapshot)?;
        saved = revision;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decoder, Dictionary, Session};
    use std::{
        io::Cursor,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn directory() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "yanliu-learning-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn candidate(text: &str, consumed: usize) -> Candidate {
        Candidate {
            text: text.into(),
            consumed,
            source: CandidateSource::Composed,
            score: 1.0,
        }
    }

    fn decoder(rows: &str) -> Arc<Decoder> {
        Arc::new(Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(format!("...\n{rows}"))).unwrap(),
        ))
    }

    fn write_rows(path: &Path, rows: serde_json::Value) {
        let snapshot =
            serde_json::json!({"version": 1, "rows": rows, "adapter": null, "updates": 0});
        fs::write(path, serde_json::to_vec(&snapshot).unwrap()).unwrap();
    }

    fn count(learning: &Learning, input: &str, text: &str) -> Option<u32> {
        learning
            .lock()
            .counts
            .get(&(input.into(), text.into()))
            .copied()
    }

    #[test]
    fn contextual_direction_uses_post_count_scores_and_the_existing_update_norm_budget() {
        let input = "kaiyuan";
        let context = "这座城市位于辽宁省，名叫";
        // This pair has cross-family aliases, so d·q differs from the context norm squared.
        for (selected_language, capped) in [(-2.0, false), (-20.0, true)] {
            let learning = Learning::new(true);
            let mut candidates = [
                candidate("开园", input.len()),
                candidate("开源", input.len()),
                candidate("开原", input.len()),
            ];
            candidates[0].source = CandidateSource::ExactDictionary;
            for (candidate, score) in candidates.iter_mut().zip([3.0, 2.0, 2.0]) {
                candidate.score = score;
            }
            let prior = [(0, -10.0), (1, -0.2), (2, selected_language)];
            let old = vec![0.001; FEATURES];
            learning.lock().adapter = Some(old.clone());
            let (rows, context_rows): (Vec<_>, Vec<_>) = candidates
                .iter()
                .map(|candidate| {
                    let mut contextual = Vec::new();
                    let forward = features(input, context, &candidate.text, Some(&mut contextual));
                    (forward, contextual)
                })
                .unzip();
            let personal = |row: &Vec<(usize, f32)>| {
                row.iter()
                    .map(|&(index, value)| old[index] * value)
                    .sum::<f32>()
            };
            let selected_base = selected_language - 2.0;
            let scale = -2.2_f32 - (-10.0_f32).min(selected_base);
            let rival = -2.2 / scale + personal(&rows[1]);
            let selected = selected_base / scale + personal(&rows[2]) + 0.15 * 1.0_f32.ln_1p();
            let mut direction = vec![0.0; FEATURES];
            for &(index, value) in &rows[2] {
                direction[index] += value;
            }
            for &(index, value) in &rows[1] {
                direction[index] -= value;
            }
            let mut contextual = vec![0.0; FEATURES];
            for &(index, value) in &context_rows[2] {
                contextual[index] += value;
            }
            for &(index, value) in &context_rows[1] {
                contextual[index] -= value;
            }
            let slope: f32 = direction.iter().zip(&contextual).map(|(d, q)| d * q).sum();
            let norm_direction: f32 = direction.iter().map(|value| value * value).sum();
            let norm_context: f32 = contextual.iter().map(|value| value * value).sum();
            let budget = 0.5 * (norm_direction / norm_context).sqrt();
            let rate =
                ((crate::model::PROMOTION_MARGIN / scale + rival - selected) / slope).min(budget);
            assert!((slope - norm_context).abs() > 1e-4);
            assert!(rate > 0.0);
            if capped {
                assert!(rate > 0.5);
                assert_eq!(rate, budget);
            } else {
                assert!(rate < budget);
            }
            assert!(learning.record_choice(input, context, &candidates, 2, None, Some(&prior)));
            let state = learning.lock();
            assert_eq!(state.counts.get(&(input.into(), "开原".into())), Some(&1));
            assert_eq!(state.updates, 1);
            let mut actual_norm_squared = 0.0;
            for ((&actual, old), contextual) in state
                .adapter
                .as_ref()
                .unwrap()
                .iter()
                .zip(old)
                .zip(contextual)
            {
                assert!((actual - (old + rate * contextual)).abs() < 1e-7);
                actual_norm_squared += (actual - old).powi(2);
            }
            assert!(actual_norm_squared.sqrt() <= 0.5 * norm_direction.sqrt() + 1e-6);
        }
    }

    #[test]
    fn an_already_satisfied_post_count_margin_does_not_invent_an_update() {
        let learning = Learning::new(true);
        let candidates = [candidate("你", 2), candidate("泥", 2), candidate("拟", 2)];
        let prior = [(0, -1.0), (1, -1.2), (2, -20.0)];
        assert!(learning.record_choice("ni", "道路上的", &candidates, 1, None, Some(&prior)));
        let state = learning.lock();
        assert_eq!(state.counts.get(&("ni".into(), "泥".into())), Some(&1));
        assert_eq!(state.updates, 0);
        assert!(state.adapter.is_none());
    }

    #[test]
    fn forward_feature_bits_match_the_previous_mapping_with_or_without_context_output() {
        let long = format!("旧文{}", "语".repeat(32));
        for (input, context, text, expected) in [
            ("Rust", "编程语言", "Rust", 0x93a91520537ecde6),
            ("kaiyuan", "乘高铁前往", "开原", 0x5721b2d2a5272b10),
            ("xian", long.as_str(), "西安", 0x12bd09676ca66c32),
            ("ni", "", "泥", 0xe6f775c7564f94b7),
        ] {
            let forward = features(input, context, text, None);
            let mut contextual = Vec::new();
            let with_context = features(input, context, text, Some(&mut contextual));
            assert_eq!(forward, with_context);
            let fingerprint = hash(forward.iter().flat_map(|&(index, value)| {
                (index as u64)
                    .to_le_bytes()
                    .into_iter()
                    .chain(value.to_bits().to_le_bytes())
            }));
            assert_eq!(fingerprint, expected);
            if context.is_empty() {
                assert!(contextual.is_empty());
            }
        }
    }

    #[test]
    fn features_preserve_condition_and_ascii_identity() {
        let reference = features("Rust", "编程语言", "Rust", None);
        for alternative in [
            features("rust", "编程语言", "Rust", None),
            features("Rust", "金属生锈", "Rust", None),
            features("Rust", "编程语言", "rust", None),
            features("rusi", "编程语言", "Rust", None),
        ] {
            assert_ne!(reference, alternative);
        }
        assert!(
            reference
                .iter()
                .all(|(index, value)| *index < FEATURES && value.is_finite())
        );
        let norm: f32 = reference.iter().map(|(_, value)| value * value).sum();
        assert!((norm - 1.0).abs() < 1e-5);
        assert_eq!(features("", "", "", None), Vec::new());
        let suffix = "语".repeat(32);
        assert_eq!(
            features("x", &suffix, "西", None),
            features("x", &format!("旧文{suffix}"), "西", None)
        );
    }

    #[test]
    fn contextual_corrections_distinguish_opposite_choices_and_survive_restart() {
        let dir = directory();
        write_rows(
            &dir.join("learning.json"),
            serde_json::json!([{"input": "ni", "text": "你", "count": 2}]),
        );
        let menu = [candidate("现", 4), candidate("西安", 4)];
        let reversed = [menu[1].clone(), menu[0].clone()];
        let mut writer = LearningWriter::start(&dir, true, |_| {}).unwrap();
        assert!(
            writer
                .learning
                .record_choice("xian", "我去", &menu, 1, None, None)
        );
        assert!(
            writer
                .learning
                .record_choice("xian", "此时此刻", &reversed, 1, None, None)
        );
        let travel = writer.learning.personal_scores("xian", "我去", &menu);
        let now = writer.learning.personal_scores("xian", "此时此刻", &menu);
        assert!(travel[1] > travel[0], "{travel:?}");
        assert!(now[0] > now[1], "{now:?}");
        assert_eq!(
            writer
                .learning
                .personal_scores("xian", "我去", menu.iter().rev()),
            vec![travel[1], travel[0]]
        );
        writer.finish().unwrap();
        let bytes = fs::read(dir.join("learning.json")).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["updates"], 2);
        assert_eq!(value["adapter"].as_array().unwrap().len(), FEATURES);
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains("我去"));
        assert!(!text.contains("此时此刻"));
        let summary = list(&dir).unwrap();
        assert_eq!(summary["context_model"]["updates"], 2);
        assert_eq!(summary["rows"].as_array().unwrap().len(), 3);
        assert!(summary.get("adapter").is_none());

        let mut writer = LearningWriter::start(&dir, true, |_| {}).unwrap();
        assert_eq!(
            writer.learning.personal_scores("xian", "我去", &menu),
            travel
        );
        assert_eq!(
            writer.learning.personal_scores("xian", "此时此刻", &menu),
            now
        );
        assert_eq!(writer.learning.counts("ni", &[candidate("你", 2)]), [2]);
        writer.finish().unwrap();

        // Without contextual learning the stored parameters are kept but never updated.
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        assert!(!writer.learning.contextual());
        assert!(
            writer
                .learning
                .record_choice("xian", "我去", &menu, 1, None, None)
        );
        writer.finish().unwrap();
        let summary = list(&dir).unwrap();
        assert_eq!(summary["context_model"]["updates"], 2);
        assert_eq!(summary["context_model"]["parameters"], FEATURES);
        clear(&dir).unwrap();
        let cleared = list(&dir).unwrap();
        assert_eq!(cleared["rows"], serde_json::json!([]));
        assert_eq!(cleared["context_model"]["parameters"], 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn contextual_scores_require_the_confirmed_raw_input_but_not_the_chosen_text() {
        let learning = Learning::new(true);
        let menu = [candidate("现", 4), candidate("西安", 4)];
        assert!(learning.record_choice("xian", "我去", &menu, 1, None, None));
        let transferred = learning.personal_scores("xian", "乘火车前往", &menu);
        assert!(transferred[1] > transferred[0], "{transferred:?}");

        let other_candidates = [candidate("现", 4), candidate("仙", 4)];
        assert!(
            learning
                .personal_scores("xian", "我去", &other_candidates)
                .iter()
                .any(|score| *score != 0.0),
            "a shortlist without 西安 must still use the confirmed raw input"
        );
        for input in ["XIAN", "Xian", "xi'an", "xian'", "xianhao", "ni"] {
            let candidates = [candidate("现", input.len()), candidate("西安", input.len())];
            assert_eq!(
                learning.personal_scores(input, "我去", &candidates),
                vec![0.0, 0.0],
                "unconfirmed raw input {input} received contextual scores"
            );
        }
    }

    #[test]
    fn contextual_updates_use_only_preceding_competitors_and_consumed_input() {
        let learning = Learning::new(true);
        let mut completion = candidate("你好", 3);
        completion.source = CandidateSource::Completion;
        let mut correction = candidate("拟", 3);
        correction.source = CandidateSource::Transposition {
            at: 0,
            ends: vec![2],
        };
        let menu = [
            completion,
            candidate("你好", 6),
            correction,
            candidate("你", 3),
            candidate("泥", 3),
        ];
        assert!(learning.record_choice("ni'hao", "外部前文", &menu, 3, None, None));
        assert_eq!(learning.lock().updates, 0);
        assert_eq!(count(&learning, "ni'", "你"), Some(1));
        assert!(learning.lock().adapter.is_none());
        assert!(learning.record_choice("ni'hao", "外部前文", &menu, 4, None, None));
        assert_eq!(learning.lock().updates, 1);
        assert!(learning.record_choice("ni'hao", "外部前文", &menu, 0, None, None));
        assert_eq!(learning.lock().updates, 1);
        assert_eq!(learning.revision(), 2);
    }

    #[test]
    fn joined_choice_adds_only_a_count_without_a_virtual_contextual_update() {
        let ordinary = Learning::new(true);
        let joined = Learning::new(true);
        let menu = [candidate("号", 3), candidate("好", 3)];
        assert!(ordinary.record_choice("hao", "你", &menu, 1, None, None));
        assert!(joined.record_choice("hao", "你", &menu, 1, Some(("nihao", "你好")), None));
        let a = ordinary.lock();
        let b = joined.lock();
        assert_eq!(a.adapter, b.adapter);
        assert_eq!(a.updates, 1);
        assert_eq!(a.updates, b.updates);
        assert_eq!(a.revision, b.revision);
        assert_eq!(a.counts.get(&("hao".into(), "好".into())), Some(&1));
        assert_eq!(b.counts.get(&("hao".into(), "好".into())), Some(&1));
        assert!(!a.counts.contains_key(&("nihao".into(), "你好".into())));
        assert_eq!(b.counts.get(&("nihao".into(), "你好".into())), Some(&1));
    }

    #[test]
    fn damaged_snapshots_report_invalid_data_with_their_path() {
        let dir = directory();
        let path = dir.join("learning.json");
        let valid = serde_json::json!({
            "version": 1, "rows": [], "adapter": vec![0.0; FEATURES], "updates": 0,
        });
        let mut damaged = Vec::new();
        for (field, value) in [
            ("version", serde_json::json!(2)),
            ("adapter", serde_json::json!([0.0])),
            ("adapter", serde_json::json!(vec![9.0; FEATURES])),
            (
                "rows",
                serde_json::json!([{"input": "n1", "text": "泥", "count": 1}]),
            ),
            (
                "rows",
                serde_json::json!([{"input": "ni", "text": "泥", "count": 0}]),
            ),
            (
                "rows",
                serde_json::json!([
                    {"input": "ni", "text": "泥", "count": 1},
                    {"input": "ni", "text": "泥", "count": 2},
                ]),
            ),
        ] {
            let mut snapshot = valid.clone();
            snapshot[field] = value;
            damaged.push(serde_json::to_string(&snapshot).unwrap());
        }
        // Too large for f32, so it parses as infinity.
        damaged.push(
            serde_json::to_string(&valid)
                .unwrap()
                .replacen("[0.0", "[1e100", 1),
        );
        for body in damaged {
            fs::write(&path, &body).unwrap();
            let error = list(&dir).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{body:.80}");
            assert!(error.to_string().contains(&path.display().to_string()));
        }
        fs::write(&path, "broken").unwrap();
        assert!(list(&dir).is_err());
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        let error = list(&dir).unwrap_err().to_string();
        assert!(error.contains("must be a regular file"), "{error}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn uppercase_pinyin_explicit_selection_survives_persistence() {
        let dir = directory();
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        let mut session = Session::new(decoder("你\tni\t100\n"), Some(writer.learning.clone()));
        session.insert("NI").unwrap();
        assert_eq!(session.select(0).as_deref(), Some("你"));
        writer.finish().unwrap();
        let rows = list(&dir).unwrap()["rows"].clone();
        assert_eq!(
            rows,
            serde_json::json!([{"input": "NI", "text": "你", "count": 1}])
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn persisted_array_preferences_survive_selection_and_object_save() {
        let dir = directory();
        let path = dir.join("learning.json");
        fs::write(&path, r#"[{"input":"ni","text":"泥","count":2}]"#).unwrap();
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let mut session = Session::new(
            decoder("你\tni\t100\n泥\tni\t1\n"),
            Some(writer.learning.clone()),
        );
        session.insert("ni").unwrap();
        assert_eq!(session.candidates()[0].text, "泥");
        assert_eq!(session.select(0).as_deref(), Some("泥"));
        writer.finish().unwrap();
        let snapshot: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(snapshot["version"], 1);
        assert_eq!(
            snapshot["rows"],
            serde_json::json!([{"input":"ni","text":"泥","count":3}])
        );
        assert_eq!(list(&dir).unwrap()["rows"], snapshot["rows"]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn stored_count_never_matches_a_completion_but_full_preferences_apply() {
        let dir = directory();
        let path = dir.join("learning.json");
        write_rows(
            &path,
            serde_json::json!([
                {"input": "ni", "text": "娘", "count": 99},
                {"input": "hao", "text": "号", "count": 2},
            ]),
        );
        let snapshot = fs::read(&path).unwrap();
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        let learning = writer.learning.clone();
        let decoder = decoder("你\tni\t100\n泥\tni\t1\n娘\tniang\t100\n好\thao\t100\n号\thao\t1\n");
        let mut baseline = Session::new(decoder.clone(), None);
        let mut session = Session::new(decoder, Some(learning.clone()));
        baseline.insert("ni").unwrap();
        session.insert("ni").unwrap();
        assert_eq!(session.candidates(), baseline.candidates());
        let completion = session
            .candidates()
            .iter()
            .position(|c| c.text == "娘" && c.source == CandidateSource::Completion)
            .unwrap();
        assert_eq!(learning.counts("ni", session.candidates())[completion], 0);
        assert_eq!(session.select(completion).as_deref(), Some("娘"));
        assert_eq!(learning.revision(), 0);
        assert_eq!(count(&learning, "ni", "娘"), Some(99));
        session.insert("hao").unwrap();
        assert_eq!(session.candidates()[0].text, "号");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        writer.finish().unwrap();
        assert_eq!(fs::read(&path).unwrap(), snapshot);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn clear_excludes_writer_and_recovers_corruption_idempotently() {
        let dir = directory();
        let path = dir.join("learning.json");
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        assert!(LearningWriter::start(&dir, false, |_| {}).is_err());
        assert!(
            writer
                .learning
                .record_choice("ni", "", &[candidate("你", 2)], 0, None, None)
        );
        let error = clear(&dir).unwrap_err().to_string();
        assert!(error.contains("stop yanliu-ibus and retry"), "{error}");
        writer.finish().unwrap();
        assert_eq!(list(&dir).unwrap()["rows"][0]["count"], 1);
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::write(&path, b"broken").unwrap();
        assert!(list(&dir).is_err());
        clear(&dir).unwrap();
        clear(&dir).unwrap();
        assert!(dir.join("learning.lock").is_file());
        assert_eq!(list(&dir).unwrap()["rows"], serde_json::json!([]));
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        assert!(
            writer
                .learning
                .record_choice("ni", "", &[candidate("泥", 2)], 0, None, None)
        );
        writer.finish().unwrap();
        assert_eq!(list(&dir).unwrap()["rows"][0]["text"], "泥");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn list_missing_state_does_not_create_directory() {
        let dir = directory().join("missing");
        assert_eq!(list(&dir).unwrap()["rows"], serde_json::json!([]));
        assert!(!dir.exists());
        fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    #[test]
    fn long_dictionary_macro_roundtrips_through_selection_and_reload() {
        let dir = directory();
        let text = "字".repeat(342);
        let decoder = decoder(&format!("{text}\tni\t100\n"));
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        let mut session = Session::new(decoder.clone(), Some(writer.learning.clone()));
        session.insert("ni").unwrap();
        assert_eq!(session.select(0), Some(text.clone()));
        writer.finish().unwrap();
        let writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        let mut session = Session::new(decoder, Some(writer.learning.clone()));
        session.insert("ni").unwrap();
        assert_eq!(session.candidates()[0].text, text);
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        drop(writer);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_save_keeps_previous_snapshot_and_reports_the_cause() {
        let dir = directory();
        let path = dir.join("learning.json");
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        assert!(
            writer
                .learning
                .record_choice("ni", "", &[candidate("你", 2)], 0, None, None)
        );
        writer.finish().unwrap();
        let previous = fs::read(&path).unwrap();
        let blocked = dir.join("learning.json.tmp");
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("occupied"), b"").unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut writer = LearningWriter::start(&dir, false, move |cause| {
            tx.send(cause).unwrap();
        })
        .unwrap();
        assert!(
            writer
                .learning
                .record_choice("ni", "", &[candidate("泥", 2)], 0, None, None)
        );
        let cause = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            cause.starts_with("learning writer: save learning snapshot"),
            "{cause}"
        );
        assert!(cause.contains("Is a directory"), "{cause}");
        assert!(
            !writer
                .learning
                .record_choice("ni", "", &[candidate("泥", 2)], 0, None, None)
        );
        assert!(writer.finish().is_err());
        assert_eq!(fs::read(&path).unwrap(), previous);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn final_flush_rejects_late_selection_without_committing() {
        let dir = directory();
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        let mut session = Session::new(decoder("你\tni\t100\n"), Some(writer.learning.clone()));
        session.insert("ni").unwrap();
        writer.finish().unwrap();
        assert!(session.select(0).is_none());
        assert_eq!(session.input(), "ni");
        assert!(session.context().is_empty());
        assert!(writer.learning.lock().counts.is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn stopped_writer_rejects_the_last_segment_without_recording_a_joined_choice() {
        let dir = directory();
        let mut writer = LearningWriter::start(&dir, false, |_| {}).unwrap();
        let mut session = Session::new(
            decoder("你\tni\t100\n好\thao\t100\n"),
            Some(writer.learning.clone()),
        );
        session.insert("nihao").unwrap();
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "你" && c.consumed == 2)
            .unwrap();
        assert_eq!(session.select(index).as_deref(), Some("你"));
        writer.finish().unwrap();
        let before = session.snapshot();
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "好" && c.consumed == 3)
            .unwrap();
        assert!(session.select(index).is_none());
        assert_eq!(session.input(), before.input);
        assert_eq!(session.context(), before.context);
        assert_eq!(session.revision(), before.revision);
        assert_eq!(count(&writer.learning, "ni", "你"), Some(1));
        assert_eq!(count(&writer.learning, "hao", "好"), None);
        assert_eq!(count(&writer.learning, "nihao", "你好"), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw_commit_and_external_context_do_not_learn() {
        let learning = Arc::new(Learning::new(true));
        let mut session = Session::new(decoder("你\tni\t100\n"), Some(learning.clone()));
        session.set_context("外部上下文");
        session.insert("ni").unwrap();
        assert_eq!(session.commit_raw().as_deref(), Some("ni"));
        assert_eq!(learning.revision(), 0);
    }

    #[test]
    fn bounded_table_evicts_deterministically_and_saturates_counts() {
        let learning = Learning::new(false);
        let choose = |text: &str| {
            assert!(learning.record_choice("ni", "", &[candidate(text, 2)], 0, None, None))
        };
        for index in 0..MAX_ROWS {
            choose(&format!("词{index:04}"));
        }
        choose("词0000");
        choose("新");
        assert_eq!(learning.lock().counts.len(), MAX_ROWS);
        assert_eq!(count(&learning, "ni", "词0000"), Some(2));
        assert_eq!(count(&learning, "ni", "词0001"), None);
        *learning
            .lock()
            .counts
            .get_mut(&("ni".into(), "新".into()))
            .unwrap() = u32::MAX;
        choose("新");
        assert_eq!(count(&learning, "ni", "新"), Some(u32::MAX));
    }

    #[test]
    fn personal_recall_has_a_fixed_budget_and_prefers_full_then_closed_prefixes() {
        let learning = Learning::new(false);
        for index in 0..16 {
            let text = format!("词{index:02}");
            for _ in 0..index + 1 {
                assert!(learning.record_choice("ni", "", &[candidate(&text, 2)], 0, None, None));
            }
        }
        assert!(learning.record_choice("nihao'", "", &[candidate("边界", 6)], 0, None, None));
        assert!(learning.record_choice("nihao'hao", "", &[candidate("完整", 9)], 0, None, None));
        let recalled = learning.recall("nihao'hao", &[]);
        assert_eq!(recalled.len(), RECALL_LIMIT);
        assert_eq!(recalled[0].text, "完整");
        assert_eq!(recalled[0].consumed, 9);
        assert_eq!(recalled[1].text, "边界");
        assert_eq!(recalled[1].consumed, 6);
        assert_eq!(
            recalled[2..]
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            ["词15", "词14", "词13", "词12", "词11", "词10"]
        );
        assert!(
            recalled
                .iter()
                .all(|candidate| candidate.source == CandidateSource::User)
        );
        assert!(learning.recall("", &[]).is_empty());
        let baseline = vec![candidate("完整", 9), candidate("边界", 6)];
        let recalled = learning.recall("nihao'hao", &baseline);
        assert_eq!(recalled.len(), RECALL_LIMIT);
        assert!(
            recalled
                .iter()
                .all(|candidate| !baseline.iter().any(|existing| {
                    candidate.text == existing.text && candidate.consumed == existing.consumed
                }))
        );

        let decoder = decoder("你\tni\t100\n好\thao\t100\n");
        let public = decoder.decode("nihao'hao");
        let mut session = Session::new(decoder, Some(Arc::new(learning)));
        session.insert("nihao'hao").unwrap();
        assert_eq!(session.candidates().len(), public.len() + RECALL_LIMIT);
        assert_eq!(
            session
                .candidates()
                .iter()
                .filter(|c| c.source == CandidateSource::User)
                .count(),
            RECALL_LIMIT
        );
        assert!(
            public
                .iter()
                .all(|candidate| session.candidates().contains(candidate))
        );
    }
}
