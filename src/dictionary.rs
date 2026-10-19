use std::{
    fmt,
    fs::File,
    io::{self, BufRead, BufReader, Read},
    ops::Range,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub enum DictionaryError {
    Io(io::Error),
    InvalidRow {
        line: usize,
        detail: String,
    },
    AtPath {
        path: PathBuf,
        cause: Box<DictionaryError>,
    },
    Empty,
}
impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(cause) => cause.fmt(f),
            Self::InvalidRow { line, detail } => write!(f, "dictionary line {line}: {detail}"),
            Self::AtPath { path, cause } => write!(f, "{}: {cause}", path.display()),
            Self::Empty => write!(f, "dictionary has no usable entries"),
        }
    }
}
impl std::error::Error for DictionaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(cause) => Some(cause),
            Self::AtPath { cause, .. } => Some(cause.as_ref()),
            _ => None,
        }
    }
}
impl From<io::Error> for DictionaryError {
    fn from(cause: io::Error) -> Self {
        Self::Io(cause)
    }
}

const USER_MAX_BYTES: u64 = 8 * 1024 * 1024;
const USER_MAX_ENTRIES: usize = 10_000;
const USER_MAX_KEY_BYTES: usize = 128;

#[derive(Debug)]
pub(crate) struct Entry {
    pub text: Box<str>,
    pub key: Box<str>,
    pub ends: Box<[u16]>,
    pub weight: f32,
}
impl Entry {
    pub(crate) fn syllables(&self) -> impl Iterator<Item = &str> {
        let mut start = 0;
        self.ends.iter().map(move |&end| {
            let syllable = &self.key[start..usize::from(end)];
            start = usize::from(end);
            syllable
        })
    }
    fn initials(&self) -> impl Iterator<Item = u8> {
        self.syllables().map(|syllable| syllable.as_bytes()[0])
    }
}
#[derive(Debug)]
struct CompletionIndex {
    entry: u32,
    prefix_len: u16,
}

/// Sorted concatenated-pinyin index. Each lookup uses binary bounds rather than scanning the corpus.
#[derive(Debug)]
pub struct Dictionary {
    entries: Vec<Entry>,
    /// Prefixes ending inside a final syllable, so querying "n" never visits words beginning with "na".
    completions: Vec<CompletionIndex>,
    /// Entry indices ordered by syllable initials, then weight.
    abbrev: Vec<u32>,
    /// Trie of every syllable spelling.
    syllables: Vec<SyllableNode>,
}
#[derive(Debug, Default)]
struct SyllableNode {
    next: [u32; 26],
    done: bool,
}
impl Dictionary {
    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        Self::load_dir_with_user_dictionary(path, None)
    }
    /// Merge an explicitly selected user table before building the shared indexes.
    pub fn load_dir_with_user_dictionary(
        path: impl AsRef<Path>,
        user_dictionary: Option<&Path>,
    ) -> Result<Self, DictionaryError> {
        let mut entries = Vec::new();
        for name in ["8105.dict.yaml", "base.dict.yaml", "ext.dict.yaml"] {
            entries.extend(read_file(&path.as_ref().join(name), false)?);
        }
        if let Some(path) = user_dictionary {
            entries.extend(read_file(path, true)?);
        }
        Self::new(entries)
    }
    pub fn from_rime_reader(reader: impl BufRead) -> Result<Self, DictionaryError> {
        Self::new(read(reader, false)?)
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn new(mut entries: Vec<Entry>) -> Result<Self, DictionaryError> {
        if entries.is_empty() {
            return Err(DictionaryError::Empty);
        }
        entries.sort_unstable_by(|a, b| {
            a.key
                .cmp(&b.key)
                .then(a.text.cmp(&b.text))
                .then(a.ends.cmp(&b.ends))
                .then(b.weight.total_cmp(&a.weight))
        });
        entries.dedup_by(|a, b| a.key == b.key && a.text == b.text && a.ends == b.ends);
        entries.sort_unstable_by(|a, b| {
            a.key
                .cmp(&b.key)
                .then(b.weight.total_cmp(&a.weight))
                .then(a.text.cmp(&b.text))
        });
        let mut completions = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            let last_start = entry.ends.iter().rev().nth(1).copied().unwrap_or(0);
            for prefix_len in last_start + 1..*entry.ends.last().unwrap() {
                completions.push(CompletionIndex {
                    entry: index as u32,
                    prefix_len,
                });
            }
        }
        completions.sort_unstable_by(|a, b| {
            let a_entry = &entries[a.entry as usize];
            let b_entry = &entries[b.entry as usize];
            a_entry.key[..usize::from(a.prefix_len)]
                .cmp(&b_entry.key[..usize::from(b.prefix_len)])
                .then(b_entry.weight.total_cmp(&a_entry.weight))
                .then(a_entry.text.cmp(&b_entry.text))
        });
        let mut syllables = vec![SyllableNode::default()];
        for syllable in entries.iter().flat_map(Entry::syllables) {
            let mut node = 0;
            for byte in syllable.bytes() {
                let slot = usize::from(byte - b'a');
                if syllables[node].next[slot] == 0 {
                    syllables[node].next[slot] = syllables.len() as u32;
                    syllables.push(SyllableNode::default());
                }
                node = syllables[node].next[slot] as usize;
            }
            syllables[node].done = true;
        }
        let mut abbrev: Vec<u32> = (0..entries.len() as u32).collect();
        abbrev.sort_unstable_by(|&a, &b| {
            let (a, b) = (&entries[a as usize], &entries[b as usize]);
            a.initials()
                .cmp(b.initials())
                .then(b.weight.total_cmp(&a.weight))
                .then(a.text.cmp(&b.text))
        });
        Ok(Self {
            entries,
            completions,
            abbrev,
            syllables,
        })
    }
    pub(crate) fn exact(&self, key: &str) -> &[Entry] {
        &self.entries[self.exact_range(key)]
    }
    /// Exact rows for a span starting at the input, plus whether any key has that prefix.
    pub(crate) fn exact_prefix<'a>(&'a self, key: &str) -> (&'a [Entry], bool) {
        let range = self.exact_range(key);
        let is_prefix =
            range.start < self.entries.len() && self.entries[range.start].key.starts_with(key);
        (&self.entries[range], is_prefix)
    }
    fn exact_range(&self, key: &str) -> Range<usize> {
        let start = self.entries.partition_point(|e| e.key.as_ref() < key);
        let end = self.entries.partition_point(|e| e.key.as_ref() <= key);
        start..end
    }
    // Every row in parent shares the preceding depth bytes. Shorter exact keys
    // sort before extensions, so Option<byte> retains the existing key ordering.
    fn next_prefix_range(&self, parent: Range<usize>, depth: usize, byte: u8) -> Range<usize> {
        let rows = &self.entries[parent.clone()];
        let start =
            rows.partition_point(|entry| entry.key.as_bytes().get(depth).copied() < Some(byte));
        let end =
            rows.partition_point(|entry| entry.key.as_bytes().get(depth).copied() <= Some(byte));
        parent.start + start..parent.start + end
    }
    fn exact_prefix_rows(&self, range: Range<usize>, depth: usize) -> &[Entry] {
        let rows = &self.entries[range];
        &rows[..rows.partition_point(|entry| entry.key.len() == depth)]
    }
    /// Visit every exact raw prefix reachable by inserting one lowercase byte.
    /// For 24 letters: at most 8474 range transitions and 8424 exact buckets.
    pub(crate) fn visit_missing_letter_prefixes(
        &self,
        key: &str,
        mut visit: impl FnMut(usize, usize, u8, &[Entry]),
    ) {
        let mut parent = 0..self.entries.len();
        for p in 0..=key.len() {
            for to in b'a'..=b'z' {
                let mut branch = self.next_prefix_range(parent.clone(), p, to);
                if branch.is_empty() {
                    continue;
                }
                if p > 0 {
                    let rows = self.exact_prefix_rows(branch.clone(), p + 1);
                    if !rows.is_empty() {
                        visit(p, p, to, rows);
                    }
                }
                for end in p + 1..=key.len() {
                    branch = self.next_prefix_range(branch, end, key.as_bytes()[end - 1]);
                    if branch.is_empty() {
                        break;
                    }
                    let rows = self.exact_prefix_rows(branch.clone(), end + 1);
                    if !rows.is_empty() {
                        visit(end, p, to, rows);
                    }
                }
            }
            if p < key.len() {
                parent = self.next_prefix_range(parent, p, key.as_bytes()[p]);
                if parent.is_empty() {
                    break;
                }
            }
        }
    }
    pub(crate) fn completions(&self, prefix: &str) -> impl Iterator<Item = &Entry> {
        let key = |index: &CompletionIndex| {
            &self.entries[index.entry as usize].key[..usize::from(index.prefix_len)]
        };
        let start = self
            .completions
            .partition_point(|index| key(index) < prefix);
        let end = self
            .completions
            .partition_point(|index| key(index) <= prefix);
        self.completions[start..end]
            .iter()
            .map(|index| &self.entries[index.entry as usize])
    }
    /// Longest leading syllable, measured in bytes. Pinyin keys are ASCII.
    pub(crate) fn longest_syllable(&self, text: &str) -> Option<usize> {
        let mut node = 0;
        let mut best = None;
        for (index, byte) in text.bytes().enumerate() {
            node = self.syllables[node].next[usize::from(byte - b'a')] as usize;
            if node == 0 {
                break;
            }
            if self.syllables[node].done {
                best = Some(index + 1);
            }
        }
        best
    }
    pub(crate) fn by_initials<'a>(&'a self, initials: &[u8]) -> impl Iterator<Item = &'a Entry> {
        let order = |&index: &u32| {
            self.entries[index as usize]
                .initials()
                .cmp(initials.iter().copied())
        };
        let start = self.abbrev.partition_point(|index| order(index).is_lt());
        let end = self.abbrev.partition_point(|index| order(index).is_le());
        self.abbrev[start..end]
            .iter()
            .map(|&index| &self.entries[index as usize])
    }
}

/// Reads one table and names its path in any failure.
fn read_file(path: &Path, user: bool) -> Result<Vec<Entry>, DictionaryError> {
    let read_path = || {
        if user {
            // Checked before opening, so a FIFO cannot block startup.
            if !std::fs::metadata(path)?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "user dictionary must be a regular file",
                )
                .into());
            }
            let mut bytes = Vec::new();
            File::open(path)?
                .take(USER_MAX_BYTES + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() as u64 > USER_MAX_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "user dictionary exceeds 8 MiB",
                )
                .into());
            }
            return read(BufReader::new(bytes.as_slice()), true);
        }
        read(BufReader::new(File::open(path)?), false)
    };
    read_path().map_err(|cause| DictionaryError::AtPath {
        path: path.to_owned(),
        cause: Box::new(cause),
    })
}

/// Rows of a Rime table after its `...` header terminator. Pure English words never
/// enter pinyin recall: public tables drop them, while a user table rejects them.
fn read(reader: impl BufRead, user: bool) -> Result<Vec<Entry>, DictionaryError> {
    let mut entries = Vec::new();
    let mut body = false;
    for (index, row) in reader.lines().enumerate() {
        let invalid = |detail: String| DictionaryError::InvalidRow {
            line: index + 1,
            detail,
        };
        let row = row?;
        if !body {
            body = row.trim() == "...";
            continue;
        }
        if row.trim().is_empty() || row.starts_with('#') {
            continue;
        }
        let entry = parse_row(&row).map_err(invalid)?;
        if entry.text.bytes().all(|byte| byte.is_ascii_alphabetic()) {
            if user {
                return Err(invalid(
                    "English words are unsupported in user dictionaries".into(),
                ));
            }
            continue;
        }
        if user && entry.key.len() > USER_MAX_KEY_BYTES {
            return Err(invalid("user pinyin exceeds 128 bytes".into()));
        }
        if user && entries.len() == USER_MAX_ENTRIES {
            return Err(invalid("user dictionary exceeds 10000 entries".into()));
        }
        entries.push(entry);
    }
    if !body {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "missing ... header terminator").into(),
        );
    }
    Ok(entries)
}

/// Text, whitespace-separated lowercase syllables and an optional nonnegative weight.
fn parse_row(row: &str) -> Result<Entry, String> {
    let mut fields = row.split('\t');
    let (Some(text), Some(pinyin), weight, None) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return Err("expected two or three TSV columns".into());
    };
    if text.chars().any(char::is_control) {
        return Err("text contains a control character".into());
    }
    let mut key = String::new();
    let mut ends = Vec::new();
    for syllable in pinyin.split_whitespace() {
        if !syllable.bytes().all(|byte| byte.is_ascii_lowercase()) {
            return Err(format!("invalid pinyin {syllable:?}"));
        }
        key.push_str(syllable);
        ends.push(u16::try_from(key.len()).map_err(|_| "pinyin exceeds 65535 bytes")?);
    }
    if text.is_empty() || key.is_empty() {
        return Err("empty text or pinyin".into());
    }
    let weight = match weight {
        None | Some("") => 1.0,
        Some(weight) => weight
            .trim_end_matches('%')
            .parse::<f32>()
            .map_err(|_| format!("invalid weight {weight:?}"))?,
    };
    if !weight.is_finite() || weight < 0.0 {
        return Err("weight must be finite and nonnegative".into());
    }
    Ok(Entry {
        text: text.into(),
        key: key.into(),
        ends: ends.into(),
        weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CandidateSource, Decoder, Session};
    use std::{io::Cursor, sync::Arc};

    #[test]
    fn english_words_are_dropped_from_public_tables_and_rejected_from_user_tables() {
        let chinese = "...\n你\tni\t10\n公婆\tgong po\t30\nB站\tbi zhan\t40\n";
        let public = Dictionary::from_rime_reader(Cursor::new(format!(
            "{chinese}GPU\tgpu\t100\nRust\trust\t100\n"
        )))
        .unwrap();
        assert_eq!(public.len(), 3);
        for row in ["Rust\trust", "GPU\tgpu", "abc\ta b c"] {
            let error = read(Cursor::new(format!("...\n{row}\n")), true).unwrap_err();
            assert!(
                matches!(error, DictionaryError::InvalidRow { line: 2, ref detail }
                    if detail.contains("English words are unsupported")),
                "{error}"
            );
        }
        let digits = read(Cursor::new("...\n123\tyi er san\n"), true).unwrap();
        assert_eq!(digits[0].text.as_ref(), "123");
    }

    #[test]
    fn table_validation_keeps_user_limits_and_original_io_causes() {
        for user in [false, true] {
            for row in [
                "你\tNI",
                "你\tni\tNaN",
                "你\tni\t-1",
                "你\tni\t1\textra",
                "你\0\tni",
            ] {
                assert!(
                    matches!(
                        read(Cursor::new(format!("...\n{row}\n")), user),
                        Err(DictionaryError::InvalidRow { line: 2, .. })
                    ),
                    "{row:?}, user={user}"
                );
            }
            assert!(read(Cursor::new("no header\n你\tni\n"), user).is_err());
        }
        assert!(read(Cursor::new("...\n"), true).unwrap().is_empty());
        assert!(matches!(
            Dictionary::from_rime_reader(Cursor::new("...\n")),
            Err(DictionaryError::Empty)
        ));
        let max_key = format!("...\n词\t{}\n", "a".repeat(USER_MAX_KEY_BYTES));
        assert_eq!(read(Cursor::new(&max_key), true).unwrap().len(), 1);
        let too_long = format!("...\n词\t{}\n", "a".repeat(USER_MAX_KEY_BYTES + 1));
        assert!(read(Cursor::new(too_long), true).is_err());
        let max_rows = format!("...\n{}", "词\tni\n".repeat(USER_MAX_ENTRIES));
        assert_eq!(
            read(Cursor::new(&max_rows), true).unwrap().len(),
            USER_MAX_ENTRIES
        );
        assert!(read(Cursor::new(max_rows + "词\tni\n"), true).is_err());

        let error = read(Cursor::new(b"...\n\xff"), false).unwrap_err();
        let cause = std::error::Error::source(&error).unwrap();
        assert_eq!(
            cause.downcast_ref::<io::Error>().unwrap().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn user_dictionary_merge_preserves_partial_selection_and_completion() {
        let directory =
            std::env::temp_dir().join(format!("yanliu-user-dictionary-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        for name in ["8105.dict.yaml", "base.dict.yaml", "ext.dict.yaml"] {
            std::fs::write(directory.join(name), "...\n你\tni\t1\n").unwrap();
        }
        let user = directory.join("user.dict.yaml");
        std::fs::write(&user, "...\n砚流\tyan liu\t10\n砚流\tyan liu\t20\n").unwrap();
        let dictionary =
            Dictionary::load_dir_with_user_dictionary(&directory, Some(&user)).unwrap();
        assert_eq!(dictionary.len(), 2);
        assert_eq!(dictionary.exact("yanliu")[0].weight, 20.0);
        let decoder = Arc::new(Decoder::new(dictionary));
        let mut session = Session::new(Arc::clone(&decoder), None);
        session.insert("yan'liu'ni").unwrap();
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "砚流")
            .unwrap();
        let candidate = &session.candidates()[index];
        assert_eq!(candidate.consumed, 8);
        assert_eq!(candidate.source, CandidateSource::ExactDictionary);
        assert_eq!(session.select(index).as_deref(), Some("砚流"));
        assert_eq!(session.input(), "ni");
        let mut session = Session::new(decoder, None);
        session.insert("yanli").unwrap();
        let candidate = session
            .candidates()
            .iter()
            .find(|c| c.text == "砚流")
            .unwrap();
        assert_eq!(candidate.consumed, 5);
        assert_eq!(candidate.source, CandidateSource::Completion);
        assert!(!session.candidates().iter().any(|c| c.text == "砚流"
            && c.consumed == 5
            && c.source != CandidateSource::Completion));
        // Public tables name the failing file too.
        let base = directory.join("base.dict.yaml");
        std::fs::write(&base, "...\n你\tNI\n").unwrap();
        let error = Dictionary::load_dir_with_user_dictionary(&directory, None)
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with(&base.display().to_string()) && error.contains("line 2"),
            "{error}"
        );
        std::fs::write(&user, vec![b'#'; USER_MAX_BYTES as usize + 1]).unwrap();
        assert!(
            read_file(&user, true)
                .unwrap_err()
                .to_string()
                .contains("exceeds 8 MiB")
        );
        let error = read_file(&directory.join("missing.dict.yaml"), true).unwrap_err();
        let cause = std::error::Error::source(&error).unwrap();
        let cause = cause.source().unwrap().downcast_ref::<io::Error>().unwrap();
        assert_eq!(cause.kind(), io::ErrorKind::NotFound);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn missing_letter_prefixes_match_exhaustive_exact_buckets() {
        let d = Dictionary::from_rime_reader(Cursor::new(
            "...\n甲\ta\t1\n乙\ta b\t2\n丙\ta b c\t3\n丁\tb a c\t4\n戊\tc a b\t5\n己\ta b\t6\n庚\tx b c d\t7\n辛\ta x b c\t8\n",
        ))
        .unwrap();
        for key in ["a", "ab", "bc", "xbc", "abc", "zz", "bcd"] {
            let mut actual = Vec::new();
            d.visit_missing_letter_prefixes(key, |end, p, to, rows| {
                for row in rows {
                    actual.push((end, p, to, row.text.to_string()));
                }
            });
            let mut expected = Vec::new();
            for end in 1..=key.len() {
                for p in 0..=end {
                    for to in b'a'..=b'z' {
                        let mut changed = key[..end].to_owned();
                        changed.insert(p, char::from(to));
                        for row in d.exact(&changed) {
                            expected.push((end, p, to, row.text.to_string()));
                        }
                    }
                }
            }
            actual.sort();
            expected.sort();
            assert_eq!(actual, expected, "{key}");
        }
    }
}
