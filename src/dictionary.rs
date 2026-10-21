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
pub(crate) struct Entry<'a> {
    pub text: &'a str,
    pub key: &'a str,
    pub ends: &'a [u16],
    pub weight: f32,
}
impl Entry<'_> {
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
struct Record {
    text_start: u32,
    key_start: u32,
    key_end: u32,
    ends: Range<u32>,
    weight: f32,
}
impl Record {
    fn key<'a>(&self, strings: &'a str) -> &'a str {
        &strings[self.key_start as usize..self.key_end as usize]
    }
    fn entry<'a>(&self, strings: &'a str, ends: &'a [u16]) -> Entry<'a> {
        Entry {
            text: &strings[self.text_start as usize..self.key_start as usize],
            key: self.key(strings),
            ends: &ends[self.ends.start as usize..self.ends.end as usize],
            weight: self.weight,
        }
    }
}
#[derive(Clone)]
pub(crate) struct Rows<'a> {
    dictionary: &'a Dictionary,
    records: std::slice::Iter<'a, Record>,
}
impl Rows<'_> {
    fn is_empty(&self) -> bool {
        self.records.as_slice().is_empty()
    }
}
impl<'a> Iterator for Rows<'a> {
    type Item = Entry<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.records
            .next()
            .map(|record| record.entry(&self.dictionary.strings, &self.dictionary.ends))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.records.size_hint()
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
    entries: Vec<Record>,
    strings: String,
    ends: Vec<u16>,
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
    fn empty() -> Self {
        Self {
            entries: Vec::new(),
            strings: String::new(),
            ends: Vec::new(),
            completions: Vec::new(),
            abbrev: Vec::new(),
            syllables: Vec::new(),
        }
    }
    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self, DictionaryError> {
        Self::load_dir_with_user_dictionary(path, None)
    }
    /// Merge an explicitly selected user table before building the shared indexes.
    pub fn load_dir_with_user_dictionary(
        path: impl AsRef<Path>,
        user_dictionary: Option<&Path>,
    ) -> Result<Self, DictionaryError> {
        let mut dictionary = Self::empty();
        for name in ["8105.dict.yaml", "base.dict.yaml", "ext.dict.yaml"] {
            read_file(&path.as_ref().join(name), false, &mut dictionary)?;
        }
        if let Some(path) = user_dictionary {
            read_file(path, true, &mut dictionary)?;
        }
        dictionary.build_indexes()
    }
    pub fn from_rime_reader(reader: impl BufRead) -> Result<Self, DictionaryError> {
        let mut dictionary = Self::empty();
        read(reader, false, &mut dictionary)?;
        dictionary.build_indexes()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn build_indexes(mut self) -> Result<Self, DictionaryError> {
        if self.entries.is_empty() {
            return Err(DictionaryError::Empty);
        }
        let strings = self.strings.as_str();
        let ends = self.ends.as_slice();
        self.entries.sort_unstable_by(|a, b| {
            let (a, b) = (a.entry(strings, ends), b.entry(strings, ends));
            a.key
                .cmp(b.key)
                .then_with(|| a.text.cmp(b.text))
                .then_with(|| a.ends.cmp(b.ends))
                .then_with(|| b.weight.total_cmp(&a.weight))
        });
        self.entries.dedup_by(|a, b| {
            let (a, b) = (a.entry(strings, ends), b.entry(strings, ends));
            a.key == b.key && a.text == b.text && a.ends == b.ends
        });
        self.entries.sort_unstable_by(|a, b| {
            let (a, b) = (a.entry(strings, ends), b.entry(strings, ends));
            a.key
                .cmp(b.key)
                .then_with(|| b.weight.total_cmp(&a.weight))
                .then_with(|| a.text.cmp(b.text))
        });
        self.entries.shrink_to_fit();
        self.strings.shrink_to_fit();
        self.ends.shrink_to_fit();
        let mut completions = Vec::new();
        for (index, record) in self.entries.iter().enumerate() {
            let entry = record.entry(&self.strings, &self.ends);
            let last_start = entry.ends.iter().rev().nth(1).copied().unwrap_or(0);
            for prefix_len in last_start + 1..*entry.ends.last().unwrap() {
                completions.push(CompletionIndex {
                    entry: index as u32,
                    prefix_len,
                });
            }
        }
        completions.sort_unstable_by(|a, b| {
            let a_entry = self.entry(a.entry);
            let b_entry = self.entry(b.entry);
            a_entry.key[..usize::from(a.prefix_len)]
                .cmp(&b_entry.key[..usize::from(b.prefix_len)])
                .then_with(|| b_entry.weight.total_cmp(&a_entry.weight))
                .then_with(|| a_entry.text.cmp(b_entry.text))
        });
        completions.shrink_to_fit();
        let mut syllables = vec![SyllableNode::default()];
        for record in &self.entries {
            for syllable in record.entry(&self.strings, &self.ends).syllables() {
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
        }
        let mut abbrev: Vec<u32> = (0..self.entries.len() as u32).collect();
        abbrev.sort_unstable_by(|&a, &b| {
            let (a, b) = (self.entry(a), self.entry(b));
            a.initials()
                .cmp(b.initials())
                .then_with(|| b.weight.total_cmp(&a.weight))
                .then_with(|| a.text.cmp(b.text))
        });
        self.completions = completions;
        self.abbrev = abbrev;
        self.syllables = syllables;
        Ok(self)
    }
    fn entry(&self, index: u32) -> Entry<'_> {
        self.entries[index as usize].entry(&self.strings, &self.ends)
    }
    fn rows<'a>(&'a self, records: &'a [Record]) -> Rows<'a> {
        Rows {
            dictionary: self,
            records: records.iter(),
        }
    }
    pub(crate) fn exact(&self, key: &str) -> Rows<'_> {
        self.rows(&self.entries[self.exact_range(key)])
    }
    /// Exact rows for a span starting at the input, plus whether any key has that prefix.
    pub(crate) fn exact_prefix(&self, key: &str) -> (Rows<'_>, bool) {
        let range = self.exact_range(key);
        let is_prefix = range.start < self.entries.len()
            && self.entries[range.start]
                .key(&self.strings)
                .starts_with(key);
        (self.rows(&self.entries[range]), is_prefix)
    }
    fn exact_range(&self, key: &str) -> Range<usize> {
        let start = self.entries.partition_point(|e| e.key(&self.strings) < key);
        let end = self
            .entries
            .partition_point(|e| e.key(&self.strings) <= key);
        start..end
    }
    // Every row in parent shares the preceding depth bytes. Shorter exact keys
    // sort before extensions, so Option<byte> retains the existing key ordering.
    fn next_prefix_range(&self, parent: Range<usize>, depth: usize, byte: u8) -> Range<usize> {
        let rows = &self.entries[parent.clone()];
        let start = rows.partition_point(|entry| {
            entry.key(&self.strings).as_bytes().get(depth).copied() < Some(byte)
        });
        let end = rows.partition_point(|entry| {
            entry.key(&self.strings).as_bytes().get(depth).copied() <= Some(byte)
        });
        parent.start + start..parent.start + end
    }
    fn exact_prefix_rows(&self, range: Range<usize>, depth: usize) -> Rows<'_> {
        let rows = &self.entries[range];
        self.rows(
            &rows[..rows
                .partition_point(|entry| (entry.key_end - entry.key_start) as usize == depth)],
        )
    }
    /// Visit every exact raw prefix reachable by inserting one lowercase byte.
    /// For 24 letters: at most 8474 range transitions and 8424 exact buckets.
    pub(crate) fn visit_missing_letter_prefixes(
        &self,
        key: &str,
        mut visit: impl FnMut(usize, usize, u8, Rows<'_>),
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
    pub(crate) fn completions(&self, prefix: &str) -> impl Iterator<Item = Entry<'_>> {
        let key = |index: &CompletionIndex| {
            &self.entries[index.entry as usize].key(&self.strings)[..usize::from(index.prefix_len)]
        };
        let start = self
            .completions
            .partition_point(|index| key(index) < prefix);
        let end = self
            .completions
            .partition_point(|index| key(index) <= prefix);
        self.completions[start..end]
            .iter()
            .map(|index| self.entry(index.entry))
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
    pub(crate) fn by_initials<'a>(&'a self, initials: &[u8]) -> impl Iterator<Item = Entry<'a>> {
        let order = |&index: &u32| self.entry(index).initials().cmp(initials.iter().copied());
        let start = self.abbrev.partition_point(|index| order(index).is_lt());
        let end = self.abbrev.partition_point(|index| order(index).is_le());
        self.abbrev[start..end]
            .iter()
            .map(|&index| self.entry(index))
    }
}

/// Reads one table and names its path in any failure.
fn read_file(path: &Path, user: bool, dictionary: &mut Dictionary) -> Result<(), DictionaryError> {
    let mut read_path = || {
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
            return read(BufReader::new(bytes.as_slice()), true, dictionary);
        }
        read(BufReader::new(File::open(path)?), false, dictionary)
    };
    read_path().map_err(|cause| DictionaryError::AtPath {
        path: path.to_owned(),
        cause: Box::new(cause),
    })
}

/// Rows of a Rime table after its `...` header terminator. Pure English words never
/// enter pinyin recall: public tables drop them, while a user table rejects them.
fn read(
    mut reader: impl BufRead,
    user: bool,
    dictionary: &mut Dictionary,
) -> Result<(), DictionaryError> {
    let mut row = String::new();
    let mut body = false;
    let mut count = 0;
    let mut line = 0;
    loop {
        row.clear();
        if reader.read_line(&mut row)? == 0 {
            break;
        }
        line += 1;
        if row.ends_with('\n') {
            row.pop();
            if row.ends_with('\r') {
                row.pop();
            }
        }
        let invalid = |detail: String| DictionaryError::InvalidRow { line, detail };
        if !body {
            body = row.trim() == "...";
            continue;
        }
        if row.trim().is_empty() || row.starts_with('#') {
            continue;
        }
        let record =
            parse_row(&row, &mut dictionary.strings, &mut dictionary.ends).map_err(invalid)?;
        let entry = record.entry(&dictionary.strings, &dictionary.ends);
        if entry.text.bytes().all(|byte| byte.is_ascii_alphabetic()) {
            if user {
                return Err(invalid(
                    "English words are unsupported in user dictionaries".into(),
                ));
            }
            dictionary.strings.truncate(record.text_start as usize);
            dictionary.ends.truncate(record.ends.start as usize);
            continue;
        }
        if user && entry.key.len() > USER_MAX_KEY_BYTES {
            return Err(invalid("user pinyin exceeds 128 bytes".into()));
        }
        if user && count == USER_MAX_ENTRIES {
            return Err(invalid("user dictionary exceeds 10000 entries".into()));
        }
        dictionary
            .entries
            .len()
            .checked_add(1)
            .and_then(|len| u32::try_from(len).ok())
            .ok_or_else(|| invalid("dictionary exceeds 4294967295 entries".into()))?;
        dictionary.entries.push(record);
        count += 1;
    }
    if !body {
        return Err(
            io::Error::new(io::ErrorKind::InvalidData, "missing ... header terminator").into(),
        );
    }
    Ok(())
}

fn string_end(start: usize, extra: usize) -> Result<u32, &'static str> {
    u32::try_from(
        start
            .checked_add(extra)
            .ok_or("dictionary string pool length overflow")?,
    )
    .map_err(|_| "dictionary string pool exceeds 4294967295 bytes")
}

/// Text, whitespace-separated lowercase syllables and an optional nonnegative weight.
fn parse_row(row: &str, strings: &mut String, ends: &mut Vec<u16>) -> Result<Record, String> {
    let mut fields = row.split('\t');
    let (Some(text), Some(pinyin), weight, None) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return Err("expected two or three TSV columns".into());
    };
    if text.chars().any(char::is_control) {
        return Err("text contains a control character".into());
    }
    let text_start = string_end(strings.len(), 0)?;
    let key_start = string_end(strings.len(), text.len())?;
    strings.push_str(text);
    let ends_start = u32::try_from(ends.len())
        .map_err(|_| "dictionary syllable pool exceeds 4294967295 elements")?;
    let mut ends_end = ends_start;
    for syllable in pinyin.split_whitespace() {
        if !syllable.bytes().all(|byte| byte.is_ascii_lowercase()) {
            return Err(format!("invalid pinyin {syllable:?}"));
        }
        let key_end = string_end(strings.len(), syllable.len())?;
        let end = u16::try_from(key_end - key_start).map_err(|_| "pinyin exceeds 65535 bytes")?;
        ends_end = ends_end
            .checked_add(1)
            .ok_or("dictionary syllable pool exceeds 4294967295 elements")?;
        strings.push_str(syllable);
        ends.push(end);
    }
    let key_end = string_end(strings.len(), 0)?;
    if text.is_empty() || key_end == key_start {
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
    Ok(Record {
        text_start,
        key_start,
        key_end,
        ends: ends_start..ends_end,
        weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CandidateSource, Decoder, Session};
    use std::{io::Cursor, sync::Arc};

    fn read_table(reader: impl BufRead, user: bool) -> Result<Dictionary, DictionaryError> {
        let mut dictionary = Dictionary::empty();
        super::read(reader, user, &mut dictionary)?;
        Ok(dictionary)
    }
    fn read_path(path: &Path, user: bool) -> Result<Dictionary, DictionaryError> {
        let mut dictionary = Dictionary::empty();
        super::read_file(path, user, &mut dictionary)?;
        Ok(dictionary)
    }

    #[test]
    fn english_words_are_dropped_from_public_tables_and_rejected_from_user_tables() {
        let chinese = "...\n你\tni\t10\n公婆\tgong po\t30\nB站\tbi zhan\t40\n";
        let public = Dictionary::from_rime_reader(Cursor::new(format!(
            "{chinese}GPU\tgpu\t100\nRust\trust\t100\n"
        )))
        .unwrap();
        assert_eq!(public.len(), 3);
        for row in ["Rust\trust", "GPU\tgpu", "abc\ta b c"] {
            let error = read_table(Cursor::new(format!("...\n{row}\n")), true).unwrap_err();
            assert!(
                matches!(error, DictionaryError::InvalidRow { line: 2, ref detail }
                    if detail.contains("English words are unsupported")),
                "{error}"
            );
        }
        let digits = read_table(Cursor::new("...\n123\tyi er san\n"), true).unwrap();
        assert_eq!(digits.entry(0).text, "123");
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
                        read_table(Cursor::new(format!("...\n{row}\n")), user),
                        Err(DictionaryError::InvalidRow { line: 2, .. })
                    ),
                    "{row:?}, user={user}"
                );
            }
            assert!(read_table(Cursor::new("no header\n你\tni\n"), user).is_err());
        }
        assert!(read_table(Cursor::new("...\n"), true).unwrap().is_empty());
        assert!(matches!(
            Dictionary::from_rime_reader(Cursor::new("...\n")),
            Err(DictionaryError::Empty)
        ));
        let max_key = format!("...\n词\t{}\n", "a".repeat(USER_MAX_KEY_BYTES));
        assert_eq!(read_table(Cursor::new(&max_key), true).unwrap().len(), 1);
        let too_long = format!("...\n词\t{}\n", "a".repeat(USER_MAX_KEY_BYTES + 1));
        assert!(read_table(Cursor::new(too_long), true).is_err());
        let max_rows = format!("...\n{}", "词\tni\n".repeat(USER_MAX_ENTRIES));
        assert_eq!(
            read_table(Cursor::new(&max_rows), true).unwrap().len(),
            USER_MAX_ENTRIES
        );
        assert!(read_table(Cursor::new(max_rows + "词\tni\n"), true).is_err());

        let error = read_table(Cursor::new(b"...\n\xff"), false).unwrap_err();
        let cause = std::error::Error::source(&error).unwrap();
        assert_eq!(
            cause.downcast_ref::<io::Error>().unwrap().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn pooled_rows_preserve_unicode_syllabifications_and_weight_bits() {
        let dictionary = Dictionary::from_rime_reader(Cursor::new(
            "...\r\n西安\txi an\t1.125\r\n西安\txi an\t2.25\r\n西安\txian\t2.25\r\n先\txian\t-0.0\r\n",
        )).unwrap();
        assert_eq!(dictionary.len(), 3);
        let mut rows: Vec<_> = dictionary
            .exact("xian")
            .map(|entry| (entry.text, entry.ends.to_vec(), entry.weight.to_bits()))
            .collect();
        rows.sort();
        let mut expected = vec![
            ("西安", vec![2, 4], 2.25f32.to_bits()),
            ("西安", vec![4], 2.25f32.to_bits()),
            ("先", vec![4], (-0.0f32).to_bits()),
        ];
        expected.sort();
        assert_eq!(rows, expected);
        for row in ["GPU\tGPU", "Rust\trust\tNaN"] {
            assert!(matches!(
                Dictionary::from_rime_reader(Cursor::new(format!("...\n{row}\n"))),
                Err(DictionaryError::InvalidRow { line: 2, .. })
            ));
        }
    }

    #[test]
    fn pool_offsets_and_pinyin_length_reject_overflow() {
        assert_eq!(string_end(u32::MAX as usize - 1, 1), Ok(u32::MAX));
        assert!(string_end(u32::MAX as usize, 1).is_err());
        assert!(string_end(usize::MAX, 1).is_err());
        let longest = "a".repeat(usize::from(u16::MAX));
        let dictionary = read_table(Cursor::new(format!("...\n词\t{longest}\n")), false).unwrap();
        assert_eq!(dictionary.entry(0).ends, &[u16::MAX]);
        let error = read_table(Cursor::new(format!("...\n词\t{longest}a\n")), false).unwrap_err();
        assert!(
            matches!(error, DictionaryError::InvalidRow { line: 2, ref detail }
            if detail == "pinyin exceeds 65535 bytes")
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
        assert_eq!(dictionary.exact("yanliu").next().unwrap().weight, 20.0);
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
            read_path(&user, true)
                .unwrap_err()
                .to_string()
                .contains("exceeds 8 MiB")
        );
        let error = read_path(&directory.join("missing.dict.yaml"), true).unwrap_err();
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
