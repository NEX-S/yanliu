use super::{
    BEAM, Candidate, CandidateSource, Decoder, SPAN_ENTRIES, ScannedInput, allowed, scan_input,
};
use chinese_ime_lm::{ModelError, Prefix, Reranker, SentenceModel};
use std::{collections::HashSet, sync::Arc};

#[derive(Clone, Copy)]
struct Edge {
    end: usize,
    character: char,
    weight: f32,
}

struct Path {
    text: String,
    end: usize,
    language: f32,
    score: f32,
    prefix: Option<Arc<Prefix>>,
}

struct Extension {
    text: String,
    end: usize,
    language: f32,
    score: f32,
    parent: usize,
    character: char,
}

struct Finished {
    candidate: Candidate,
    language: f32,
    parent: Option<Arc<Prefix>>,
}

impl Decoder {
    /// Complete dictionary-constrained paths; `None` means the caller cancelled this search.
    pub fn neural_candidates(
        &self,
        raw: &str,
        context: &str,
        reranker: &mut Reranker,
        mut still_current: impl FnMut() -> bool,
    ) -> Result<Option<Vec<Candidate>>, ModelError> {
        if !still_current() {
            return Ok(None);
        }
        let Some(input) = scan_input(raw) else {
            return Ok(Some(Vec::new()));
        };
        let (arcs, longest) = self.neural_arcs(&input, reranker.model());
        // Static recall already retains every one-syllable character. Long targets must not be truncated.
        if longest <= 1 || longest >= reranker.model().context_length() {
            return Ok(Some(Vec::new()));
        }
        if !still_current() {
            return Ok(None);
        }
        let (prepared, root) = reranker.prefix(context, longest)?;
        let model = prepared.model();
        let mut paths = vec![Path {
            text: String::new(),
            end: 0,
            language: 0.0,
            score: 0.0,
            prefix: None,
        }];
        let mut finished: Vec<Finished> = Vec::new();
        for length in 1..=longest {
            if !still_current() {
                return Ok(None);
            }
            let mut extensions = Vec::new();
            for (parent, path) in paths.iter().enumerate() {
                if !still_current() {
                    return Ok(None);
                }
                let prefix = path.prefix.as_deref().unwrap_or(root);
                for arc in &arcs[path.end] {
                    let probability = model
                        .next_log_probability(prefix, arc.character)
                        .expect("DAG characters belong to the model vocabulary");
                    let mut text = path.text.clone();
                    text.push(arc.character);
                    extensions.push(Extension {
                        text,
                        end: arc.end,
                        language: path.language + probability,
                        score: path.score + (arc.weight + 1.0).ln() / 14.0 - 2.0,
                        parent,
                        character: arc.character,
                    });
                }
            }
            extensions.sort_by(|a, b| {
                b.language
                    .total_cmp(&a.language)
                    .then(b.score.total_cmp(&a.score))
                    .then(a.text.cmp(&b.text))
                    .then(a.end.cmp(&b.end))
            });
            let mut seen = HashSet::new();
            extensions.retain(|extension| seen.insert((extension.text.clone(), extension.end)));
            extensions.truncate(BEAM);
            let mut continuing = Vec::with_capacity(BEAM);
            for extension in extensions {
                if !still_current() {
                    return Ok(None);
                }
                if extension.end == input.key.len() {
                    finished.push(Finished {
                        candidate: Candidate {
                            text: extension.text,
                            consumed: input.raw_ends[extension.end],
                            source: if length == 1 {
                                CandidateSource::ExactDictionary
                            } else {
                                CandidateSource::Composed
                            },
                            score: extension.score,
                        },
                        language: extension.language / length as f32,
                        parent: paths[extension.parent].prefix.clone(),
                    });
                } else {
                    continuing.push(extension);
                }
            }
            let mut prefixes = Vec::with_capacity(continuing.len());
            for chunk in continuing.chunks(2) {
                if !still_current() {
                    return Ok(None);
                }
                let cached: Vec<_> = chunk
                    .iter()
                    .map(|extension| prepared.reuse_prefix(&extension.text))
                    .collect();
                let missing: Vec<_> = chunk
                    .iter()
                    .zip(&cached)
                    .filter(|(_, cached)| cached.is_none())
                    .map(|(extension, _)| {
                        (
                            paths[extension.parent].prefix.as_deref().unwrap_or(root),
                            extension.character,
                        )
                    })
                    .collect();
                let fresh = if missing.is_empty() {
                    Vec::new()
                } else {
                    model
                        .advance(&missing)
                        .expect("DAG characters belong to the model vocabulary")
                };
                let mut fresh = fresh.into_iter();
                prefixes.extend(cached.into_iter().map(|cached| {
                    cached.unwrap_or_else(|| {
                        fresh.next().expect("every uncached branch was advanced")
                    })
                }));
            }
            let next = continuing
                .into_iter()
                .zip(prefixes)
                .map(|(extension, prefix)| Path {
                    text: extension.text,
                    end: extension.end,
                    language: extension.language,
                    score: extension.score,
                    prefix: Some(Arc::new(prefix)),
                })
                .collect();
            finished.sort_by(|a, b| {
                b.language
                    .total_cmp(&a.language)
                    .then(b.candidate.score.total_cmp(&a.candidate.score))
                    .then(a.candidate.text.cmp(&b.candidate.text))
            });
            let mut seen = HashSet::new();
            finished.retain(|path| seen.insert(path.candidate.text.clone()));
            finished.truncate(BEAM);
            paths = next;
            if paths.is_empty() {
                break;
            }
        }
        if !still_current() {
            return Ok(None);
        }
        drop(paths);
        let mut candidates = Vec::with_capacity(finished.len());
        let mut parents = Vec::with_capacity(finished.len());
        let mut seen = HashSet::new();
        for path in finished {
            if let Some(prefix) = path.parent {
                let mut text = path.candidate.text.clone();
                text.pop();
                if seen.insert(text.clone()) {
                    parents.push((text, prefix));
                }
            }
            candidates.push(path.candidate);
        }
        reranker.retain_search_prefixes(
            parents
                .into_iter()
                .map(|(text, prefix)| {
                    let prefix = Arc::try_unwrap(prefix)
                        .unwrap_or_else(|_| panic!("search released all other parent references"));
                    (text, prefix)
                })
                .collect(),
        );
        Ok(Some(candidates))
    }

    fn neural_arcs(
        &self,
        input: &ScannedInput<'_>,
        model: &SentenceModel,
    ) -> (Vec<Vec<Edge>>, usize) {
        let n = input.key.len();
        let mut arcs: Vec<Vec<Edge>> = (0..=n).map(|_| Vec::new()).collect();
        let mut longest = vec![0; n + 1];
        let mut reachable = vec![false; n + 1];
        reachable[n] = true;
        for start in (0..n).rev() {
            let Some(maximum) = self.dictionary.longest_syllable(&input.key[start..]) else {
                continue;
            };
            for end in start + 1..=start + maximum {
                if !reachable[end] {
                    continue;
                }
                let mut seen = HashSet::new();
                for entry in self
                    .dictionary
                    .exact(&input.key[start..end])
                    .filter(|entry| {
                        entry.ends.len() == 1
                            && allowed(entry, start, end, &input.boundaries)
                            && entry.text.chars().count() == 1
                            && model.contains_character(entry.text.chars().next().unwrap())
                    })
                    .filter(|entry| seen.insert(entry.text))
                    .take(SPAN_ENTRIES)
                {
                    arcs[start].push(Edge {
                        end,
                        character: entry.text.chars().next().unwrap(),
                        weight: entry.weight,
                    });
                    longest[start] = longest[start].max(longest[end] + 1);
                    reachable[start] = true;
                }
            }
        }
        (arcs, longest[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dictionary;
    use std::{io::Cursor, sync::Arc};

    fn model() -> Reranker {
        let bytes = std::fs::read("data/models/sentence-model.safetensors").unwrap();
        Reranker::new(Arc::new(SentenceModel::load(&bytes).unwrap()))
    }

    fn decoder() -> Decoder {
        Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n西\txi\t100\n安\tan\t80\n现\txian\t120\n先\txian\t90\n西安\txi an\t200\n你好\tni hao\t100\n",
            ))
            .unwrap(),
        )
    }

    #[test]
    fn arcs_honor_readings_boundaries_and_require_a_complete_path() {
        let decoder = decoder();
        let reranker = model();
        for (input, expected) in [("xian", 2), ("xi'an", 2), ("xian'", 2)] {
            let scanned = scan_input(input).unwrap();
            let (arcs, longest) = decoder.neural_arcs(&scanned, reranker.model());
            assert_eq!(longest, expected);
            assert!(arcs[0].iter().any(|arc| arc.character == '西'));
            assert_eq!(
                arcs[0].iter().any(|arc| arc.character == '现'),
                input != "xi'an"
            );
        }
        for input in ["xiani", "nihao", "x"] {
            let (_, longest) = decoder.neural_arcs(&scan_input(input).unwrap(), reranker.model());
            assert_eq!(longest, 0, "{input}");
        }
        assert!(decoder.decode("nihao").iter().any(|c| c.text == "你好"));
    }

    #[test]
    fn one_character_paths_need_no_neural_recall_even_with_a_closing_apostrophe() {
        let decoder = decoder();
        let mut reranker = model();
        for input in ["xi", "xi'", "XI'"] {
            assert!(
                decoder
                    .decode(input)
                    .iter()
                    .any(|candidate| candidate.text == "西"
                        && candidate.consumed == input.len()
                        && candidate.source != CandidateSource::Completion)
            );
            assert!(
                decoder
                    .neural_candidates(input, "我去", &mut reranker, || true)
                    .unwrap()
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn paths_preserve_raw_consumption_static_scores_and_cancellation() {
        let decoder = decoder();
        let mut reranker = model();
        for input in ["xian", "xi'an", "Xi'An'"] {
            let found = decoder
                .neural_candidates(input, "我去", &mut reranker, || true)
                .unwrap()
                .unwrap();
            assert!(found.iter().any(|candidate| candidate.text == "西安"));
            assert!(found.len() <= BEAM);
            assert!(
                found
                    .iter()
                    .all(|candidate| candidate.consumed == input.len()
                        && candidate.source != CandidateSource::Completion)
            );
            let west = decoder.dictionary.exact("xi").next().unwrap();
            let peace = decoder.dictionary.exact("an").next().unwrap();
            let score = super::super::extended_score(
                super::super::extended_score(0.0, &west, false),
                &peace,
                false,
            );
            assert_eq!(
                found
                    .iter()
                    .find(|c| c.text == "西安")
                    .unwrap()
                    .score
                    .to_bits(),
                score.to_bits()
            );
        }
        let mut calls = 0;
        assert!(
            decoder
                .neural_candidates("xian", "我去", &mut reranker, || {
                    calls += 1;
                    calls < 5
                })
                .unwrap()
                .is_none()
        );
        assert!(
            decoder
                .neural_candidates(&"xi".repeat(64), "", &mut reranker, || true)
                .unwrap()
                .unwrap()
                .is_empty()
        );
    }
}
