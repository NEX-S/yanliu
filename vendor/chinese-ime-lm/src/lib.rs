//! Character language-model inference and session-local scoring caches.
//! Callers select comparable candidates and combine model scores with engine evidence.

mod weights;

use std::collections::HashMap;
use std::sync::Arc;

use weights::{Tensor, Weights};

fn log_softmax_terms(logits: &[f32]) -> (f32, f64) {
    let mut highest = f32::NEG_INFINITY;
    for logit in logits.iter() {
        highest = highest.max(*logit);
    }
    let sum: f64 = logits
        .iter()
        .map(|logit| f64::from((logit - highest).exp()))
        .sum();
    (highest, sum.ln())
}

/// Stable terminal normalization; retain f32 exp, accumulate and subtract in f64.
fn log_softmax_in_place(logits: &mut [f32]) {
    let (highest, lnsum) = log_softmax_terms(logits);
    for logit in logits.iter_mut() {
        *logit = ((f64::from(*logit) - f64::from(highest)) - lnsum) as f32;
    }
}

/// A weight matrix, kept in whatever precision the file stored it in.
///
/// Quantized matrices are not expanded on load. A per-output-row scale is a constant factor for the
/// whole row, so it lifts out of the dot product and multiplies the result once — identical
/// arithmetic, a quarter of the memory. The model runs inside an iOS keyboard extension, where
/// turning a 4.5 MB file into 24.5 MB resident is most of what the process is allowed.
struct Matrix {
    inner: Tensor,
}

impl Matrix {
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// `out[row] = dot(x, row) * scale[row]`. Several quantized rows share one pass over `x` so
    /// their independent sums overlap; each row still uses the same eight-lane sum as `dot_i8`.
    fn write_gemv(&self, x: &[f32], width: usize, out: &mut [f32]) {
        match &self.inner {
            Tensor::Float(values) => {
                for (row, slot) in out.iter_mut().enumerate() {
                    let start = row * width;
                    *slot = dot(x, &values[start..start + width]);
                }
            }
            Tensor::Quantized { values, scales } => {
                quantized_gemv(x, values, scales, width, out);
            }
        }
    }

    fn write_gemm(&self, x: &[f32], width: usize, outputs: usize, out: &mut [f32]) {
        if let Tensor::Quantized { values, scales } = &self.inner {
            let pairs = x.len() / width / 2;
            for pair in 0..pairs {
                let source = &x[pair * 2 * width..(pair + 1) * 2 * width];
                let target = &mut out[pair * 2 * outputs..(pair + 1) * 2 * outputs];
                let (first, second) = target.split_at_mut(outputs);
                quantized_gemv_pair(
                    [&source[..width], &source[width..]],
                    values,
                    scales,
                    width,
                    [first, second],
                );
            }
            if x.len() / width % 2 != 0 {
                self.write_gemv(
                    &x[pairs * 2 * width..],
                    width,
                    &mut out[pairs * 2 * outputs..],
                );
            }
        } else {
            for (source, target) in x.chunks_exact(width).zip(out.chunks_exact_mut(outputs)) {
                self.write_gemv(source, width, target);
            }
        }
    }

    /// One row as `f32`, for the token table, which is read as an embedding as well as multiplied
    /// as an output projection. A row is `n_embd` values, so materializing it costs nothing.
    fn row(&self, index: usize, width: usize) -> Vec<f32> {
        let span = index * width..(index + 1) * width;
        match &self.inner {
            Tensor::Float(values) => values[span].to_vec(),
            Tensor::Quantized { values, scales } => {
                let scale = scales[index];
                values[span].iter().map(|&v| f32::from(v) * scale).collect()
            }
        }
    }
}

const UNK: u32 = 1;
const BOS: u32 = 2;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model file is truncated")]
    Truncated,
    #[error("model header: {0}")]
    Header(String),
    #[error("model is missing tensor {0}")]
    MissingTensor(String),
    #[error("model is missing metadata {0}")]
    MissingMetadata(String),
    #[error("model geometry: {0}")]
    Geometry(String),
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
struct Config {
    vocab: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    context: usize,
}

struct Block {
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    qkv_weight: Matrix,
    qkv_bias: Vec<f32>,
    proj_weight: Matrix,
    proj_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
    fc_weight: Matrix,
    fc_bias: Vec<f32>,
    out_weight: Matrix,
    out_bias: Vec<f32>,
}

pub struct SentenceModel {
    config: Config,
    index: HashMap<char, u32>,
    token: Matrix,
    position: Vec<f32>,
    blocks: Vec<Block>,
    final_weight: Vec<f32>,
    final_bias: Vec<f32>,
    attribution: String,
    license: String,
}

impl std::fmt::Debug for SentenceModel {
    /// The shape and provenance, never the weights: they run to tens of millions of values and
    /// nothing useful is learned from seeing them.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SentenceModel")
            .field("vocab", &self.config.vocab)
            .field("n_layer", &self.config.n_layer)
            .field("n_head", &self.config.n_head)
            .field("n_embd", &self.config.n_embd)
            .field("context", &self.config.context)
            .field("attribution", &self.attribution)
            .finish()
    }
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Reranker")
            .field("model", &self.model)
            .field("prefix_cached", &self.cached.is_some())
            .finish()
    }
}

/// The per-layer keys and values for a prefix, so that scoring several candidates against the same
/// committed text runs the prefix once instead of once per candidate.
pub struct Prefix {
    length: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Log probabilities from this prefix state, reused for every candidate opening.
    last_distribution: Vec<f32>,
    /// Only the uncommitted path, so a search state can become a ranking resume point.
    hidden: Vec<f32>,
    cumulative: Vec<f32>,
}

/// Everything one candidate's run passed through, so that a later run agreeing with part of it can
/// start from where they stop agreeing instead of from the beginning.
struct TailRun {
    /// How many prefix positions sit in front of `keys` and `values`, which hold the prefix's
    /// entries followed by this run's.
    base: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Contiguous normalized hidden rows, each `n_embd` wide, which predict the next position.
    hidden: Vec<f32>,
    /// Log-probability of the run's first `i + 1` tokens, summed left to right.
    cumulative: Vec<f32>,
    /// Search already projects its final state; retain that result for a complete resume.
    last_distribution: Option<Vec<f32>>,
}

impl TailRun {
    /// The state after `taken` of this run's tokens, as a prefix a continuation can run against.
    ///
    /// The keys and values are truncated rather than rebuilt: position `taken`'s cache is the
    /// leading part of the one already held.
    fn resume(&self, taken: usize, n_embd: usize, model: &SentenceModel) -> (Prefix, f32) {
        let width = (self.base + taken) * n_embd;
        let last_distribution = match &self.last_distribution {
            Some(distribution) if taken == self.cumulative.len() => distribution.clone(),
            _ => model.distribution(&self.hidden[(taken - 1) * n_embd..taken * n_embd]),
        };
        (
            Prefix {
                length: self.base + taken,
                keys: self.keys.iter().map(|key| key[..width].to_vec()).collect(),
                values: self
                    .values
                    .iter()
                    .map(|value| value[..width].to_vec())
                    .collect(),
                last_distribution,
                hidden: self.hidden[..taken * n_embd].to_vec(),
                cumulative: self.cumulative[..taken].to_vec(),
            },
            self.cumulative[taken - 1],
        )
    }

    /// This run's leading `taken` positions followed by `rest`, as one run over the whole
    /// candidate, so the next keystroke can resume anywhere along it.
    fn joined(&self, taken: usize, n_embd: usize, carried: f32, rest: TailRun) -> TailRun {
        TailRun {
            base: self.base,
            keys: rest.keys,
            values: rest.values,
            hidden: self.hidden[..taken * n_embd]
                .iter()
                .copied()
                .chain(rest.hidden)
                .collect(),
            cumulative: self.cumulative[..taken]
                .iter()
                .copied()
                .chain(rest.cumulative.iter().map(|value| carried + value))
                .collect(),
            last_distribution: rest.last_distribution,
        }
    }
}

impl SentenceModel {
    pub fn load(bytes: &[u8]) -> Result<Self, ModelError> {
        let mut weights = Weights::parse(bytes)?;
        let config: Config = serde_json::from_str(weights.metadata("config")?)
            .map_err(|error| ModelError::Header(error.to_string()))?;
        let vocabulary: Vec<String> = serde_json::from_str(weights.metadata("vocab")?)
            .map_err(|error| ModelError::Header(error.to_string()))?;
        let attribution = weights
            .metadata("attribution")
            .unwrap_or_default()
            .to_owned();
        let license = weights.metadata("license").unwrap_or_default().to_owned();

        // Every dimension below is multiplied out while checking the tensors, and all of them come
        // from the file. Bounding them here means the later arithmetic cannot overflow, which is
        // simpler to be sure of than checking each multiplication.
        const LARGEST: usize = 1 << 20;
        if config.vocab > LARGEST
            || config.n_layer > LARGEST
            || config.n_embd > LARGEST
            || config.context > LARGEST
        {
            return Err(ModelError::Geometry(
                "a dimension is larger than any real model".into(),
            ));
        }
        // `%` rather than `is_multiple_of`, which needs a far newer compiler than anything else here.
        if config.n_head == 0 || config.n_embd == 0 || config.n_embd % config.n_head != 0 {
            return Err(ModelError::Geometry(format!(
                "{} channels do not divide into {} heads",
                config.n_embd, config.n_head
            )));
        }
        if vocabulary.len() != config.vocab {
            return Err(ModelError::Geometry(format!(
                "vocabulary holds {} tokens but the configuration says {}",
                vocabulary.len(),
                config.vocab
            )));
        }

        let index = vocabulary
            .iter()
            .enumerate()
            .filter_map(|(id, token)| {
                let mut chars = token.chars();
                match (chars.next(), chars.next()) {
                    // Reserved tokens are spelled `<pad>` and are addressed by constant, not by text.
                    (Some(ch), None) => Some((ch, id as u32)),
                    _ => None,
                }
            })
            .collect();

        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let name = |suffix: &str| format!("blocks.{layer}.{suffix}");
            blocks.push(Block {
                ln1_weight: weights.take(&name("ln1.weight"))?,
                ln1_bias: weights.take(&name("ln1.bias"))?,
                qkv_weight: Matrix {
                    inner: weights.take_tensor(&name("qkv.weight"))?,
                },
                qkv_bias: weights.take(&name("qkv.bias"))?,
                proj_weight: Matrix {
                    inner: weights.take_tensor(&name("proj.weight"))?,
                },
                proj_bias: weights.take(&name("proj.bias"))?,
                ln2_weight: weights.take(&name("ln2.weight"))?,
                ln2_bias: weights.take(&name("ln2.bias"))?,
                fc_weight: Matrix {
                    inner: weights.take_tensor(&name("fc.weight"))?,
                },
                fc_bias: weights.take(&name("fc.bias"))?,
                out_weight: Matrix {
                    inner: weights.take_tensor(&name("out.weight"))?,
                },
                out_bias: weights.take(&name("out.bias"))?,
            });
        }

        let model = Self {
            config,
            index,
            token: Matrix {
                inner: weights.take_tensor("tok.weight")?,
            },
            position: weights.take("pos.weight")?,
            blocks,
            final_weight: weights.take("ln_f.weight")?,
            final_bias: weights.take("ln_f.bias")?,
            attribution,
            license,
        };
        model.check_geometry()?;
        Ok(model)
    }

    fn check_geometry(&self) -> Result<(), ModelError> {
        let Config {
            vocab,
            n_embd,
            context,
            ..
        } = self.config;
        let expect = |name: &str, actual: usize, wanted: usize| {
            (actual == wanted).then_some(()).ok_or_else(|| {
                ModelError::Geometry(format!("{name} holds {actual}, wanted {wanted}"))
            })
        };
        expect("tok.weight", self.token.len(), vocab * n_embd)?;
        expect("pos.weight", self.position.len(), context * n_embd)?;
        expect("ln_f.weight", self.final_weight.len(), n_embd)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            expect(
                &format!("blocks.{layer}.qkv"),
                block.qkv_weight.len(),
                3 * n_embd * n_embd,
            )?;
            expect(
                &format!("blocks.{layer}.proj"),
                block.proj_weight.len(),
                n_embd * n_embd,
            )?;
            expect(
                &format!("blocks.{layer}.fc"),
                block.fc_weight.len(),
                4 * n_embd * n_embd,
            )?;
            expect(
                &format!("blocks.{layer}.out"),
                block.out_weight.len(),
                4 * n_embd * n_embd,
            )?;
        }
        Ok(())
    }

    /// The corpus attribution carried in the weights file. Both corpora require it to travel with a
    /// model trained on them, which is why it lives in the file rather than beside it.
    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// The licence of the **weights**, as an SPDX identifier, which is not the licence of the code
    /// that produced them. Empty when the file does not say — older files do not, and a caller that
    /// needs to know should treat silence as unknown rather than as permission.
    pub fn license(&self) -> &str {
        &self.license
    }

    pub fn context_length(&self) -> usize {
        self.config.context
    }

    pub fn contains_character(&self, character: char) -> bool {
        self.index.contains_key(&character)
    }

    /// Probability of a known character after this prefix; unknowns do not become UNK paths.
    pub fn next_log_probability(&self, prefix: &Prefix, character: char) -> Option<f32> {
        self.index
            .get(&character)
            .map(|&token| prefix.last_distribution[token as usize])
    }

    /// Advance independent branches together so their projections share weight reads.
    pub fn advance(&self, branches: &[(&Prefix, char)]) -> Option<Vec<Prefix>> {
        let n_embd = self.config.n_embd;
        let mut x = Vec::with_capacity(branches.len() * n_embd);
        let mut advanced = Vec::with_capacity(branches.len());
        for &(prefix, character) in branches {
            let &token = self.index.get(&character)?;
            assert!(prefix.length < self.config.context);
            x.extend(self.embed(&[token], prefix.length));
            advanced.push(Prefix {
                length: prefix.length + 1,
                keys: Vec::with_capacity(self.blocks.len()),
                values: Vec::with_capacity(self.blocks.len()),
                last_distribution: Vec::new(),
                hidden: prefix.hidden.clone(),
                cumulative: {
                    let mut cumulative = prefix.cumulative.clone();
                    cumulative.push(
                        cumulative.last().copied().unwrap_or(0.0)
                            + prefix.last_distribution[token as usize],
                    );
                    cumulative
                },
            });
        }
        for (layer, block) in self.blocks.iter().enumerate() {
            let normed = layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
            let qkv = linear(
                &normed,
                &block.qkv_weight,
                &block.qkv_bias,
                n_embd,
                3 * n_embd,
            );
            let mut attended = Vec::with_capacity(x.len());
            // Rows are different sequences: causal attention must never see another branch's KV.
            for (row, &(prefix, _)) in branches.iter().enumerate() {
                let qkv = &qkv[row * 3 * n_embd..(row + 1) * 3 * n_embd];
                let mut keys = Vec::with_capacity(prefix.keys[layer].len() + n_embd);
                let mut values = Vec::with_capacity(prefix.values[layer].len() + n_embd);
                keys.extend_from_slice(&prefix.keys[layer]);
                values.extend_from_slice(&prefix.values[layer]);
                keys.extend_from_slice(&qkv[n_embd..2 * n_embd]);
                values.extend_from_slice(&qkv[2 * n_embd..]);
                attended.extend(self.attention(&qkv[..n_embd], &keys, &values, 1, prefix.length));
                advanced[row].keys.push(keys);
                advanced[row].values.push(values);
            }
            let projected = linear(
                &attended,
                &block.proj_weight,
                &block.proj_bias,
                n_embd,
                n_embd,
            );
            for (slot, value) in x.iter_mut().zip(projected) {
                *slot += value;
            }
            self.feed_forward(&mut x, block);
        }
        let normed = layer_norm(&x, &self.final_weight, &self.final_bias, n_embd);
        let mut logits = vec![0.0; branches.len() * self.config.vocab];
        self.token
            .write_gemm(&normed, n_embd, self.config.vocab, &mut logits);
        for ((prefix, distribution), hidden) in advanced
            .iter_mut()
            .zip(logits.chunks_exact_mut(self.config.vocab))
            .zip(normed.chunks_exact(n_embd))
        {
            log_softmax_in_place(distribution);
            prefix.last_distribution = distribution.to_vec();
            prefix.hidden.extend_from_slice(hidden);
        }
        Some(advanced)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|ch| self.index.get(&ch).copied().unwrap_or(UNK))
            .collect()
    }

    #[cfg(test)]
    fn score(&self, context: &str, candidates: &[&str]) -> Vec<f32> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let longest = candidates
            .iter()
            .map(|text| text.chars().count())
            .max()
            .unwrap_or(0);
        let prefix = self.run_prefix(&self.prefix_tokens(context, longest));
        self.score_against(&prefix, candidates)
    }

    /// The committed text as tokens, opened with `BOS` and trimmed to leave room for the longest
    /// candidate.
    ///
    /// The position table is the hard limit on sequence length, so the context yields rather than
    /// the candidate: a truncated candidate would be scored on text the user did not type, whereas
    /// a truncated context only weakens the conditioning.
    fn prefix_tokens(&self, context: &str, longest: usize) -> Vec<u32> {
        let room = self.config.context.saturating_sub(longest + 1);
        let encoded = self.encode(context);
        let start = encoded.len().saturating_sub(room);
        let mut tokens = Vec::with_capacity(encoded.len() - start + 1);
        tokens.push(BOS);
        tokens.extend_from_slice(&encoded[start..]);
        tokens
    }

    #[cfg(test)]
    fn score_against(&self, prefix: &Prefix, candidates: &[&str]) -> Vec<f32> {
        candidates
            .iter()
            .map(|text| {
                let tail = self.encode(text);
                if tail.is_empty() {
                    return f32::NEG_INFINITY;
                }
                let limit = self.config.context.saturating_sub(prefix.length).max(1);
                let tail = &tail[..tail.len().min(limit)];
                self.score_tail(prefix, tail) / tail.len() as f32
            })
            .collect()
    }

    fn embed(&self, tokens: &[u32], offset: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut x = vec![0.0; tokens.len() * n_embd];
        for (step, &token) in tokens.iter().enumerate() {
            let token = (token as usize).min(self.config.vocab - 1);
            let position = (offset + step).min(self.config.context - 1);
            let row = &mut x[step * n_embd..(step + 1) * n_embd];
            let token_row = self.token.row(token, n_embd);
            let position_row = &self.position[position * n_embd..(position + 1) * n_embd];
            for ((out, &t), &p) in row.iter_mut().zip(&token_row).zip(position_row) {
                *out = t + p;
            }
        }
        x
    }

    /// Runs the prefix and keeps every layer's keys and values.
    fn run_prefix(&self, tokens: &[u32]) -> Prefix {
        let n_embd = self.config.n_embd;
        let mut x = self.embed(tokens, 0);
        let mut keys = Vec::with_capacity(self.blocks.len());
        let mut values = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let normed = layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
            let qkv = linear(
                &normed,
                &block.qkv_weight,
                &block.qkv_bias,
                n_embd,
                3 * n_embd,
            );
            let (q, k, v) = split_qkv(&qkv, tokens.len(), n_embd);
            let attended = self.attention(&q, &k, &v, tokens.len(), 0);
            let projected = linear(
                &attended,
                &block.proj_weight,
                &block.proj_bias,
                n_embd,
                n_embd,
            );
            for (slot, value) in x.iter_mut().zip(projected) {
                *slot += value;
            }
            self.feed_forward(&mut x, block);
            keys.push(k);
            values.push(v);
        }
        let last = (tokens.len() - 1) * n_embd;
        let last_hidden = layer_norm(&x[last..], &self.final_weight, &self.final_bias, n_embd);
        let last_distribution = self.distribution(&last_hidden);
        Prefix {
            length: tokens.len(),
            keys,
            values,
            last_distribution,
            hidden: Vec::new(),
            cumulative: Vec::new(),
        }
    }

    /// Runs a candidate against a cached prefix and returns its total log-probability.
    #[cfg(test)]
    fn score_tail(&self, prefix: &Prefix, tail: &[u32]) -> f32 {
        let run = self.run_tail(prefix, tail);
        run.cumulative[run.cumulative.len() - 1]
    }

    /// Runs a candidate, retaining keys, values and normalized hidden rows at every position so
    /// later candidates can resume at any shared prefix.
    ///
    /// Resuming adds a carried score to each new local cumulative sum. Keep this grouping: f32
    /// addition cannot be reassociated without changing score bits.
    fn run_tail(&self, prefix: &Prefix, tail: &[u32]) -> TailRun {
        let n_embd = self.config.n_embd;
        let mut x = self.embed(tail, prefix.length);
        let mut keys_out = Vec::with_capacity(self.blocks.len());
        let mut values_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let normed = layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
            let qkv = linear(
                &normed,
                &block.qkv_weight,
                &block.qkv_bias,
                n_embd,
                3 * n_embd,
            );
            let (q, k, v) = split_qkv(&qkv, tail.len(), n_embd);
            let mut keys = Vec::with_capacity(prefix.keys[layer].len() + k.len());
            let mut values = Vec::with_capacity(prefix.values[layer].len() + v.len());
            keys.extend_from_slice(&prefix.keys[layer]);
            values.extend_from_slice(&prefix.values[layer]);
            keys.extend_from_slice(&k);
            values.extend_from_slice(&v);
            let attended = self.attention(&q, &keys, &values, tail.len(), prefix.length);
            let projected = linear(
                &attended,
                &block.proj_weight,
                &block.proj_bias,
                n_embd,
                n_embd,
            );
            for (slot, value) in x.iter_mut().zip(projected) {
                *slot += value;
            }
            self.feed_forward(&mut x, block);
            keys_out.push(keys);
            values_out.push(values);
        }
        let normed = layer_norm(&x, &self.final_weight, &self.final_bias, n_embd);

        // Position i predicts token i + 1. The opening character is therefore predicted by the last
        // prefix position, and every later one by the tail position before it.
        let first_token = (tail[0] as usize).min(self.config.vocab - 1);
        let mut running = prefix.last_distribution[first_token];
        let mut cumulative = Vec::with_capacity(tail.len());
        cumulative.push(running);
        for step in 0..tail.len() {
            let row = &normed[step * n_embd..(step + 1) * n_embd];
            if step + 1 < tail.len() {
                running += self.log_probability(row, tail[step + 1]);
                cumulative.push(running);
            }
        }
        TailRun {
            base: prefix.length,
            keys: keys_out,
            values: values_out,
            hidden: normed,
            cumulative,
            last_distribution: None,
        }
    }

    fn feed_forward(&self, x: &mut [f32], block: &Block) {
        let n_embd = self.config.n_embd;
        let normed = layer_norm(x, &block.ln2_weight, &block.ln2_bias, n_embd);
        let mut hidden = linear(
            &normed,
            &block.fc_weight,
            &block.fc_bias,
            n_embd,
            4 * n_embd,
        );
        for value in &mut hidden {
            *value = gelu(*value);
        }
        let out = linear(
            &hidden,
            &block.out_weight,
            &block.out_bias,
            4 * n_embd,
            n_embd,
        );
        for (slot, value) in x.iter_mut().zip(out) {
            *slot += value;
        }
    }

    /// log P(token) under the tied output projection, computed as the logit minus the log of the
    /// summed exponentials. The full vocabulary is needed for the denominator, which is why this is
    /// the most expensive step per scored character.
    fn log_probability(&self, hidden: &[f32], token: u32) -> f32 {
        let token = (token as usize).min(self.config.vocab - 1);
        let n_embd = self.config.n_embd;
        let mut logits = vec![0.0; self.config.vocab];
        self.token.write_gemv(hidden, n_embd, &mut logits);
        let (highest, lnsum) = log_softmax_terms(&logits);
        ((f64::from(logits[token]) - f64::from(highest)) - lnsum) as f32
    }

    /// log P over the whole vocabulary for the state at `hidden`.
    fn distribution(&self, hidden: &[f32]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut logits = vec![0.0; self.config.vocab];
        self.token.write_gemv(hidden, n_embd, &mut logits);
        log_softmax_in_place(&mut logits);
        logits
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], rows: usize, offset: usize) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let heads = self.config.n_head;
        let width = n_embd / heads;
        let scale = 1.0 / (width as f32).sqrt();
        let total = k.len() / n_embd;
        let mut out = vec![0.0; rows * n_embd];
        let mut scores = vec![0.0; total];
        for row in 0..rows {
            // Causal: a query at absolute position `offset + row` may attend to itself and earlier.
            let visible = offset + row + 1;
            for head in 0..heads {
                let base = head * width;
                let query = &q[row * n_embd + base..row * n_embd + base + width];
                let mut highest = f32::NEG_INFINITY;
                for (step, score) in scores[..visible].iter_mut().enumerate() {
                    let key = &k[step * n_embd + base..step * n_embd + base + width];
                    *score = dot(query, key) * scale;
                    highest = highest.max(*score);
                }
                let mut sum = 0.0;
                for score in &mut scores[..visible] {
                    *score = (*score - highest).exp();
                    sum += *score;
                }
                let target = &mut out[row * n_embd + base..row * n_embd + base + width];
                for (step, &score) in scores[..visible].iter().enumerate() {
                    let weight = score / sum;
                    let value = &v[step * n_embd + base..step * n_embd + base + width];
                    for (slot, &item) in target.iter_mut().zip(value) {
                        *slot += weight * item;
                    }
                }
            }
        }
        out
    }
}

/// A model plus the prefix it last ran, so that typing does not re-run the committed text.
///
/// The prefix is the expensive half of a short decision — nineteen characters of context cost about
/// as much as the candidates themselves — and it only changes when something is committed, not on
/// every keystroke. Holding it across keystrokes is what puts the decision inside a frame.
pub struct Reranker {
    /// Shared, because the weights are read-only and a host opens a session per focus change.
    /// Only the cached prefix belongs to one session.
    model: Arc<SentenceModel>,
    cached: Option<(Vec<u32>, Prefix)>,
    /// Previous ranking runs followed by the current search's finished parents. Scoring replaces
    /// both with its actual candidates; keeping older frames would grow without a ceiling.
    resume: Vec<(Vec<u32>, Arc<TailRun>)>,
    /// Search seeds live only until ranking or the next search, including neutral-score frames.
    ranked_len: usize,
}

impl Reranker {
    pub fn new(model: Arc<SentenceModel>) -> Self {
        Self {
            model,
            cached: None,
            resume: Vec::new(),
            ranked_len: 0,
        }
    }

    pub fn model(&self) -> &SentenceModel {
        &self.model
    }

    /// Reuse the committed prefix while reserving room for a complete constrained path.
    pub fn prefix(
        &mut self,
        context: &str,
        reserve: usize,
    ) -> Result<(&Self, &Prefix), ModelError> {
        self.resume.truncate(self.ranked_len);
        if reserve >= self.model.config.context {
            return Err(ModelError::Geometry(format!(
                "{reserve} output characters leave no BOS position in a {}-position model",
                self.model.config.context
            )));
        }
        self.cache_prefix(context, reserve);
        Ok((
            self,
            &self.cached.as_ref().expect("prefix was just computed").1,
        ))
    }

    /// Reuse a known nonempty path from the previous ranking under the prepared root.
    pub fn reuse_prefix(&self, text: &str) -> Option<Prefix> {
        let tokens = self.model.encode(text);
        let (_, run) = self
            .resume
            .iter()
            .rev()
            .find(|(previous, _)| previous.starts_with(&tokens))?;
        Some(
            run.resume(tokens.len(), self.model.config.n_embd, &self.model)
                .0,
        )
    }

    /// Retain the parents of the final search beam without running its last characters twice.
    pub fn retain_search_prefixes(&mut self, prefixes: Vec<(String, Prefix)>) {
        self.resume
            .extend(prefixes.into_iter().map(|(text, prefix)| {
                let run = TailRun {
                    base: prefix.length - prefix.cumulative.len(),
                    keys: prefix.keys,
                    values: prefix.values,
                    hidden: prefix.hidden,
                    cumulative: prefix.cumulative,
                    last_distribution: Some(prefix.last_distribution),
                };
                (self.model.encode(&text), Arc::new(run))
            }));
    }

    fn cache_prefix(&mut self, context: &str, longest: usize) {
        // Keyed on the tokens rather than the context string, because two different contexts that
        // truncate to the same window are the same prefix and must not invalidate each other.
        let tokens = self.model.prefix_tokens(context, longest);
        let matched = self
            .cached
            .as_ref()
            .is_some_and(|(cached, _)| *cached == tokens);
        if !matched {
            let prefix = self.model.run_prefix(&tokens);
            self.cached = Some((tokens, prefix));
            // Every resume point hangs off the old prefix and means nothing against a new one.
            self.resume.clear();
            self.ranked_len = 0;
        }
    }

    /// Mean log-probability per character for each candidate, in the order given.
    ///
    /// Exposed because the model's opinion is only one term in the decision: combining it with the
    /// evidence the engine already has is a different question from letting it decide alone, and
    /// answering that question needs the numbers rather than the verdict.
    pub fn log_probabilities(&mut self, context: &str, candidates: &[&str]) -> Vec<f32> {
        let longest = candidates
            .iter()
            .map(|text| text.chars().count())
            .max()
            .unwrap_or(0);
        self.cache_prefix(context, longest);
        let (_, prefix) = self.cached.as_ref().expect("prefix was just computed");
        let n_embd = self.model.config.n_embd;

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum SourceKind {
            Previous,
            Current,
        }
        // Candidates may resume the same run at the same position. Project its hidden state once.
        let mut resumed: Vec<(SourceKind, usize, usize, Prefix, f32)> = Vec::new();
        let mut next: Vec<(Vec<u32>, Arc<TailRun>)> = Vec::with_capacity(candidates.len());
        let mut scores = Vec::with_capacity(candidates.len());
        for text in candidates {
            let tail = self.model.encode(text);
            if tail.is_empty() {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let limit = self
                .model
                .config
                .context
                .saturating_sub(prefix.length)
                .max(1);
            let tail = &tail[..tail.len().min(limit)];
            // The prefix already predicts every first character; a single character has no
            // continuation to run or state worth retaining for the next decision.
            if tail.len() == 1 {
                let token = (tail[0] as usize).min(self.model.config.vocab - 1);
                scores.push(prefix.last_distribution[token]);
                continue;
            }

            // A complete match needs neither a new last-token run nor a new output projection.
            if let Some((_, run)) = self
                .resume
                .iter()
                .rev()
                .chain(next.iter().rev())
                .find(|(tokens, _)| tokens.as_slice() == tail)
            {
                let run = Arc::clone(run);
                scores.push(run.cumulative[tail.len() - 1] / tail.len() as f32);
                next.push((tail.to_vec(), run));
                continue;
            }

            // Complete matches were handled above; a partial match leaves at least one token
            // for run_tail, using the furthest-agreeing previous or current candidate.
            let previous = self
                .resume
                .iter()
                .map(|(tokens, _)| {
                    let agreed = tokens
                        .iter()
                        .zip(tail)
                        .take_while(|(left, right)| left == right)
                        .count();
                    agreed.min(tail.len() - 1)
                })
                .enumerate()
                .max_by_key(|(_, agreed)| *agreed);
            let current = next
                .iter()
                .map(|(tokens, _)| {
                    tokens
                        .iter()
                        .zip(tail)
                        .take_while(|(left, right)| left == right)
                        .count()
                        .min(tail.len() - 1)
                })
                .enumerate()
                .max_by_key(|(_, agreed)| *agreed);
            // Keep the previous-call source on ties, including its existing max_by_key tie rule.
            let shared = match (previous, current) {
                (old, Some((index, agreed))) if agreed > old.map_or(0, |(_, n)| n) => {
                    Some((SourceKind::Current, index, agreed))
                }
                (Some((index, agreed)), _) => Some((SourceKind::Previous, index, agreed)),
                _ => None,
            };

            let run = match shared {
                Some((kind, index, agreed)) if agreed > 0 => {
                    let source = match kind {
                        SourceKind::Previous => &self.resume[index].1,
                        SourceKind::Current => &next[index].1,
                    };
                    let cached = match resumed.iter().position(|(source_kind, old, taken, _, _)| {
                        *source_kind == kind && *old == index && *taken == agreed
                    }) {
                        Some(cached) => cached,
                        None => {
                            let (from, carried) = source.resume(agreed, n_embd, &self.model);
                            resumed.push((kind, index, agreed, from, carried));
                            resumed.len() - 1
                        }
                    };
                    let (_, _, _, from, carried) = &resumed[cached];
                    let rest = self.model.run_tail(from, &tail[agreed..]);
                    source.joined(agreed, n_embd, *carried, rest)
                }
                _ => self.model.run_tail(prefix, tail),
            };
            let total = run.cumulative[run.cumulative.len() - 1];
            scores.push(total / tail.len() as f32);
            next.push((tail.to_vec(), Arc::new(run)));
        }
        self.resume = next;
        self.ranked_len = self.resume.len();
        scores
    }
}

/// Independent accumulators rather than one running sum.
///
/// A single accumulator makes every multiply-add wait for the previous one, so the loop runs at the
/// latency of the fused multiply-add instead of its throughput and leaves most of the core idle.
/// Eight partial sums give the pipeline eight independent chains to interleave. The summation order
/// differs from a serial reduction, which is immaterial here: the result feeds a ranking, and the
/// order was never guaranteed to match the trainer's anyway.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut partial = [0.0f32; LANES];
    let left = a.chunks_exact(LANES);
    let right = b.chunks_exact(LANES);
    let (tail_left, tail_right) = (left.remainder(), right.remainder());
    for (x, y) in left.zip(right) {
        for lane in 0..LANES {
            partial[lane] += x[lane] * y[lane];
        }
    }
    let mut total = partial.iter().sum::<f32>();
    for (x, y) in tail_left.iter().zip(tail_right) {
        total += x * y;
    }
    total
}

/// `out[m][o] = dot(x[m], w[o]) + bias[o]`, with both operands laid out row-major so every dot
/// product runs over contiguous memory.
fn linear(x: &[f32], w: &Matrix, bias: &[f32], inputs: usize, outputs: usize) -> Vec<f32> {
    let rows = x.len() / inputs;
    let mut out = vec![0.0; rows * outputs];
    w.write_gemm(x, inputs, outputs, &mut out);
    for target in out.chunks_exact_mut(outputs) {
        for (slot, &term) in target.iter_mut().zip(bias) {
            *slot += term;
        }
    }
    out
}

/// Eight independent output rows. Each row keeps the eight-lane sum of `dot_i8`; the rows do not
/// share accumulators, so a row's bits match scoring it alone.
fn dot8(x: &[f32], rows: [&[i8]; 8]) -> [f32; 8] {
    const LANES: usize = 8;
    let mut p0 = [0.0f32; LANES];
    let mut p1 = [0.0f32; LANES];
    let mut p2 = [0.0f32; LANES];
    let mut p3 = [0.0f32; LANES];
    let mut p4 = [0.0f32; LANES];
    let mut p5 = [0.0f32; LANES];
    let mut p6 = [0.0f32; LANES];
    let mut p7 = [0.0f32; LANES];
    let n = x.len() / LANES;
    for chunk in 0..n {
        let at = chunk * LANES;
        let x = &x[at..at + LANES];
        let y0 = &rows[0][at..at + LANES];
        let y1 = &rows[1][at..at + LANES];
        let y2 = &rows[2][at..at + LANES];
        let y3 = &rows[3][at..at + LANES];
        let y4 = &rows[4][at..at + LANES];
        let y5 = &rows[5][at..at + LANES];
        let y6 = &rows[6][at..at + LANES];
        let y7 = &rows[7][at..at + LANES];
        for lane in 0..LANES {
            p0[lane] += x[lane] * f32::from(y0[lane]);
            p1[lane] += x[lane] * f32::from(y1[lane]);
            p2[lane] += x[lane] * f32::from(y2[lane]);
            p3[lane] += x[lane] * f32::from(y3[lane]);
            p4[lane] += x[lane] * f32::from(y4[lane]);
            p5[lane] += x[lane] * f32::from(y5[lane]);
            p6[lane] += x[lane] * f32::from(y6[lane]);
            p7[lane] += x[lane] * f32::from(y7[lane]);
        }
    }
    let mut totals = [
        sum_lanes(p0),
        sum_lanes(p1),
        sum_lanes(p2),
        sum_lanes(p3),
        sum_lanes(p4),
        sum_lanes(p5),
        sum_lanes(p6),
        sum_lanes(p7),
    ];
    for index in n * LANES..x.len() {
        totals[0] += x[index] * f32::from(rows[0][index]);
        totals[1] += x[index] * f32::from(rows[1][index]);
        totals[2] += x[index] * f32::from(rows[2][index]);
        totals[3] += x[index] * f32::from(rows[3][index]);
        totals[4] += x[index] * f32::from(rows[4][index]);
        totals[5] += x[index] * f32::from(rows[5][index]);
        totals[6] += x[index] * f32::from(rows[6][index]);
        totals[7] += x[index] * f32::from(rows[7][index]);
    }
    totals
}

fn sum_lanes(lanes: [f32; 8]) -> f32 {
    let mut total = 0.0;
    for lane in lanes {
        total += lane;
    }
    total
}

fn quantized_gemv(x: &[f32], values: &[i8], scales: &[f32], width: usize, out: &mut [f32]) {
    let rows = out.len();
    let mut row = 0;
    while row + 8 <= rows {
        let at = |offset: usize| &values[(row + offset) * width..(row + offset + 1) * width];
        let dots = dot8(x, [at(0), at(1), at(2), at(3), at(4), at(5), at(6), at(7)]);
        for offset in 0..8 {
            out[row + offset] = dots[offset] * scales[row + offset];
        }
        row += 8;
    }
    while row < rows {
        let start = row * width;
        out[row] = dot_i8(x, &values[start..start + width]) * scales[row];
        row += 1;
    }
}

// Two input rows share each weight conversion without changing either dot product's reduction.
fn dot2x4(x: [&[f32]; 2], rows: [&[i8]; 4]) -> [[f32; 4]; 2] {
    const LANES: usize = 8;
    let mut p00 = [0.0f32; LANES];
    let mut p01 = [0.0f32; LANES];
    let mut p02 = [0.0f32; LANES];
    let mut p03 = [0.0f32; LANES];
    let mut p10 = [0.0f32; LANES];
    let mut p11 = [0.0f32; LANES];
    let mut p12 = [0.0f32; LANES];
    let mut p13 = [0.0f32; LANES];
    let n = x[0].len() / LANES;
    for chunk in 0..n {
        let at = chunk * LANES;
        let x0 = &x[0][at..at + LANES];
        let x1 = &x[1][at..at + LANES];
        let y0 = &rows[0][at..at + LANES];
        let y1 = &rows[1][at..at + LANES];
        let y2 = &rows[2][at..at + LANES];
        let y3 = &rows[3][at..at + LANES];
        for lane in 0..LANES {
            let y0 = f32::from(y0[lane]);
            let y1 = f32::from(y1[lane]);
            let y2 = f32::from(y2[lane]);
            let y3 = f32::from(y3[lane]);
            p00[lane] += x0[lane] * y0;
            p01[lane] += x0[lane] * y1;
            p02[lane] += x0[lane] * y2;
            p03[lane] += x0[lane] * y3;
            p10[lane] += x1[lane] * y0;
            p11[lane] += x1[lane] * y1;
            p12[lane] += x1[lane] * y2;
            p13[lane] += x1[lane] * y3;
        }
    }
    let mut totals = [
        [
            sum_lanes(p00),
            sum_lanes(p01),
            sum_lanes(p02),
            sum_lanes(p03),
        ],
        [
            sum_lanes(p10),
            sum_lanes(p11),
            sum_lanes(p12),
            sum_lanes(p13),
        ],
    ];
    for index in n * LANES..x[0].len() {
        for (row, weight) in rows.iter().enumerate() {
            let weight = f32::from(weight[index]);
            totals[0][row] += x[0][index] * weight;
            totals[1][row] += x[1][index] * weight;
        }
    }
    totals
}

fn quantized_gemv_pair(
    x: [&[f32]; 2],
    values: &[i8],
    scales: &[f32],
    width: usize,
    out: [&mut [f32]; 2],
) {
    let [first, second] = out;
    let rows = first.len();
    let mut row = 0;
    while row + 4 <= rows {
        let at = |offset: usize| &values[(row + offset) * width..(row + offset + 1) * width];
        let dots = dot2x4(x, [at(0), at(1), at(2), at(3)]);
        for offset in 0..4 {
            first[row + offset] = dots[0][offset] * scales[row + offset];
            second[row + offset] = dots[1][offset] * scales[row + offset];
        }
        row += 4;
    }
    while row < rows {
        let start = row * width;
        let weights = &values[start..start + width];
        first[row] = dot_i8(x[0], weights) * scales[row];
        second[row] = dot_i8(x[1], weights) * scales[row];
        row += 1;
    }
}

/// The quantized counterpart of `dot`, with the same eight independent accumulators and the same
/// reason for them. The row's scale is applied by the caller, once, rather than to every value:
/// that is the whole point of keeping the weights as `i8`.
fn dot_i8(a: &[f32], b: &[i8]) -> f32 {
    const LANES: usize = 8;
    let mut partial = [0.0f32; LANES];
    let left = a.chunks_exact(LANES);
    let right = b.chunks_exact(LANES);
    let (tail_left, tail_right) = (left.remainder(), right.remainder());
    for (x, y) in left.zip(right) {
        for lane in 0..LANES {
            partial[lane] += x[lane] * f32::from(y[lane]);
        }
    }
    let mut total = partial.iter().sum::<f32>();
    for (x, y) in tail_left.iter().zip(tail_right) {
        total += x * f32::from(*y);
    }
    total
}

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (source, target) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let mean = source.iter().sum::<f32>() / width as f32;
        let variance = source
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + 1e-5).sqrt();
        for (index, slot) in target.iter_mut().enumerate() {
            *slot = (source[index] - mean) * inverse * weight[index] + bias[index];
        }
    }
    out
}

fn split_qkv(qkv: &[f32], rows: usize, n_embd: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; rows * n_embd];
    let mut k = vec![0.0; rows * n_embd];
    let mut v = vec![0.0; rows * n_embd];
    for row in 0..rows {
        let source = &qkv[row * 3 * n_embd..(row + 1) * 3 * n_embd];
        q[row * n_embd..(row + 1) * n_embd].copy_from_slice(&source[..n_embd]);
        k[row * n_embd..(row + 1) * n_embd].copy_from_slice(&source[n_embd..2 * n_embd]);
        v[row * n_embd..(row + 1) * n_embd].copy_from_slice(&source[2 * n_embd..]);
    }
    (q, k, v)
}

/// The exact GELU, matching `torch.nn.functional.gelu` with its default settings. The tanh
/// approximation is a different function and the model was not trained with it.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz and Stegun 7.1.26, whose error stays below 1.5e-7 across the range.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::{erf, gelu};

    #[test]
    fn stable_normalizer_matches_independent_f64_mass_and_shifts() {
        fn ulp32(value: f64) -> f64 {
            let exponent = ((value as f32).to_bits() >> 23) & 255;
            2.0_f64.powi(if exponent == 0 {
                -149
            } else {
                exponent as i32 - 127 - 23
            })
        }
        for size in [3, 257, 12417] {
            for family in 0..4 {
                let original: Vec<f32> = (0..size)
                    .map(|i| match family {
                        0 => 0.0,
                        1 => {
                            if i == 0 {
                                0.0
                            } else {
                                -16.0
                            }
                        }
                        2 => -((i % 257) as f32) / 8.0,
                        _ => {
                            if i == 0 {
                                0.0
                            } else {
                                [-2.0, -16.0, -80.0, -1000.0][(i - 1) % 4]
                            }
                        }
                    })
                    .collect();
                let mut base = original.clone();
                super::log_softmax_in_place(&mut base);
                for shift in [0.0, -1024.0, 32.0, 1024.0] {
                    let supplied: Vec<f32> = original.iter().map(|x| x + shift).collect();
                    assert!(supplied
                        .iter()
                        .zip(&original)
                        .all(|(&x, &y)| x - shift == y));
                    // Independent reference: exact supplied f32 logits, f64 subtraction AND exp.
                    let highest = supplied.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let log_sum = supplied
                        .iter()
                        .map(|&x| (f64::from(x) - highest).exp())
                        .sum::<f64>()
                        .ln();
                    let mut actual = supplied.clone();
                    super::log_softmax_in_place(&mut actual);
                    for ((&got, &input), &unshifted) in actual.iter().zip(&supplied).zip(&base) {
                        let reference = (f64::from(input) - highest) - log_sum;
                        assert!(got.is_finite());
                        assert!(
                            (f64::from(got) - reference).abs() <= 2.0 * ulp32(reference) + 2e-6
                        );
                        assert!((got - unshifted).abs() <= 2e-6);
                    }
                    let mass = actual.iter().map(|&x| f64::from(x).exp()).sum::<f64>();
                    assert!((mass - 1.0).abs() <= 5e-6);
                }
            }
        }
    }

    fn reranker() -> super::Reranker {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/models/sentence-model.safetensors");
        let model = super::SentenceModel::load(&std::fs::read(path).expect("model asset"))
            .expect("valid model");
        super::Reranker::new(std::sync::Arc::new(model))
    }

    #[test]
    fn constrained_steps_match_full_tail_and_reject_unknown_characters() {
        let mut scorer = reranker();
        let (prepared, root) = scorer.prefix("昨天", 3).unwrap();
        let model = prepared.model();
        assert!(model.next_log_probability(root, '\u{10ffff}').is_none());
        assert!(model.advance(&[(root, '\u{10ffff}')]).is_none());
        let first = model.advance(&[(root, '我')]).unwrap().remove(0);
        let second = model.advance(&[(&first, '今')]).unwrap().remove(0);
        let actual = model.next_log_probability(root, '我').unwrap()
            + model.next_log_probability(&first, '今').unwrap()
            + model.next_log_probability(&second, '天').unwrap();
        let expected = model.score_tail(root, &model.encode("我今天"));
        assert!((actual - expected).abs() <= 1e-5);
        assert!(scorer.prefix("", scorer.model().context_length()).is_err());
    }

    #[test]
    fn search_parents_feed_ranking_and_expire_after_a_skipped_decision() {
        fn parent(scorer: &mut super::Reranker, context: &str, reserve: usize) -> super::Prefix {
            let (prepared, root) = scorer.prefix(context, reserve).unwrap();
            let first = prepared.model().advance(&[(root, '我')]).unwrap().remove(0);
            prepared
                .model()
                .advance(&[(&first, '们')])
                .unwrap()
                .remove(0)
        }
        let mut scorer = reranker();
        let candidates = ["我们去", "我们来", "我来"];
        let expected = reranker().log_probabilities("昨天", &candidates);
        scorer.log_probabilities("昨天", &["今天去", "明天来"]);
        let ranked = scorer.resume.len();
        let seed = parent(&mut scorer, "昨天", 3);
        scorer.retain_search_prefixes(vec![("我们".into(), seed)]);
        assert_eq!(scorer.resume.len(), ranked + 1);
        // A protected/neutral frame never scores. Starting another search must drop its seeds.
        let seed = parent(&mut scorer, "昨天", 3);
        assert_eq!(scorer.resume.len(), ranked);
        scorer.retain_search_prefixes(vec![("我们".into(), seed)]);
        let actual = scorer.log_probabilities("昨天", &candidates);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5);
        }
        assert_eq!(scorer.resume.len(), candidates.len());
        assert_eq!(scorer.ranked_len, candidates.len());

        let seed = {
            let (prepared, _) = scorer.prefix("昨天", 4).unwrap();
            let reused = prepared.reuse_prefix("我们").unwrap();
            prepared
                .model()
                .advance(&[(&reused, '一')])
                .unwrap()
                .remove(0)
        };
        scorer.retain_search_prefixes(vec![("我们一".into(), seed)]);
        let candidates = ["我们一起", "我们一同"];
        let actual = scorer.log_probabilities("昨天", &candidates);
        let expected = reranker().log_probabilities("昨天", &candidates);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5);
        }

        // Reserving a different output length near the window edge changes the effective root.
        let context = "昨天".repeat(40);
        let seed = parent(&mut scorer, &context, 5);
        scorer.retain_search_prefixes(vec![("我们".into(), seed)]);
        let actual = scorer.log_probabilities(&context, &candidates);
        let expected = reranker().log_probabilities(&context, &candidates);
        assert_eq!(actual, expected);
    }

    #[test]
    fn search_reuses_only_the_matching_root_and_character_prefix() {
        let mut scorer = reranker();
        scorer.log_probabilities("昨天", &["我今天去上班", "我今天去工作"]);
        let (prepared, root) = scorer.prefix("昨天", 8).unwrap();
        let model = prepared.model();
        assert!(prepared.reuse_prefix("我们").is_none());
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        for text in ["我", "我今", "我今天", "我今天去上班"] {
            let reused = prepared.reuse_prefix(text).unwrap();
            let mut fresh = None;
            for character in text.chars() {
                let parent = fresh.as_ref().unwrap_or(root);
                fresh = Some(model.advance(&[(parent, character)]).unwrap().remove(0));
            }
            let fresh = fresh.unwrap();
            assert_eq!(reused.length, fresh.length);
            assert_eq!(bits(&reused.keys), bits(&fresh.keys));
            assert_eq!(bits(&reused.values), bits(&fresh.values));
            assert_eq!(
                reused
                    .last_distribution
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                fresh
                    .last_distribution
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let (prepared, _) = scorer.prefix("前天", 8).unwrap();
        assert!(prepared.reuse_prefix("我").is_none());

        let context = "昨天".repeat(40);
        scorer.log_probabilities(&context, &["我今天去上班"]);
        let (prepared, _) = scorer.prefix(&context, 3).unwrap();
        assert!(
            prepared.reuse_prefix("我").is_none(),
            "different window must drop old runs"
        );
    }

    #[test]
    fn independent_branches_keep_single_step_bits() {
        let mut scorer = reranker();
        let (prepared, root) = scorer.prefix("昨天", 6).unwrap();
        let model = prepared.model();
        let parents = model
            .advance(&[(root, '我'), (root, '今'), (root, '天')])
            .unwrap();
        // Different lengths and repeated parents expose accidental cross-branch attention.
        let branches = [
            (root, '他'),
            (&parents[0], '们'),
            (&parents[1], '天'),
            (&parents[0], '今'),
            (&parents[2], '气'),
        ];
        let actual = model.advance(&branches).unwrap();
        for ((prefix, character), actual) in branches.iter().zip(actual) {
            let run = model.run_tail(prefix, &[model.index[character]]);
            let expected = model.distribution(&run.hidden);
            assert_eq!(actual.length, prefix.length + 1);
            for (actual, expected) in actual.keys.iter().flatten().zip(run.keys.iter().flatten()) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            for (actual, expected) in actual
                .values
                .iter()
                .flatten()
                .zip(run.values.iter().flatten())
            {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            assert_eq!(
                actual
                    .last_distribution
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn batched_quantized_dots_match_single_rows() {
        let width = 192;
        let rows = 17;
        let mut values = vec![0i8; rows * width];
        let mut scales = vec![0.0; rows];
        let mut x = vec![0.0; width];
        for (index, value) in values.iter_mut().enumerate() {
            *value = (index * 17).wrapping_sub(40) as i8;
        }
        for (index, scale) in scales.iter_mut().enumerate() {
            *scale = 0.01 * (index as f32 + 1.0);
        }
        for (index, value) in x.iter_mut().enumerate() {
            *value = ((index % 9) as f32 - 4.0) * 0.125;
        }
        let mut out = vec![0.0; rows];
        super::quantized_gemv(&x, &values, &scales, width, &mut out);
        for row in 0..rows {
            let start = row * width;
            let alone = super::dot_i8(&x, &values[start..start + width]) * scales[row];
            assert_eq!(out[row].to_bits(), alone.to_bits());
        }
        let width = 20;
        let rows = 9;
        let mut values = vec![0i8; rows * width];
        for (index, value) in values.iter_mut().enumerate() {
            *value = index.wrapping_sub(3) as i8;
        }
        let scales = vec![0.5; rows];
        let x: Vec<f32> = (0..width).map(|index| index as f32 * -0.25).collect();
        let mut out = vec![0.0; rows];
        super::quantized_gemv(&x, &values, &scales, width, &mut out);
        for row in 0..rows {
            let start = row * width;
            let alone = super::dot_i8(&x, &values[start..start + width]) * scales[row];
            assert_eq!(out[row].to_bits(), alone.to_bits());
        }
    }

    #[test]
    fn matrix_batches_preserve_single_row_bits_and_bias() {
        for (width, outputs) in [
            (192_usize, 17_usize),
            (768, 9),
            (195, 7),
            (771, 5),
            (192, 9358),
        ] {
            let values: Vec<i8> = (0..width * outputs)
                .map(|index| (index * 17).wrapping_sub(40) as i8)
                .collect();
            let scales: Vec<f32> = (0..outputs)
                .map(|index| 0.0007 * (index as f32 + 1.0))
                .collect();
            let bias: Vec<f32> = (0..outputs)
                .map(|index| ((index * 53 % 101) as f32 - 50.0) / 29.0)
                .collect();
            for quantized in [true, false] {
                let inner = if quantized {
                    super::Tensor::Quantized {
                        values: values.clone(),
                        scales: scales.clone(),
                    }
                } else {
                    super::Tensor::Float(
                        values
                            .iter()
                            .map(|&value| f32::from(value) / 37.0)
                            .collect(),
                    )
                };
                let matrix = super::Matrix { inner };
                for batch in [0, 1, 2, 3, 7, 12] {
                    let x: Vec<f32> = (0..batch * width)
                        .map(|index| ((index * 37 % 251) as f32 - 125.0) / 13.0)
                        .collect();
                    let mut expected = vec![0.0; batch * outputs];
                    for (source, target) in x
                        .chunks_exact(width)
                        .zip(expected.chunks_exact_mut(outputs))
                    {
                        matrix.write_gemv(source, width, target);
                    }
                    let mut actual = vec![0.0; expected.len()];
                    matrix.write_gemm(&x, width, outputs, &mut actual);
                    assert_eq!(
                        actual
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "width={width}, outputs={outputs}, batch={batch}, quantized={quantized}"
                    );
                    for target in expected.chunks_exact_mut(outputs) {
                        for (slot, term) in target.iter_mut().zip(&bias) {
                            *slot += term;
                        }
                    }
                    let actual = super::linear(&x, &matrix, &bias, width, outputs);
                    assert_eq!(
                        actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        "biased width={width}, outputs={outputs}, batch={batch}, quantized={quantized}"
                    );
                }
            }
        }
    }

    #[test]
    fn tail_cumulative_matches_serial_distributions_bit_for_bit() {
        let scorer = reranker();
        let model = scorer.model();
        let tokens = model.encode("我今天去上班回来");
        for context in ["", "昨天", "医生调整了药物的"] {
            let prefix = model.run_prefix(&model.prefix_tokens(context, tokens.len()));
            for length in 1..=tokens.len() {
                let tail = &tokens[..length];
                let run = model.run_tail(&prefix, tail);
                assert_eq!(run.hidden.len(), length * model.config.n_embd);
                assert_eq!(run.cumulative.len(), length);
                let mut expected = prefix.last_distribution[tail[0] as usize];
                assert_eq!(run.cumulative[0].to_bits(), expected.to_bits());
                for (step, (&token, row)) in tail[1..]
                    .iter()
                    .zip(run.hidden.chunks_exact(model.config.n_embd))
                    .enumerate()
                {
                    expected += model.distribution(row)[token as usize];
                    assert_eq!(
                        run.cumulative[step + 1].to_bits(),
                        expected.to_bits(),
                        "context={context:?}, length={length}, step={}",
                        step + 1
                    );
                }
            }
        }
    }

    #[test]
    fn resumed_scores_keep_exact_bits() {
        let mut scorer = reranker();
        let first = ["我今天去上班", "我今天去工作"];
        let cold = scorer.log_probabilities("昨天", &first);
        let extended = scorer.log_probabilities("昨天", &["我今天去上班了", "我今天去工作"]);
        assert_eq!(cold[1].to_bits(), extended[1].to_bits());
        let mut again = reranker();
        let forward = again.log_probabilities("昨天", &first);
        let backward = again.log_probabilities("昨天", &["我今天去工作", "我今天去上班"]);
        assert_eq!(forward[0].to_bits(), backward[1].to_bits());
        assert_eq!(forward[1].to_bits(), backward[0].to_bits());
    }

    #[test]
    fn single_characters_use_the_prefix_distribution_with_exact_score_bits() {
        let mut scorer = reranker();
        for context in ["", "医生调整了药物的", &"昨天".repeat(50)] {
            for candidates in [vec!["剂", "纸", "我"], vec!["我", "我们", "我今天", "剂"]]
            {
                let expected = scorer.model.score(context, &candidates);
                let actual = scorer.log_probabilities(context, &candidates);
                for ((text, actual), expected) in candidates.iter().zip(actual).zip(expected) {
                    if text.chars().count() == 1 {
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
                assert!(scorer.resume.iter().all(|(tail, _)| tail.len() > 1));
            }
        }
        scorer.log_probabilities("昨天", &["我", "今"]);
        assert!(scorer.resume.is_empty());
        let candidates = ["我今天", "今天", "我"];
        let mut fresh = super::Reranker::new(std::sync::Arc::clone(&scorer.model));
        let expected = fresh.log_probabilities("昨天", &candidates);
        let actual = scorer.log_probabilities("昨天", &candidates);
        assert_eq!(
            actual
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn resume_cache_retains_only_previous_call() {
        let mut scorer = reranker();
        let candidates = ["我今天去上班", "我今天去上班了", "我今天去工作"];
        scorer.log_probabilities("昨天", &candidates);
        assert_eq!(scorer.resume.len(), candidates.len());
        scorer.log_probabilities("昨天", &["今天", "明天"]);
        assert_eq!(scorer.resume.len(), 2);
        let after = scorer.log_probabilities("昨天", &candidates);
        let mut fresh = super::Reranker::new(std::sync::Arc::clone(&scorer.model));
        assert_eq!(after, fresh.log_probabilities("昨天", &candidates));
    }

    #[test]
    fn changed_effective_prefix_invalidates_resume_state() {
        let mut scorer = reranker();
        let candidates = ["今天", "明天"];
        scorer.log_probabilities("昨天", &candidates);
        let changed = scorer.log_probabilities("我", &candidates);
        let mut fresh = super::Reranker::new(std::sync::Arc::clone(&scorer.model));
        assert_eq!(changed, fresh.log_probabilities("我", &candidates));
        assert_eq!(scorer.resume.len(), 2);
    }

    #[test]
    fn shared_candidate_branches_match_independent_runs() {
        let scorer = reranker();
        // Cold reuse, a previous short prefix beaten by a current longer one, and a
        // previous longer prefix or tie retained while current candidates branch.
        for (previous, candidates) in [
            (
                vec![],
                vec!["我今天去上班", "我今天去上班了", "我今天去工作"],
            ),
            (vec!["我今"], vec!["我今天去上班", "我今天去上班了"]),
            (vec!["我今天去上班"], vec!["我今天去工作", "我今天去上班了"]),
            (vec!["我今天去上班"], vec!["我今天去上课", "我今天去工作"]),
            (
                vec!["我今天去上班", "我今天去工作"],
                vec!["我今天去工作", "我今天去上班", "我今天去上班了"],
            ),
            (vec![], vec!["我今天", "我今天", "我今", "", "我"]),
        ] {
            let mut scorer = super::Reranker::new(std::sync::Arc::clone(&scorer.model));
            if !previous.is_empty() {
                scorer.log_probabilities("昨天", &previous);
            }
            let actual = scorer.log_probabilities("昨天", &candidates);
            let expected = scorer.model.score("昨天", &candidates);
            for (a, b) in actual.iter().zip(&expected) {
                if b.is_finite() {
                    assert!((a - b).abs() <= 0.000_01, "{a} != {b}");
                } else {
                    assert_eq!(a, b);
                }
            }
            assert_eq!(
                scorer.resume.len(),
                candidates.iter().filter(|t| t.chars().count() > 1).count()
            );
        }
        let mut scorer = super::Reranker::new(std::sync::Arc::clone(&scorer.model));
        assert_eq!(
            scorer.log_probabilities("", &["", "我"])[0],
            f32::NEG_INFINITY
        );
        assert_eq!(scorer.log_probabilities("", &[""]), vec![f32::NEG_INFINITY]);
    }

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(0.5) - 0.520_499_9).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-6);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-6);
        assert!((erf(3.0) - 0.999_977_9).abs() < 1e-6);
    }

    #[test]
    fn gelu_matches_torch_reference() {
        // Values taken from torch.nn.functional.gelu with its default exact formulation.
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_25).abs() < 1e-6);
        assert!((gelu(2.0) - 1.954_499_7).abs() < 1e-6);
    }
}
