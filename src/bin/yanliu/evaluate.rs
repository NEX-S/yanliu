//! Candidate checks for JSONL cases; every case stays in the denominator.
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fmt::Display,
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::Arc,
};
use yanliu::{Candidate, CandidateSource, Decoder, Session, model::Model};

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    id: String,
    input: String,
    #[serde(default)]
    context: String,
    #[serde(rename = "gold", alias = "golds", deserialize_with = "targets")]
    golds: Vec<String>,
    #[serde(default)]
    bucket: String,
}

fn targets<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Targets {
        One(String),
        Many(Vec<String>),
    }
    Ok(match Targets::deserialize(de)? {
        Targets::One(text) => vec![text],
        Targets::Many(texts) => texts,
    })
}

#[derive(Default, Serialize)]
struct Metrics {
    n: usize,
    top1: usize,
    top5: usize,
    recall: usize,
    static_top1: usize,
    static_top5: usize,
    rescues: usize,
    breakages: usize,
}

impl Metrics {
    fn add(&mut self, static_rank: Option<usize>, rank: Option<usize>) {
        self.n += 1;
        self.top1 += usize::from(rank == Some(1));
        self.top5 += usize::from(rank.is_some_and(|rank| rank <= 5));
        self.recall += usize::from(rank.is_some());
        self.static_top1 += usize::from(static_rank == Some(1));
        self.static_top5 += usize::from(static_rank.is_some_and(|rank| rank <= 5));
        self.rescues += usize::from(static_rank != Some(1) && rank == Some(1));
        self.breakages += usize::from(static_rank == Some(1) && rank != Some(1));
    }
}

/// One-based rank of a target converting the whole input; a Completion includes untyped text.
fn rank(session: &Session, golds: &[String]) -> Option<usize> {
    session
        .candidates()
        .iter()
        .position(|candidate| {
            golds.contains(&candidate.text)
                && candidate.consumed == session.input().len()
                && candidate.source != CandidateSource::Completion
        })
        .map(|index| index + 1)
}

fn candidate_values(candidates: &[Candidate]) -> Vec<Value> {
    candidates
        .iter()
        .map(|candidate| {
            json!({
                "text": candidate.text, "consumed": candidate.consumed,
                "source": format!("{:?}", candidate.source), "score_bits": candidate.score.to_bits(),
            })
        })
        .collect()
}

pub fn run(
    data: &Path,
    output: &Path,
    decoder: &Arc<Decoder>,
    mut model: Option<Model>,
) -> super::Result<()> {
    let reader: Box<dyn BufRead> = if data == Path::new("-") {
        Box::new(io::stdin().lock())
    } else {
        let file = File::open(data).map_err(|error| format!("{}: {error}", data.display()))?;
        Box::new(BufReader::new(file))
    };
    let mut buckets: BTreeMap<String, Metrics> = BTreeMap::new();
    let mut rows = Vec::new();
    for (line, text) in reader.lines().enumerate() {
        let at = |error: &dyn Display| format!("{}:{}: {error}", data.display(), line + 1);
        let case: Case =
            serde_json::from_str(&text.map_err(|error| at(&error))?).map_err(|error| at(&error))?;
        if case.input.is_empty() || case.golds.is_empty() || case.golds.iter().any(String::is_empty)
        {
            return Err(at(&"input and every gold must be nonempty").into());
        }
        let mut session = Session::new(decoder.clone(), None);
        session.set_context(&case.context);
        // Type one character at a time, as IBus does.
        for ch in case.input.chars() {
            session
                .insert(&ch.to_string())
                .map_err(|error| at(&error))?;
        }
        let static_rank = rank(&session, &case.golds);
        let mut scoring = None;
        if let Some(model) = &mut model {
            let mut snapshot = session.snapshot();
            let promotion = model
                .refine(&mut snapshot, decoder, || true)?
                .expect("synchronous CLI inference cannot be cancelled");
            scoring = Some(json!({
                "candidates": candidate_values(&snapshot.candidates),
                "promotion": promotion,
            }));
            session.apply_candidates(
                snapshot.revision,
                snapshot.candidates,
                promotion.index,
                &promotion.order,
            );
        }
        let rank = rank(&session, &case.golds);
        buckets
            .entry("all".into())
            .or_default()
            .add(static_rank, rank);
        if !case.bucket.is_empty() && case.bucket != "all" {
            buckets
                .entry(case.bucket)
                .or_default()
                .add(static_rank, rank);
        }
        let candidates = candidate_values(session.candidates());
        let id = if case.id.is_empty() {
            (line + 1).to_string()
        } else {
            case.id
        };
        rows.push(json!({"id": id, "rank": rank, "static_rank": static_rank,
                   "candidates": candidates, "scoring": scoring}));
    }
    if rows.is_empty() {
        return Err(format!("{}: no cases", data.display()).into());
    }
    let mut report = json!({"format": "yanliu-candidate-check-v1", "buckets": buckets});
    // Move rows: json! would clone the full candidate tree.
    report["rows"] = Value::Array(rows);
    if output == Path::new("-") {
        let mut writer = BufWriter::new(io::stdout().lock());
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writeln!(writer)?;
        writer.flush()?;
    } else {
        // Never replace an earlier report.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output)
            .map_err(|error| format!("{}: {error}", output.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.flush()?;
        println!("{}", serde_json::to_string_pretty(&report["buckets"])?);
    }
    Ok(())
}
