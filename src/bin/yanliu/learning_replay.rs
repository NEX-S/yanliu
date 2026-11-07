//! Explicit public corrections in a newly created state, never the desktop's preferences.
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::{self, DirBuilder, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
    sync::Arc,
};
use yanliu::{Decoder, Session, learning::LearningWriter, model::Model};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    text: String,
    consumed: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    input: String,
    #[serde(default)]
    context: String,
    select: Option<Selection>,
    selects: Option<Vec<Selection>>,
}

fn refine(
    session: &mut Session,
    decoder: &Arc<Decoder>,
    model: &mut Model,
) -> super::Result<Value> {
    let mut snapshot = session.snapshot();
    let explanation = model
        .refine(&mut snapshot, decoder, || true)?
        .expect("synchronous CLI inference cannot be cancelled");
    let _ = session.apply_candidates(snapshot.revision, snapshot.candidates.clone(), &explanation);
    let candidates: Vec<_> = session.candidates().iter().map(|candidate| json!({
        "text": candidate.text, "consumed": candidate.consumed,
        "source": format!("{:?}", candidate.source), "score_bits": candidate.score.to_bits(),
        "before_rank": snapshot.candidates.iter().position(|c| c == candidate).unwrap() + 1,
    })).collect();
    let mut result = json!({"promotion": explanation});
    result["candidates"] = Value::Array(candidates);
    Ok(result)
}

fn query(
    case: &Case,
    decoder: &Arc<Decoder>,
    writer: &LearningWriter,
    model: &mut Model,
) -> super::Result<(Session, Value)> {
    let mut session = Session::new(decoder.clone(), Some(writer.learning.clone()));
    session.set_context(&case.context);
    for character in case.input.chars() {
        session.insert(&character.to_string())?;
    }
    let result = refine(&mut session, decoder, model)?;
    Ok((session, result))
}

pub fn run(
    data: &Path,
    state_dir: &Path,
    output: &Path,
    decoder: &Arc<Decoder>,
    mut model: Model,
) -> super::Result<()> {
    let mut cases = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, line) in BufReader::new(File::open(data)?).lines().enumerate() {
        let case: Case = serde_json::from_str(&line?)
            .map_err(|error| format!("{}:{}: {error}", data.display(), index + 1))?;
        if case.id.is_empty()
            || !ids.insert(case.id.clone())
            || case.input.is_empty()
            || !case
                .input
                .bytes()
                .all(|b| b.is_ascii_alphabetic() || b == b'\'')
        {
            return Err(format!(
                "{}:{}: nonempty unique id, ASCII pinyin and valid selection required",
                data.display(),
                index + 1
            )
            .into());
        }
        if case.select.is_some() && case.selects.is_some() {
            return Err(format!("{}: select and selects are mutually exclusive", case.id).into());
        }
        if case.selects.as_ref().is_some_and(Vec::is_empty) {
            return Err(format!("{}: selects must not be empty", case.id).into());
        }
        let choices = case.selects.as_deref().unwrap_or(case.select.as_slice());
        let mut remaining = case.input.len();
        for (step, choice) in choices.iter().enumerate() {
            if choice.text.is_empty() || choice.consumed == 0 || choice.consumed > remaining {
                return Err(format!(
                    "{}: selection step {} has invalid text or consumed bytes",
                    case.id,
                    step + 1
                )
                .into());
            }
            remaining -= choice.consumed;
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return Err("learning replay requires at least one case".into());
    }
    // Resolve an existing parent before creating anything; symlinks cannot redirect this run
    // into the system learning directory, and an existing state is never accepted.
    let parent = state_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let directory = parent.canonicalize()?.join(
        state_dir
            .file_name()
            .ok_or("state directory needs a name")?,
    );
    let system = yanliu::learning::state_directory()?;
    let mut ancestor = system.as_path();
    let system = loop {
        match ancestor.canonicalize() {
            Ok(path) => break path.join(system.strip_prefix(ancestor)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = ancestor
                    .parent()
                    .expect("the absolute system state path has a root");
            }
            Err(error) => return Err(error.into()),
        }
    };
    let mut normalized_system = std::path::PathBuf::new();
    for component in system.components() {
        if component == std::path::Component::ParentDir {
            normalized_system.pop();
        } else {
            normalized_system.push(component);
        }
    }
    let output_parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let output_path = output_parent
        .canonicalize()?
        .join(output.file_name().ok_or("output needs a filename")?);
    if directory.starts_with(&normalized_system) || output_path.starts_with(&normalized_system) {
        return Err("learning replay cannot use the system learning directory".into());
    }
    match fs::symlink_metadata(&directory) {
        Ok(_) => {
            return Err(format!("{}: state directory already exists", directory.display()).into());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&output_path)?;
    let mut report_file = BufWriter::new(report_file);
    DirBuilder::new().mode(0o700).create(&directory)?;
    let mut writer = LearningWriter::start(&directory, true, |error| eprintln!("{error}"))?;
    let mut rows = Vec::new();
    for case in &cases {
        let (mut session, before) = query(case, decoder, &writer, &mut model)?;
        let choices = case.selects.as_deref().unwrap_or(case.select.as_slice());
        let mut selections = Vec::new();
        for (step, selection) in choices.iter().enumerate() {
            let index = session
                .candidates()
                .iter()
                .position(|candidate| {
                    candidate.text == selection.text && candidate.consumed == selection.consumed
                })
                .ok_or_else(|| {
                    if case.selects.is_some() {
                        format!(
                            "{}: selection step {}: selected candidate not recalled",
                            case.id,
                            step + 1
                        )
                    } else {
                        format!("{}: selected candidate not recalled", case.id)
                    }
                })?;
            let committed = session.select(index).ok_or_else(|| {
                format!(
                    "{}: selection step {}: learning writer stopped before selection",
                    case.id,
                    step + 1
                )
            })?;
            selections.push(
                json!({"text": committed, "consumed": selection.consumed, "remaining_input": session.input()}),
            );
            if step + 1 < choices.len() {
                refine(&mut session, decoder, &mut model)?;
            }
        }
        let selection = if case.select.is_some() {
            selections.pop()
        } else {
            None
        };
        let (_, after) = query(case, decoder, &writer, &mut model)?;
        let mut row = json!({"id": case.id, "input": case.input, "context": case.context});
        row["before"] = before;
        row["selection"] = selection.unwrap_or(Value::Null);
        row["after"] = after;
        if case.selects.is_some() {
            row["selections"] = Value::Array(selections);
        }
        rows.push(row);
    }
    let mut query_all = |writer: &LearningWriter| -> super::Result<Vec<Value>> {
        let mut seen = BTreeSet::new();
        cases
            .iter()
            .filter(|case| seen.insert((&case.input, &case.context)))
            .map(|case| {
                let (_, result) = query(case, decoder, writer, &mut model)?;
                let mut row = json!({"input": case.input, "context": case.context});
                row["result"] = result;
                Ok(row)
            })
            .collect()
    };
    let read_snapshot = || match fs::read(directory.join("learning.json")) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    };
    let final_queries = query_all(&writer)?;
    writer.finish()?;
    let saved = read_snapshot()?;
    let mut writer = LearningWriter::start(&directory, true, |error| eprintln!("{error}"))?;
    let reloaded = query_all(&writer)?;
    writer.finish()?;
    let unchanged = read_snapshot()? == saved;
    let rankings_unchanged = reloaded == final_queries;
    let queries_after_reload = reloaded.len();
    let mut report = json!({"format": "yanliu-learning-replay-v1", "state_dir": directory,
        "reload_preserved_snapshot": unchanged, "reload_preserved_rankings": rankings_unchanged});
    report["rows"] = Value::Array(rows);
    report["reloaded"] = Value::Array(reloaded);
    report["before_reload"] = Value::Array(final_queries);
    serde_json::to_writer_pretty(&mut report_file, &report)?;
    report_file.flush()?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"cases": cases.len(), "queries_after_reload": queries_after_reload, "reload_preserved_snapshot": unchanged, "reload_preserved_rankings": rankings_unchanged})
        )?
    );
    Ok(())
}
