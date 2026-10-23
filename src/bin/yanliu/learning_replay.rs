//! Explicit public corrections in a newly created state, never the desktop's preferences.
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::{self, DirBuilder, File, OpenOptions},
    io::{BufRead, BufReader, Write},
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
    let mut snapshot = session.snapshot();
    let explanation = model
        .refine(&mut snapshot, decoder, || true)?
        .expect("synchronous CLI inference cannot be cancelled");
    session.apply_candidates(
        snapshot.revision,
        snapshot.candidates.clone(),
        explanation.index,
        &explanation.order,
    );
    let candidates: Vec<_> = session.candidates().iter().map(|candidate| json!({
        "text": candidate.text, "consumed": candidate.consumed,
        "source": format!("{:?}", candidate.source), "score_bits": candidate.score.to_bits(),
        "before_rank": snapshot.candidates.iter().position(|c| c == candidate).unwrap() + 1,
    })).collect();
    Ok((
        session,
        json!({"candidates": candidates, "promotion": explanation}),
    ))
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
            || case.select.as_ref().is_some_and(|s| {
                s.text.is_empty() || s.consumed == 0 || s.consumed > case.input.len()
            })
        {
            return Err(format!(
                "{}:{}: nonempty unique id, ASCII pinyin and valid selection required",
                data.display(),
                index + 1
            )
            .into());
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
    let mut report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&output_path)?;
    DirBuilder::new().mode(0o700).create(&directory)?;
    let mut writer = LearningWriter::start(&directory, true, |error| eprintln!("{error}"))?;
    let mut rows = Vec::new();
    for case in &cases {
        let (mut session, before) = query(case, decoder, &writer, &mut model)?;
        let selection = if let Some(selection) = &case.select {
            let index = session
                .candidates()
                .iter()
                .position(|candidate| {
                    candidate.text == selection.text && candidate.consumed == selection.consumed
                })
                .ok_or_else(|| format!("{}: selected candidate not recalled", case.id))?;
            let committed = session
                .select(index)
                .ok_or("learning writer stopped before selection")?;
            Some(
                json!({"text": committed, "consumed": selection.consumed, "remaining_input": session.input()}),
            )
        } else {
            None
        };
        let (_, after) = query(case, decoder, &writer, &mut model)?;
        rows.push(
            json!({"id": case.id, "input": case.input, "context": case.context,
                         "before": before, "selection": selection, "after": after}),
        );
    }
    let mut seen = BTreeSet::new();
    let mut final_queries = Vec::new();
    for case in &cases {
        if seen.insert((&case.input, &case.context)) {
            let (_, result) = query(case, decoder, &writer, &mut model)?;
            final_queries
                .push(json!({"input": case.input, "context": case.context, "result": result}));
        }
    }
    writer.finish()?;
    let saved = fs::read(directory.join("learning.json")).or_else(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Ok(Vec::new())
        } else {
            Err(error)
        }
    })?;
    let mut writer = LearningWriter::start(&directory, true, |error| eprintln!("{error}"))?;
    seen.clear();
    let mut reloaded = Vec::new();
    for case in &cases {
        if seen.insert((&case.input, &case.context)) {
            let (_, result) = query(case, decoder, &writer, &mut model)?;
            reloaded.push(json!({"input": case.input, "context": case.context, "result": result}));
        }
    }
    writer.finish()?;
    let unchanged = match fs::read(directory.join("learning.json")) {
        Ok(bytes) => bytes == saved,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => saved.is_empty(),
        Err(error) => return Err(error.into()),
    };
    let rankings_unchanged = reloaded == final_queries;
    let report = json!({"format": "yanliu-learning-replay-v1", "state_dir": directory,
        "rows": rows, "reloaded": reloaded, "before_reload": final_queries,
        "reload_preserved_snapshot": unchanged, "reload_preserved_rankings": rankings_unchanged});
    report_file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"cases": cases.len(), "queries_after_reload": reloaded.len(), "reload_preserved_snapshot": unchanged, "reload_preserved_rankings": rankings_unchanged})
        )?
    );
    Ok(())
}
