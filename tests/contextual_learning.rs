use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use yanliu::learning::{Learning, LearningWriter};
use yanliu::model::Model;
use yanliu::{CandidateSource, Decoder, Dictionary, Session};

struct StateDirectory(PathBuf);
impl StateDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!(
                "contextual-path-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn snapshot(&self) -> serde_json::Value {
        serde_json::from_slice(&fs::read(self.0.join("learning.json")).unwrap()).unwrap()
    }
    fn writer(&self, contextual: bool) -> LearningWriter {
        LearningWriter::start(&self.0, contextual, |error| panic!("{error}")).unwrap()
    }
}
impl Drop for StateDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn decoder() -> Arc<Decoder> {
    Arc::new(Decoder::new(Dictionary::from_rime_reader(Cursor::new(
        "...\n现\txian\t1000\n西安\txi an\t100\n你\tni\t100\n泥\tni\t90\n好\thao\t100\n年\tnian\t90\n"
    )).unwrap()))
}

fn model() -> Model {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
    Model::load(&path, "cpu", None).unwrap()
}

fn menu(session: &Session) -> Vec<(String, usize, CandidateSource, u32)> {
    let mut rows: Vec<_> = session
        .candidates()
        .iter()
        .map(|candidate| {
            (
                candidate.text.clone(),
                candidate.consumed,
                candidate.source.clone(),
                candidate.score.to_bits(),
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    rows
}

fn promoted_session(
    decoder: &Arc<Decoder>,
    learning: &Arc<Learning>,
    model: &mut Model,
    input: &str,
    context: &str,
) -> Session {
    let mut session = Session::new(decoder.clone(), Some(learning.clone()));
    session.set_context(context);
    session.insert(input).unwrap();
    let before = menu(&session);
    let snapshot = session.snapshot();
    if let Some(index) = model.promotion(&snapshot).unwrap() {
        assert!(session.apply_promotion(snapshot.revision, index));
    }
    assert_eq!(
        menu(&session),
        before,
        "promotion changed candidate identity or static data"
    );
    session
}

fn choose(session: &mut Session, text: &str, consumed: usize) {
    let selected = session
        .candidates()
        .iter()
        .position(|candidate| candidate.text == text && candidate.consumed == consumed)
        .unwrap_or_else(|| panic!("{text} absent from {:?}", session.candidates()));
    assert_eq!(session.select(selected).as_deref(), Some(text));
}

#[test]
fn explicit_contextual_choices_change_real_session_order_and_survive_restart() {
    let directory = StateDirectory::new();
    let decoder = decoder();
    let mut model = model();
    let mut writer = directory.writer(true);
    let choices = [("乘火车前往", "西安"), ("此时此刻", "现")];
    for _ in 0..4 {
        for (context, text) in choices {
            let mut session =
                promoted_session(&decoder, &writer.learning, &mut model, "xian", context);
            choose(&mut session, text, 4);
            assert_eq!(session.input(), "");
        }
    }
    for (context, text) in choices {
        let session = promoted_session(&decoder, &writer.learning, &mut model, "xian", context);
        assert_eq!(
            session.candidates()[0].text,
            text,
            "context={context}; menu={:?}",
            session.candidates()
        );
    }
    writer.finish().unwrap();
    let snapshot = directory.snapshot();
    assert!(snapshot["updates"].as_u64().unwrap() > 0);
    let rows = snapshot["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row["input"], "xian");
        assert_eq!(row["count"], 4);
    }
    let persisted = serde_json::to_string(&snapshot).unwrap();
    for (context, _) in choices {
        assert!(!persisted.contains(context));
    }
    let mut writer = directory.writer(true);
    for (context, text) in choices {
        let session = promoted_session(&decoder, &writer.learning, &mut model, "xian", context);
        assert_eq!(
            session.candidates()[0].text,
            text,
            "after restart; context={context}"
        );
    }
    writer.finish().unwrap();
    assert_eq!(directory.snapshot(), snapshot);

    // Disabling contextual learning still retains ordinary explicit-choice counts,
    // but the stored adapter must not affect model promotion.
    let mut ordinary = directory.writer(false);
    for (context, _) in choices {
        let mut session = Session::new(decoder.clone(), Some(ordinary.learning.clone()));
        session.set_context(context);
        session.insert("xian").unwrap();
        assert_eq!(model.promotion(&session.snapshot()).unwrap(), None);
    }
    ordinary.finish().unwrap();
}

#[test]
fn partial_choice_learns_consumed_bytes_while_raw_and_completion_do_not_train() {
    let directory = StateDirectory::new();
    let decoder = decoder();
    let mut writer = directory.writer(true);
    let mut session = Session::new(decoder.clone(), Some(writer.learning.clone()));
    session.set_context("道路上的");
    session.insert("ni'hao").unwrap();
    choose(&mut session, "泥", 3);
    assert_eq!(session.input(), "hao");
    assert_eq!(session.commit_raw().as_deref(), Some("hao"));
    session.reset();
    session.insert("ni").unwrap();
    let completion = session
        .candidates()
        .iter()
        .position(|candidate| {
            candidate.text == "年" && candidate.source == CandidateSource::Completion
        })
        .unwrap();
    assert_eq!(session.select(completion).as_deref(), Some("年"));
    writer.finish().unwrap();
    let snapshot = directory.snapshot();
    let rows = snapshot["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "Enter raw input or completion was persisted");
    assert_eq!(
        rows[0],
        serde_json::json!({"input": "ni'", "text": "泥", "count": 1})
    );
    assert_eq!(snapshot["updates"], 1);
    let persisted = serde_json::to_string(&snapshot).unwrap();
    assert!(!persisted.contains("道路上的"));
    assert!(!persisted.contains("hao"));
}

#[test]
fn contextual_correction_can_overcome_a_large_stored_count_without_reversing_other_contexts() {
    let directory = StateDirectory::new();
    fs::write(
        directory.0.join("learning.json"),
        serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "rows": [{"input": "xian", "text": "西安", "count": 1000}],
            "adapter": null,
            "updates": 0,
        }))
        .unwrap(),
    )
    .unwrap();
    let decoder = decoder();
    let mut model = model();
    let mut writer = directory.writer(true);
    let travel = "乘火车前往";
    let now = "此时此刻";
    // The real LM also prefers 西安 here, so correcting 现 must overcome both priors.
    for context in [travel, now] {
        let session = promoted_session(&decoder, &writer.learning, &mut model, "xian", context);
        assert_eq!(
            session.candidates()[0].text,
            "西安",
            "the stored preference must initially be active"
        );
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
    }
    let mut first_adapted = None;
    for previous_choices in 0..8 {
        let mut session = promoted_session(&decoder, &writer.learning, &mut model, "xian", now);
        if session.candidates()[0].text == "现" && first_adapted.is_none() {
            first_adapted = Some(previous_choices);
        }
        choose(&mut session, "现", 4);
    }
    eprintln!(
        "stored count=1000; first observed contextual promotion after {first_adapted:?} choices (8 choices total)"
    );
    for (context, expected) in [(travel, "西安"), (now, "现")] {
        let session = promoted_session(&decoder, &writer.learning, &mut model, "xian", context);
        assert_eq!(
            session.candidates()[0].text,
            expected,
            "context={context}; menu={:?}",
            session.candidates()
        );
    }
    writer.finish().unwrap();
    let snapshot = directory.snapshot();
    let rows = snapshot["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows.iter().find(|row| row["text"] == "西安").unwrap()["count"],
        1000
    );
    assert_eq!(
        rows.iter().find(|row| row["text"] == "现").unwrap()["count"],
        8
    );
}
