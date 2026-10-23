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
        let path = std::env::temp_dir().join(format!(
            "yanliu-contextual-{}-{}",
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
    let path = std::env::var_os("YANLIU_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors")
        });
    Model::load(&path, "cpu", None).unwrap()
}

fn menu(candidates: &[yanliu::Candidate]) -> Vec<(String, usize, CandidateSource, u32)> {
    let mut rows: Vec<_> = candidates
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

fn refined_session(
    decoder: &Arc<Decoder>,
    learning: &Arc<Learning>,
    model: &mut Model,
    input: &str,
    context: &str,
) -> Session {
    let mut session = Session::new(decoder.clone(), Some(learning.clone()));
    session.set_context(context);
    session.insert(input).unwrap();
    let mut snapshot = session.snapshot();
    let promotion = model
        .refine(&mut snapshot, decoder, || true)
        .unwrap()
        .unwrap();
    for existing in session.candidates().iter().filter(|candidate| {
        matches!(
            candidate.source,
            CandidateSource::ExactDictionary | CandidateSource::Composed | CandidateSource::User
        )
    }) {
        assert!(
            snapshot.candidates.contains(existing),
            "refinement changed an existing conversion"
        );
    }
    let before = menu(&snapshot.candidates);
    assert!(session.apply_candidates(
        snapshot.revision,
        snapshot.candidates,
        promotion.index,
        &promotion.order
    ));
    assert_eq!(
        menu(session.candidates()),
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
                refined_session(&decoder, &writer.learning, &mut model, "xian", context);
            choose(&mut session, text, 4);
            assert_eq!(session.input(), "");
        }
    }
    for (context, text) in choices {
        let session = refined_session(&decoder, &writer.learning, &mut model, "xian", context);
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
        let session = refined_session(&decoder, &writer.learning, &mut model, "xian", context);
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
        let leader = session.candidates()[0].clone();
        let session = refined_session(&decoder, &ordinary.learning, &mut model, "xian", context);
        assert_eq!(session.candidates()[0], leader);
    }
    ordinary.finish().unwrap();
}

#[test]
fn explicit_composed_choices_learn_and_survive_restart() {
    let directory = StateDirectory::new();
    let decoder = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new(
            "...\n现\txian\t1000\n西\txi\t100\n安\tan\t100\n",
        ))
        .unwrap(),
    ));
    let mut model = model();
    let mut writer = directory.writer(true);
    let mut session = Session::new(decoder.clone(), Some(writer.learning.clone()));
    session.set_context("我去");
    session.insert("xian").unwrap();
    assert_eq!(session.candidates()[0].text, "现");
    assert_eq!(
        session.candidates()[0].source,
        CandidateSource::ExactDictionary
    );
    assert_eq!(
        session
            .candidates()
            .iter()
            .find(|candidate| candidate.text == "西安")
            .unwrap()
            .source,
        CandidateSource::Composed
    );
    choose(&mut session, "西安", 4);
    session = refined_session(&decoder, &writer.learning, &mut model, "xian", "我去");
    assert_eq!(session.candidates()[0].text, "西安");
    assert_eq!(session.candidates()[0].source, CandidateSource::User);
    let learned_menu = menu(session.candidates());
    writer.finish().unwrap();
    let saved = directory.snapshot();
    assert_eq!(
        saved["rows"],
        serde_json::json!([{"input": "xian", "text": "西安", "count": 1}])
    );
    assert!(saved["updates"].as_u64().unwrap() > 0);

    let mut writer = directory.writer(true);
    let session = refined_session(&decoder, &writer.learning, &mut model, "xian", "我去");
    assert_eq!(session.candidates()[0].text, "西安");
    assert_eq!(menu(session.candidates()), learned_menu);
    writer.finish().unwrap();
    assert_eq!(directory.snapshot(), saved);
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
        let session = refined_session(&decoder, &writer.learning, &mut model, "xian", context);
        assert_eq!(
            session.candidates()[0].text,
            "西安",
            "the stored preference must initially be active"
        );
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
    }
    let mut first_adapted = None;
    for previous_choices in 0..8 {
        let mut session = refined_session(&decoder, &writer.learning, &mut model, "xian", now);
        if session.candidates()[0].text == "现" && first_adapted.is_none() {
            first_adapted = Some(previous_choices);
        }
        choose(&mut session, "现", 4);
    }
    eprintln!(
        "stored count=1000; first observed contextual promotion after {first_adapted:?} choices (8 choices total)"
    );
    for (context, expected) in [(travel, "西安"), (now, "现")] {
        let session = refined_session(&decoder, &writer.learning, &mut model, "xian", context);
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

#[test]
fn contextual_habits_can_reverse_and_relearn_without_replacing_saved_parameters() {
    let directory = StateDirectory::new();
    let decoder = decoder();
    let mut model = model();
    let mut writer = directory.writer(true);
    let context = "同一处前文";
    for expected in ["西安", "现", "西安"] {
        for _ in 0..4 {
            let mut session =
                refined_session(&decoder, &writer.learning, &mut model, "xian", context);
            choose(&mut session, expected, 4);
        }
        let session = refined_session(&decoder, &writer.learning, &mut model, "xian", context);
        assert_eq!(
            session.candidates()[0].text,
            expected,
            "habit phase {expected}"
        );
    }
    writer.finish().unwrap();
    let snapshot = directory.snapshot();
    assert_eq!(snapshot["version"], 1);
    assert_eq!(snapshot["adapter"].as_array().unwrap().len(), 4096);
    let mut writer = directory.writer(true);
    let session = refined_session(&decoder, &writer.learning, &mut model, "xian", context);
    assert_eq!(session.candidates()[0].text, "西安");
    writer.finish().unwrap();
    assert_eq!(directory.snapshot(), snapshot);
}

#[test]
fn confirmed_words_are_recalled_when_public_candidates_change_and_survive_restart() {
    let directory = StateDirectory::new();
    let original = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new("...\n你\tni\t100\n泥\tni\t1\n")).unwrap(),
    ));
    let changed = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new("...\n你\tni\t100\n好\thao\t100\n")).unwrap(),
    ));
    assert!(
        changed
            .decode("ni")
            .iter()
            .all(|candidate| candidate.text != "泥")
    );
    let mut writer = directory.writer(false);
    let mut session = Session::new(original, Some(writer.learning.clone()));
    let mut other_engine = Session::new(changed.clone(), Some(writer.learning.clone()));
    session.insert("ni").unwrap();
    choose(&mut session, "泥", 2);

    let public = changed.decode("ni");
    other_engine.insert("ni").unwrap();
    assert_eq!(other_engine.candidates()[0].text, "泥");
    assert_eq!(other_engine.candidates()[0].source, CandidateSource::User);
    assert_eq!(other_engine.candidates()[0].consumed, 2);
    for candidate in &public {
        assert!(other_engine.candidates().contains(candidate));
    }
    assert_eq!(other_engine.candidates().len(), public.len() + 1);
    let before = menu(other_engine.candidates());
    writer.finish().unwrap();
    assert_eq!(
        directory.snapshot()["rows"],
        serde_json::json!([{"input": "ni", "text": "泥", "count": 1}])
    );

    let mut writer = directory.writer(false);
    let mut session = Session::new(changed.clone(), Some(writer.learning.clone()));
    session.insert("ni").unwrap();
    assert_eq!(menu(session.candidates()), before);
    session.reset();
    session.insert("nihao").unwrap();
    choose(&mut session, "泥", 2);
    assert_eq!(session.input(), "hao");
    assert_eq!(session.context(), "泥");
    writer.finish().unwrap();
    assert_eq!(directory.snapshot()["rows"][0]["count"], 2);

    let mut isolated = Session::new(changed, None);
    isolated.insert("ni").unwrap();
    assert!(
        isolated
            .candidates()
            .iter()
            .all(|candidate| candidate.text != "泥")
    );
}

#[test]
fn personal_prefix_recall_preserves_raw_case_and_apostrophe_consumption() {
    let directory = StateDirectory::new();
    let original = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new("...\n泥\tni\t10\n好\thao\t100\n")).unwrap(),
    ));
    let changed = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new("...\n你\tni\t100\n好\thao\t100\n")).unwrap(),
    ));
    let mut writer = directory.writer(true);
    let mut session = Session::new(original, Some(writer.learning.clone()));
    session.insert("NI'hao").unwrap();
    choose(&mut session, "泥", 3);
    assert_eq!(session.input(), "hao");
    writer.finish().unwrap();
    assert_eq!(
        directory.snapshot()["rows"],
        serde_json::json!([{"input": "NI'", "text": "泥", "count": 1}])
    );
    let mut writer = directory.writer(true);
    for input in ["NI'", "NI'hao"] {
        let mut session = Session::new(changed.clone(), Some(writer.learning.clone()));
        session.insert(input).unwrap();
        let candidate = session
            .candidates()
            .iter()
            .find(|candidate| candidate.text == "泥")
            .unwrap();
        assert_eq!(candidate.source, CandidateSource::User);
        assert_eq!(candidate.consumed, 3);
        choose(&mut session, "泥", 3);
        assert_eq!(session.input(), &input[3..]);
    }
    for input in ["NI", "ni'", "ni'hao", "NIhao"] {
        let mut session = Session::new(changed.clone(), Some(writer.learning.clone()));
        session.insert(input).unwrap();
        assert!(
            session
                .candidates()
                .iter()
                .all(|candidate| candidate.text != "泥"),
            "input={input}"
        );
    }
    writer.finish().unwrap();
}

#[test]
fn single_initial_choice_persists_without_leaking_into_longer_input_or_training_completions() {
    let decoder = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new(
            "...\n了\tle\t1000\n来\tlai\t100\n拉\tla\t90\n里\tli\t80\n梁\tliang\t70\n",
        ))
        .unwrap(),
    ));
    for contextual in [false, true] {
        let directory = StateDirectory::new();
        let mut writer = directory.writer(contextual);
        let mut session = Session::new(decoder.clone(), Some(writer.learning.clone()));
        session.set_context("我马上");
        session.insert("l").unwrap();
        assert_eq!(session.candidates()[0].text, "了");
        assert_eq!(
            session.candidates()[0].source,
            CandidateSource::ExactDictionary
        );
        choose(&mut session, "来", 1);
        session.reset();
        session.insert("li").unwrap();
        let completion = session
            .candidates()
            .iter()
            .position(|c| c.text == "梁")
            .unwrap();
        assert_eq!(
            session.candidates()[completion].source,
            CandidateSource::Completion
        );
        assert_eq!(session.select(completion).as_deref(), Some("梁"));
        session.reset();
        session.insert("l").unwrap();
        assert_eq!(session.commit_raw().as_deref(), Some("l"));
        writer.finish().unwrap();
        let saved = directory.snapshot();
        assert_eq!(
            saved["rows"],
            serde_json::json!([{"input":"l", "text":"来", "count":1}])
        );
        assert_eq!(saved["updates"], u64::from(contextual));

        let mut writer = directory.writer(contextual);
        let mut session = Session::new(decoder.clone(), Some(writer.learning.clone()));
        session.insert("l").unwrap();
        assert_eq!(session.candidates()[0].text, "来");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        for suffix in ["a", "i"] {
            session.insert(suffix).unwrap();
            let mut baseline = Session::new(decoder.clone(), None);
            baseline.insert(session.input()).unwrap();
            assert_eq!(
                menu(session.candidates()),
                menu(baseline.candidates()),
                "{}",
                session.input()
            );
            assert!(
                session
                    .candidates()
                    .iter()
                    .all(|c| c.consumed != 1 || c.source != CandidateSource::User)
            );
        }
        session.reset();
        session.insert("L").unwrap();
        assert_eq!(
            session.candidates()[0].text,
            "了",
            "raw case remains separate"
        );
        writer.finish().unwrap();
        assert_eq!(directory.snapshot(), saved);
    }
}

#[test]
fn personal_recall_requires_valid_input_even_when_the_public_pool_is_empty() {
    let directory = StateDirectory::new();
    let original = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new("...\n你好\tni hao\t100\n")).unwrap(),
    ));
    let changed = Arc::new(Decoder::new(
        Dictionary::from_rime_reader(Cursor::new("...\n字\tzi\t100\n")).unwrap(),
    ));
    assert!(changed.decode("nihao").is_empty());
    let mut writer = directory.writer(true);
    let mut session = Session::new(original, Some(writer.learning.clone()));
    session.insert("nihao").unwrap();
    choose(&mut session, "你好", 5);
    let mut session = Session::new(changed, Some(writer.learning.clone()));
    session.insert("nihao").unwrap();
    assert_eq!(session.candidates()[0].text, "你好");
    assert_eq!(session.candidates()[0].source, CandidateSource::User);
    for input in ["nihao''", "'nihao", "ni''hao", "nihao'''", "''", ""] {
        session.reset();
        for byte in input.bytes() {
            session.insert(&char::from(byte).to_string()).unwrap();
        }
        assert!(session.candidates().is_empty(), "{input}");
        assert_eq!(session.select(0), None, "{input}");
        assert_eq!(session.input(), input);
    }
    session.reset();
    session.insert("nihao").unwrap();
    assert_eq!(
        session.candidates()[0].text,
        "你好",
        "valid input recovers after invalid edits"
    );
    writer.finish().unwrap();
    assert_eq!(
        directory.snapshot()["rows"],
        serde_json::json!([{"input":"nihao", "text":"你好", "count":1}])
    );
}
